"""Torque-control runtime for the PiPER-X: bring your own control law.

This module is INFRASTRUCTURE, deliberately free of physics: it owns the MIT
session (mode entry, 200 Hz streaming thread, runaway watchdog, recovery into
position hold) and calls YOUR control law every tick. Gravity compensation,
impedance, and whatever you design next are all just laws:

    from piperx_teleop import TorqueSession, PiperModel

    mdl = PiperModel()

    def my_law(s):                       # s.q, s.qdot, s.t  (ArmState)
        return mdl.gravity_torque(s.q) + my_research_term(s.q, s.qdot)

    with TorqueSession(my_law) as sess:  # arm runs your controller
        ...

A law returns either a plain (6,) torque array, or a MitCommand to use the
firmware's full per-joint impedance interface (p_des/v_des/kp/kd/t_ff - the
PD then runs in the motor drivers at their own rate, above the CAN rate):

    def spring(s):
        return MitCommand(t_ff=mdl.gravity_torque(s.q),
                          p_des=q_ref, kp=KP, kd=KD)

Hard-won facts encapsulated here (do not relearn them on hardware):
  * stock piper_sdk misencodes the MIT frame for firmware S-V1.8-8+ - checked
    at construction, fails loudly
  * the firmware silently stays in MOVE_J if any joint rests past its soft
    range (a parked arm does) - entry is verified, range recovered first
  * a glitched feedback sample reads as an impossible velocity - the runaway
    watchdog requires sustained violation before tripping
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .arm import JOINT_LIMITS, PiperArm

RAD = np.pi / 180.0

# Runaway thresholds, rad/s per joint. Human dragging peaks ~3 rad/s at the
# wrist (measured); a fault sustains far beyond these.
VMAX = np.array([3.0, 3.0, 3.5, 6.0, 7.0, 7.0])
VMAX_TICKS = 4


def require_patched_sdk():
    """Raise (with the fix) if this piper_sdk speaks the pre-V1.8-8 MIT frame."""
    from piper_sdk.piper_msgs.msg_v2.transmit.arm_motion_ctrl_2 import ArmMsgMotionCtrl_2
    try:
        ArmMsgMotionCtrl_2(0x01, 0x06, 0, 0xAD)
    except ValueError:
        raise RuntimeError(
            "this piper_sdk misencodes the MIT frame for firmware S-V1.8-8+ "
            "(8-bit+CRC instead of 12-bit/no-CRC); the arm will silently "
            "ignore every torque command. Use the patched piper_sdk "
            "(piperctl env).") from None


@dataclass
class ArmState:
    """What a control law sees each tick."""
    q: np.ndarray        # joint positions, rad
    qdot: np.ndarray     # joint velocities, rad/s (finite-differenced)
    t: float             # seconds since the session went live


@dataclass
class MitCommand:
    """Full per-joint MIT command: tau = kp*(p_des-q) + kd*(v_des-qdot) + t_ff.

    Any field left None is sent as zeros; a law returning a bare array is
    equivalent to MitCommand(t_ff=array).
    """
    t_ff: np.ndarray = None
    p_des: np.ndarray = None
    v_des: np.ndarray = None
    kp: np.ndarray = None
    kd: np.ndarray = None

    def arrays(self):
        z = np.zeros(6)
        f = lambda x: z if x is None else np.asarray(x, float) * np.ones(6)
        return f(self.t_ff), f(self.p_des), f(self.v_des), f(self.kp), f(self.kd)


class TorqueSession:
    """Runs a control law on the arm in MIT (torque) mode.

    Parameters
    ----------
    law : callable(ArmState) -> ndarray | MitCommand
        The controller. Called at `hz` from the session thread.
    arm : PiperArm, optional
        Connected arm to use (lets sessions share a connection); else `can`.
    """

    def __init__(self, law, arm=None, can="can0", hz=200.0):
        require_patched_sdk()
        self.law = law
        self.arm = arm if arm is not None else PiperArm(can).connect(require_control=False)
        self.piper = self.arm.piper
        self.hz = float(hz)
        self.trip = None
        self._thread = None
        self._stop = threading.Event()
        self._recovered = threading.Event()

    # ---------- state (thread-safe: SDK getters are mutex-guarded) ----------

    def q(self):
        return self.arm.q()

    def gripper(self):
        return self.arm.gripper()[0]

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    # ---------- lifecycle ----------

    def start(self):
        if self.running:
            return self
        self._ensure_ready()
        if not self._enter_mit():
            raise RuntimeError("firmware refused MIT mode (mode_feed=%s) - "
                               "joint out of range or arm error state"
                               % self.piper.GetArmStatus().arm_status.mode_feed)
        self.trip = None
        self._stop.clear()
        self._recovered.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self, hold=True):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if hold and not self._recovered.is_set():
            self._recover(self.q())

    def run(self, duration=0.0):
        self.start()
        t0 = time.time()
        try:
            while self.running and not (duration and time.time() - t0 > duration):
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        self.stop()
        return self.trip

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---------- internals ----------

    def _ensure_ready(self):
        if not all(self.arm.is_enabled()):
            self.piper.EnableArm(7)
            time.sleep(1.5)
        q = self.q()
        margin = np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q)
        if margin.min() < np.radians(2):
            safe = np.clip(q, JOINT_LIMITS[:, 0] + np.radians(6),
                           JOINT_LIMITS[:, 1] - np.radians(6))
            self._goto_position(safe, secs=2.5)

    def _goto_position(self, q, secs=2.0):
        self.piper.MotionCtrl_2(0x01, 0x01, 15, 0x00, 0, 0x01)
        time.sleep(0.05)
        q0, t0 = self.q(), time.time()
        while time.time() - t0 < secs + 0.3:
            f = min((time.time() - t0) / secs, 1.0)
            self.arm.move_j(q0 + f * (q - q0), speed_pct=15)
            time.sleep(0.01)

    def _enter_mit(self, timeout=2.5):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.piper.MotionCtrl_2(0x01, 0x06, 0, 0xAD, 0, 0x01)
            time.sleep(0.1)
            if "0x6" in str(self.piper.GetArmStatus().arm_status.mode_feed):
                return True
        return False

    def _loop(self):
        try:
            self._loop_body()
        except Exception as e:
            self.trip = "loop error: %r" % (e,)   # never die silently
            try:
                self._recover(self.q())
            except Exception:
                pass

    def _loop_body(self):
        period = 1.0 / self.hz
        q_prev, t_prev = self.q(), time.time()
        t_start = t_prev
        over = 0
        while not self._stop.is_set():
            tick = time.time()
            q = self.q()
            qdot = (q - q_prev) / max(tick - t_prev, 1e-3)
            q_prev, t_prev = q, tick

            out = self.law(ArmState(q=q, qdot=qdot, t=tick - t_start))
            cmd = out if isinstance(out, MitCommand) else MitCommand(t_ff=out)
            t_ff, p_des, v_des, kp, kd = cmd.arrays()
            for j in range(6):
                self.piper.JointMitCtrl(j + 1, float(p_des[j]), float(v_des[j]),
                                        float(kp[j]), float(kd[j]), float(t_ff[j]))

            if (np.abs(qdot) > VMAX).any():
                over += 1
                if over >= VMAX_TICKS:
                    j = int((np.abs(qdot) / VMAX).argmax())
                    self.trip = "J%d %.1f rad/s sustained" % (j + 1, abs(qdot[j]))
                    self._recover(self.q())
                    return
            else:
                over = 0
            elapsed = time.time() - tick
            if elapsed < period:
                time.sleep(period - elapsed)

    def _recover(self, q):
        """AgileX's own exit path: MIT PD catch, then the position loop."""
        self._recovered.set()
        t0 = time.time()
        while time.time() - t0 < 0.8:
            for j in range(6):
                self.piper.JointMitCtrl(j + 1, float(q[j]), 0.0, 10.0, 0.8, 0.0)
            time.sleep(0.005)
        self.piper.MotionCtrl_2(0x01, 0x01, 20, 0x00, 0, 0x01)
        time.sleep(0.05)
        for _ in range(30):
            self.arm.move_j(q, speed_pct=20)
            time.sleep(0.01)
