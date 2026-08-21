"""Gravity compensation: torque-mode compliance, the arm becomes draggable.

The control law is AgileX's reference architecture (their
agilex-arm-gravity-compensation repo), reduced to its essence:

    tau = gravity(q)                       # rigid-body model, piper_x + gripper
    JointMitCtrl(j, 0, 0, 0, 0, tau[j])    # torque only: kp = 0, kd = 0

Three hard-won facts live in this module so no caller rediscovers them:
  * firmware S-V1.8-8+ uses a 12-bit / no-CRC MIT frame that stock piper_sdk
    misencodes - checked at construction with a clear error, never silently
  * the model must be the PiPER-X one (firmware FK matches it to 0.01 mm);
    the plain-piper model makes J4 fight a phantom and J5 run away
  * MIT entry must be VERIFIED: the firmware silently stays in MOVE_J when a
    joint rests past its soft limit, which a parked arm does naturally

Blocking use:

    from piperx_teleop import GravityCompensator
    GravityCompensator().run()                     # drag until Ctrl-C

Background use (recording, experiments):

    with GravityCompensator() as gc:               # arm goes compliant
        while recording:
            log(gc.q(), gc.gripper())
    # position hold restored; gc.trip explains any abnormal ending

Leader-follower teleoperation (the design driver for this module):

    leader = GravityCompensator(can="can0")        # human drags this arm
    follower = PiperArm("can1").connect()
    with leader:
        while True:
            follower.move_j(leader.q())            # follower mirrors, ~100 Hz
            time.sleep(0.01)

Carrying a payload: pass payload_mass / payload_com (kg, metres in the
gripper_base frame) and the model compensates it too.
"""

import threading
import time

import numpy as np

from .arm import JOINT_LIMITS, PiperArm
from .model import PiperModel

RAD = np.pi / 180.0

# Runaway thresholds, rad/s per joint. Human dragging peaks ~3 rad/s at the
# wrist (measured mid-session); a fault sustains far beyond these.
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
            "(8-bit+CRC instead of 12-bit/no-CRC) and the arm will silently "
            "ignore every torque command. Install the patched piper_sdk "
            "(see the piperctl env) before using GravityCompensator.") from None


class GravityCompensator:
    """Torque-mode gravity compensation on a PiperArm.

    Parameters
    ----------
    arm : PiperArm, optional
        An already-connected arm to make compliant (lets a teleop session and
        the compensator share one connection). If omitted, connects to `can`.
    can : str
        CAN interface, used only when `arm` is None.
    hz : float
        Torque streaming rate.
    payload_mass, payload_com :
        Extra rigid payload on the gripper, added to the gravity model.
    """

    def __init__(self, arm=None, can="can0", model=None, hz=200.0,
                 payload_mass=0.0, payload_com=(0.0, 0.0, 0.0)):
        require_patched_sdk()
        self.arm = arm if arm is not None else PiperArm(can).connect(require_control=False)
        self.piper = self.arm.piper
        self.mdl = model or PiperModel(payload_mass=payload_mass,
                                       payload_com=payload_com)
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

    def gravity(self, q=None):
        return self.mdl.gravity_torque(self.q() if q is None else q)

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    # ---------- lifecycle ----------

    def start(self):
        """Go compliant: enter MIT and stream gravity torque in a thread."""
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
        """Leave compliance; by default freeze in position hold where it stands."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if hold and not self._recovered.is_set():
            self._recover(self.q())

    def run(self, duration=0.0):
        """Blocking convenience: compliant until Ctrl-C (or `duration` seconds)."""
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
        """Enable motors; nudge joints resting past their soft range back in
        (a parked Piper settles just outside J2/J3's limits, and the firmware
        refuses MIT while any joint is out of range)."""
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
            # a library thread must never die silently
            self.trip = "loop error: %r" % (e,)
            try:
                self._recover(self.q())
            except Exception:
                pass

    def _loop_body(self):
        period = 1.0 / self.hz
        q_prev, t_prev = self.q(), time.time()
        over = 0
        while not self._stop.is_set():
            tick = time.time()
            q = self.q()
            qdot = (q - q_prev) / max(tick - t_prev, 1e-3)
            q_prev, t_prev = q, tick

            tau = self.mdl.gravity_torque(q)
            for j in range(6):
                self.piper.JointMitCtrl(j + 1, 0.0, 0.0, 0.0, 0.0, float(tau[j]))

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
