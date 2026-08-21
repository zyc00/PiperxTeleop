"""Unit-safe wrapper around the Piper SDK.

Everything crossing this boundary is SI: radians, rad/s, N.m, metres, and
degrees only where the firmware itself uses them (end-pose orientation).
"""

import time

import numpy as np
from piper_sdk import C_PiperInterface_V2

RAD2CMD = 1000.0 * 180.0 / np.pi       # rad -> SDK 0.001 deg
CMD2RAD = 1.0 / RAD2CMD

# URDF joint limits, applied as a clamp on every outgoing joint command.
# PiPER-X limits (agx_arm_urdf piper_x_description.urdf). The plain piper's
# table (J4 +/-100 deg, J5 +/-70 deg) is a different robot: on this arm J4 and
# J5 are both +/-89 deg - the firmware clamps there itself.
JOINT_LIMITS = np.array([
    [-2.618, 2.618],
    [0.0, 3.14],
    [-2.967, 0.0],
    [-1.553, 1.553],
    [-1.553, 1.553],
    [-2.0944, 2.0944],
])

MOVE_P, MOVE_J, MOVE_L = 0x00, 0x01, 0x02


class TeachModeError(RuntimeError):
    """The arm is in drag-teach mode and is ignoring every command.

    Entered with the button on the arm (or by hand-dragging it).  The firmware
    will NOT leave it on request - MotionCtrl_1 with grag_teach_ctrl 0x00, 0x02
    and 0x06, and track_ctrl 0x04, are all accepted and ignored.  The only ways
    out are the button again, a power cycle, or ResetPiper (which de-powers the
    arm and drops it).  While in this mode JointCtrl and EndPoseCtrl are
    silently discarded, which looks exactly like broken motion control.
    """


class PiperArm:
    def __init__(self, can_name="can0"):
        self.can_name = can_name
        self.piper = C_PiperInterface_V2(can_name)
        self._connected = False

    # ---------- lifecycle ----------

    def connect(self, settle=0.3, require_control=True):
        self.piper.ConnectPort()
        time.sleep(settle)
        self._connected = True
        if require_control and self.in_teach_mode():
            raise TeachModeError(
                "arm on %s is in %s and will ignore all commands; press the "
                "teach button on the arm or power-cycle it"
                % (self.can_name, self.control_mode()))
        return self

    def close(self):
        try:
            self.piper.DisconnectPort()
        except Exception:
            pass
        self._connected = False

    def enable(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.piper.EnablePiper():
                return True
            time.sleep(0.01)
        return all(self.is_enabled())

    def disable(self):
        """De-energise arm motors AND gripper. THE ARM WILL FALL."""
        for _ in range(10):
            self.piper.GripperCtrl(0, 0, 0x00, 0)
            time.sleep(0.01)
        self.piper.DisableArm(7)

    def is_enabled(self):
        return list(self.piper.GetArmEnableStatus())

    # ---------- status ----------

    def control_mode(self):
        return str(self.piper.GetArmStatus().arm_status.ctrl_mode)

    def in_teach_mode(self):
        return "TEACHING" in self.control_mode()

    def status(self):
        return self.piper.GetArmStatus().arm_status

    # ---------- state ----------

    def q(self):
        j = self.piper.GetArmJointMsgs().joint_state
        return np.array([j.joint_1, j.joint_2, j.joint_3,
                         j.joint_4, j.joint_5, j.joint_6]) * CMD2RAD

    def _motors(self):
        h = self.piper.GetArmHighSpdInfoMsgs()
        return [h.motor_1, h.motor_2, h.motor_3, h.motor_4, h.motor_5, h.motor_6]

    def dq(self):
        return np.array([m.motor_speed for m in self._motors()]) * 1e-3

    def effort(self):
        """Joint torque from motor current (SDK's fixed coefficient, N.m).

        Note the SDK applies one coefficient per joint group; measured against a
        gravity model the true constants differ per joint, so treat this as
        relative rather than absolute.
        """
        return np.array([m.effort for m in self._motors()]) * 1e-3

    def end_pose(self):
        """Firmware end pose: (position m, orientation deg as extrinsic xyz)."""
        e = self.piper.GetArmEndPoseMsgs().end_pose
        return (np.array([e.X_axis, e.Y_axis, e.Z_axis]) * 1e-6,
                np.array([e.RX_axis, e.RY_axis, e.RZ_axis]) * 1e-3)

    def gripper(self):
        g = self.piper.GetArmGripperMsgs().gripper_state
        return g.grippers_angle * 1e-6, g.grippers_effort * 1e-3

    def obs_time(self):
        """Wall-clock time the most recent joint-feedback frame arrived.

        The SDK stamps each parsed CAN frame, so this is when the observation was
        actually sampled - not when we got round to reading it.
        """
        return float(self.piper.GetArmJointMsgs().time_stamp)

    def stream_ages(self, now=None):
        """Age in seconds of each feedback stream, for spotting a stalled bus."""
        now = time.time() if now is None else now
        return dict(
            joint=now - float(self.piper.GetArmJointMsgs().time_stamp),
            motor=now - float(self.piper.GetArmHighSpdInfoMsgs().time_stamp),
            gripper=now - float(self.piper.GetArmGripperMsgs().time_stamp),
        )

    def stream_rates(self):
        return dict(joint=float(self.piper.GetArmJointMsgs().Hz),
                    motor=float(self.piper.GetArmHighSpdInfoMsgs().Hz),
                    gripper=float(self.piper.GetArmGripperMsgs().Hz))

    def limit_margin(self):
        """Degrees of remaining travel on each joint."""
        q = self.q()
        return np.degrees(np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q))

    # ---------- control ----------

    def clamp(self, q):
        return np.clip(np.asarray(q, float), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    def _install_pos(self):
        """installation_pos for MotionCtrl_2: assert 0x01 (upright) exactly
        ONCE, on the first commanded motion. Streaming it at the 100 Hz
        command rate makes the firmware re-seed IK/gravity state mid-motion
        (measured: J1 branch-flip jumps during Cartesian teleop that vanish
        on the pre-0.2.0 package). 0x00 afterwards means "no change"."""
        if getattr(self, "_install_sent", False):
            return 0x00
        self._install_sent = True
        return 0x01

    def move_j(self, q, speed_pct=20):
        q = self.clamp(q)
        self.piper.MotionCtrl_2(0x01, MOVE_J, int(speed_pct), 0x00, 0, self._install_pos())
        self.piper.JointCtrl(*[int(round(v * RAD2CMD)) for v in q])

    def move_to(self, q, speed_pct=15, tol_deg=1.2, timeout=20.0):
        """Blocking joint move. Returns True if the pose was reached."""
        q = self.clamp(q)
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.move_j(q, speed_pct)
            if np.abs(self.q() - q).max() < np.radians(tol_deg):
                return True
            time.sleep(0.01)
        return False

    def end_pose_ctrl(self, pos, rpy, move_mode=MOVE_P, speed_pct=20):
        self.piper.MotionCtrl_2(0x01, move_mode, int(speed_pct), 0x00, 0, self._install_pos())
        self.piper.EndPoseCtrl(int(round(pos[0] * 1e6)), int(round(pos[1] * 1e6)),
                               int(round(pos[2] * 1e6)), int(round(rpy[0] * 1e3)),
                               int(round(rpy[1] * 1e3)), int(round(rpy[2] * 1e3)))

    def set_gripper(self, closed, effort=1000, opening_m=0.07):
        angle = 0 if closed else int(opening_m * 1e6)
        self.piper.GripperCtrl(abs(angle), int(effort), 0x01, 0)

    def open_gripper(self, opening_m=0.07):
        self.set_gripper(False, opening_m=opening_m)

    def hold(self, seconds=0.3, speed_pct=10):
        """Pin the arm where it stands, in joint space."""
        q = self.q()
        t0 = time.time()
        while time.time() - t0 < seconds:
            self.move_j(q, speed_pct)
            time.sleep(0.01)
