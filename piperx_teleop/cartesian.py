"""Cartesian teleop sink: anchored operator displacement -> arm end pose.

Drives the arm through the firmware's own Cartesian interface, in the firmware's
own frame.  We never convert into a URDF frame, which matters: the vendor URDF
does not match this hardware (best rigid fit leaves 51 mm mean / 107 mm max
residual), so any model-based route would inherit that error.  Reading the
firmware's pose and adding a delta to it sidesteps the question entirely.

Two design choices are load-bearing, both learned by getting them wrong:

ANCHORED mapping.  The goal pose is recomputed every tick from the operator's
TOTAL displacement since the clutch engaged, not accumulated from per-tick
deltas.  Integrating deltas and then clamping the target permanently deletes
every millimetre the rate limit clips - measured: a 150 mm hand motion produced
20 mm of arm motion, losing 87%.  Anchoring makes the rate limit govern only how
fast the command chases the goal, so nothing is lost and dropped samples heal.

The command LEASH.  The command may lead the arm's actual pose by only
`lead_limit`.  Without it, a stall against a joint limit lets the command run
away, and the operator must then retrace the whole gap before anything responds
- which feels like total unresponsiveness in every direction, not just the
blocked one.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .arm import JOINT_LIMITS, MOVE_P
from .filters import make_filter

# Identified on hardware by single-joint rotation self-consistency: rotating one
# wrist joint must produce rotation about a constant axis through that joint's
# own angle.  Extrinsic "xyz" with angles (RX, RY, RZ) reproduced that with
# 0.00 deg angle error; the next-best convention was 16 deg off.
EULER_SEQ = "xyz"
# Extrinsic xyz is singular at RY = +/-90, where RX and RZ stop being separable.
GIMBAL_WARN_DEG = 80.0


@dataclass
class TeleopState:
    """One control tick: what was commanded, what was measured, and diagnostics.

    ACTION      target, rpy, gripper_closed   - what we asked the arm for
    OBSERVATION q, dq, effort, actual, actual_rpy, gripper_pos
    EPISODE     t, clutch                     - clutch segments the episode
    DIAGNOSTIC  lag, clamps, notes, aborted

    Use `.action()` and `.observation()` rather than picking fields by hand, so
    a change here does not silently reshape your dataset.

    Note there is no commanded JOINT target: the firmware runs its own IK and
    does not report the joint solution it chose.  For policies that want joint
    actions, the usual substitute is the next tick's measured `q`.
    """
    # Wall clock stamped on entry to follow()/idle(), i.e. immediately before
    # this tick's action is computed and sent - so it is the COMMAND time.
    # Measured on hardware, everything else in the tick sits within 0.25 ms of
    # it (input poll 0.002 ms before, state assembled 0.075 ms after), so it is
    # a sound anchor for the whole tick.  The one real offset is obs_age.
    # Use this to line up with cameras and anything else outside this process.
    t: float = 0.0
    # Monotonic clock. Use this for intervals, rates and jitter: wall clock can
    # step under NTP and would silently corrupt them.
    t_mono: float = 0.0
    # When the observation was actually SAMPLED - the SDK's arrival stamp on the
    # joint-feedback frame, not when we read it.  Precedes `t`, because the CAN
    # frame arrived before we acted on it.
    obs_t: float = 0.0
    # Age of that observation at tick start; spikes mean the CAN stream stalled.
    obs_age: float = 0.0
    clutch: bool = False
    # action
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper_closed: bool = False
    # The operator's intent BEFORE rate limiting, box/floor clamping and the
    # leash.  `target` is what the arm was actually told and replays faithfully;
    # `goal` is what the operator asked for.  They diverge when the arm is
    # stalled or you move faster than the rate limit, which is exactly when the
    # demonstration and the intent disagree.
    goal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # observation
    q: np.ndarray = field(default_factory=lambda: np.zeros(6))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(6))
    effort: np.ndarray = field(default_factory=lambda: np.zeros(6))
    actual: np.ndarray = field(default_factory=lambda: np.zeros(3))
    actual_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper_pos: float = 0.0
    # diagnostics
    lag: float = 0.0
    clamps: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    aborted: str = None

    def action(self):
        """Commanded end-effector pose + gripper, as a flat 7-vector."""
        return np.concatenate([self.target, self.rpy, [1.0 if self.gripper_closed else 0.0]])

    def observation(self):
        """Proprioception, as a dict of arrays."""
        return dict(q=self.q, dq=self.dq, effort=self.effort,
                    ee_pos=self.actual, ee_rpy=self.actual_rpy,
                    gripper_pos=self.gripper_pos)

    def intent(self):
        """Unclamped commanded pose + gripper, as a flat 7-vector."""
        return np.concatenate([self.goal, self.rpy, [1.0 if self.gripper_closed else 0.0]])

    ACTION_DIM = 7
    ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")


class CartesianTeleop:
    def __init__(self, arm, config=None, move_mode=MOVE_P):
        from .config import Config
        cfg = config or Config()
        self.arm = arm
        self.cfg = cfg
        self.move_mode = move_mode

        # Both gains are applied HERE, not in the sources, so every source is
        # scaled identically. Sources emit raw operator displacement.
        self.gain = cfg.motion.gain
        self.rot_gain = cfg.rotation.gain
        # Smooth the operator's displacement before it becomes a goal, so the
        # rate limit and leash act on a clean signal rather than on tremor.
        f = cfg.filter
        self._pos_filter = make_filter(f.kind, min_cutoff=f.min_cutoff, beta=f.beta,
                                       process_var=f.process_var, meas_var=f.meas_var)
        self._rot_filter = make_filter(f.kind, min_cutoff=f.rot_min_cutoff, beta=f.rot_beta,
                                       process_var=f.rot_process_var, meas_var=f.rot_meas_var)
        self._last_filter_t = None
        self.max_step = cfg.motion.max_step
        self.speed_pct = cfg.motion.speed
        self.max_reach = cfg.workspace.max_reach
        self.min_z = cfg.workspace.min_z
        self.lock_rotation = not cfg.rotation.unlock
        self.max_rot_step = np.radians(cfg.rotation.max_step)
        self.max_joint_step = np.radians(cfg.safety.max_joint_step)
        self.lead_limit = cfg.safety.lead_limit

        self.origin = None
        self.anchor_pos = None
        self.anchor_rot = None
        self.cmd_pos = None
        self.cmd_rot = None
        self.target = None
        self.rot = None

        self.aborted = None
        self.notes = []
        self._announced = set()
        self._last_q = None
        self.clamp_step = self.clamp_box = self.clamp_floor = self.clamp_rot = 0
        self._gimbal_warned = False
        self.gripper_closed = False

    # ---------- anchoring ----------

    def start(self):
        pos, rot = self.arm.end_pose()
        self.origin = pos.copy()
        self._set_anchor(pos, rot)
        self._last_q = self.arm.q()
        self.aborted = None
        return self

    def relatch(self):
        """Re-anchor to where the arm is now. Call on every clutch engage."""
        pos, rot = self.arm.end_pose()
        self._set_anchor(pos, rot)
        self._last_q = self.arm.q()

    def _set_anchor(self, pos, rot):
        self.anchor_pos = pos.copy()
        self.anchor_rot = rot.copy()
        self.cmd_pos = pos.copy()
        self.cmd_rot = rot.copy()
        self.target = pos.copy()
        self.rot = rot.copy()

    # ---------- guards ----------

    def _announce(self, key, msg):
        if key not in self._announced:
            self._announced.add(key)
            self.notes.append(msg)

    def check_joints(self):
        """False (and sets .aborted) if the arm jumped, e.g. an IK branch flip."""
        q = self.arm.q()
        margin = np.degrees(np.minimum(q - JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1] - q))
        for j in np.where(margin < 3.0)[0]:
            self._announce("limit%d" % j,
                           "JOINT LIMIT: J%d is %.1f deg from its end of travel. The "
                           "firmware's IK has run out of room this way; re-clutching will "
                           "not help. Move back the way you came." % (j + 1, margin[j]))
        if self._last_q is not None:
            jump = np.abs(q - self._last_q)
            if jump.max() > self.max_joint_step:
                self.aborted = ("J%d jumped %.1f deg in one cycle - likely a firmware IK "
                                "branch flip or a singularity"
                                % (int(jump.argmax()) + 1, np.degrees(jump.max())))
                return False
        self._last_q = q
        return True

    def hold(self):
        self.arm.hold()

    # ---------- the control law ----------

    def follow(self, disp_pos, disp_rotvec=None, dry_run=False):
        """Advance toward the goal implied by the operator's total displacement.

        `disp_pos` and `disp_rotvec` are RAW operator displacement since the
        clutch engaged; motion.gain and rotation.gain are applied here, so every
        source is scaled the same way.  A source already emitting robot metres
        should run with gain 1.0.
        """
        tick = (time.time(), time.monotonic())
        if self.aborted:
            return self.state(clutch=True, t=tick)

        disp_pos = np.asarray(disp_pos, float)
        if self._pos_filter is not None:
            now = tick[1]
            dt = 0.01 if self._last_filter_t is None else max(now - self._last_filter_t, 1e-4)
            self._last_filter_t = now
            disp_pos = self._pos_filter(disp_pos, dt)
            if disp_rotvec is not None:
                disp_rotvec = self._rot_filter(np.asarray(disp_rotvec, float), dt)
        goal = self.anchor_pos + disp_pos * self.gain
        raw_goal = goal.copy()
        clipped = np.clip(goal, self.origin - self.max_reach, self.origin + self.max_reach)
        if np.any(clipped != goal):
            self.clamp_box += 1
            self._announce("box", "WORKSPACE BOX limit (+/-%.2f m from where the session "
                                  "started). Release the clutch, move back, re-clutch."
                           % self.max_reach)
            goal = clipped
        if goal[2] < self.min_z:
            goal[2] = self.min_z
            self.clamp_floor += 1
            self._announce("floor", "Z FLOOR reached (%.3f m); cannot go lower." % self.min_z)

        step = goal - self.cmd_pos
        n = np.linalg.norm(step)
        if n > self.max_step:
            step = step / n * self.max_step
            self.clamp_step += 1
        self.cmd_pos = self.cmd_pos + step

        actual, actual_rpy = self.arm.end_pose()
        lead = self.cmd_pos - actual
        ln = np.linalg.norm(lead)
        if ln > self.lead_limit:
            self.cmd_pos = actual + lead / ln * self.lead_limit
            self._announce("stall",
                           "ARM STALLED: not reaching the commanded pose (held %.0f mm "
                           "behind). Usually a joint limit - see any JOINT LIMIT note."
                           % (self.lead_limit * 1000))
        self.target = self.cmd_pos

        if disp_rotvec is not None and not self.lock_rotation:
            self._advance_rotation(np.asarray(disp_rotvec, float) * self.rot_gain)

        if not dry_run:
            self.arm.end_pose_ctrl(self.target, self.rot, self.move_mode, self.speed_pct)
        return self.state(clutch=True, actual=actual, actual_rpy=actual_rpy,
                           goal=raw_goal, t=tick)

    def _advance_rotation(self, disp_rotvec):
        R_goal = Rot.from_rotvec(disp_rotvec) * Rot.from_euler(EULER_SEQ, self.anchor_rot,
                                                               degrees=True)
        R_cmd = Rot.from_euler(EULER_SEQ, self.cmd_rot, degrees=True)
        err = (R_goal * R_cmd.inv()).as_rotvec()
        m = np.linalg.norm(err)
        if m > self.max_rot_step:
            err = err / m * self.max_rot_step
            self.clamp_rot += 1
        self.cmd_rot = (Rot.from_rotvec(err) * R_cmd).as_euler(EULER_SEQ, degrees=True)
        self.rot = self.cmd_rot
        if abs(self.rot[1]) > GIMBAL_WARN_DEG and not self._gimbal_warned:
            self._gimbal_warned = True
            self._announce("gimbal", "NEAR GIMBAL LOCK (RY=%.0f deg): orientation control "
                                     "is ill-conditioned here." % self.rot[1])

    def set_gripper(self, closed):
        self.gripper_closed = bool(closed)
        self.arm.set_gripper(self.gripper_closed)

    # ---------- reporting ----------

    def state(self, clutch=False, actual=None, actual_rpy=None, goal=None, t=None):
        if actual is None:
            actual, actual_rpy = self.arm.end_pose()
        notes, self.notes = self.notes, []
        try:
            grip_pos, _ = self.arm.gripper()
        except Exception:
            grip_pos = 0.0
        t_wall, t_mono = t if t is not None else (time.time(), time.monotonic())
        try:
            obs_t = self.arm.obs_time()
        except Exception:
            obs_t = t_wall
        return TeleopState(
            t=t_wall, t_mono=t_mono, obs_t=obs_t, obs_age=max(0.0, t_wall - obs_t),
            clutch=clutch,
            target=self.target.copy(), rpy=np.asarray(self.rot).copy(),
            goal=(self.target.copy() if goal is None else np.asarray(goal).copy()),
            actual=actual.copy(), actual_rpy=np.asarray(actual_rpy).copy(),
            q=self.arm.q(), dq=self.arm.dq(), effort=self.arm.effort(),
            gripper_pos=float(grip_pos),
            gripper_closed=self.gripper_closed,
            lag=float(np.linalg.norm(self.target - actual)),
            clamps=dict(step=self.clamp_step, box=self.clamp_box,
                        floor=self.clamp_floor, rot=self.clamp_rot),
            notes=notes, aborted=self.aborted)

    def idle(self):
        """One tick with the clutch open: no motion, but state for logging."""
        return self.state(clutch=False, t=(time.time(), time.monotonic()))
