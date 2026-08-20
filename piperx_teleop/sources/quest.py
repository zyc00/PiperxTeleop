"""Quest 3 controller source. Requires the optional `quest` extra (robovr).

Frames
------
Quest/OpenXR world is +X right, +Y up, -Z forward and gravity-aligned; the robot
base is +X forward, +Y left, +Z up.  Its YAW, however, is arbitrary - fixed when
tracking starts - so the frame must be aimed at the operator.  Deriving that aim
from the headset is unreliable: set down it points wherever it was placed, and
worn it follows the operator's gaze rather than their body.  Prefer measuring it
once from a deliberate forward reach (`calibrate_forward`) and passing the
resulting heading.

Tracking
--------
Quest 3 controllers have no tracking ring; they are tracked optically by the
headset's cameras.  Out of view the IMU dead-reckons a smooth, `valid=true`,
full-rate pose that does not follow the hand - measured once at 85 mm reported
for ~250 mm of real motion.  It fails silently, so `tracking_suspect()` is
provided and the session warns on it.
"""

import time
from collections import deque

import numpy as np

from .base import TeleopSample

POSITION_VALID_BIT = 0x2


def _norm(q):
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    return np.array([0.0, 0.0, 0.0, 1.0]) if n < 1e-12 else q / n


def _conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def _rotvec(q):
    q = _norm(q)
    if q[3] < 0:
        q = -q
    v = q[:3]
    s = np.linalg.norm(v)
    return np.zeros(3) if s < 1e-9 else v / s * (2.0 * np.arctan2(s, q[3]))


class QuestSource:
    def __init__(self, config=None, heading_deg=0.0, server=None):
        from ..config import Config
        cfg = config or Config()
        self.cfg = cfg.quest
        self.rot = cfg.rotation
        self._server = server
        self._owns = server is None
        self.set_heading(heading_deg)
        self._anchor_pos = None
        self._anchor_quat = None
        self._was_clutched = False
        self._trigger_was = False
        self.gripper_closed = False
        self._hist = deque(maxlen=400)

    # ---------- frame ----------

    def set_heading(self, deg):
        """Aim the frame along the operator's forward, as a compass heading."""
        a = np.radians(float(deg))
        c, s = np.cos(a), np.sin(a)
        self._right = np.array([c, 0.0, -s])
        self._up = np.array([0.0, 1.0, 0.0])
        self._forward = np.array([-s, 0.0, -c])
        self.heading_deg = float(deg)

    def _to_robot(self, v):
        local = np.array([v @ self._right, v @ self._up, v @ self._forward])
        return np.array([local[2], -local[0], local[1]])

    # ---------- lifecycle ----------

    def start(self):
        if self._server is None:
            try:
                from robovr.quest3 import Quest3Server
            except ImportError as exc:
                raise ImportError(
                    "QuestSource needs the 'quest' extra: pip install piperx-teleop[quest]"
                ) from exc
            self._server = Quest3Server(host=self.cfg.host, port=self.cfg.port,
                                        adb_reverse=self.cfg.adb_reverse)
        start = getattr(self._server, "start", None)
        if callable(start):
            start()
        return self

    def stop(self):
        if self._owns and self._server is not None:
            close = getattr(self._server, "close", None)
            if callable(close):
                close()

    def wait_connected(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.poll().connected:
                return True
            time.sleep(0.05)
        return False

    # ---------- tracking health ----------

    def tracking_suspect(self, span_m=0.05):
        """True when the controller has barely moved over the recent window.

        Loss of optical tracking is silent, so this is the only warning the
        operator gets that the headset cannot see the controller.
        """
        if len(self._hist) < self._hist.maxlen:
            return False
        a = np.array(self._hist)
        return float((a.max(0) - a.min(0)).max()) < span_m

    def controller_position(self):
        """Raw right-controller position as the headset reports it, or None."""
        st = self._server.latest() if self._server is not None else None
        g = getattr(st, "right_grip", None) if st is not None else None
        return None if g is None else np.asarray(g.position, float)

    def _reset(self):
        self._anchor_pos = None
        self._anchor_quat = None
        self._was_clutched = False
        self._hist.clear()

    # ---------- read ----------

    def poll(self):
        st = self._server.latest() if self._server is not None else None
        if st is None or not getattr(st, "connected", False):
            self._reset()
            return TeleopSample(connected=False)

        trig = float(getattr(st, "right_trigger", 0.0)) >= self.cfg.gripper_threshold
        if trig and not self._trigger_was:
            self.gripper_closed = not self.gripper_closed
        self._trigger_was = trig

        grip = getattr(st, "right_grip", None)
        valid = (grip is not None and bool(getattr(grip, "valid", False))
                 and (int(getattr(st, "right_grip_flags", 0)) & POSITION_VALID_BIT) != 0)
        clutch = valid and float(getattr(st, "right_squeeze", 0.0)) >= self.cfg.clutch_threshold

        out = TeleopSample(connected=True, clutch=clutch, tracking_valid=valid,
                           gripper_closed=self.gripper_closed,
                           info=dict(heading_deg=self.heading_deg))
        if not clutch:
            self._reset()
            return out

        pos = np.asarray(grip.position, float)
        quat = _norm(np.asarray(grip.quat_xyzw, float))
        self._hist.append(pos)
        if self._anchor_pos is None or not self._was_clutched or bool(getattr(st, "button_b", False)):
            self._anchor_pos, self._anchor_quat = pos, quat
            self._was_clutched = True
            out.reset_reference = True
            return out

        out.disp_pos = self._to_robot(pos - self._anchor_pos)
        if self.rot.unlock:
            rv = _rotvec(_mul(quat, _conj(self._anchor_quat)))
            mag = np.linalg.norm(rv)
            db = np.radians(self.rot.deadband)
            # Shrink rather than hard-threshold, so crossing the deadband does
            # not jump.  Without a deadband here, incidental wrist rotation while
            # translating drove the wrist through 122 deg.
            rv = np.zeros(3) if mag <= db else rv / mag * (mag - db)
            # Raw operator rotation; rotation.gain is applied by CartesianTeleop
            # so that every source is scaled the same way.
            out.disp_rotvec = self._to_robot(rv)
        out.info["tracking_suspect"] = self.tracking_suspect()
        return out


def calibrate_forward(source, timeout=15.0, min_travel=0.05):
    """Measure the operator's forward heading from one deliberate reach.

    Ask the operator to clutch, push straight away from their chest, and release
    while still extended.  Returns the heading in degrees, for `set_heading`.

    Measured rather than taken from the headset because the headset points
    wherever it was set down, and when worn it follows the operator's gaze
    rather than their body.
    """
    total = np.zeros(3)
    prev = None
    seen = False
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = source.poll()
        p = source.controller_position()
        if s.clutch and p is not None:
            if prev is not None:
                total += p - prev
            prev = p
            seen = True
        elif seen and np.linalg.norm(total) > min_travel:
            break
        time.sleep(0.005)
    horiz = np.array([total[0], 0.0, total[2]])
    if np.linalg.norm(horiz) < 0.03:
        raise RuntimeError("not enough horizontal motion to measure a heading")
    horiz /= np.linalg.norm(horiz)
    return float(np.degrees(np.arctan2(-horiz[0], -horiz[2])))
