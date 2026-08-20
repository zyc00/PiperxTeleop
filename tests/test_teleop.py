"""Offline tests. No hardware, no headset."""
import sys
import time as _time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from piperx_teleop import CartesianTeleop, Config, TeleopSession, load_config
from piperx_teleop.cartesian import EULER_SEQ
from piperx_teleop.sources import TeleopSample


class FakeArm:
    """Arm stub. `wall` optionally hard-stops motion past a given +x offset."""

    def __init__(self, wall=None):
        self.q_ = np.radians([0, 50, -75, 0, 20, 0])
        self.pos = np.array([0.30, 0.0, 0.35])
        self.rpy = np.array([-170.0, 0.0, -90.0])
        self.wall = wall
        self.p0 = self.pos.copy()
        self.gripper_closed = False
        self.sent = 0

    def end_pose(self): return self.pos.copy(), self.rpy.copy()
    def q(self): return self.q_.copy()
    def dq(self): return np.zeros(6)
    def effort(self): return np.arange(6) * 0.1
    def gripper(self): return (0.035, 0.2)
    def in_teach_mode(self): return False
    def obs_time(self): return _time.time() - 0.002      # 2 ms stale, as on hardware
    def stream_ages(self, now=None): return dict(joint=0.002, motor=0.001, gripper=0.002)
    def stream_rates(self): return dict(joint=200.0, motor=100.0, gripper=200.0)
    def control_mode(self): return "CAN_CTRL(0x1)"
    def is_enabled(self): return [True] * 6
    def move_j(self, q, speed_pct=0): pass
    def move_to(self, q, **kw): return True
    def hold(self, *a, **k): pass
    def set_gripper(self, closed, **kw): self.gripper_closed = closed

    def end_pose_ctrl(self, pos, rpy, move_mode=0, speed_pct=0):
        self.sent += 1
        p = np.asarray(pos, float).copy()
        if self.wall is not None:
            p[0] = min(p[0], self.p0[0] + self.wall)
        self.pos = p
        self.rpy = np.asarray(rpy, float).copy()


fails = []


def check(name, cond):
    if not cond:
        fails.append(name)


# --- anchored mapping recovers motion the rate limit clips -------------------
cfg = Config()
cfg.motion.gain = 1.0
cfg.motion.max_step = 0.004
arm = FakeArm()
ctl = CartesianTeleop(arm, cfg).start()
for k in range(200):
    ctl.follow(np.array([min(0.150, 0.03 * k), 0, 0]))
check("anchored mapping reaches the goal", abs((arm.pos - ctl.origin)[0] - 0.150) < 0.002)

# and still does with a fifth of the samples dropped
arm = FakeArm()
ctl = CartesianTeleop(arm, cfg).start()
for k in range(200):
    if k % 5 == 0:
        continue
    ctl.follow(np.array([min(0.150, 0.03 * k), 0, 0]))
check("survives dropped samples", abs((arm.pos - ctl.origin)[0] - 0.150) < 0.002)

# --- workspace box and floor -------------------------------------------------
cfg2 = Config(); cfg2.motion.gain = 1.0; cfg2.workspace.max_reach = 0.05
arm = FakeArm(); ctl = CartesianTeleop(arm, cfg2).start()
for _ in range(300):
    ctl.follow(np.array([0.5, 0, 0]))
check("box clamps", (ctl.target - ctl.origin)[0] <= 0.0501)
check("box announced", ctl.clamp_box > 0)

cfg3 = Config(); cfg3.motion.gain = 1.0; cfg3.workspace.min_z = 0.34
arm = FakeArm(); ctl = CartesianTeleop(arm, cfg3).start()
for _ in range(300):
    ctl.follow(np.array([0, 0, -0.5]))
check("z floor clamps", ctl.target[2] >= 0.3399)

# --- the leash: a stall must not let the command run away --------------------
cfg4 = Config(); cfg4.motion.gain = 1.0; cfg4.safety.lead_limit = 0.04
arm = FakeArm(wall=0.03); ctl = CartesianTeleop(arm, cfg4).start()
for k in range(300):
    ctl.follow(np.array([min(0.20, 0.005 * k), 0, 0]))
check("command leashed to actual", (ctl.cmd_pos[0] - arm.pos[0]) <= 0.041)
for _ in range(200):
    ctl.follow(np.array([0.02, 0, 0]))
check("recovers immediately after a stall", abs((arm.pos - ctl.origin)[0] - 0.02) < 0.003)

# --- joint watchdog ----------------------------------------------------------
arm = FakeArm(); ctl = CartesianTeleop(arm, Config()).start()
ctl.check_joints()
arm.q_ = arm.q_ + np.radians([0, 20, 0, 0, 0, 0])
check("watchdog fires on a joint jump", (not ctl.check_joints()) and ctl.aborted)

# --- rotation ----------------------------------------------------------------
cfg5 = Config(); cfg5.rotation.unlock = True; cfg5.rotation.max_step = 90.0
cfg5.rotation.gain = 1.0        # this case tests composition, not scaling
arm = FakeArm(); ctl = CartesianTeleop(arm, cfg5).start()
want = np.radians([0, 0, 20.0])
for _ in range(50):
    ctl.follow(np.zeros(3), want)
R_end = Rot.from_euler(EULER_SEQ, arm.rpy, degrees=True)
R_ref = Rot.from_rotvec(want) * Rot.from_euler(EULER_SEQ, ctl.anchor_rot, degrees=True)
check("rotation reaches the commanded attitude",
      np.degrees(np.linalg.norm((R_end * R_ref.inv()).as_rotvec())) < 1.0)

arm = FakeArm(); ctl = CartesianTeleop(arm, Config()).start()   # locked by default
for _ in range(50):
    ctl.follow(np.zeros(3), want)
check("rotation locked by default", np.allclose(arm.rpy, [-170.0, 0.0, -90.0]))

# --- session yields one state per tick and can be driven by a stub source ----
class StubSource:
    def __init__(self, n=30):
        self.n = n
        self.i = 0
    def start(self): return self
    def stop(self): pass
    def poll(self):
        self.i += 1
        s = TeleopSample(connected=True, clutch=True)
        s.disp_pos = np.array([0.001 * self.i, 0, 0])
        if self.i > self.n:
            s.quit = True
        return s


arm = FakeArm()
sess = TeleopSession(arm, CartesianTeleop(arm, Config()), StubSource(30),
                     rate=1000.0, on_note=None)
states = list(sess.step())
check("session yields per-tick states", len(states) >= 30)
check("session reports why it stopped", sess.stopped_reason == "source quit")
check("state carries joints and target", states[-1].q.shape == (6,) and states[-1].target.shape == (3,))
st = states[-1]
check("action is a 7-vector", st.action().shape == (7,))
check("action names match dim", len(st.ACTION_NAMES) == st.ACTION_DIM)
obs = st.observation()
check("observation has proprio keys",
      set(obs) == {"q", "dq", "effort", "ee_pos", "ee_rpy", "gripper_pos"})
check("effort recorded", obs["effort"].shape == (6,))
check("measured gripper recorded", obs["gripper_pos"] == 0.035)
check("intent is a 7-vector", st.intent().shape == (7,))

# goal (intent) and target (what was sent) must diverge when the arm stalls
cfg6 = Config(); cfg6.motion.gain = 1.0; cfg6.safety.lead_limit = 0.04
arm = FakeArm(wall=0.03); ctl = CartesianTeleop(arm, cfg6).start()
last = None
for k in range(300):
    last = ctl.follow(np.array([min(0.20, 0.005 * k), 0, 0]))
check("goal keeps advancing past the stall", (last.goal - ctl.origin)[0] > 0.15)
check("target is held back by the leash", (last.target - ctl.origin)[0] < 0.08)
check("intent and action differ when stalled",
      np.linalg.norm(last.intent()[:3] - last.action()[:3]) > 0.05)

# --- timing contract ---------------------------------------------------------
arm = FakeArm(); ctl = CartesianTeleop(arm, Config()).start()
s1 = ctl.follow(np.zeros(3))
_time.sleep(0.02)
s2 = ctl.follow(np.zeros(3))
check("wall clock present", s1.t > 1e9)
check("monotonic clock advances", s2.t_mono > s1.t_mono)
check("monotonic interval is sane", 0.015 < (s2.t_mono - s1.t_mono) < 0.2)
check("observation time recorded", s1.obs_t > 1e9 and s1.obs_t <= s1.t)
check("observation age is positive and small", 0.0 <= s1.obs_age < 0.05)
check("tick stamped before the CAN reads", s1.t <= s1.obs_t + s1.obs_age + 1e-6)

# --- config ------------------------------------------------------------------
c = load_config()
check("packaged defaults load", c.motion.gain == 0.5 and c.workspace.max_reach == 0.30)
c2 = c.with_(**{"motion.gain": 0.2, "workspace.min_z": 0.02})
check("overrides apply", c2.motion.gain == 0.2 and c2.workspace.min_z == 0.02)
check("overrides do not mutate the original", c.motion.gain == 0.5)

# --- filters: remove tremor, keep intent ------------------------------------
from piperx_teleop.filters import ConstantVelocityFilter, OneEuroFilter

_fs = 200.0
_t = np.arange(0, 8, 1 / _fs)
_intent = 0.15 * np.sin(2 * np.pi * 0.4 * _t)          # deliberate reach
_sig = np.stack([_intent + 0.004 * np.sin(2 * np.pi * 10.0 * _t),
                 0 * _t, 0 * _t], axis=1)              # + 10 Hz tremor
_faxis = np.fft.rfftfreq(len(_t), 1 / _fs)


def _amp_at(x, f0):
    """Amplitude of the single bin nearest f0.

    Summing a band instead leaks the large 0.4 Hz component into the tremor
    measurement and makes a working filter look useless - which it did.
    """
    X = np.fft.rfft(x - x.mean())
    return 2 * np.abs(X[np.argmin(np.abs(_faxis - f0))]) / len(x)


_raw_tremor = _amp_at(_sig[:, 0], 10.0)
_raw_intent = _amp_at(_sig[:, 0], 0.4)

_oe = OneEuroFilter(3.0, 1.5)
_out = np.array([_oe(_sig[i], 1 / _fs) for i in range(len(_t))])[:, 0]
check("one-euro attenuates tremor", _amp_at(_out, 10.0) < 0.5 * _raw_tremor)
check("one-euro preserves the intended motion",
      abs(_amp_at(_out, 0.4) - _raw_intent) / _raw_intent < 0.05)

_cv = ConstantVelocityFilter(1.0, 1e-4)
_outk = np.array([_cv(_sig[i], 1 / _fs) for i in range(len(_t))])[:, 0]
check("kalman attenuates tremor", _amp_at(_outk, 10.0) < 0.5 * _raw_tremor)

# Prediction is what buys the Kalman its zero lag, and it is also why it sails
# past a hard stop. Guard the property that made us prefer one-euro.
_ts = np.arange(0, 6, 1 / _fs)
_ramp = np.clip((_ts - 1) / 2, 0, 1) * 0.15
_stop = np.stack([_ramp + 0.004 * np.sin(2 * np.pi * 10 * _ts), 0 * _ts, 0 * _ts], 1)


def _overshoot(filt):
    o = np.array([filt(_stop[i], 1 / _fs) for i in range(len(_ts))])[:, 0]
    return (o[_ts >= 3].max() - 0.15) * 1000


check("one-euro overshoots a stop by under 2 mm", _overshoot(OneEuroFilter(3.0, 1.5)) < 2.0)
check("kalman overshoots more than one-euro",
      _overshoot(ConstantVelocityFilter(1.0, 1e-4)) > _overshoot(OneEuroFilter(3.0, 1.5)))

check("filtering off by default", CartesianTeleop(FakeArm(), Config())._pos_filter is None)
_cfgf = Config(); _cfgf.filter.kind = "oneeuro"
check("filter constructed when configured",
      CartesianTeleop(FakeArm(), _cfgf)._pos_filter is not None)

# every kind selectable from config must actually construct: the config carries
# both parameter sets, and passing the wrong set to a filter used to raise
for _kind in ("oneeuro", "kalman", "none"):
    _c = Config(); _c.filter.kind = _kind
    try:
        _t2 = CartesianTeleop(FakeArm(), _c)
        _ok = (_t2._pos_filter is None) if _kind == "none" else (_t2._pos_filter is not None)
    except Exception:
        _ok = False
    check("filter kind %r selectable from config" % _kind, _ok)

_cb = Config(); _cb.filter.kind = "nonsense"
try:
    CartesianTeleop(FakeArm(), _cb); _ok = False
except ValueError:
    _ok = True
check("unknown filter kind raises a clear error", _ok)

_ck = Config(); _ck.filter.kind = "kalman"; _ck.filter.process_var = 0.2
check("kalman params reach the filter",
      CartesianTeleop(FakeArm(), _ck)._pos_filter.q == 0.2)

# --- both gains applied in one place, identically for every source ----------
cfgg = Config(); cfgg.motion.gain = 0.5; cfgg.rotation.unlock = True
cfgg.rotation.gain = 0.5; cfgg.rotation.max_step = 90.0
arm = FakeArm(); ctl = CartesianTeleop(arm, cfgg).start()
for _ in range(80):
    ctl.follow(np.array([0.10, 0, 0]), np.radians([0, 0, 20.0]))
check("motion gain applied by the controller",
      abs((ctl.target - ctl.origin)[0] - 0.05) < 0.002)
R_end = Rot.from_euler(EULER_SEQ, arm.rpy, degrees=True)
R_ref = Rot.from_euler(EULER_SEQ, ctl.anchor_rot, degrees=True)
ang = np.degrees(np.linalg.norm((R_end * R_ref.inv()).as_rotvec()))
check("rotation gain applied by the controller", abs(ang - 10.0) < 1.0)

# --- keyboard reads every key, not just the first --------------------------
import os as _os
import pty as _pty
import sys as _sys

from piperx_teleop.sources import KeyboardSource

_master, _slave = _pty.openpty()
_old = _sys.stdin
_sys.stdin = _os.fdopen(_slave, "r")
try:
    ks = KeyboardSource(Config()).start()
    _os.write(_master, b"w\x1b[A" + b"s")      # w, arrow-up, s
    _time.sleep(0.05)
    seen = []
    for _ in range(20):
        k = ks._getkey()
        if k:
            seen.append(k)
        _time.sleep(0.005)
    ks.stop()
finally:
    _sys.stdin = _old
check("keyboard reads all keys, not just the first", seen == ["w", "up", "s"])

print("%d passed, %d failed" % (43 - len(fails), len(fails)))
for f in fails:
    print("  FAIL:", f)
sys.exit(1 if fails else 0)
