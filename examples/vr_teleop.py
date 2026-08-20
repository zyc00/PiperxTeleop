"""Quest 3 teleop. Needs the `quest` extra and the RoboVR client running.

    python examples/vr_teleop.py --calibrate     # measure your forward once
    python examples/vr_teleop.py --heading -16   # or pass a known heading
"""
import argparse
import os

import numpy as np

from piperx_teleop import CartesianTeleop, PiperArm, TeleopSession, load_config
from piperx_teleop.sources import QuestSource, calibrate_forward

ap = argparse.ArgumentParser()
ap.add_argument("--can", default="can0")
ap.add_argument("--config", default=None)
ap.add_argument("--home", default=None)
ap.add_argument("--heading", type=float, default=0.0)
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--unlock-rotation", action="store_true")
a = ap.parse_args()

cfg = load_config(a.config)
if a.unlock_rotation:
    cfg.rotation.unlock = True

src = QuestSource(cfg, heading_deg=a.heading).start()
if not src.wait_connected(15.0):
    raise SystemExit("Quest not connected - is the RoboVR client running?")

if a.calibrate:
    input(">>> Squeeze the grip, push your hand STRAIGHT AWAY from your chest,\n"
          "    release while it is still out there, then press ENTER here. ")
    h = calibrate_forward(src)
    src.set_heading(h)
    print("measured forward heading: %.1f deg" % h)

home = np.load(a.home)["q"] if a.home and os.path.exists(a.home) else None
arm = PiperArm(a.can).connect()
ctl = CartesianTeleop(arm, cfg)
sess = TeleopSession(arm, ctl, src, home_q=home)

print("squeeze the right grip to clutch; Ctrl-C to stop")
try:
    for _ in sess.step():
        pass
except KeyboardInterrupt:
    pass
print("stopped:", sess.stopped_reason)
