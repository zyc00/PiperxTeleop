"""Keyboard teleop. No VR, no external hardware.

    python examples/keyboard_teleop.py [--config my.toml] [--home]
"""
import argparse
import os
import sys
import time

import numpy as np

from piperx_teleop import CartesianTeleop, PiperArm, TeleopSession, load_config
from piperx_teleop.sources import KeyboardSource
from piperx_teleop.sources.keyboard import HELP

ap = argparse.ArgumentParser()
ap.add_argument("--can", default="can0")
ap.add_argument("--config", default=None)
ap.add_argument("--home", default=None, help="npz with a 'q' array to home to")
ap.add_argument("--unlock-rotation", action="store_true")
a = ap.parse_args()

cfg = load_config(a.config)
if a.unlock_rotation:
    cfg.rotation.unlock = True

home = np.load(a.home)["q"] if a.home and os.path.exists(a.home) else None
arm = PiperArm(a.can).connect()
ctl = CartesianTeleop(arm, cfg)
src = KeyboardSource(cfg)
sess = TeleopSession(arm, ctl, src, home_q=home)

print(HELP)
last = 0.0
try:
    for st in sess.step():
        if time.time() - last > 0.15:
            last = time.time()
            sys.stdout.write("\r  pos %s | rpy %s | lag %3.0f mm | grip %-6s "
                             % (st.actual.round(3), np.round(st.rpy, 1),
                                st.lag * 1000, "closed" if st.gripper_closed else "open"))
            sys.stdout.flush()
finally:
    print("\nstopped:", sess.stopped_reason)
