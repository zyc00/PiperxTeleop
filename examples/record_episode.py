"""Data collection: teleoperate while recording every tick.

This is the point of exposing step() - the loop hands control back each tick, so
you can grab camera frames alongside the arm state instead of the teleop owning
your process.  Swap the writer for whatever dataset format you target.
"""
import argparse

import numpy as np

from piperx_teleop import CartesianTeleop, PiperArm, TeleopSession, load_config
from piperx_teleop.sources import KeyboardSource

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="episode.npz")
ap.add_argument("--can", default="can0")
a = ap.parse_args()

cfg = load_config()
arm = PiperArm(a.can).connect()
sess = TeleopSession(arm, CartesianTeleop(arm, cfg), KeyboardSource(cfg))

rec = {"t": [], "clutch": [], "action": [],
       "q": [], "dq": [], "effort": [], "ee_pos": [], "ee_rpy": [], "gripper_pos": []}
try:
    for st in sess.step():
        # frame = camera.read()               # <- your cameras here

        # action: commanded EE pose + gripper (x y z rx ry rz grip)
        rec["action"].append(st.action())
        # observation: proprioception
        for k, v in st.observation().items():
            rec[k].append(v)
        rec["t"].append(st.t)
        # clutch segments the episode - ticks with clutch False are the operator
        # repositioning their hand, not demonstrating. Usually drop them.
        rec["clutch"].append(st.clutch)
except KeyboardInterrupt:
    pass

kept = int(np.sum(rec["clutch"]))
print("\n%d ticks recorded, %d with the clutch engaged" % (len(rec["t"]), kept))

np.savez(a.out, **{k: np.array(v) for k, v in rec.items()})
print("\nwrote %s (%d ticks)" % (a.out, len(rec["t"])))
