"""Command-line helpers, installed as `piperx-release` and `piperx-quest`.

These live in the package rather than a project's scripts/ directory because
they are safety and setup tooling: they should work from any directory, and they
should not disappear when a project is tidied up.
"""

import argparse
import subprocess
import sys
import time


def release(argv=None):
    """Emergency release.

        piperx-release            arm motors AND gripper off - THE ARM WILL FALL
        piperx-release --hold     freeze where it stands, torque on
        piperx-release --open     open the gripper only, arm untouched

    DisableArm covers joints 1-6 only; the gripper is a separate motor reached
    through GripperCtrl, so a plain disable leaves it clamped.
    """
    ap = argparse.ArgumentParser(prog="piperx-release", description=release.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--can", default="can0")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--hold", action="store_true", help="freeze in place, torque on")
    g.add_argument("--open", dest="open_", action="store_true", help="open the gripper only")
    a = ap.parse_args(argv)

    import numpy as np

    from .arm import PiperArm
    arm = PiperArm(a.can).connect(require_control=False)

    if a.open_:
        for _ in range(25):
            arm.open_gripper()
            time.sleep(0.02)
        print("gripper opened -> %.1f mm" % (arm.gripper()[0] * 1000))
    elif a.hold:
        arm.hold(seconds=0.5)
        print("holding at", np.degrees(arm.q()).round(2), "deg")
    else:
        # Gripper first, so the tool opens while the arm still holds pose rather
        # than after it has gone slack.
        arm.disable()
        time.sleep(0.3)
        print("motors OFF - arm is limp. enabled:", arm.is_enabled())
    return 0


def quest(argv=None):
    """Quest headset setup helpers.

        piperx-quest awake        stay awake off-head (proximity sleep off)
        piperx-quest awake --off
        piperx-quest guardian     pause the boundary system
        piperx-quest guardian --off

    Neither survives a headset reboot.
    """
    ap = argparse.ArgumentParser(prog="piperx-quest", description=quest.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["awake", "guardian"])
    ap.add_argument("--off", action="store_true")
    a = ap.parse_args(argv)

    def sh(*cmd):
        return subprocess.run(["adb", *cmd], capture_output=True, text=True, timeout=15)

    if a.what == "awake":
        act = "prox_open" if a.off else "prox_close"
        sh("shell", "am", "broadcast", "-a", "com.oculus.vrpowermanager." + act)
        time.sleep(1)
        r = sh("shell", "dumpsys", "power")
        state = [ln for ln in r.stdout.splitlines() if "mWakefulness=" in ln]
        print("proximity sleep %s. %s" % ("restored" if a.off else "disabled",
                                          state[0].strip() if state else ""))
    else:
        val = "0" if a.off else "1"
        for prop in ("guardian_pause", "guardian_disable"):
            sh("shell", "setprop", "debug.oculus." + prop, val)
        print("guardian %s" % ("restored" if a.off else "paused"))
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(release())


def drag():
    """Console entry: gravity-compensated drag mode until Ctrl-C."""
    import argparse

    import numpy as np

    from .gravity import GravityCompensator

    ap = argparse.ArgumentParser(
        description="Gravity-compensated drag mode: the arm becomes freely "
                    "draggable and holds wherever you leave it. Ctrl-C exits "
                    "into position hold.")
    ap.add_argument("--can", default="can0")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds; 0 = until Ctrl-C")
    ap.add_argument("--payload-mass", type=float, default=0.0,
                    help="kg carried by the gripper, added to the model")
    a = ap.parse_args()

    gc = GravityCompensator(can=a.can, payload_mass=a.payload_mass)
    print("pose (deg)   :", np.degrees(gc.q()).round(1))
    print("gravity (N.m):", gc.gravity().round(2))
    trip = gc.run(duration=a.duration)
    print("position hold restored" + (" (%s)" % trip if trip else ""))
