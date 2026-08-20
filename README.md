# piperx-teleop

Cartesian teleoperation for the AgileX Piper (piperx) arm.

Drives the arm through the firmware's own Cartesian interface, with the safety
layer that a black-box IK needs. Input is pluggable: a keyboard source ships in
the core, and a Quest 3 source comes with the optional `quest` extra.

```bash
pip install -e .            # core: keyboard teleop
pip install -e ".[quest]"   # adds Quest 3 support (needs RoboVR)
```

## Use

```python
from piperx_teleop import PiperArm, CartesianTeleop, TeleopSession, load_config
from piperx_teleop.sources import KeyboardSource

arm = PiperArm("can0").connect()          # raises TeachModeError if hand-dragged
ctl = CartesianTeleop(arm, load_config())
for state in TeleopSession(arm, ctl, KeyboardSource()).step():
    ...                                   # record alongside your cameras
```

`step()` yields one `TeleopState` per tick — time, joints, velocities, commanded
and actual pose, gripper, lag, clamp counts and warnings — so data collection
keeps control of the process. `run()` is the blocking equivalent.

Examples: `examples/keyboard_teleop.py`, `examples/vr_teleop.py`,
`examples/record_episode.py`.

## Command-line helpers

```bash
piperx-release          # motors AND gripper off - THE ARM WILL FALL
piperx-release --hold   # freeze in place, torque on
piperx-release --open   # open the gripper only
piperx-quest awake      # keep the headset awake off-head
piperx-quest guardian   # pause the boundary system
```

`DisableArm` covers joints 1-6 only - the gripper is a separate motor, so a
plain disable leaves it clamped. `piperx-release` handles both, gripper first.

## Configuration

`load_config("my.toml")` layers over the packaged defaults; see
`piperx_teleop/default_config.toml` for every key and what it does.
`config.with_(**{"motion.gain": 0.3})` for programmatic overrides.

The two you will actually tune are `workspace.max_reach` and `workspace.min_z`.
**`max_reach` is measured from where the session starts, not from the robot
base** — so to work near a table, start the session low rather than opening the
box wide, and set `min_z` just above the surface as a collision guard.

## What the safety layer is for

The firmware's Cartesian control is accurate but not robust — measured on
hardware, tracking is essentially exact (100–101% of commanded, per-cycle joint
motion p99 0.81°), but IK branch flips are real: J6 once jumped **17.9° in a
single cycle**. So:

- **joint watchdog** (2°/cycle) aborts to a joint hold before a flip becomes a swing
- **command leash** stops the command running away when the arm stalls, which
  otherwise makes it feel dead in every direction until you retrace the gap
- **workspace box, z floor**, and **named joint-limit warnings** instead of the
  arm silently going still

Two design choices are load-bearing, both learned by getting them wrong first:
the operator's displacement is **anchored** (recomputed from the clutch pose each
tick, never integrated — integrating and clamping lost 87% of a fast motion), and
the rate limit governs how fast the command *chases* the goal rather than
deleting motion.

## Gotchas that look like bugs

- **Hand-dragging the arm puts it in teach mode**, after which every command is
  silently discarded. It cannot be cleared over CAN. Press the teach button on
  the arm, or power-cycle. `PiperArm.connect()` raises `TeachModeError`.
- **Quest 3 controllers are tracked optically** and have no tracking ring. Out of
  the headset's view the IMU dead-reckons a smooth, `valid=true` pose that does
  not follow your hand. `QuestSource.tracking_suspect()` detects it and the
  session warns.
- **Descending holds tool orientation**, so the firmware's IK straightens the
  elbow until a joint jams — a few hundred mm depending on start pose. Start
  poses with **J4 near 0** keep range in every direction.
- The vendor URDF does not match the hardware (51 mm mean fit residual), which is
  why this drives the firmware's Cartesian interface rather than doing its own IK.

## Coordinates

Robot base is **+x forward, +y left, +z up**, verified against the physical arm.
End-pose orientation is **extrinsic `xyz`** (RX, RY, RZ), identified on hardware
to 0.00° by single-joint rotation self-consistency. Singular at RY = ±90°.
