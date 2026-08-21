"""Cartesian teleoperation for the AgileX Piper (piperx) arm.

    from piperx_teleop import PiperArm, CartesianTeleop, TeleopSession, load_config
    from piperx_teleop.sources import KeyboardSource      # or QuestSource

    arm = PiperArm("can0").connect()
    ctl = CartesianTeleop(arm, load_config())
    for state in TeleopSession(arm, ctl, KeyboardSource()).step():
        ...   # record state alongside your cameras
"""

from .arm import JOINT_LIMITS, MOVE_J, MOVE_L, MOVE_P, PiperArm, TeachModeError
from .cartesian import EULER_SEQ, CartesianTeleop, TeleopState
from .config import Config, load_config
from .gravity import GravityCompensator, JointImpedance, require_patched_sdk
from .model import PiperModel
from .session import TeleopSession
from .sources import TeleopSample, TeleopSource

__version__ = "0.2.0"
__all__ = [
    "PiperArm", "TeachModeError", "JOINT_LIMITS", "MOVE_P", "MOVE_J", "MOVE_L",
    "CartesianTeleop", "TeleopState", "EULER_SEQ",
    "TeleopSession", "TeleopSample", "TeleopSource",
    "Config", "load_config",
    "GravityCompensator", "JointImpedance", "require_patched_sdk", "PiperModel",
    "__version__",
]
