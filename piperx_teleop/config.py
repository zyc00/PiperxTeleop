"""Typed configuration, loaded from TOML."""

import os
from dataclasses import dataclass, field, replace

try:
    import tomllib
except ModuleNotFoundError:                          # pragma: no cover
    tomllib = None

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "default_config.toml")


@dataclass
class MotionConfig:
    gain: float = 0.5
    max_step: float = 0.004
    speed: int = 20


@dataclass
class WorkspaceConfig:
    # max_reach is measured from where the session STARTS, not from the robot
    # base: a tight box around the actual work area is safer than a wide one,
    # so start the session where you intend to work.
    max_reach: float = 0.30
    min_z: float = 0.05


@dataclass
class RotationConfig:
    unlock: bool = False
    gain: float = 0.4
    deadband: float = 6.0
    max_step: float = 1.5


@dataclass
class SafetyConfig:
    # Do not raise these to work around a stall - they are what turn a firmware
    # IK branch flip into a 2 deg twitch instead of a swing.
    max_joint_step: float = 2.0
    lead_limit: float = 0.04


@dataclass
class KeyboardConfig:
    step: float = 0.005
    rot_step: float = 3.0


@dataclass
class QuestConfig:
    port: int = 7777
    host: str = "127.0.0.1"
    adb_reverse: bool = True
    frame_mode: str = "between_clutch"
    clutch_threshold: float = 0.5
    gripper_threshold: float = 0.75


@dataclass
class Config:
    motion: MotionConfig = field(default_factory=MotionConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    quest: QuestConfig = field(default_factory=QuestConfig)

    def with_(self, **overrides):
        """Return a copy with dotted overrides, e.g. with_(**{'motion.gain': 0.3})."""
        out = Config(**{k: replace(getattr(self, k)) for k in
                        ("motion", "workspace", "rotation", "safety", "keyboard", "quest")})
        for key, val in overrides.items():
            if val is None:
                continue
            section, _, name = key.partition(".")
            setattr(getattr(out, section), name, val)
        return out


_SECTIONS = {"motion": MotionConfig, "workspace": WorkspaceConfig,
             "rotation": RotationConfig, "safety": SafetyConfig,
             "keyboard": KeyboardConfig, "quest": QuestConfig}


def load_config(path=None):
    """Load config, falling back to the packaged defaults."""
    cfg = Config()
    for src in [DEFAULT_CONFIG] + ([path] if path else []):
        if not src or not os.path.exists(src) or tomllib is None:
            continue
        with open(src, "rb") as f:
            raw = tomllib.load(f)
        for section in _SECTIONS:
            if section in raw:
                cur = getattr(cfg, section)
                for k, v in raw[section].items():
                    if hasattr(cur, k):
                        setattr(cur, k, v)
    return cfg
