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
class FilterConfig:
    # Smoothing of the operator's displacement, for hand-tracked input.
    # "oneeuro" adapts its cutoff to speed: heavy smoothing when nearly still
    # (tremor dominates, lag is imperceptible), little when moving fast (lag is
    # what you feel). "kalman" is a constant-velocity model with fixed lag.
    kind: str = "none"            # none | oneeuro | kalman

    # --- oneeuro ---
    min_cutoff: float = 3.0       # Hz, smoothing at rest. Lower = steadier.
    beta: float = 1.5             # how fast the cutoff opens with speed
    rot_min_cutoff: float = 3.0
    rot_beta: float = 1.5

    # --- kalman (constant-velocity) ---
    # process_var: how much acceleration the model expects. LOWER = smoother and
    #   more predictive, which also means more overshoot when you stop.
    # meas_var: assumed input noise. HIGHER = trusts the model over the input.
    process_var: float = 1.0
    meas_var: float = 1e-4
    rot_process_var: float = 1.0
    rot_meas_var: float = 1e-4


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
class InputConfig:
    # Which TeleopSource to drive with.  The package does not construct sources
    # itself - the application does - but the preference belongs in the same
    # file as everything else it is configured with.
    source: str = "keyboard"          # keyboard | quest | scripted


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
    input: InputConfig = field(default_factory=InputConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    rotation: RotationConfig = field(default_factory=RotationConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    quest: QuestConfig = field(default_factory=QuestConfig)

    def with_(self, **overrides):
        """Return a copy with dotted overrides, e.g. with_(**{'motion.gain': 0.3})."""
        out = Config(**{k: replace(getattr(self, k)) for k in
                        ("input", "motion", "workspace", "rotation", "filter",
                         "safety", "keyboard", "quest")})
        for key, val in overrides.items():
            if val is None:
                continue
            section, _, name = key.partition(".")
            setattr(getattr(out, section), name, val)
        return out


_SECTIONS = {"input": InputConfig, "motion": MotionConfig, "workspace": WorkspaceConfig,
             "rotation": RotationConfig, "filter": FilterConfig,
             "safety": SafetyConfig,
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
