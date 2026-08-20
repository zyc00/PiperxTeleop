"""The input-side contract: anything that can drive Cartesian teleop."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class TeleopSample:
    """One reading from an input device.

    `disp_pos` / `disp_rotvec` are the operator's TOTAL displacement since the
    clutch engaged, expressed in the robot base frame (+x forward, +y left,
    +z up).  They are absolute-from-anchor, not per-tick deltas: see the note in
    piperx_teleop.cartesian for why that distinction matters.
    """
    connected: bool = False
    clutch: bool = False
    reset_reference: bool = False
    disp_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    disp_rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper_closed: bool = False
    tracking_valid: bool = True
    quit: bool = False
    info: dict = field(default_factory=dict)


@runtime_checkable
class TeleopSource(Protocol):
    def start(self) -> "TeleopSource": ...
    def poll(self) -> TeleopSample: ...
    def stop(self) -> None: ...
