from .base import TeleopSample, TeleopSource
from .keyboard import KeyboardSource

__all__ = ["TeleopSample", "TeleopSource", "KeyboardSource", "QuestSource",
           "calibrate_forward"]


def __getattr__(name):
    # Import lazily so the core does not require the optional `quest` extra.
    if name in ("QuestSource", "calibrate_forward"):
        from . import quest
        return getattr(quest, name)
    raise AttributeError(name)
