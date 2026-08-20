"""Keyboard teleop source. No VR, no external hardware.

Key auto-repeat gives continuous motion while a key is held.  Useful on its own
and as the reference implementation of TeleopSource - it isolates the arm from
any tracking question, which is how we proved the arm side was sound.
"""

import select
import sys
import termios
import time
import tty

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .base import TeleopSample

HELP = """\
POSITION
  arrow up/down      z up / down
  arrow left/right   y left / right
  w / s              x forward / back
  [ / ]              smaller / larger position step
ORIENTATION
  u / p              roll  about x
  i / k              pitch about y
  j / l              yaw   about z
  - / =              smaller / larger rotation step
  r                  reset orientation
OTHER
  space              re-anchor here      h  home
  g                  toggle gripper      o  open gripper
  q                  quit
"""

ARROWS = {"[A": "up", "[B": "down", "[C": "right", "[D": "left",
          "OA": "up", "OB": "down", "OC": "right", "OD": "left"}
_ROT_KEYS = {"u": (0, +1), "p": (0, -1), "i": (1, +1),
             "k": (1, -1), "j": (2, +1), "l": (2, -1)}


class KeyboardSource:
    def __init__(self, config=None):
        from ..config import Config
        cfg = config or Config()
        self.step = cfg.keyboard.step
        self.rot_step = cfg.keyboard.rot_step
        self.disp = np.zeros(3)
        self.R = Rot.identity()
        self.gripper_closed = False
        self._fd = None
        self._old = None
        self.home_requested = False

    def start(self):
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        # setcbreak drops line buffering but leaves ECHO on, so every keystroke
        # would print into the caller's status line.
        attrs = termios.tcgetattr(self._fd)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(self._fd, termios.TCSADRAIN, attrs)
        return self

    def stop(self):
        if self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._old = None

    def _getkey(self):
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        c = sys.stdin.read(1)
        if c != "\x1b":
            return c
        # Arrow keys arrive as ESC [ A/B/C/D; the tail can lag the ESC by a few
        # milliseconds, so wait for it rather than reading a bare ESC.
        seq = ""
        t0 = time.time()
        while len(seq) < 2 and time.time() - t0 < 0.05:
            if select.select([sys.stdin], [], [], 0.01)[0]:
                seq += sys.stdin.read(1)
        return ARROWS.get(seq, "esc")

    def poll(self):
        out = TeleopSample(connected=True, clutch=True)
        k = self._getkey()
        if k in ("q", "\x03"):
            out.quit = True
        elif k == "up":     self.disp[2] += self.step
        elif k == "down":   self.disp[2] -= self.step
        elif k == "left":   self.disp[1] += self.step
        elif k == "right":  self.disp[1] -= self.step
        elif k == "w":      self.disp[0] += self.step
        elif k == "s":      self.disp[0] -= self.step
        elif k == "[":      self.step = max(0.001, self.step / 1.5)
        elif k == "]":      self.step = min(0.05, self.step * 1.5)
        elif k in _ROT_KEYS:
            ax, sgn = _ROT_KEYS[k]
            v = np.zeros(3)
            v[ax] = sgn * np.radians(self.rot_step)
            # compose, never sum rotation vectors: they only add correctly about
            # a shared axis, and roll-then-pitch is not that
            self.R = Rot.from_rotvec(v) * self.R
        elif k == "-":      self.rot_step = max(0.25, self.rot_step / 1.5)
        elif k == "=":      self.rot_step = min(20.0, self.rot_step * 1.5)
        elif k == "r":      self.R = Rot.identity()
        elif k == " ":
            self.disp[:] = 0; self.R = Rot.identity(); out.reset_reference = True
        elif k == "h":
            self.disp[:] = 0; self.R = Rot.identity(); self.home_requested = True
        elif k == "g":      self.gripper_closed = not self.gripper_closed
        elif k == "o":      self.gripper_closed = False

        out.disp_pos = self.disp.copy()
        out.disp_rotvec = self.R.as_rotvec()
        out.gripper_closed = self.gripper_closed
        out.info = dict(step=self.step, rot_step=self.rot_step)
        return out
