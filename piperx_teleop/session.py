"""The loop that joins a TeleopSource to a CartesianTeleop.

Two entry points:

    session.run()      blocking, for interactive use
    for st in session.step():   generator, one TeleopState per tick

`step()` is the one to build data collection on: it hands control back every
tick so you can grab camera frames alongside the arm state, instead of the loop
owning your process.
"""

import time

import numpy as np


class TeleopSession:
    def __init__(self, arm, controller, source, rate=100.0, home_q=None,
                 auto_enable=True, on_note=print):
        self.arm = arm
        self.ctl = controller
        self.src = source
        self.dt = 1.0 / rate
        self.home_q = None if home_q is None else np.asarray(home_q, float)
        self.auto_enable = auto_enable
        self.on_note = on_note
        self._was_clutched = False
        self._last_track_warn = 0.0
        self.stopped_reason = None

    # ---------- setup ----------

    def prepare(self):
        if self.arm.in_teach_mode():
            raise RuntimeError("arm is in %s and will ignore commands; press the teach "
                               "button on the arm" % self.arm.control_mode())
        if self.auto_enable and not all(self.arm.is_enabled()):
            self.arm.enable()
            self.arm.hold(seconds=1.0)
        if self.home_q is not None:
            self.arm.move_to(self.home_q)
            time.sleep(0.6)
        self.src.start()
        self.ctl.start()
        return self

    # ---------- the loop ----------

    def step(self):
        """Yield one TeleopState per tick until the source quits or we abort."""
        self.prepare()
        try:
            while True:
                t0 = time.time()
                s = self.src.poll()

                if getattr(s, "quit", False):
                    self.stopped_reason = "source quit"
                    break

                if getattr(self.src, "home_requested", False):
                    self.src.home_requested = False
                    if self.home_q is not None:
                        self.arm.move_to(self.home_q)
                        time.sleep(0.4)
                    self.ctl.start()

                if not s.connected:
                    yield self.ctl.idle()
                    time.sleep(0.05)
                    continue

                if s.clutch and not self._was_clutched:
                    self.ctl.relatch()
                if s.reset_reference:
                    self.ctl.relatch()
                self._was_clutched = s.clutch

                if s.gripper_closed != self.ctl.gripper_closed:
                    self.ctl.set_gripper(s.gripper_closed)

                if s.clutch and not s.reset_reference:
                    if not self.ctl.check_joints():
                        st = self.ctl.state(clutch=True)
                        self._emit(st)
                        self.stopped_reason = self.ctl.aborted
                        yield st
                        break
                    st = self.ctl.follow(s.disp_pos, s.disp_rotvec)
                else:
                    st = self.ctl.idle()

                self._warn_tracking(s)
                self._emit(st)
                yield st
                time.sleep(max(0.0, self.dt - (time.time() - t0)))
        finally:
            self.close()

    def run(self):
        """Blocking loop. Returns the reason it stopped."""
        for _ in self.step():
            pass
        return self.stopped_reason

    # ---------- helpers ----------

    def _emit(self, state):
        if self.on_note:
            for n in state.notes:
                self.on_note("  !! " + n)

    def _warn_tracking(self, sample):
        if not sample.info.get("tracking_suspect"):
            return
        if time.time() - self._last_track_warn < 8.0:
            return
        self._last_track_warn = time.time()
        if self.on_note:
            self.on_note("  !! CONTROLLER BARELY MOVING while clutched. If your hand is "
                         "moving, the headset cannot see the controller and is "
                         "dead-reckoning. Point it at your hands, or wear it.")

    def close(self):
        try:
            self.src.stop()
        except Exception:
            pass
        try:
            self.arm.hold()
        except Exception:
            pass
