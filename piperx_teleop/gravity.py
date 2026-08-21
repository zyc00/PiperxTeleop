"""Gravity compensation and joint impedance, as control LAWS on TorqueSession.

The layering: piperx_teleop.mit.TorqueSession is the physics-free runtime
(MIT session, safety, loop); PiperModel provides torque terms; the classes
here are thin conveniences that plug a specific law into the runtime. Your
own controllers compose the same way - see mit.py's docstring.

    GravityCompensator  = TorqueSession( tau = G(q) )              # drag mode
    JointImpedance      = TorqueSession( tau = G(q) + firmware PD ) # springy

Both keep the ergonomic API: run() / with-block / q() / gripper() / trip.
"""

import time

import numpy as np

from .mit import MitCommand, TorqueSession, require_patched_sdk  # noqa: F401  (re-export)
from .model import PiperModel


class GravityCompensator(TorqueSession):
    """Pure gravity feed-forward: the arm becomes freely draggable.

        GravityCompensator().run()              # drag until Ctrl-C

        with GravityCompensator() as gc:        # background compliance
            log(gc.q(), gc.gripper())

    Equivalent to TorqueSession(lambda s: model.gravity_torque(s.q)); use
    that form directly when composing gravity with your own torque terms.
    """

    def __init__(self, arm=None, can="can0", model=None, hz=200.0,
                 payload_mass=0.0, payload_com=(0.0, 0.0, 0.0)):
        self.mdl = model or PiperModel(payload_mass=payload_mass,
                                       payload_com=payload_com)
        super().__init__(self._law, arm=arm, can=can, hz=hz)

    def _law(self, s):
        return self.mdl.gravity_torque(s.q)

    def gravity(self, q=None):
        return self.mdl.gravity_torque(self.q() if q is None else q)


class JointImpedance(GravityCompensator):
    """Virtual spring-damper on top of gravity comp (compliant hold/follow).

        tau = G(q) + Kp*(q_ref - q) - Kd*qdot

    The PD runs IN THE FIRMWARE at motor rate (MIT frame p_des/kp/kd fields);
    only gravity and the target stream over CAN. Kp=0 is drag mode. This is
    the compliant-follower building block: stream a leader's pose into
    set_target() and the follower tracks softly.

    Gain units are the firmware's (~N.m/rad, but per-joint scaling is not
    uniform - J3 responds stronger than J2). Start soft.
    """

    def __init__(self, arm=None, can="can0", model=None, hz=200.0,
                 kp=8.0, kd=0.8, q_ref=None,
                 payload_mass=0.0, payload_com=(0.0, 0.0, 0.0)):
        super().__init__(arm=arm, can=can, model=model, hz=hz,
                         payload_mass=payload_mass, payload_com=payload_com)
        self.kp = np.asarray(kp, float) * np.ones(6)
        self.kd = np.asarray(kd, float) * np.ones(6)
        self._q_ref = None if q_ref is None else np.asarray(q_ref, float).copy()

    def start(self):
        if self._q_ref is None:
            self._q_ref = self.q()          # hold where the arm stands
        return super().start()

    def set_target(self, q):
        """Retarget the spring; the arm moves compliantly toward q."""
        self._q_ref = np.asarray(q, float).copy()   # atomic swap (GIL)

    @property
    def target(self):
        return None if self._q_ref is None else self._q_ref.copy()

    def move_to(self, q, secs=2.0):
        """Blocking compliant move: ramp the spring target to q over secs."""
        q = np.asarray(q, float)
        q0, t0 = self.target, time.time()
        while time.time() - t0 < secs:
            f = min((time.time() - t0) / secs, 1.0)
            self.set_target(q0 + f * (q - q0))
            time.sleep(0.01)
        self.set_target(q)

    def _law(self, s):
        return MitCommand(t_ff=self.mdl.gravity_torque(s.q),
                          p_des=self._q_ref, kp=self.kp, kd=self.kd)
