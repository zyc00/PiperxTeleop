"""Input smoothing for hand-tracked teleop.

Physiological hand tremor sits around 8-12 Hz; deliberate motion is below about
2-3 Hz, so they separate cleanly with a low-pass.  The difficulty is that a
FIXED low-pass trades jitter against lag at a single operating point: enough
smoothing to kill tremor while reaching for something adds lag you feel during
fast motion, and little enough to feel responsive leaves the shake in.

OneEuroFilter solves that by adapting its cutoff to speed - heavy smoothing when
the hand is nearly still (where tremor dominates and lag is imperceptible),
almost none when moving fast (where lag is what you notice).  Two parameters:

    min_cutoff  Hz, smoothing at rest.  Lower = steadier, more lag when slow.
    beta        how fast the cutoff opens with speed. Higher = more responsive.

Reference: Casiez, Roussel, Vogel, "1 euro filter" (CHI 2012).

ConstantVelocityFilter is the Kalman-style alternative: a constant-velocity
model with fixed process/measurement noise.  Included for comparison; its lag is
constant regardless of how fast you move, which is usually the worse trade here.
"""

import numpy as np


class _LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        self.y = x if self.y is None else alpha * x + (1 - alpha) * self.y
        return self.y


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * np.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """Speed-adaptive low-pass over a vector signal."""

    def __init__(self, min_cutoff=1.0, beta=0.7, d_cutoff=1.0, dim=3):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dim = dim
        self.reset()

    def reset(self):
        self._x = _LowPass()
        self._dx = _LowPass()
        self._prev = None

    def __call__(self, x, dt):
        x = np.asarray(x, float)
        if self._prev is None:
            self._prev = x.copy()
        dx = (x - self._prev) / max(dt, 1e-6)
        self._prev = x.copy()
        edx = self._dx(dx, _alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(edx)
        return self._x(x, _alpha(cutoff, dt))


class ConstantVelocityFilter:
    """Kalman filter with a constant-velocity model, per axis.

    Fixed gains mean fixed lag: steadier than raw input, but it does not back off
    when you move deliberately the way OneEuroFilter does.
    """

    def __init__(self, process_var=25.0, meas_var=1e-4, dim=3):
        self.q = float(process_var)
        self.r = float(meas_var)
        self.dim = dim
        self.reset()

    def reset(self):
        self.x = None          # (dim, 2): position, velocity
        self.P = None

    def __call__(self, z, dt):
        z = np.asarray(z, float)
        if self.x is None:
            self.x = np.stack([z, np.zeros_like(z)], axis=1)
            self.P = np.tile(np.eye(2) * 1e-3, (len(z), 1, 1))
            return z.copy()
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        H = np.array([1.0, 0.0])
        out = np.empty(len(z))
        for i in range(len(z)):
            x = F @ self.x[i]
            P = F @ self.P[i] @ F.T + Q
            y = z[i] - H @ x
            S = H @ P @ H + self.r
            K = P @ H / S
            x = x + K * y
            P = (np.eye(2) - np.outer(K, H)) @ P
            self.x[i], self.P[i] = x, P
            out[i] = x[0]
        return out


# Each filter takes only its own parameters, so the config can carry both sets
# and `kind` selects which are used. Passing the wrong set used to raise.
_PARAMS = {
    "oneeuro": ("min_cutoff", "beta", "d_cutoff"),
    "kalman": ("process_var", "meas_var"),
}
_ALIASES = {"one_euro": "oneeuro", "1euro": "oneeuro",
            "cv": "kalman", "constant_velocity": "kalman"}


def make_filter(kind, dim=3, **kw):
    kind = _ALIASES.get((kind or "none").lower(), (kind or "none").lower())
    if kind in ("none", "off"):
        return None
    if kind not in _PARAMS:
        raise ValueError("unknown filter %r; expected one of none, %s"
                         % (kind, ", ".join(sorted(_PARAMS))))
    args = {k: v for k, v in kw.items() if k in _PARAMS[kind]}
    cls = OneEuroFilter if kind == "oneeuro" else ConstantVelocityFilter
    return cls(dim=dim, **args)
