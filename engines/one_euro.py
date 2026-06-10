# =============================================================================
# engines/one_euro.py
# -----------------------------------------------------------------------------
# The 1€ (One Euro) filter — the standard adaptive low-pass filter for real-time
# noisy signals (Casiez, Roussel & Vogel, CHI 2012). It removes jitter at low
# speed (when humans notice shake) while keeping lag low at high speed (when
# humans notice latency), by raising the cutoff frequency with the signal's
# speed.
#
# We use it to stabilise the driving motion (head pitch/yaw/roll, translation,
# scale, expression) and the face-crop box, so the avatar tracks smoothly with
# no shimmer/jitter — what previously read as "tracking errors" on movement.
#
# Works elementwise on Python floats, NumPy arrays, AND torch tensors (the kp
# fields are torch tensors on the GPU — kept on-device, no host transfer).
#
# Reference: https://jaantollander.com/post/noise-filtering-using-one-euro-filter/
# =============================================================================

import math
import time


def _abs(x):
    """abs() that works for floats, numpy arrays, and torch tensors."""
    a = getattr(x, "abs", None)
    return a() if callable(a) else abs(x)


class OneEuroFilter:
    """Adaptive low-pass filter for one (possibly vector/tensor) signal.

    min_cutoff : lower = more smoothing of slow movement (less jitter)
    beta       : higher = less lag during fast movement (more responsive)
    d_cutoff   : cutoff for the derivative estimate (usually 1.0)
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        """Forget history — the next sample passes through unfiltered."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t=None):
        """Filter one sample x at time t (seconds). Returns the smoothed value."""
        if t is None:
            t = time.monotonic()
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = x * 0.0
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-3
        # After a long gap (e.g. face was lost), don't let a stale derivative
        # fight the new value — snap toward it.
        if dt > 0.25:
            self.x_prev = x
            self.dx_prev = x * 0.0
            self.t_prev = t
            return x

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * _abs(dx_hat)
        a = self._alpha(cutoff, dt)          # scalar or elementwise array/tensor
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


if __name__ == "__main__":
    import random
    # Tuned light (high cutoff, modest beta) so it denoises without lagging the
    # signal — the regime we use for head motion.
    f = OneEuroFilter(min_cutoff=3.0, beta=0.01)
    t = 0.0
    jittery = smoothed_err = 0.0
    for i in range(300):
        t += 1 / 30.0
        true = math.sin(i / 35.0)
        noisy = true + random.uniform(-0.08, 0.08)
        out = f(noisy, t)
        jittery += abs(noisy - true)
        smoothed_err += abs(out - true)
    print(f"[ONE_EURO] mean error  raw={jittery/300:.4f}  filtered={smoothed_err/300:.4f}")
    print("[ONE_EURO] OK — filter reduces noise" if smoothed_err < jittery else "?? over-smoothed")
