import numpy as np

from .base import WaveSimulation


class SWaveSimulation(WaveSimulation):
    """Finite-difference solver for shear (S) waves."""

    def __init__(self, f0=15.0, source_pos=None, source_func=None, **kwargs):
        kwargs.setdefault("c", 1500.0)
        kwargs.setdefault("dx", 5.0)
        kwargs.setdefault("dt", 0.0005)
        super().__init__(**kwargs)
        self.f0 = f0
        if source_pos is None:
            source_pos = (self.n // 2, self.n // 2)
        self.source_pos = source_pos
        self.initialize(amplitude=0.0, source_func=source_func)

    @staticmethod
    def ricker_wavelet(t, f0):
        tau = 1.0 / f0
        return (1.0 - 2.0 * (np.pi ** 2) * (f0 ** 2) * (t - tau) ** 2) * np.exp(
            -(np.pi ** 2) * (f0 ** 2) * (t - tau) ** 2
        )

    def step(self):
        c2 = (self.c * self.dt / self.dx) ** 2
        laplacian = (
            np.roll(self.u_curr, 1, axis=0)
            + np.roll(self.u_curr, -1, axis=0)
            + np.roll(self.u_curr, 1, axis=1)
            + np.roll(self.u_curr, -1, axis=1)
            - 4 * self.u_curr
        )
        u_next = 2 * self.u_curr - self.u_prev + c2 * laplacian

        amp = self.ricker_wavelet(self.time, self.f0)
        sx, sy = self.source_pos
        u_next[sx, sy] += amp

        if self.boundary == "reflective":
            u_next[0, :] = 0
            u_next[-1, :] = 0
            u_next[:, 0] = 0
            u_next[:, -1] = 0
        elif self.boundary == "periodic":
            pass
        elif self.boundary == "absorbing":
            u_next[0, :] = u_next[1, :]
            u_next[-1, :] = u_next[-2, :]
            u_next[:, 0] = u_next[:, 1]
            u_next[:, -1] = u_next[:, -2]
        else:
            raise ValueError(f"Unknown boundary condition {self.boundary}")

        self.u_prev, self.u_curr = self.u_curr, u_next
        self.time += self.dt
        return u_next


class SHWaveSimulation(SWaveSimulation):
    """Horizontally polarised shear wave.

    This subclass does not change the numerical scheme from
    :class:`SWaveSimulation` but serves to emphasise that the simulated field
    represents the out-of-plane displacement ``u_y``.  All solver parameters and
    the ``step`` method are inherited unchanged.
    """

    # No additional functionality required; the base SWaveSimulation already
    # implements the finite-difference update for a shear wave field.
    pass


class SVWaveSimulation(SWaveSimulation):
    """Vertically polarised shear wave.

    As with :class:`SHWaveSimulation`, this class simply inherits the scalar
    shear-wave solver from :class:`SWaveSimulation`.  The field can be
    interpreted as a potential whose spatial derivatives give the in-plane
    displacement components ``(u_x, u_z)``.  The numerical scheme remains the
    same; the class exists mainly for clarity when constructing simulations.
    """

    pass
