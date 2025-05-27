import numpy as np
from .base import WaveSimulation


class SeismicPWaveSimulation(WaveSimulation):
    """P-wave modeled via linear elasticity."""

    def __init__(self, density=2500.0, bulk_modulus=30e9, **kwargs):
        c = np.sqrt(bulk_modulus / density)
        super().__init__(c=c, **kwargs)
        self.density = density
        self.bulk_modulus = bulk_modulus
        self.initialize(amplitude=1.0)


class SeismicSWaveSimulation(WaveSimulation):
    """S-wave modeled via linear elasticity."""

    def __init__(self, density=2500.0, shear_modulus=30e9, **kwargs):
        c = np.sqrt(shear_modulus / density)
        super().__init__(c=c, **kwargs)
        self.density = density
        self.shear_modulus = shear_modulus
        self.initialize(amplitude=1.0)


class InternalGravityWaveSimulation(WaveSimulation):
    """Simplified internal gravity wave using a damping term."""

    def __init__(self, density=1025.0, buoyancy_freq=0.01, damping=0.01, **kwargs):
        c = 1.0 / buoyancy_freq
        super().__init__(c=c, **kwargs)
        self.density = density
        self.buoyancy_freq = buoyancy_freq
        self.damping = damping
        self.initialize(amplitude=0.1)

    def step(self):
        c2 = (self.c * self.dt / self.dx) ** 2
        laplacian = (
            np.roll(self.u_curr, 1, axis=0)
            + np.roll(self.u_curr, -1, axis=0)
            + np.roll(self.u_curr, 1, axis=1)
            + np.roll(self.u_curr, -1, axis=1)
            - 4 * self.u_curr
        )
        u_next = ((2 - self.damping) * self.u_curr
                  - (1 - self.damping) * self.u_prev
                  + c2 * laplacian)
        u_next[0, :] = 0
        u_next[-1, :] = 0
        u_next[:, 0] = 0
        u_next[:, -1] = 0
        self.u_prev, self.u_curr = self.u_curr, u_next
        return u_next


class ElectromagneticWaveSimulation(WaveSimulation):
    """Electromagnetic wave solved via scalar wave equation."""

    def __init__(self, permittivity=8.854e-12, permeability=4 * np.pi * 1e-7, **kwargs):
        c = 1.0 / np.sqrt(permittivity * permeability)
        super().__init__(c=c, **kwargs)
        self.permittivity = permittivity
        self.permeability = permeability
        self.initialize(amplitude=1.0)
