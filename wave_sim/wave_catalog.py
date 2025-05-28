"""Wave catalog with configurations for the high quality solver."""

from __future__ import annotations

import numpy as np

from .high_quality import ConstantSpeed, PointSource
from .initial_conditions import gaussian_2d

# --- NEW: analytical phase-speed helpers ------------------------------------
from .dispersion import (
    rayleigh_wave_speed,
    love_wave_dispersion,
    lamb_s0_mode,
    lamb_a0_mode,
    stoneley_wave_speed,
    scholte_wave_speed,
)

# Default elastic constants (quick-demo values)
_ALPHA = 3000.0      # m s-1  P-wave
_BETA  = 1500.0      # m s-1  S-wave
_RHO   = 2000.0      # kg m-3
_WATER_C   = 1480.0  # m s-1
_WATER_RHO = 1000.0  # kg m-3


def gaussian_initial_condition(X: np.ndarray, Y: np.ndarray, sigma: float = 5.0):
    """Return a Gaussian pulse centered in the grid."""
    return gaussian_2d(X, Y, sigma=sigma)


class Wave2DConfig:
    """Base configuration for 2-D waves using ``WaveSimulator2D``."""

    is_2d_fd = True
    default_speed = 1.0

    def __init__(self, c: float | None = None, **kwargs):
        self.c = c if c is not None else self.default_speed
        self.source_params = kwargs.get("source_params", {})
        self.initial_condition = kwargs.get("initial_condition")

    def get_scene_builder(self):
        def builder(resolution):
            w, h = resolution
            objs = [ConstantSpeed(self.c)]
            if self.initial_condition is not None:
                X, Y = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
                init = self.initial_condition(X, Y)
            else:
                init = None
            sp = self.source_params
            if sp:
                x = sp.get("x", w // 2)
                y = sp.get("y", h // 2)
                freq = sp.get("freq", 0.1)
                amp = sp.get("amplitude", 5.0)
                objs.append(PointSource(x, y, freq=freq, amplitude=amp))
            return objs, w, h, init

        return builder


class PrimaryWave(Wave2DConfig):
    """Compressional body wave."""

    default_speed = 3000.0


class SecondaryWave(Wave2DConfig):
    """Shear body wave."""

    default_speed = 1500.0


class SHWave(Wave2DConfig):
    """Horizontally polarised shear."""

    default_speed = 1500.0

    def get_displacement_y(self, frame: np.ndarray) -> np.ndarray:
        return frame


class SVWave(Wave2DConfig):
    """Vertically polarised shear using scalar potential."""

    default_speed = 1500.0

    def get_displacement_components(self, frame: np.ndarray, dx: float = 1.0):
        dpsi_dz, dpsi_dx = np.gradient(frame, dx, dx, edge_order=2)
        ux = dpsi_dz
        uz = -dpsi_dx
        return ux, uz


class RayleighWave(Wave2DConfig):
    """Rayleigh surface wave – scalar surrogate with realistic c_R."""

    def __init__(self, **kw):
        c_R = rayleigh_wave_speed(_ALPHA, _BETA)
        super().__init__(c=c_R, **kw)


class LoveWave(Wave2DConfig):
    """Love surface wave – fundamental mode phase speed (~10 Hz)."""

    def __init__(self, h: float = 100.0, **kw):
        c_L = love_wave_dispersion(freq=2*np.pi*10,
                                   beta1=_BETA*0.8,
                                   beta2=_BETA, h=h, n_modes=1)[0]
        super().__init__(c=c_L, **kw)


class LambS0Mode(Wave2DConfig):
    """Symmetric Lamb plate mode – scalar surrogate."""

    def __init__(self, plate_h: float = 5.0, **kw):
        c = lamb_s0_mode(freq=2*np.pi*5, alpha=_ALPHA, beta=_BETA,
                         thickness=plate_h)
        super().__init__(c=c or 2000.0, **kw)


class LambA0Mode(Wave2DConfig):
    """Antisymmetric Lamb plate mode."""

    def __init__(self, plate_h: float = 5.0, **kw):
        c = lamb_a0_mode(freq=2*np.pi*5, alpha=_ALPHA, beta=_BETA,
                         thickness=plate_h)
        super().__init__(c=c or 1000.0, **kw)


class StoneleyWave(Wave2DConfig):
    """Stoneley solid–solid interface – scalar surrogate."""

    def __init__(self, **kw):
        c = stoneley_wave_speed(_ALPHA, _BETA, _RHO,
                                _ALPHA*0.6, _BETA*0.6, _RHO*1.2)
        super().__init__(c=c or 1200.0, **kw)


class ScholteWave(Wave2DConfig):
    """Scholte solid–fluid interface – scalar surrogate."""

    def __init__(self, **kw):
        c = scholte_wave_speed(_ALPHA, _BETA, _RHO,
                               _WATER_C, _WATER_RHO)
        super().__init__(c=c or 900.0, **kw)




__all__ = [
    "PrimaryWave",
    "SecondaryWave",
    "SHWave",
    "SVWave",
    "RayleighWave",
    "LoveWave",
    "LambS0Mode",
    "LambA0Mode",
    "StoneleyWave",
    "ScholteWave",
    "gaussian_initial_condition",
]
