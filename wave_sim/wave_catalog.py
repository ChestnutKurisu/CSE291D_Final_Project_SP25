"""Wave catalog with configurations for the high quality solver."""

from __future__ import annotations

import numpy as np

from .high_quality import ConstantSpeed, PointSource
from .initial_conditions import gaussian_2d


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
    """PLACEHOLDER: uses scalar solver with tuned speed only.

    A real Rayleigh wave arises from coupled P-- and S--wave motion
    in an elastic half‐space.  This configuration merely reuses the
    scalar ``WaveSimulator2D`` and so lacks the correct elliptic
    particle motion or dispersion.  It is included for illustrative
    purposes only.
    """

    default_speed = 1300.0


class LoveWave(Wave2DConfig):
    """PLACEHOLDER: horizontally polarised surface shear.

    The model again relies on the scalar wave equation and therefore
    omits the vector shear‐strain behaviour that characterises real
    Love waves.  Only the propagation speed is representative.
    """

    default_speed = 1400.0


class LambS0Mode(Wave2DConfig):
    """PLACEHOLDER: symmetric Lamb mode.

    This class simply sets a constant phase speed for a plate wave but
    does not implement the frequency‐dependent dispersion of true Lamb
    modes in elastic plates.
    """

    default_speed = 2000.0


class LambA0Mode(Wave2DConfig):
    """PLACEHOLDER: antisymmetric Lamb mode.

    As with ``LambS0Mode`` this uses the scalar solver and ignores the
    dispersive nature of real A0 plate waves.  It should be viewed as an
    illustrative demo rather than a physically faithful model.
    """

    default_speed = 1000.0


class StoneleyWave(Wave2DConfig):
    """PLACEHOLDER: interface wave at a solid–solid boundary.

    Stoneley waves involve coupled shear motion across a material
    interface.  This simplified configuration propagates a scalar field
    with a single constant speed and therefore lacks the characteristic
    vector displacements.
    """

    default_speed = 1200.0


class ScholteWave(Wave2DConfig):
    """PLACEHOLDER: ocean–bottom interface wave.

    Real Scholte waves arise from an elastic–fluid boundary and exhibit
    strong dispersion.  Here we merely reuse the scalar solver with a
    fixed speed, so the behaviour is only qualitatively similar.
    """

    default_speed = 900.0




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
