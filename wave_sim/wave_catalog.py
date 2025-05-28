"""Wave catalog with configurations for the high quality solver."""

from __future__ import annotations

import numpy as np

from .solvers import (
    PlaneAcousticWave,
    SphericalAcousticWave,
    DeepWaterGravityWave,
    ShallowWaterGravityWave,
    CapillaryWave,
    InternalGravityWave as InternalGravityWaveSolver,
    KelvinWave as KelvinWaveSolver,
    RossbyPlanetaryWave as RossbyPlanetaryWaveSolver,
    FlexuralBeamWave as FlexuralBeamWaveSolver,
    AlfvenWave as AlfvenWaveSolver,
)
from .high_quality import ConstantSpeed, PointSource


def gaussian_initial_condition(X: np.ndarray, Y: np.ndarray, sigma: float = 5.0):
    """Return a Gaussian pulse centered in the grid."""
    x0 = X.shape[0] // 2
    y0 = Y.shape[1] // 2
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))


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
    default_speed = 1300.0


class LoveWave(Wave2DConfig):
    default_speed = 1400.0


class LambS0Mode(Wave2DConfig):
    default_speed = 2000.0


class LambA0Mode(Wave2DConfig):
    default_speed = 1000.0


class StoneleyWave(Wave2DConfig):
    default_speed = 1200.0


class ScholteWave(Wave2DConfig):
    default_speed = 900.0


class InternalGravityWave(InternalGravityWaveSolver):
    is_2d_fd = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_conditions(lambda x: np.exp(-100 * (x - self.L / 2) ** 2))


class KelvinWave(KelvinWaveSolver):
    is_2d_fd = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_conditions(lambda y: np.exp(-((y - self.L / 2) / (self.L/20)) ** 2))


class RossbyPlanetaryWave(RossbyPlanetaryWaveSolver):
    is_2d_fd = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_conditions(
            lambda X, Y: np.exp(-((X - self.Lx / 2) ** 2 + (Y - self.Ly / 2) ** 2) / ((self.Lx/10) ** 2))
        )


class FlexuralBeamWave(FlexuralBeamWaveSolver):
    is_2d_fd = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_conditions(lambda x: np.exp(-100 * (x - self.L / 2) ** 2))


class AlfvenWave(AlfvenWaveSolver):
    is_2d_fd = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_conditions(lambda x: np.sin(2 * np.pi * x / self.L))


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
    "PlaneAcousticWave",
    "SphericalAcousticWave",
    "DeepWaterGravityWave",
    "ShallowWaterGravityWave",
    "CapillaryWave",
    "InternalGravityWave",
    "KelvinWave",
    "RossbyPlanetaryWave",
    "FlexuralBeamWave",
    "AlfvenWave",
    "gaussian_initial_condition",
]
