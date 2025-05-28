"""Subset of wave simulations.

The classes here are **simple demonstration wrappers** around
:class:`~wave_sim.base.WaveSimulation`.  They merely choose convenient default
wave speeds and initialise the field with a unit impulse.  The specialised
surface/interface wave names (Rayleigh, Love, Stoneley, Scholte, etc.) do not
implement the full elastic or multi-layer physics that would be required for
realistic models.  They should therefore be viewed as *placeholders* useful for
illustrative animations only.
"""

import numpy as np
from .base import WaveSimulation
from .solvers import (
    PWaveSimulation,
    SWaveSimulation,
    SHWaveSimulation,
    SVWaveSimulation,
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


###############################################################################
# SEISMIC BODY WAVES
###############################################################################

class PrimaryWave(PWaveSimulation):
    """Compressional body wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 3000.0)
        super().__init__(**kwargs)


class SecondaryWave(SWaveSimulation):
    """Shear body wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1500.0)
        super().__init__(**kwargs)


class SHWave(SHWaveSimulation):
    """Horizontally polarised shear."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1500.0)
        super().__init__(**kwargs)


class SVWave(SVWaveSimulation):
    """Vertically polarised shear."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1500.0)
        super().__init__(**kwargs)


###############################################################################
# SURFACE AND GUIDED WAVES
###############################################################################

class RayleighWave(WaveSimulation):
    """Placeholder Rayleigh surface wave.

    The real Rayleigh wave involves coupled traction-free elastic equations.
    This simplified version merely solves a single scalar wave equation with a
    default wave speed.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.53)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class LoveWave(WaveSimulation):
    """Placeholder Love surface wave.

    As with :class:`RayleighWave`, this is a minimal scalar solver and does not
    implement layered shear wave physics.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.50)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class LambS0Mode(WaveSimulation):
    """Placeholder Lamb S0 (symmetric) mode."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.65)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class LambA0Mode(WaveSimulation):
    """Placeholder Lamb A0 (antisymmetric) mode."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.60)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class StoneleyWave(WaveSimulation):
    """Placeholder Stoneley solid-solid interface wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.45)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class ScholteWave(WaveSimulation):
    """Placeholder Scholte solid-fluid interface wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.40)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


###############################################################################
# ACOUSTIC WAVES IN FLUIDS
###############################################################################

###############################################################################
# FLUID SURFACE AND INTERNAL WAVES
###############################################################################

class InternalGravityWave(InternalGravityWaveSolver):
    """Internal gravity wave."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_conditions(lambda x: np.exp(-100 * (x - self.L / 2) ** 2))


class KelvinWave(KelvinWaveSolver):
    """Kelvin wave."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_conditions(lambda y: np.exp(-((y - self.L / 4) / 0.5) ** 2))


class RossbyPlanetaryWave(RossbyPlanetaryWaveSolver):
    """Rossby planetary wave."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_conditions(lambda X, Y: np.exp(-((X - self.Lx / 2) ** 2 + (Y - self.Ly / 2) ** 2) / 0.2))


###############################################################################
# STRUCTURAL AND PLASMA WAVES
###############################################################################

class FlexuralBeamWave(FlexuralBeamWaveSolver):
    """Flexural beam wave."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_conditions(lambda x: np.exp(-100 * (x - self.L / 2) ** 2))


class AlfvenWave(AlfvenWaveSolver):
    """Alfv\u00e9n wave."""

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
]
