"""Subset of wave simulations.

This module defines minimal :class:`~wave_sim.base.WaveSimulation` subclasses
for a curated list of wave types spanning seismic, acoustic and fluid
phenomena.  Each class simply sets a default wave speed ``c`` and initialises
the field with a unit impulse in the centre of the grid.
"""

from .base import WaveSimulation
from .p_wave import PWaveSimulation
from .s_wave import SWaveSimulation, SHWaveSimulation, SVWaveSimulation
from .basic_wave_solvers import (
    PlaneAcousticWave,
    SphericalAcousticWave,
    DeepWaterGravityWave,
    ShallowWaterGravityWave,
    CapillaryWave,
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
    """Rayleigh surface wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.53)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class LoveWave(WaveSimulation):
    """Love surface wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.50)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class LambS0Mode(WaveSimulation):
    """Lamb S0 (symmetric) mode."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.65)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class LambA0Mode(WaveSimulation):
    """Lamb A0 (antisymmetric) mode."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.60)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class StoneleyWave(WaveSimulation):
    """Stoneley solid-solid interface wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.45)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class ScholteWave(WaveSimulation):
    """Scholte solid-fluid interface wave."""

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

class InternalGravityWave(WaveSimulation):
    """Internal gravity wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.2)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class KelvinWave(WaveSimulation):
    """Kelvin wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.2)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class RossbyPlanetaryWave(WaveSimulation):
    """Rossby planetary wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.05)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


###############################################################################
# STRUCTURAL AND PLASMA WAVES
###############################################################################

class FlexuralBeamWave(WaveSimulation):
    """Flexural beam wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.4)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class AlfvenWave(WaveSimulation):
    """Alfv\u00e9n wave."""

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


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
