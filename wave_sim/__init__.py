"""Public interface for the :mod:`wave_sim` package."""

# Explicitly re-export the supported wave simulation classes so that external
# users can simply do ``from wave_sim import PWaveSimulation``.
from .base import WaveSimulation
from .p_wave import PWaveSimulation
from .s_wave import SWaveSimulation

__all__ = [
    'WaveSimulation',
    'PWaveSimulation',
    'SWaveSimulation',
]
