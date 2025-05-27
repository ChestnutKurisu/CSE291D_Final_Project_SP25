from .base import WaveSimulation
from .p_wave import PWaveSimulation
from .s_wave import SWaveSimulation
from .wave_catalog import (
    SeismicPWaveSimulation,
    SeismicSWaveSimulation,
    InternalGravityWaveSimulation,
    ElectromagneticWaveSimulation,
)

__all__ = [
    'WaveSimulation',
    'PWaveSimulation',
    'SWaveSimulation',
    'SeismicPWaveSimulation',
    'SeismicSWaveSimulation',
    'InternalGravityWaveSimulation',
    'ElectromagneticWaveSimulation',
]
