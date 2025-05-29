"""Wave simulation utilities."""

from .simulation import run_simulation
from .elastic_waves import simulate_p_wave, simulate_s_wave

__all__ = [
    "run_simulation",
    "simulate_p_wave",
    "simulate_s_wave",
]

