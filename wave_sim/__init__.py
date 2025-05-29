"""Wave simulation utilities."""

from .simulation import run_simulation
from .wave_equations import solve_p_wave, solve_s_wave

__all__ = [
    "run_simulation",
    "solve_p_wave",
    "solve_s_wave",
]
