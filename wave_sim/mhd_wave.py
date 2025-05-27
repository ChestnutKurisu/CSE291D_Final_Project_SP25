from .base import WaveSimulation
from .mhd_solver import MHDSolver

class MHDWave(WaveSimulation):
    """Simplified MHD wave using a stub solver."""

    def __init__(self, solver=None, source_func=None, **kwargs):
        kwargs.setdefault("c", 1.0)
        kwargs.setdefault("boundary", "periodic")
        super().__init__(**kwargs)
        self.solver = solver or MHDSolver()
        self.initialize(amplitude=1.0, source_func=source_func)
