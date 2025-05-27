from .base import WaveSimulation
from .em_solver import EMSolver

class EMWave(WaveSimulation):
    """Simplified electromagnetic wave using a stub solver."""

    def __init__(self, solver=None, source_func=None, **kwargs):
        kwargs.setdefault("c", 1.0)
        kwargs.setdefault("boundary", "periodic")
        super().__init__(**kwargs)
        self.solver = solver or EMSolver()
        self.initialize(amplitude=1.0, source_func=source_func)
