from .base import WaveSimulation

class PWaveSimulation(WaveSimulation):
    """Simple representation of a seismic P-wave."""

    def __init__(self, source_func=None, **kwargs):
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0, source_func=source_func)
