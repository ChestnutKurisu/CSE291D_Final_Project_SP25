from .base import WaveSimulation

class SWaveSimulation(WaveSimulation):
    """Simple representation of a seismic S-wave using lower wave speed."""

    def __init__(self, source_func=None, **kwargs):
        kwargs.setdefault('c', 0.6)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0, source_func=source_func)
