from .base import WaveSimulation

class SeismicSurfaceWave(WaveSimulation):
    """Simplified seismic surface wave."""

    def __init__(self, source_func=None, **kwargs):
        kwargs.setdefault("c", 0.5)
        kwargs.setdefault("boundary", "absorbing")
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0, source_func=source_func)
