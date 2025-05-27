from .base import WaveSimulation

class GuidedWave(WaveSimulation):
    """Wave confined in a guiding structure."""

    def __init__(self, source_func=None, **kwargs):
        kwargs.setdefault("c", 0.8)
        kwargs.setdefault("boundary", "reflective")
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0, source_func=source_func)
