from .base import WaveSimulation

class AcousticMode(WaveSimulation):
    """Acoustic mode in a resonant cavity."""

    def __init__(self, source_func=None, **kwargs):
        kwargs.setdefault("c", 0.9)
        kwargs.setdefault("boundary", "reflective")
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0, source_func=source_func)
