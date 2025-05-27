from .base import WaveSimulation

class FluidInternalWave(WaveSimulation):
    """Idealized internal wave within a stratified fluid."""

    def __init__(self, source_func=None, **kwargs):
        kwargs.setdefault("c", 0.3)
        kwargs.setdefault("boundary", "absorbing")
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0, source_func=source_func)
