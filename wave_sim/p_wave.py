from .base import WaveSimulation


class PWaveSimulation(WaveSimulation):
    """Simple representation of a seismic P-wave (compressional).

    The propagation speed defaults to ``1.0`` which mirrors the reference
    value used for the fastest body waves in elastic solids.
    """

    def __init__(self, source_func=None, **kwargs):
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0, source_func=source_func)
