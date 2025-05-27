"""Collection of wave simulations grouped by physical phenomena.

This module contains concrete subclasses of :class:`~wave_sim.base.WaveSimulation`
for different physical wave types. Each class sets sensible defaults for the
wave speed ``c`` and initializes with a unit amplitude source in the middle of
the grid. Use these classes directly to experiment with various wave
phenomena without needing to configure the base solver each time.
"""

from .base import WaveSimulation


# ---------------------------------------------------------------------------
# Seismic waves
# ---------------------------------------------------------------------------

class SeismicPWave(WaveSimulation):
    """Primary (P) seismic wave.

    Represents the fastest seismic body wave which propagates via
    compressions and expansions in the material.

    Parameters
    ----------
    grid_size : int, optional
        Number of grid points along one dimension. Defaults to ``100``.
    c : float, optional
        Propagation speed of the P-wave. Defaults to ``1.0``.
    dx : float, optional
        Spatial discretization step. Defaults to ``1.0``.
    dt : float, optional
        Time step for the simulation. Defaults to ``0.1``.

    Examples
    --------
    >>> from wave_sim.wave_catalog import SeismicPWave
    >>> sim = SeismicPWave(grid_size=50)
    >>> frames = sim.simulate(steps=10)
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


class SeismicSWave(WaveSimulation):
    """Secondary (S) seismic wave.

    S-waves travel slower than P-waves and involve shear motion
    perpendicular to the propagation direction.

    Parameters are the same as :class:`SeismicPWave` but with a lower
    default wave speed.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.6)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


# ---------------------------------------------------------------------------
# Acoustic waves
# ---------------------------------------------------------------------------

class AcousticWave(WaveSimulation):
    """Pressure wave in an acoustic medium (e.g. sound in air).

    This simulation simply models a generic acoustic wave using the base
    solver. The default wave speed is chosen to roughly mimic sound in air
    in arbitrary units.

    Examples
    --------
    >>> from wave_sim.wave_catalog import AcousticWave
    >>> sim = AcousticWave(grid_size=100, c=0.9)
    >>> sim.animate(steps=20)
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.9)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


# ---------------------------------------------------------------------------
# Fluid waves
# ---------------------------------------------------------------------------

class FluidSurfaceWave(WaveSimulation):
    """Simple representation of a surface wave on a fluid.

    The wave speed is typically slower than acoustic or seismic body waves.
    This class can be used to mimic ripple propagation on water in a very
    idealized manner.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 0.4)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


# ---------------------------------------------------------------------------
# Electromagnetic waves
# ---------------------------------------------------------------------------

class ElectromagneticWave(WaveSimulation):
    """Propagation of an electromagnetic wave in vacuum.

    The default wave speed is set to ``1.0`` which corresponds to the speed
    of light in normalized units. Although the base solver does not model
    electromagnetic fields explicitly, it can visualize the propagation of a
    wavefront at constant speed.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("c", 1.0)
        super().__init__(**kwargs)
        self.initialize(amplitude=1.0)


__all__ = [
    "SeismicPWave",
    "SeismicSWave",
    "AcousticWave",
    "FluidSurfaceWave",
    "ElectromagneticWave",
]
