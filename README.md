# Wave Simulation Examples

This repository contains a minimal framework for simulating and animating simple wave phenomena.

The repository now includes a simple subclass for each of **70** different wave
types collected in ``wave_sim.wave_catalog``.  These range from seismic and
acoustic examples to fluid, electromagnetic and plasma waves, all reusing the
same underlying 2‑D solver for illustration.

## Usage Notes

`WaveSimulation` now supports several boundary conditions and flexible initial
sources. The boundary condition can be selected when constructing a simulation
via the `boundary` keyword with one of:

* `"reflective"` – fixed edges (default)
* `"periodic"` – domain repeats at the edges
* `"absorbing"` – simple absorbing boundary

The initial disturbance can be provided through `initialize(source_func=...)`
where `source_func` is a function of coordinate arrays ``(X, Y)``. For example,
a Gaussian pulse centered in the domain can be created with:

```python
def gaussian(X, Y, sigma=5):
    x0 = X.shape[0] // 2
    y0 = Y.shape[1] // 2
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

sim = PWaveSimulation(boundary="absorbing")
sim.initialize(source_func=gaussian)
```

The solver emits a warning if the CFL condition ``c * dt / dx`` exceeds
``1 / sqrt(2)`` to help maintain stability.

Animations can be generated via ``examples/all_waves_collage.py`` which iterates
over every class in the catalog and writes a short MP4 file for each one into an
``output`` directory.
