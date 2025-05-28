# Wave Simulation Examples

This repository contains a minimal framework for simulating and animating simple wave phenomena.  The core solver now supports optional GPU acceleration via ``cupy`` for higher resolution and smoother animations when available.

The repository now ships with implementations for **20** illustrative wave
types collected in ``wave_sim.wave_catalog``.  These cover a mix of seismic,
acoustic and fluid phenomena while all reusing the same underlying 2‑D solver
for simplicity.

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

For the specialised shear-wave solvers ``SHWaveSimulation`` and
``SVWaveSimulation`` the physical displacement can be retrieved directly from
the simulation state:

```python
sh = SHWaveSimulation()
uy = sh.displacement()               # out-of-plane displacement

sv = SVWaveSimulation()
ux, uz = sv.displacement_components()  # in-plane components
```

Animations can be generated via ``examples/all_waves_collage.py`` which
iterates over every class in the catalog and writes a short MP4 file for each
one into an ``output`` directory.  The script now also assembles these movies
into a single collage video.  GPU acceleration via ``cupy`` is used by default
for smoother, high-resolution output when available.


Additional standalone scripts in the ``examples`` directory demonstrate simple
1‑D or spectral solvers for several textbook waves:

* ``internal_gravity_wave.py`` – finite difference scheme for a stratified fluid.
* ``kelvin_wave.py`` – rotating shallow water Kelvin wave.
* ``rossby_planetary_wave.py`` – linear Rossby wave via FFTs.
* ``flexural_beam_wave.py`` – Euler–Bernoulli beam equation.
* ``alfven_wave.py`` – 1‑D Alfv\u00e9n wave along a magnetic field.

Each script runs a short simulation and plots the final state when executed.
