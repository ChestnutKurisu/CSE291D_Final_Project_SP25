# Wave Simulation Examples

This repository contains a minimal framework for simulating and animating simple wave phenomena.  The code has recently been consolidated around a single high quality pipeline.  The core solver now supports optional GPU acceleration via ``cupy`` for higher resolution and smoother animations when available.

The focus of this repository is purely on wave propagation. Earlier project
notes about simulating elastic bodies via incremental potentials and Hessians
are not implemented here.

The repository now ships with implementations for **20** illustrative wave
types collected in ``wave_sim.wave_catalog``.  These cover a mix of seismic,
acoustic and fluid phenomena.  Basic body waves share a common 2‑D finite
difference solver, while other entries use one‑dimensional or spectral
schemes specialised for each equation.  Surface and interface waves
(Rayleigh, Love, Lamb, Stoneley, Scholte) are **highly simplified**: they are
animated using a generic scalar wave model with preset speeds and do **not**
capture the true dispersive or vector nature of these waves.

Legacy modules under ``wave_sim2d`` have been removed.  All animations now
use the GPU optimised utilities found in ``wave_sim.high_quality`` which are
also able to fall back to NumPy when a CUDA device is not available.

## Usage Notes

`WaveSimulator2D` supports several boundary conditions and flexible initial
sources. The boundary condition can be selected when constructing a simulation
via the `boundary_condition` keyword with one of:

* `"reflective"` – zero normal derivative at the edges (mirror boundary)
* `"periodic"` – domain repeats at the edges
* `"absorbing"` – damped sponge layer near the borders

The initial disturbance is passed via ``initial_field`` when creating a
`WaveSimulator2D` or through the ``scene_builder`` used by
`simulate_wave`. A Gaussian pulse centered in the domain can be created with:

```python
def gaussian(X, Y, sigma=5):
    x0 = X.shape[0] // 2
    y0 = Y.shape[1] // 2
    return np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

from wave_sim.high_quality import simulate_wave, ConstantSpeed
from wave_sim.wave_catalog import PrimaryWave, gaussian_initial_condition

scene = PrimaryWave(initial_condition=gaussian).get_scene_builder()
simulate_wave(scene, "out.mp4", steps=200, resolution=(256, 256))
```

The solver emits a warning if the CFL condition ``c * dt / dx`` exceeds
``1 / sqrt(2)`` to help maintain stability.

For the specialised shear-wave configurations ``SHWave`` and ``SVWave`` the
physical displacement can be retrieved from simulation frames via helper
methods:

```python
sh = SHWave()
uy = sh.get_displacement_y(frame)

sv = SVWave()
ux, uz = sv.get_displacement_components(frame, dx=1.0)
```

Animations can be generated via ``examples/high_quality_collage.py`` which
uses the :mod:`wave_sim.high_quality` package to produce GPU accelerated movies
for several wave classes.  The script writes an MP4 per wave type and then
assembles all clips into a collage video.  Because the underlying animator is a
generic scalar wave solver, only the basic body-wave examples are physically
meaningful; the more complex surface/interface waves are shown merely for
illustration.


Additional standalone scripts in the ``examples`` directory demonstrate simple
1‑D or spectral solvers for several textbook waves:

* ``internal_gravity_wave.py`` – finite difference scheme for a stratified fluid.
* ``kelvin_wave.py`` – rotating shallow water Kelvin wave.
* ``rossby_planetary_wave.py`` – linear Rossby wave via FFTs.
* ``flexural_beam_wave.py`` – Euler–Bernoulli beam equation.
* ``alfven_wave.py`` – 1‑D Alfv\u00e9n wave along a magnetic field.

Each script runs a short simulation and plots the final state when executed.
