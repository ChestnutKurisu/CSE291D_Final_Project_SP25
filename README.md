# Wave Simulation Examples

This repository contains a minimal framework for simulating and animating simple wave phenomena.  The code has recently been consolidated around a single high quality pipeline.  The core solver now supports optional GPU acceleration via ``cupy`` for higher resolution and smoother animations when available.

The focus of this repository is purely on wave propagation. Earlier project
notes about simulating elastic bodies via incremental potentials and Hessians
are not implemented here.

The repository provides **20** illustrative wave types.  Configurations for
ten 2‑D phenomena (P, S and several surface/interface waves) are implemented in
``wave_sim.wave_catalog`` for use with the GPU‑accelerated solver.  A further ten
textbook waves are available via specialised one‑dimensional or spectral
schemes in ``wave_sim.solvers``.  Surface and interface wave animations are
**highly simplified**: they use a scalar wave model with preset speeds and do
not capture the true dispersive or vector nature of these waves.

Legacy modules under ``wave_sim2d`` have been removed.  All animations now
use the GPU optimised utilities found in ``wave_sim.high_quality`` which are
also able to fall back to NumPy when a CUDA device is not available.

## Usage Notes

`WaveSimulator2D` supports several boundary conditions and flexible initial
sources. The boundary condition is specified with a
``BoundaryCondition`` value:

* ``BoundaryCondition.REFLECTIVE`` – zero normal derivative at the edges
* ``BoundaryCondition.PERIODIC`` – domain repeats at the edges
* ``BoundaryCondition.ABSORBING`` – damped sponge layer near the borders

The initial disturbance is passed via ``initial_field`` when creating a
`WaveSimulator2D` or through the ``scene_builder`` used by
`simulate_wave`. A Gaussian pulse centered in the domain can be created with:

```python
from wave_sim.initial_conditions import gaussian_2d as gaussian

from wave_sim.high_quality import simulate_wave, ConstantSpeed
from wave_sim.wave_catalog import PrimaryWave, gaussian_initial_condition

scene = PrimaryWave(initial_condition=gaussian).get_scene_builder()
simulate_wave(scene, "out.mp4", steps=200, resolution=(256, 256))
```

The scene object collection also includes ``LineSource`` for emitting waves
along an arbitrary segment and ``ModulatorSmoothSquare`` for smoothly pulsing
source amplitudes.

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
uses :mod:`wave_sim.high_quality` to produce movies for all twenty wave types.
The script writes an MP4 per wave and then assembles the clips into a single
collage video.  Because the 2‑D solver is a generic scalar model, only the
body-wave examples are physically meaningful; the surface and interface waves
are included purely for illustration.


Additional scripts in the ``examples`` directory demonstrate the specialised
one‑dimensional and spectral solvers:

* ``internal_gravity_wave.py`` – finite difference scheme for a stratified fluid.
* ``kelvin_wave.py`` – rotating shallow water Kelvin wave.
* ``rossby_planetary_wave.py`` – linear Rossby wave via FFTs.
* ``flexural_beam_wave.py`` – Euler–Bernoulli beam equation.
* ``alfven_wave.py`` – 1‑D Alfv\u00e9n wave along a magnetic field.
Each script now generates an MP4 animation that can also be incorporated into
the collage.
