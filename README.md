# Wave Simulation Examples

This repository contains a minimal framework for simulating and animating simple wave phenomena.  The code has recently been consolidated around a single high quality pipeline.  The core solver now supports optional GPU acceleration via ``cupy`` for higher resolution and smoother animations when available.

The focus of this repository is purely on wave propagation. Earlier project
notes about simulating elastic bodies via incremental potentials and Hessians
are not implemented here.

The repository provides **20** illustrative wave types. Configurations for
ten 2‑D phenomena (P, S, Rayleigh, Love, Lamb, Stoneley, Scholte waves) are implemented in
``wave_sim.wave_catalog`` for use with the GPU‑accelerated ``WaveSimulator2D`` solver.
A further ten textbook waves are available via specialised one‑dimensional or spectral
schemes in ``wave_sim.solvers``.

**Important Note on 2D Wave Fidelity:** The ``WaveSimulator2D`` is a generic scalar wave
equation solver. While it reasonably models P-waves (compressional), its application to
S-waves (shear) and particularly to complex phenomena like Rayleigh, Love, Lamb, Stoneley,
and Scholte waves involves significant simplification. For these latter wave types, the
simulations primarily demonstrate propagation at a pre-configured speed. They **do not**
capture the true vector displacement fields, coupled P-SV motion, frequency-dependent
dispersion, or the specific boundary conditions that give rise to these waves' unique
characteristics. These examples are included for illustrative breadth rather than
rigorous physical accuracy.

Legacy modules under ``wave_sim2d`` have been removed.  Animations for the ten 2‑D wave phenomena from ``wave_sim.wave_catalog`` (e.g., P, S, Rayleigh waves) are generated using the GPU‑optimised utilities in ``wave_sim.high_quality``.
These can fall back to NumPy if a CUDA device is not available.  The other ten 1‑D or spectral wave types (e.g., Alfven, Rossby waves) from ``wave_sim.solvers`` are animated using ``matplotlib`` directly within their respective example scripts.

## Usage Notes

`WaveSimulator2D` supports several boundary conditions and flexible initial
sources. The boundary condition is specified with a
``BoundaryCondition`` value:

* ``BoundaryCondition.REFLECTIVE`` – zero normal derivative at the edges
* ``BoundaryCondition.PERIODIC`` – domain repeats at the edges
* ``BoundaryCondition.ABSORBING`` – damped sponge layer near the borders, handled internally by the simulator using ``sponge_thickness``

Note: If ``BoundaryCondition.ABSORBING`` is active, the simulator applies its own
sponge layer whose width is ``sponge_thickness`` (set to ``0`` to disable). For more
customized absorption profiles, set the boundary to ``BoundaryCondition.REFLECTIVE``
and add a ``StaticDampening`` scene object with a desired ``border_thickness`` and profile.

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
along an arbitrary segment, ``GaussianBlobSource`` for a soft circular emitter
using a Gaussian envelope, and ``ModulatorSmoothSquare`` for smoothly pulsing
source amplitudes. ``LineSource`` accepts an optional ``amp_modulator`` to vary
its strength over time.

The solver emits a warning if the CFL condition ``c * dt / dx`` exceeds
``1 / sqrt(2)`` to help maintain stability. The spatial step ``dx`` and time step ``dt``
for `WaveSimulator2D` default to `1.0`. These values define the simulation's
discretization and directly affect the interpretation of wave speeds (``c``) and source
frequencies. Ensure these are consistent with your desired physical scales.

For the specialised shear-wave configurations ``SHWave`` and ``SVWave`` the
physical displacement can be retrieved from simulation frames via helper
methods:

```python
sh = SHWave()
uy = sh.get_displacement_y(frame)

sv = SVWave()
ux, uz = sv.get_displacement_components(frame, dx=1.0)
```

Animations for all twenty wave types can be generated and combined into a single collage video using ``examples/high_quality_collage.py``.
This script utilizes :mod:`wave_sim.high_quality` for the 2‑D waves and calls the individual example scripts for the 1‑D/spectral waves.
Because the 2‑D solver is a generic scalar model, only the body‑wave examples are physically meaningful; the surface and interface waves are included purely for illustration.

Colour lookup tables are retrieved with ``get_colormap_lut``.  Preset names
include ``wave1``–``wave4`` and ``icefire`` which correspond to built-in
palettes.


Additional scripts in the ``examples`` directory demonstrate the specialised
one‑dimensional and spectral solvers:

* ``internal_gravity_wave.py`` – finite difference scheme for a stratified fluid.
* ``kelvin_wave.py`` – rotating shallow water Kelvin wave.
* ``rossby_planetary_wave.py`` – linear Rossby wave via FFTs.
* ``flexural_beam_wave.py`` – Euler–Bernoulli beam equation.
* ``alfven_wave.py`` – 1‑D Alfv\u00e9n wave along a magnetic field.
Each script now generates a consistent MP4 animation that can also be incorporated
into the collage.
