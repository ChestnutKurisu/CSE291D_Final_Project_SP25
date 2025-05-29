# 2-D Wave Simulation

This project contains a small Python implementation of a 2‑D wave equation solver.  The code evolves a wave field in time and writes an MP4 movie showing the amplitude.

## Governing Equation

The solver models the classic second‑order wave equation for a field $u(x,y,t)$ with constant wave speed $c$:

$$
\frac{\partial^2 u}{\partial t^2} = c^2\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right).
$$

Initial conditions consist of a compact Gaussian pulse and the simulation assumes zero displacement at the boundaries.

## Numerical Method

A uniform Cartesian grid is used with spacing $\Delta x = \Delta y$.  Spatial derivatives are approximated with second‑order central differences.  Time integration follows a standard explicit three‑level scheme

$$
 u^{n+1}_{i,j} = 2u^{n}_{i,j} - u^{n-1}_{i,j} + S^2\,\nabla^2 u^{n}_{i,j},
$$

where $S = c\,\Delta t/\Delta x$.  To satisfy the Courant–Friedrichs–Lewy (CFL) stability condition the time step is chosen as

$$
 \Delta t = \text{CFL}\; \frac{\Delta x}{c},
$$

with `CFL = 0.7` in `simulation.py`.

## Running the Simulation

Install the Python dependencies

```bash
pip install -r requirements.txt
```

The video writer relies on the `ffmpeg` binary being available on the `PATH`.

Execute the driver script

```bash
python main.py --steps 60 --output wave_2d.mp4
```

### Command‑line Options

```
--steps        Number of time steps to simulate (default: 20)
--output       Output MP4 path (default: wave_2d.mp4)
--log_interval How often to log diagnostics (default: every step)
--wave_type    acoustic | P | S_SH | S_SV_potential
--c_acoustic   Wave speed for acoustic run
--vp           P-wave velocity
--vs           S-wave velocity
```

Example: run a P-wave simulation for 100 steps

```bash
python main.py --wave_type P --steps 100 --output p_wave.mp4
```

### Elastic potentials demo

The module `wave_sim.elastic_waves` now includes a helper function
`simulate_elastic_potentials` that evolves coupled P- and S-wave potentials and
returns the corresponding displacement fields. This routine illustrates a more
complete elastic formulation where both wave modes propagate simultaneously.

## Output Files

The program writes the MP4 animation specified by `--output`.  A log file containing ring‑average metrics is also generated in the `logs/` directory with a timestamped name such as `simulation_acoustic_YYYYMMDD_HHMMSS.log`.

