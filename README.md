This repository contains a simple 2‑D wave simulation that writes an mp4 file.

## Usage

Run the simulation from the command line:

```bash
python main.py [--steps N] [--output PATH]
```

- `--steps` controls the number of simulation steps (default: 19).
- `--output` sets the path to the generated video.

If CuPy or a compatible GPU is not present, the code falls back to NumPy.

## Additional Wave Solvers

Standalone finite difference solvers for acoustic P-waves and shear S-waves are
provided in `wave_sim/elastic_waves.py`:

```python
from wave_sim.elastic_waves import simulate_p_wave, simulate_s_wave
final_p, snaps_p = simulate_p_wave()
final_s, snaps_s = simulate_s_wave()
```

Each function returns the final wavefield and a list of snapshots collected
every 50 steps.

