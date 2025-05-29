This repository contains a simple 2‑D wave simulation.

## Usage

Run the simulation from the command line:

```bash
python main.py [--steps N] [--output PATH] [--wave TYPE]
```

- `--steps` controls the number of simulation steps (default: 19).
- `--output` sets the path to the generated video for the baseline solver.
- `--wave` selects which solver to run: `baseline` (default), `p` (P-wave), or `s` (S-wave).

The baseline solver produces an MP4 animation using `matplotlib`. The P- and S-wave
solvers run purely in NumPy and report the final amplitude at the source location.
