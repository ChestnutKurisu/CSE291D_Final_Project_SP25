This repository contains a simple 2‑D wave simulation that writes an mp4 file.

## Usage

Run the simulation from the command line:

```bash
python main.py [--gpu] [--steps N] [--output PATH] [--log_interval M]
```

- `--gpu` enables GPU acceleration via CuPy when available.
- `--steps` controls the number of simulation steps (default: 20).
- `--output` sets the path to the generated video.
- `--log_interval` controls how often simulation metrics are written to the log
  file (in steps).

If CuPy or a compatible GPU is not present, the code falls back to NumPy.

Each run creates a log file under the `logs/` directory named with the
timestamp of when the simulation started. The log records the time, velocity,
amplitude, and energy at the specified interval.
