This repository contains a simple 2‑D wave simulation that writes an mp4 file.

## Usage

Run the simulation from the command line:

```bash
python main.py [--gpu] [--steps N] [--output PATH]
```

- `--gpu` enables GPU acceleration via CuPy when available.
- `--steps` controls the number of simulation steps (default: 19).
- `--output` sets the path to the generated video.

If CuPy or a compatible GPU is not present, the code falls back to NumPy.
