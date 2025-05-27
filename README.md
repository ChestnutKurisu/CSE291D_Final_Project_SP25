# Wave Simulation Examples

This repository contains a minimal framework for simulating and animating simple wave phenomena.  
The code now exposes placeholder classes for **70 different wave types**. Each class is a thin wrapper
around a basic 2D finite difference solver found in `wave_sim.base`.  The numerical model is the same
for all waves and does **not** attempt to capture the true physics of the named phenomenon.  The
classes merely provide distinct names and default wave speeds so that example animations can be
generated for demonstration purposes.

Run `examples/collage.py` to generate an animation for every available wave class.  The script will
create individual MP4 files in an `output/` directory.

The implementation is intentionally lightweight and meant only as a starting point for a more
comprehensive project.
