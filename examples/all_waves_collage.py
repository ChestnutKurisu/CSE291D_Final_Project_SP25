"""Generate animations for a subset of wave classes.

This script iterates over a curated set of simulations defined in
``wave_sim.wave_catalog`` and writes a short MP4 file for each one into an
``output`` directory.  The results are purely illustrative.
"""

import os
import numpy as np

import matplotlib

matplotlib.use("Agg")  # allow running without a display

from wave_sim import (
    PrimaryWave,
    SecondaryWave,
    SHWave,
    SVWave,
    RayleighWave,
    LoveWave,
    LambS0Mode,
    LambA0Mode,
    StoneleyWave,
    ScholteWave,
    PlaneAcousticWave,
    SphericalAcousticWave,
    DeepWaterGravityWave,
    ShallowWaterGravityWave,
    CapillaryWave,
    InternalGravityWave,
    KelvinWave,
    RossbyPlanetaryWave,
    FlexuralBeamWave,
    AlfvenWave,
)

from wave_sim.collage import collage_videos


def gaussian_source(X, Y, sigma=5.0):
    cx = X.shape[0] // 2
    cy = Y.shape[1] // 2
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


def run_and_save(sim, name, steps=50, fps=60):
    ani = sim.animate(steps=steps, interval=1000 / fps)
    path = f"{name}.mp4"
    ani.save(path, fps=fps)
    return path


def main():
    os.makedirs("output", exist_ok=True)

    wave_classes = [
        PrimaryWave,
        SecondaryWave,
        SHWave,
        SVWave,
        RayleighWave,
        LoveWave,
        LambS0Mode,
        LambA0Mode,
        StoneleyWave,
        ScholteWave,
        PlaneAcousticWave,
        SphericalAcousticWave,
        DeepWaterGravityWave,
        ShallowWaterGravityWave,
        CapillaryWave,
        InternalGravityWave,
        KelvinWave,
        RossbyPlanetaryWave,
        FlexuralBeamWave,
        AlfvenWave,
    ]

    files = []
    for wave_cls in wave_classes:
        name = wave_cls.__name__
        print(f"Running {name}...")
        sim = wave_cls(grid_size=512, boundary="absorbing", source_func=gaussian_source)
        outfile = os.path.join("output", name.lower())
        files.append(run_and_save(sim, outfile, steps=300, fps=60))

    print("Generated files:")
    for path in files:
        print("  ", path)

    collage_out = os.path.join("output", "collage.mp4")
    print("Creating collage video...")
    collage_videos(files, collage_out)
    print("Collage saved to", collage_out)


if __name__ == "__main__":
    main()
