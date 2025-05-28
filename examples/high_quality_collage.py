"""Generate high quality animations for all wave classes."""
import os

from wave_sim.wave_catalog import (
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

from wave_sim.high_quality import simulate_wave, PointSource, ConstantSpeed
from wave_sim.collage import collage_videos


WAVE_CLASSES = [
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


def get_wave_speed(cls):
    obj = cls(grid_size=8, backend="cpu")
    return getattr(obj, "c", 1.0)


def build_scene_factory(speed):
    def builder(resolution):
        w, h = resolution
        objs = [ConstantSpeed(speed), PointSource(w // 2, h // 2, freq=0.1, amplitude=5.0)]
        return objs, w, h
    return builder


def run_all():
    os.makedirs("output_hq", exist_ok=True)
    files = []
    for cls in WAVE_CLASSES:
        speed = get_wave_speed(cls)
        name = cls.__name__.lower()
        outfile = os.path.join("output_hq", f"{name}.mp4")
        simulate_wave(build_scene_factory(speed), outfile, steps=600, sim_steps_per_frame=4)
        files.append(outfile)
    collage_videos(files, os.path.join("output_hq", "collage.mp4"))


if __name__ == "__main__":
    run_all()
