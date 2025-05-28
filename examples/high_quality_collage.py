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
)

import argparse

from wave_sim.high_quality import simulate_wave, PointSource, ConstantSpeed
from wave_sim.collage import collage_videos
from wave_sim.core.boundary import BoundaryCondition

# Import generators for 1D/spectral animations
from examples.alfven_wave import generate_animation as generate_alfven_animation
from examples.flexural_beam_wave import generate_animation as generate_flexural_animation
from examples.internal_gravity_wave import generate_animation as generate_internal_gravity_animation
from examples.kelvin_wave import generate_animation as generate_kelvin_animation
from examples.rossby_planetary_wave import generate_animation as generate_rossby_animation
from examples.plane_acoustic_wave import generate_animation as generate_plane_acoustic_animation
from examples.spherical_acoustic_wave import generate_animation as generate_spherical_acoustic_animation
from examples.deep_water_gravity_wave import generate_animation as generate_deep_water_animation
from examples.shallow_water_gravity_wave import generate_animation as generate_shallow_water_animation
from examples.capillary_wave import generate_animation as generate_capillary_animation


WAVE_CLASSES_2D = [
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
]

ONE_D_SPECTRAL_ANIMATIONS_CONFIG = [
    (generate_alfven_animation, "alfven_wave"),
    (generate_flexural_animation, "flexural_beam_wave"),
    (generate_internal_gravity_animation, "internal_gravity_wave"),
    (generate_kelvin_animation, "kelvin_wave"),
    (generate_rossby_animation, "rossby_planetary_wave"),
    (generate_plane_acoustic_animation, "plane_acoustic_wave"),
    (generate_spherical_acoustic_animation, "spherical_acoustic_wave"),
    (generate_deep_water_animation, "deep_water_gravity_wave"),
    (generate_shallow_water_animation, "shallow_water_gravity_wave"),
    (generate_capillary_animation, "capillary_wave"),
]


def get_wave_speed(cls):
    return getattr(cls, "default_speed", 1.0)


def build_scene_factory(speed):
    def builder(resolution):
        w, h = resolution
        objs = [ConstantSpeed(speed), PointSource(w // 2, h // 2, freq=0.1, amplitude=5.0)]
        return objs, w, h, None
    return builder


def run_all(args):
    os.makedirs(args.outdir, exist_ok=True)
    files = []
    print("Generating 2D wave animations...")
    for cls in WAVE_CLASSES_2D:
        speed = get_wave_speed(cls)
        name = cls.__name__.lower()
        outfile = os.path.join(args.outdir, f"{name}.mp4")
        simulate_wave(
            build_scene_factory(speed),
            outfile,
            steps=args.steps,
            sim_steps_per_frame=args.sim_steps_per_frame,
            resolution=(args.resolution, args.resolution),
            fps=args.fps,
            backend=args.backend,
            boundary_condition=BoundaryCondition(args.boundary),
        )
        files.append(outfile)

    print("\nGenerating 1D/spectral wave animations...")
    for generator_func, base_name in ONE_D_SPECTRAL_ANIMATIONS_CONFIG:
        out_name_1d = f"{base_name}.mp4"
        generated_path = generator_func(output_dir=args.outdir, out_name=out_name_1d)
        files.append(generated_path)

    collage_videos(files, os.path.join(args.outdir, "collage_all.mp4"), fps=args.fps, mode=args.mode)


def parse_args():
    p = argparse.ArgumentParser(description="Run high quality wave simulations and build a collage")
    p.add_argument("--outdir", default="output_all_waves_collage", help="directory for individual videos and collage")
    p.add_argument("--resolution", type=int, default=256, help="simulation width/height in pixels")
    p.add_argument("--steps", type=int, default=300, help="number of simulation steps")
    p.add_argument("--sim-steps-per-frame", type=int, default=2, help="steps per rendered frame")
    p.add_argument("--fps", type=int, default=30, help="frames per second")
    p.add_argument("--mode", choices=["grid", "horizontal", "vertical", "overlay"], default="grid", help="collage layout mode")
    p.add_argument("--boundary", choices=["reflective", "periodic", "absorbing"], default="absorbing", help="boundary condition")
    p.add_argument("--backend", choices=["gpu", "cpu", "auto"], default="auto", help="computation backend for 2D waves")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args)
