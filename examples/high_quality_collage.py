"""Generate high quality animations for all wave classes."""
import os

from wave_sim.wave_catalog import (
    PrimaryWave,
    SecondaryWave,
    SHWave,
    SVWave,
)

import argparse

from wave_sim.high_quality import simulate_wave, PointSource, ConstantSpeed
from wave_sim.collage import collage_videos


WAVE_CLASSES = [
    PrimaryWave,
    SecondaryWave,
    SHWave,
    SVWave,
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
    for cls in WAVE_CLASSES:
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
            backend="gpu",
        )
        files.append(outfile)
    collage_videos(files, os.path.join(args.outdir, "collage.mp4"), fps=args.fps, mode=args.mode)


def parse_args():
    p = argparse.ArgumentParser(description="Run high quality wave simulations and build a collage")
    p.add_argument("--outdir", default="output_hq", help="directory for individual videos")
    p.add_argument("--resolution", type=int, default=512, help="simulation width/height in pixels")
    p.add_argument("--steps", type=int, default=600, help="number of simulation steps")
    p.add_argument("--sim-steps-per-frame", type=int, default=4, help="steps per rendered frame")
    p.add_argument("--fps", type=int, default=60, help="frames per second")
    p.add_argument("--mode", choices=["grid", "horizontal", "vertical", "overlay"], default="grid", help="collage layout mode")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args)
