"""Generate animations for all wave types using the GPU-based simulator."""

import os
import cv2

from wave_sim2d.wave_simulation import WaveSimulator2D
from wave_sim2d.wave_visualizer import WaveVisualizer, get_colormap_lut
from wave_sim2d.wave_catalog2d import wave_catalog_list
from wave_sim2d.collage import collage_videos


def generate_wave_animation(wave_name, scene_constructor, width=512, height=512, steps=600, dt=1.0, fps=60, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f"{wave_name}.mp4")
    scene_objs = scene_constructor(width, height)
    simulator = WaveSimulator2D(width=width, height=height, scene_objects=scene_objs, dt=dt, global_dampening=1.0)

    field_lut = get_colormap_lut("RdBu", size=256, invert=True)
    intens_lut = get_colormap_lut("afmhot", size=256, invert=False)
    visualizer = WaveVisualizer(field_colormap=field_lut, intensity_colormap=intens_lut)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videowriter = cv2.VideoWriter(outpath, fourcc, fps, (width, height))

    for step_i in range(steps):
        simulator.update_scene()
        simulator.update_field()
        visualizer.update(simulator.get_field())
        frame_bgr = visualizer.render_field(brightness_scale=1.0, overlay=None)
        videowriter.write(frame_bgr)

    videowriter.release()
    print(f"Saved {outpath}")
    return outpath


def main():
    wave_defs = wave_catalog_list()
    out_dir = "output"
    generated_files = []

    for wave_name, constructor in wave_defs:
        path = generate_wave_animation(wave_name, constructor, width=512, height=512, steps=300, dt=1.0, fps=60, out_dir=out_dir)
        generated_files.append(path)

    collage_path = os.path.join(out_dir, "all_waves_collage.mp4")
    collage_videos(generated_files, collage_path, grid=None, fps=60, out_width=1920, out_height=1080)
    print("Collage saved to", collage_path)


if __name__ == "__main__":
    main()
