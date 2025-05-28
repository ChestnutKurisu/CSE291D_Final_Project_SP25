import os
import cv2
from wave_sim2d.wave_simulation import WaveSimulator2D
from wave_sim2d.wave_visualizer import WaveVisualizer, get_colormap_lut
from wave_sim2d.wave_catalog2d import wave_catalog_list
from wave_sim2d.collage import collage_videos


def generate_wave_animation(name, constructor, width=512, height=512, steps=300, dt=1.0, fps=60, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.mp4")
    scene_objs = constructor(width, height)
    sim = WaveSimulator2D(width, height, scene_objects=scene_objs, dt=dt, global_dampening=1.0)
    field_lut = get_colormap_lut('RdBu', invert=True)
    intens_lut = get_colormap_lut('afmhot')
    vis = WaveVisualizer(field_colormap=field_lut, intensity_colormap=intens_lut)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for _ in range(steps):
        sim.update_scene()
        sim.update_field()
        vis.update(sim.get_field())
        frame = vis.render_field()
        writer.write(frame)
    writer.release()
    return path


def main():
    wave_defs = wave_catalog_list()
    out_dir = "output"
    paths = []
    for name, constructor in wave_defs:
        p = generate_wave_animation(name, constructor, out_dir=out_dir)
        paths.append(p)
    collage_path = os.path.join(out_dir, "all_waves_collage.mp4")
    collage_videos(paths, collage_path, fps=60)
    print("Collage saved to", collage_path)


if __name__ == "__main__":
    main()
