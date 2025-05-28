import os
import imageio.v2 as imageio
import numpy as np
import cv2

from .simulator import WaveSimulator2D
from ..core.boundary import BoundaryCondition
from .visualizer import WaveVisualizer, get_colormap_lut


def simulate_wave(
    scene_builder,
    out_path,
    steps=2000,
    sim_steps_per_frame=8,
    resolution=(512, 512),
    fps=60,
    backend="gpu",
    boundary_condition=BoundaryCondition.REFLECTIVE,
    global_dampening=1.0,
    sponge_thickness=8,
):
    """Run a high quality simulation and save to MP4.

    Parameters
    ----------
    sponge_thickness : int, optional
        Thickness of the absorbing boundary sponge layer when
        ``boundary_condition`` is ``ABSORBING``.
    """
    objects, w, h, init = scene_builder(resolution)
    sim = WaveSimulator2D(
        w,
        h,
        objects,
        initial_field=init,
        backend=backend,
        boundary=boundary_condition,
        sponge_thickness=sponge_thickness,
    )
    sim.global_dampening = global_dampening
    vis = WaveVisualizer(
        field_colormap=get_colormap_lut("wave1", backend=backend),
        intensity_colormap=get_colormap_lut("afmhot", backend=backend),
    )
    writer = imageio.get_writer(out_path, fps=fps)
    for i in range(steps):
        sim.update_scene()
        sim.update_field()
        vis.update(sim)
        if i % sim_steps_per_frame == 0:
            frame = vis.render_field(1.0)
            writer.append_data(frame)
    writer.close()
    return out_path
