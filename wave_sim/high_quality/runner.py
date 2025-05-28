import os
import time
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
    progress=True,
    verbose=False,
):
    """Run a high quality simulation and save to MP4.

    Parameters
    ----------
    sponge_thickness : int, optional
        Thickness of the absorbing boundary sponge layer when
        ``boundary_condition`` is ``ABSORBING``.
    progress : bool, optional
        Show a progress bar using :mod:`tqdm` if available.
    verbose : bool, optional
        Print timing information for each step and overall runtime.
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
        field_colormap=get_colormap_lut("wave1"),
        intensity_colormap=get_colormap_lut("afmhot"),
    )
    writer = imageio.get_writer(out_path, fps=fps)

    if progress:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            def tqdm(x, **k):
                return x
        iterator = tqdm(range(steps))
    else:
        iterator = range(steps)

    start_time = time.perf_counter()
    for i in iterator:
        step_start = time.perf_counter()
        sim.update_scene()
        sim.update_field()
        vis.update(sim)
        if i % sim_steps_per_frame == 0:
            frame = vis.render_field(1.0)
            writer.append_data(frame)
        if verbose:
            step_time = time.perf_counter() - step_start
            print(f"step {i+1}/{steps} took {step_time:.3f}s")

    writer.close()
    total_time = time.perf_counter() - start_time
    if verbose:
        print(f"total runtime: {total_time:.2f}s")
    return out_path
