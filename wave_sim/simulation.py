import os
import logging
from datetime import datetime

import numpy as np
from matplotlib import animation
from tqdm import tqdm
from .visualizer import WaveVisualizer


def run_simulation(
        out_path="wave_2d.mp4",
        steps=40,
        ring_radius=0.15,
        log_interval=5,
):
    L = 2.0
    dx = 0.01
    c = 1.0
    dt = 0.707 * dx / c
    nsteps = steps

    x = np.arange(0, L + dx, dx)
    y = np.arange(0, L + dx, dx)
    xx, yy = np.meshgrid(x, y)

    npts = len(x)
    f = np.zeros((npts, npts, 3))
    xc, w = L / 2, 0.05
    f[:, :, 0] = np.exp(-((xx - xc) ** 2 + (yy - xc) ** 2) / w ** 2)

    f[1:-1, 1:-1, 1] = f[1:-1, 1:-1, 0] + 0.5 * c ** 2 * (
        (f[:-2, 1:-1, 0] + f[2:, 1:-1, 0] - 2 * f[1:-1, 1:-1, 0]) +
        (f[1:-1, :-2, 0] + f[1:-1, 2:, 0] - 2 * f[1:-1, 1:-1, 0])
    ) * (dt / dx) ** 2

    viz = WaveVisualizer(
        sim_shape=f.shape[:2],
        output_video_size=(1920, 1080),
        dt=dt,
        dx=dx,
        main_plot_cmap_name="Spectral",
        dynamic_z=False,
        zlim=(-0.25, 1.0),
        font_size=16
    )

    dist = np.sqrt((xx - xc) ** 2 + (yy - xc) ** 2)
    ring_thick = 3 * dx
    ring_mask = np.abs(dist - ring_radius) <= ring_thick / 2

    velocity_hist, amp_hist = [], []
    prev_amp = f[:, :, 1][ring_mask].mean()

    # ── logging setup ─────────────────────────────────────────────────
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(
        logs_dir,
        f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    logging.info(
        "Simulation start: steps=%d, ring_radius=%.3f, log_interval=%d",
        steps,
        ring_radius,
        log_interval,
    )

    writer = animation.FFMpegWriter(fps=30, bitrate=8000)
    with writer.saving(viz.fig, out_path, dpi=100):
        for k in tqdm(range(nsteps), desc="Simulating"):
            f[1:-1, 1:-1, 2] = -f[1:-1, 1:-1, 0] + 2 * f[1:-1, 1:-1, 1] + c ** 2 * (
                (f[:-2, 1:-1, 1] + f[2:, 1:-1, 1] - 2 * f[1:-1, 1:-1, 1]) +
                (f[1:-1, :-2, 1] + f[1:-1, 2:, 1] - 2 * f[1:-1, 1:-1, 1])
            ) * (dt / dx) ** 2
            f[:, :, 0], f[:, :, 1] = f[:, :, 1], f[:, :, 2]

            amp = f[:, :, 1][ring_mask].mean()
            vel = (amp - prev_amp) / dt
            prev_amp = amp
            energy = float(np.sum(f[:, :, 1] ** 2))
            tnow = k * dt
            velocity_hist.append((tnow, vel))
            amp_hist.append(amp)

            if k % log_interval == 0:
                logging.info(
                    "step=%d t=%.3f amp=%.6f vel=%.6f energy=%.6f",
                    k,
                    tnow,
                    amp,
                    vel,
                    energy,
                )

            viz.update(f[:, :, 1], velocity_hist, amp_hist, tnow, monitor_ring=ring_radius)
            viz.render_composite_frame()
            writer.grab_frame()

    logging.info("Simulation completed. Output saved to %s", out_path)

    return out_path
