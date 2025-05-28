import numpy as np
from matplotlib import animation
from tqdm import tqdm

from .visualizer import WaveVisualizer


def run_simulation(out_path="wave_2d.mp4", steps=199):
    # physical / numerical parameters
    L = 2.0  # enlarged domain
    dx = 0.01
    c = 1.0
    dt = 0.707 * dx / c
    nsteps = steps

    # grid and initial field
    x = np.arange(0, L + dx, dx)
    y = np.arange(0, L + dx, dx)
    xx, yy = np.meshgrid(x, y)

    npts = len(x)
    f = np.zeros((npts, npts, 3))

    xc, w = L / 2, 0.05
    f[:, :, 0] = np.exp(-((xx - xc) ** 2 + (yy - xc) ** 2) / w ** 2)

    # first "kick" for leap-frog
    f[1:-1, 1:-1, 1] = f[1:-1, 1:-1, 0] + 0.5 * c ** 2 * (
        (f[:-2, 1:-1, 0] + f[2:, 1:-1, 0] - 2 * f[1:-1, 1:-1, 0]) +
        (f[1:-1, :-2, 0] + f[1:-1, 2:, 0] - 2 * f[1:-1, 1:-1, 0])
    ) * (dt / dx) ** 2

    vis = WaveVisualizer(sim_shape=f.shape[:2],
                         output_video_size=(1920, 1080),
                         dt=dt, dx=dx,
                         main_plot_cmap_name="Spectral",
                         dynamic_z=False,
                         zlim=(-0.25, 1.0))

    center_idx = npts // 2
    velocity_history = []
    amplitude_history = []
    prev_amp = f[center_idx, center_idx, 1]

    writer = animation.FFMpegWriter(fps=30, bitrate=8000)

    with writer.saving(vis.fig, out_path, dpi=100):
        for k in tqdm(range(nsteps), desc="Simulating"):
            f[1:-1, 1:-1, 2] = -f[1:-1, 1:-1, 0] + 2 * f[1:-1, 1:-1, 1] + c ** 2 * (
                (f[:-2, 1:-1, 1] + f[2:, 1:-1, 1] - 2 * f[1:-1, 1:-1, 1]) +
                (f[1:-1, :-2, 1] + f[1:-1, 2:, 1] - 2 * f[1:-1, 1:-1, 1])
            ) * (dt / dx) ** 2

            f[:, :, 0], f[:, :, 1] = f[:, :, 1], f[:, :, 2]

            amp = f[center_idx, center_idx, 1]
            vel = (amp - prev_amp) / dt
            prev_amp = amp
            velocity_history.append((k * dt, vel))
            amplitude_history.append(amp)

            vis.update(f[:, :, 1], velocity_history, amplitude_history, k * dt)
            vis.render_composite_frame()
            writer.grab_frame()

    return out_path
