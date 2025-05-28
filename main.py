import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec


class WaveVisualizer:
    """Render the wave field along with velocity and spectrum plots."""

    def __init__(self, sim_shape, output_video_size, dt, dx,
                 main_plot_cmap_name="coolwarm", brightness_scale=1.0,
                 dynamic_z=True):
        self.sim_shape = sim_shape
        self.ndim = len(sim_shape)
        self.dt, self.dx = dt, dx
        self.dynamic_z = bool(dynamic_z)
        self.cmap_name = main_plot_cmap_name

        w, h = output_video_size
        self.fig = plt.figure(figsize=(w/100, h/100), dpi=100)
        gs = GridSpec(2, 2, figure=self.fig,
                      width_ratios=[3, 1], hspace=0.25, wspace=0.05)

        self.ax_main = self.fig.add_subplot(gs[:, 0], projection='3d')
        self.ax_velocity = self.fig.add_subplot(gs[0, 1])
        self.ax_spectrum = self.fig.add_subplot(gs[1, 1])

        if self.ndim == 2:
            ny, nx = sim_shape
        else:
            ny, nx = sim_shape[1:]
        x = np.arange(nx) * dx
        y = np.arange(ny) * dx
        self.X, self.Y = np.meshgrid(x, y)

        self.ax_main.set_xlabel("X (m)")
        self.ax_main.set_ylabel("Y (m)")
        self.ax_main.set_zlabel("Amplitude")
        self.ax_main.set_box_aspect((2*nx, 2*ny, 2*max(nx, ny)))

        self.fixed_zlim = float(brightness_scale)
        if not self.dynamic_z:
            self.ax_main.set_zlim(-self.fixed_zlim, self.fixed_zlim)

        self.vel_line, = self.ax_velocity.plot([], [], '-b')
        self.spec_line, = self.ax_spectrum.plot([], [], '-r')

        self.ax_velocity.set_xlabel("Time (s)")
        self.ax_velocity.set_ylabel("Velocity")
        self.ax_spectrum.set_xlabel("Frequency (Hz)")
        self.ax_spectrum.set_ylabel("Magnitude")

        self.fig.tight_layout(pad=1.0)

        self.field_slice_to_visualize = None
        self.velocity_history = []
        self.amplitude_history_for_fft = []
        self.current_sim_time = 0.0

    def update(self, field_slice, velocity_history, amplitude_history, current_time):
        self.field_slice_to_visualize = field_slice
        self.velocity_history = velocity_history
        self.amplitude_history_for_fft = amplitude_history
        self.current_sim_time = current_time

    def render_composite_frame(self):
        field = self.field_slice_to_visualize

        self.ax_main.clear()
        if self.dynamic_z:
            z_max = max(1e-6, np.percentile(np.abs(field), 99)) * 1.05
            self.ax_main.set_zlim(-z_max, z_max)
        else:
            self.ax_main.set_zlim(-self.fixed_zlim, self.fixed_zlim)

        self.ax_main.plot_surface(self.X, self.Y, field,
                                  cmap=cm.get_cmap(self.cmap_name),
                                  rstride=4, cstride=4, alpha=0.9)
        self.ax_main.plot_wireframe(self.X, self.Y, field,
                                    rstride=10, cstride=10,
                                    color="green", linewidth=0.4, alpha=0.5)
        self.ax_main.set_title(f"Wave Field (Slice) at t={self.current_sim_time:.2f}s")

        if self.velocity_history:
            t_arr, v_arr = zip(*self.velocity_history)
            self.vel_line.set_data(t_arr, v_arr)
            self.ax_velocity.relim()
            self.ax_velocity.autoscale_view()
        self.ax_velocity.set_title(f"Velocity at Monitor Pt. (t={self.current_sim_time:.2f}s)")

        if len(self.amplitude_history_for_fft) > 1:
            sig = np.asarray(self.amplitude_history_for_fft).ravel()
            fft_vals = np.fft.rfft(sig)
            freqs = np.fft.rfftfreq(sig.size, d=self.dt)
            magnitude = np.abs(fft_vals) / sig.size
            self.spec_line.set_data(freqs[1:], magnitude[1:])
            self.ax_spectrum.relim()
            self.ax_spectrum.autoscale_view()
        self.ax_spectrum.set_title(f"Spectrum (t={self.current_sim_time:.2f}s)")

        self.fig.canvas.draw()


# ── physical / numerical parameters ───────────────────────────────────────────
L = 1.0
dx = 0.01
c = 1.0
dt = 0.707 * dx / c
nsteps = 199

# ── grid and initial field ────────────────────────────────────────────────────
x = np.arange(0, L + dx, dx)
y = np.arange(0, L + dx, dx)
xx, yy = np.meshgrid(x, y)

npts = len(x)
f = np.zeros((npts, npts, 3))

xc, w = 0.5, 0.05
f[:, :, 0] = np.exp(-((xx - xc)**2 + (yy - xc)**2) / w**2)

# first “kick” for leap-frog
f[1:-1, 1:-1, 1] = f[1:-1, 1:-1, 0] + 0.5 * c**2 * (
    (f[:-2, 1:-1, 0] + f[2:, 1:-1, 0] - 2*f[1:-1, 1:-1, 0]) +
    (f[1:-1, :-2, 0] + f[1:-1, 2:, 0] - 2*f[1:-1, 1:-1, 0])
) * (dt / dx)**2

# ── visualiser and writer setup ───────────────────────────────────────────────
vis = WaveVisualizer(sim_shape=f.shape[:2],
                    output_video_size=(1920, 1080),
                    dt=dt, dx=dx)

center_idx = npts // 2
velocity_history = []
amplitude_history = []
prev_amp = f[center_idx, center_idx, 1]

writer = animation.FFMpegWriter(fps=30, bitrate=8000)

with writer.saving(vis.fig, "wave_2d.mp4", dpi=100):
    for k in range(nsteps):
        f[1:-1, 1:-1, 2] = -f[1:-1, 1:-1, 0] + 2*f[1:-1, 1:-1, 1] + c**2 * (
            (f[:-2, 1:-1, 1] + f[2:, 1:-1, 1] - 2*f[1:-1, 1:-1, 1]) +
            (f[1:-1, :-2, 1] + f[1:-1, 2:, 1] - 2*f[1:-1, 1:-1, 1])
        ) * (dt / dx)**2

        f[:, :, 0], f[:, :, 1] = f[:, :, 1], f[:, :, 2]

        amp = f[center_idx, center_idx, 1]
        vel = (amp - prev_amp) / dt
        prev_amp = amp
        velocity_history.append((k*dt, vel))
        amplitude_history.append(amp)

        vis.update(f[:, :, 1], velocity_history, amplitude_history, k*dt)
        vis.render_composite_frame()
        writer.grab_frame()

plt.close(vis.fig)
print("Saved → wave_2d.mp4")
