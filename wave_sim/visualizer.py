import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class WaveVisualizer:
    """Visualize wave field with velocity and spectrum plots."""

    def __init__(self, sim_shape, output_video_size, dt, dx,
                 main_plot_cmap_name="Spectral", brightness_scale=1.0,
                 dynamic_z=True, zlim=(-0.25, 1.0), font_size=14):   # ← NEW ARG
        # ── font handling ───────────────────────────────────────────
        plt.rcParams.update({'font.size': font_size})   # global default
        self.label_fs = font_size
        self.title_fs = font_size + 2
        self.tick_fs  = max(6, font_size - 2)
        self.sim_shape = sim_shape
        self.ndim = len(sim_shape)
        self.dt, self.dx = dt, dx
        self.dynamic_z = bool(dynamic_z) and (zlim is None)
        self.cmap_name = main_plot_cmap_name
        self.zlim = zlim

        w, h = output_video_size
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)

        gs = GridSpec(
            2, 2,
            figure=self.fig,
            width_ratios=[9, 4],  # ≈ 83 % of the width for the cube
            height_ratios=[1, 1],  # keep rows equal in height
            left=0.06, right=0.98,  # push the whole layout toward the left edge
            bottom=0.09, top=0.92,
            wspace=0.01, hspace=0.3
        )

        self.ax_main = self.fig.add_subplot(gs[:, 0], projection='3d')
        self.ax_main.set_anchor('W')
        self.ax_velocity = self.fig.add_subplot(gs[0, 1])
        self.ax_spectrum = self.fig.add_subplot(gs[1, 1])

        if self.ndim == 2:
            ny, nx = sim_shape
        else:
            ny, nx = sim_shape[1:]
        x = np.arange(nx) * dx
        y = np.arange(ny) * dx
        self.X, self.Y = np.meshgrid(x, y)
        self.center_coord = (nx // 2) * dx

        self.ax_main.set_xlabel("X (m)", fontsize=self.label_fs)
        self.ax_main.set_ylabel("Y (m)", fontsize=self.label_fs)
        self.ax_main.set_zlabel("Amplitude", fontsize=self.label_fs)
        self.ax_main.set_box_aspect((nx, ny, 0.6 * max(nx, ny)))

        if self.zlim is not None:
            self.ax_main.set_zlim(*self.zlim)
        self.fixed_zlim = float(brightness_scale)

        self.vel_line,  = self.ax_velocity.plot([], [], '-b')
        self.spec_line, = self.ax_spectrum.plot([], [], '-r')
        self.ax_velocity.set_xlabel("Time (s)", fontsize=self.label_fs)
        self.ax_velocity.set_ylabel("Velocity", fontsize=self.label_fs)
        self.ax_spectrum.set_xlabel("Frequency (Hz)", fontsize=self.label_fs)
        self.ax_spectrum.set_ylabel("Magnitude", fontsize=self.label_fs)

        for _ax in (self.ax_main, self.ax_velocity, self.ax_spectrum):
            _ax.tick_params(labelsize=self.tick_fs)

        self.fig.tight_layout(pad=1.0)

        # add a persistent colorbar axis in the main plot
        self.cbar_ax = inset_axes(
            self.ax_main,
            width="3%", height="25%",
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98, 1, 1),
            bbox_transform=self.ax_main.transAxes,
            borderpad=0.0,
        )
        self.cbar = None

        self.field_slice_to_visualize = None
        self.velocity_history = []
        self.amplitude_history_for_fft = []
        self.current_sim_time = 0.0
        self.monitor_ring_radius = None

    def update(self, field_slice, velocity_history, amplitude_history, current_time, monitor_ring=None):
        self.field_slice_to_visualize = field_slice
        self.velocity_history = velocity_history
        self.amplitude_history_for_fft = amplitude_history
        self.current_sim_time = current_time
        self.monitor_ring_radius = monitor_ring

    def render_composite_frame(self):
        field = self.field_slice_to_visualize

        self.ax_main.clear()
        if self.dynamic_z:
            z_max = max(1e-6, np.percentile(np.abs(field), 99)) * 1.05
            self.ax_main.set_zlim(-z_max, z_max)
        elif self.zlim is not None:
            self.ax_main.set_zlim(*self.zlim)

        norm = plt.Normalize(vmin=field.min(), vmax=field.max())
        self.ax_main.plot_surface(
            self.X, self.Y, field,
            cmap=cm.get_cmap(self.cmap_name),
            rstride=4, cstride=4, alpha=0.9, norm=norm
        )
        self.ax_main.plot_wireframe(self.X, self.Y, field,
                                    rstride=10, cstride=10,
                                    color="grey", linewidth=0.4, alpha=0.5)

        if self.monitor_ring_radius is not None:
            theta = np.linspace(0, 2 * np.pi, 200)
            xs = self.center_coord + self.monitor_ring_radius * np.cos(theta)
            ys = self.center_coord + self.monitor_ring_radius * np.sin(theta)
            zs = np.zeros_like(xs)
            self.ax_main.plot(xs, ys, zs, linestyle=':', linewidth=3,
                              color='red')
        self.ax_main.set_title(f"Wave Field (Slice) at t={self.current_sim_time:.2f}s", fontsize=self.title_fs)

        # update colorbar indicating the mapping of colors to amplitude
        mappable = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(self.cmap_name))
        if self.cbar is None:
            self.cbar = self.fig.colorbar(mappable, cax=self.cbar_ax)
            self.cbar.ax.set_ylabel("Amplitude", fontsize=self.label_fs)
            self.cbar.ax.tick_params(labelsize=self.tick_fs)
        else:
            self.cbar.update_normal(mappable)

        if self.velocity_history:
            t_arr, v_arr = zip(*self.velocity_history)
            self.vel_line.set_data(t_arr, v_arr)
            self.ax_velocity.relim(); self.ax_velocity.autoscale_view()
        self.ax_velocity.set_title(f"Velocity at Monitor Pt. (t={self.current_sim_time:.2f}s)", fontsize=self.title_fs)

        if len(self.amplitude_history_for_fft) > 1:
            sig = np.asarray(self.amplitude_history_for_fft).ravel()
            fft_vals = np.fft.rfft(sig)
            freqs = np.fft.rfftfreq(sig.size, d=self.dt)
            magnitude = np.abs(fft_vals) / sig.size
            self.spec_line.set_data(freqs[1:], magnitude[1:])
            self.ax_spectrum.relim(); self.ax_spectrum.autoscale_view()
        self.ax_spectrum.set_title(f"Spectrum (t={self.current_sim_time:.2f}s)", fontsize=self.title_fs)

        self.fig.canvas.draw()
