# wave_sim/visualizer.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes


class WaveVisualizer:
    """Visualise the 3-D wave slice together with ring-average velocity and spectrum."""

    def __init__(self,
                 sim_shape,
                 output_video_size,
                 dt,
                 dx,
                 main_plot_cmap_name="Spectral",
                 brightness_scale=1.0,
                 dynamic_z=True,
                 zlim=(-0.25, 1.0),
                 font_size=14):
        # ── layout basics ───────────────────────────────────────────────
        plt.rcParams.update({'font.size': font_size})
        self.label_fs, self.title_fs = font_size, font_size + 2
        self.tick_fs = max(6, font_size - 2)

        self.sim_shape = sim_shape
        self.dt, self.dx = dt, dx
        self.dynamic_z = bool(dynamic_z) and (zlim is None)
        self.zlim = zlim
        self.cmap_name = main_plot_cmap_name

        w, h = output_video_size
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)

        gs = GridSpec(
            2, 3,                     # << one extra column for the colour-bar
            figure=self.fig,
            width_ratios=[1, 49, 19],   # | c-bar | 3-D | two 2-D plots |
            height_ratios=[1, 1],
            left=0.03, right=0.98, bottom=0.09, top=0.92,
            wspace=0.01, hspace=0.3
        )

        # axes order: colour-bar, 3-D, 2-D upper, 2-D lower
        self.cbar_ax     = self.fig.add_subplot(gs[:, 0])
        self.ax_main     = self.fig.add_subplot(gs[:, 1], projection='3d')
        self.ax_velocity = self.fig.add_subplot(gs[0, 2])
        self.ax_spectrum = self.fig.add_subplot(gs[1, 2])

        # ── spatial mesh for plotting ──────────────────────────────────
        if self.sim_shape:
            ny, nx = sim_shape
            x = np.arange(nx) * dx
            y = np.arange(ny) * dx
            self.X, self.Y = np.meshgrid(x, y)
            self.cx, self.cy = x.mean(), y.mean()

        # labels + aspect for main 3-D plot
        self.ax_main.set_xlabel("X (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_ylabel("Y (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_zlabel("Amplitude", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_box_aspect((np.ptp(x), np.ptp(y), 0.6 * max(np.ptp(x), np.ptp(y))))
        if not self.dynamic_z and self.zlim:
            self.ax_main.set_zlim(*self.zlim)

        # lines for the 2-D graphs
        self.vel_line,  = self.ax_velocity.plot([], [], '-b', label='Velocity')
        self.spec_line, = self.ax_spectrum.plot([], [], '-r', label='Magnitude')
        self.ax_velocity.set_xlabel("Time (s)",     fontsize=self.label_fs)
        self.ax_velocity.set_ylabel("Velocity",      fontsize=self.label_fs)
        self.ax_spectrum.set_xlabel("Frequency (Hz)", fontsize=self.label_fs)
        self.ax_spectrum.set_ylabel("Magnitude",      fontsize=self.label_fs)
        self.ax_velocity.legend(fontsize=self.tick_fs)
        self.ax_spectrum.legend(fontsize=self.tick_fs)

        # colour-bar initialised once; updated later
        self.cbar = None

        # runtime placeholders
        target_cells = 5000
        self._surf_stride = max(
            1,
            int(np.sqrt((sim_shape[0] * sim_shape[1]) / target_cells))
        )
        # use a slightly coarser grid for the wire-frame
        self._wire_stride = max(1, self._surf_stride * 2)
        self.field_slice_to_visualise = None
        self.velocity_history = []
        self.amplitude_history_for_fft = []
        self.current_time = 0.0
        self.monitor_ring_radius = None

    # ------------------------------------------------------------------
    # API called by the simulation loop
    # ------------------------------------------------------------------
    def update(self,
               field_slice,
               velocity_history,
               amplitude_history,
               current_time,
               monitor_ring=None):
        self.field_slice_to_visualise = field_slice
        self.velocity_history         = velocity_history
        self.amplitude_history_for_fft = amplitude_history
        self.current_time             = current_time
        self.monitor_ring_radius      = monitor_ring

    # ------------------------------------------------------------------
    # redraw everything
    # ------------------------------------------------------------------
    def render_composite_frame(self):
        fld = self.field_slice_to_visualise
        if fld is None:
            return

        # ── 3-D SLICE ────────────────────────────────────────────────
        self.ax_main.cla()
        self.ax_main.set_xlabel("X (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_ylabel("Y (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_zlabel("Amplitude", fontsize=self.label_fs, labelpad=10)

        if self.dynamic_z:
            zlim = 1.05 * np.percentile(np.abs(fld), 99)
            self.ax_main.set_zlim(-zlim, zlim)
        elif self.zlim:
            self.ax_main.set_zlim(*self.zlim)

        ds = self._surf_stride
        self.ax_main.plot_surface(
            self.X[::ds, ::ds], self.Y[::ds, ::ds], fld[::ds, ::ds],
            cmap=self.cmap_name, norm=plt.Normalize(fld.min(), fld.max()),
            rstride=1, cstride=1, antialiased=False, linewidth=0, alpha=0.9
        )
        self.ax_main.plot_wireframe(
            self.X[::ds*2, ::ds*2], self.Y[::ds*2, ::ds*2], fld[::ds*2, ::ds*2],
            color='grey', linewidth=0.4, alpha=0.5
        )

        # draw the monitoring ring on top of the surface
        if self.monitor_ring_radius and self.monitor_ring_radius > 0:
            θ = np.linspace(0, 2 * np.pi, 240)
            xs = self.cx + self.monitor_ring_radius * np.cos(θ)
            ys = self.cy + self.monitor_ring_radius * np.sin(θ)

            z_top, z_bot = self.ax_main.get_zlim()
            z0 = z_top - 0.2 * (z_top - z_bot)  # 2 % above the highest point

            # depthshade removed — keep zorder & clip_on to ensure visibility
            self.ax_main.plot(
                xs, ys, z0, ':', color='red', linewidth=2,
                zorder=100, clip_on=False
            )

            # legend (unchanged)
            surf_proxy = plt.Rectangle((0, 0), 1, 1, fc=cm.get_cmap(self.cmap_name)(0.7),
                                       ec='none', alpha=0.9)
            ring_proxy = plt.Line2D([0], [0], linestyle=':', color='red', linewidth=2)
            self.ax_main.legend([surf_proxy, ring_proxy],
                                ['Wave amplitude', 'Monitor ring'],
                                bbox_to_anchor=(-0.10, 1.00),
                                loc='upper left',
                                borderaxespad=0.0,
                                fontsize=self.tick_fs)

        self.ax_main.set_title(f"Wave Field (t={self.current_time:.2f} s)",
                               fontsize=self.title_fs)

        # ── COLOUR-BAR – now on the far-left column ──────────────────
        norm = plt.Normalize(vmin=fld.min(), vmax=fld.max())
        if self.cbar is None:
            self.cbar = self.fig.colorbar(
                cm.ScalarMappable(norm=norm, cmap=self.cmap_name),
                cax=self.cbar_ax
            )
            self.cbar_ax.set_ylabel("Amplitude", fontsize=self.label_fs)
            self.cbar_ax.tick_params(labelsize=self.tick_fs)
        else:
            self.cbar.mappable.set_norm(norm)
            self.cbar.update_normal(self.cbar.mappable)

        # ── VELOCITY (ring-average) ──────────────────────────────────
        if self.velocity_history:
            t, v = zip(*self.velocity_history)
            self.vel_line.set_data(t, v)
            self.ax_velocity.relim(); self.ax_velocity.autoscale_view()

        self.ax_velocity.set_title(
            f"Ring-avg velocity  (r={self.monitor_ring_radius:.2f} m)",
            fontsize=self.title_fs
        )

        # ── SPECTRUM (ring-average amplitude) ────────────────────────
        if len(self.amplitude_history_for_fft) > 1:
            sig   = np.asarray(self.amplitude_history_for_fft, dtype=float).ravel()
            freq  = np.fft.rfftfreq(sig.size, d=self.dt)[1:]          # skip DC
            mag   = np.abs(np.fft.rfft(sig))[1:] / sig.size
            self.spec_line.set_data(freq, mag)
            self.ax_spectrum.relim(); self.ax_spectrum.autoscale_view()

        self.ax_spectrum.set_title(
            f"Spectrum  (ring-avg, t={self.current_time:.2f} s)",
            fontsize=self.title_fs
        )

        # ── final cosmetic tweaks ────────────────────────────────────
        for ax in (self.ax_main, self.ax_velocity, self.ax_spectrum):
            ax.tick_params(labelsize=self.tick_fs)

        self.fig.canvas.draw()
