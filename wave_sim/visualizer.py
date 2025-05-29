# wave_sim/visualizer.py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # Not used
# from mpl_toolkits.axes_grid1 import make_axes_locatable  # Not used
# import matplotlib.axes as maxes  # Not used


class WaveVisualizer:
    """Visualise the 3-D wave slice together with ring-average velocity and spectrum."""

    def __init__(self,
                 sim_shape,
                 output_video_size,
                 dt,
                 dx,
                 main_plot_cmap_name="Spectral",
                 # brightness_scale=1.0,  # Not used
                 dynamic_z=True,
                 zlim=(-0.25, 1.0),
                 font_size=14,
                 field_name_label="Amplitude"):
        # ── layout basics ───────────────────────────────────────────────
        plt.rcParams.update({'font.size': font_size})
        self.label_fs, self.title_fs = font_size, font_size + 2
        self.tick_fs = max(6, font_size - 2)

        self.sim_shape = sim_shape
        self.dt, self.dx = dt, dx
        self.dynamic_z = bool(dynamic_z)
        self.zlim_user = zlim
        self.cmap_name = main_plot_cmap_name
        self.field_name_label = field_name_label

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
            x_coords = np.arange(nx) * dx
            y_coords = np.arange(ny) * dx
            self.X, self.Y = np.meshgrid(x_coords, y_coords)
            self.cx, self.cy = x_coords.mean(), y_coords.mean()

        # labels + aspect for main 3-D plot
        self.ax_main.set_xlabel("X (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_ylabel("Y (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_zlabel(self.field_name_label, fontsize=self.label_fs, labelpad=10)
        if self.sim_shape:
            self.ax_main.set_box_aspect((
                np.ptp(x_coords),
                np.ptp(y_coords),
                0.6 * max(np.ptp(x_coords), np.ptp(y_coords))
            ))
        if not self.dynamic_z and self.zlim_user:
            self.ax_main.set_zlim(*self.zlim_user)

        # lines for the 2-D graphs
        self.vel_line,  = self.ax_velocity.plot([], [], '-b', label='Field Rate')
        self.spec_line, = self.ax_spectrum.plot([], [], '-r', label='Magnitude')
        self.ax_velocity.set_xlabel("Time (s)",     fontsize=self.label_fs)
        self.ax_velocity.set_ylabel(f"Ring-Avg {self.field_name_label} Rate", fontsize=self.label_fs)
        self.ax_spectrum.set_xlabel("Frequency (Hz)", fontsize=self.label_fs)
        self.ax_spectrum.set_ylabel(f"Ring-Avg {self.field_name_label} Mag.", fontsize=self.label_fs)
        self.ax_velocity.legend(fontsize=self.tick_fs)
        self.ax_spectrum.legend(fontsize=self.tick_fs)

        # colour-bar initialised once; updated later
        self.cbar = None

        # runtime placeholders
        target_cells = 5000
        self._surf_stride = max(
            1,
            int(np.sqrt((sim_shape[0] * sim_shape[1]) / target_cells)) if sim_shape[0] * sim_shape[1] > 0 else 1
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
        self.ax_main.set_zlabel(self.field_name_label, fontsize=self.label_fs, labelpad=10)
        if hasattr(self, 'X'):
            self.ax_main.set_box_aspect((np.ptp(self.X), np.ptp(self.Y), 0.6 * max(np.ptp(self.X), np.ptp(self.Y))))

        current_zlim = None
        if self.dynamic_z:
            min_val, max_val = np.min(fld), np.max(fld)
            if min_val == max_val:
                min_val -= 0.1
                max_val += 0.1
            current_zlim = (min_val, max_val)
            self.ax_main.set_zlim(*current_zlim)
        elif self.zlim_user:
            current_zlim = self.zlim_user
            self.ax_main.set_zlim(*current_zlim)
        else:
            current_zlim = (np.min(fld), np.max(fld)) if np.any(fld) else (-1, 1)
            if current_zlim[0] == current_zlim[1]:
                current_zlim = (current_zlim[0] - 0.1, current_zlim[1] + 0.1)
            self.ax_main.set_zlim(*current_zlim)

        ds = self._surf_stride
        norm_surf = plt.Normalize(vmin=fld.min(), vmax=fld.max()) if np.any(fld) else plt.Normalize(vmin=-1, vmax=1)
        self.ax_main.plot_surface(
            self.X[::ds, ::ds], self.Y[::ds, ::ds], fld[::ds, ::ds],
            cmap=self.cmap_name, norm=norm_surf,
            rstride=1, cstride=1, antialiased=False, linewidth=0, alpha=0.9
        )
        wire_ds = self._wire_stride
        self.ax_main.plot_wireframe(
            self.X[::wire_ds, ::wire_ds], self.Y[::wire_ds, ::wire_ds], fld[::wire_ds, ::wire_ds],
            color='grey', linewidth=0.4, alpha=0.5
        )

        # draw the monitoring ring on top of the surface
        if self.monitor_ring_radius and self.monitor_ring_radius > 0:
            θ = np.linspace(0, 2 * np.pi, 240)
            xs = self.cx + self.monitor_ring_radius * np.cos(θ)
            ys = self.cy + self.monitor_ring_radius * np.sin(θ)

            z_plot_val = current_zlim[1] - 0.02 * (current_zlim[1] - current_zlim[0])

            # depthshade removed — keep zorder & clip_on to ensure visibility
            self.ax_main.plot(
                xs, ys, z_plot_val, ':', color='red', linewidth=2,
                zorder=100, clip_on=False
            )

            surf_proxy = plt.Rectangle((0, 0), 1, 1, fc=cm.get_cmap(self.cmap_name)(0.7),
                                       ec='none', alpha=0.9)
            ring_proxy = plt.Line2D([0], [0], linestyle=':', color='red', linewidth=2)
            self.ax_main.legend([surf_proxy, ring_proxy],
                                [f'{self.field_name_label}', 'Monitor ring'],
                                bbox_to_anchor=(-0.10, 1.00),
                                loc='upper left',
                                borderaxespad=0.0,
                                fontsize=self.tick_fs)

        self.ax_main.set_title(f"{self.field_name_label} (t={self.current_time:.2f} s)",
                               fontsize=self.title_fs)

        # ── COLOUR-BAR – now on the far-left column ──────────────────
        norm_cbar = plt.Normalize(vmin=fld.min(), vmax=fld.max()) if np.any(fld) else plt.Normalize(vmin=-1, vmax=1)
        if self.cbar is None:
            self.cbar = self.fig.colorbar(
                cm.ScalarMappable(norm=norm_cbar, cmap=self.cmap_name),
                cax=self.cbar_ax
            )
            self.cbar_ax.set_ylabel(self.field_name_label, fontsize=self.label_fs)
            self.cbar_ax.tick_params(labelsize=self.tick_fs)
        else:
            self.cbar.mappable.set_norm(norm_cbar)
            self.cbar.mappable.set_array([])
            self.cbar.mappable.set_clim(fld.min(), fld.max())

        # ── VELOCITY (ring-average) ──────────────────────────────────
        if self.velocity_history:
            t, v = zip(*self.velocity_history)
            self.vel_line.set_data(t, v)
            self.ax_velocity.relim(); self.ax_velocity.autoscale_view(True, True, True)

        self.ax_velocity.set_title(
            f"Ring-Avg {self.field_name_label} Rate (r={self.monitor_ring_radius:.2f} m)",
            fontsize=self.title_fs
        )

        # ── SPECTRUM (ring-average amplitude) ────────────────────────
        if len(self.amplitude_history_for_fft) > 1:
            sig   = np.asarray(self.amplitude_history_for_fft, dtype=float).ravel()
            if len(sig) > 1 and self.dt > 0:
                freq  = np.fft.rfftfreq(sig.size, d=self.dt)
                mag   = np.abs(np.fft.rfft(sig)) / sig.size
                if len(freq) > 1:
                    self.spec_line.set_data(freq[1:], mag[1:])
                else:
                    self.spec_line.set_data(freq, mag)
                self.ax_spectrum.relim(); self.ax_spectrum.autoscale_view(True, True, True)

        self.ax_spectrum.set_title(
            f"Spectrum ({self.field_name_label}, t={self.current_time:.2f} s)",
            fontsize=self.title_fs
        )
        self.ax_spectrum.set_ylabel(f"Magnitude ({self.field_name_label})", fontsize=self.label_fs)

        # ── final cosmetic tweaks ────────────────────────────────────
        for ax in (self.ax_main, self.ax_velocity, self.ax_spectrum):
            ax.tick_params(labelsize=self.tick_fs)

        self.fig.canvas.draw()
