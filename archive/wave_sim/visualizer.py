import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import logging


class WaveVisualizer:
    """
    Visualise the 2D wave field in 3D plus ring-average plots on the side.
    """

    def __init__(self,
                 sim_shape,         # (ny, nx)
                 output_video_size, # (width, height) in pixels
                 dt,
                 dx,
                 main_plot_cmap_name="Spectral",
                 dynamic_z=True,
                 zlim=(-0.25, 1.0),
                 font_size=14,
                 field_name_label="Amplitude"):
        plt.rcParams.update({
            'font.size': font_size,
            'figure.max_open_warning': 0
        })
        self.label_fs = font_size
        self.title_fs = font_size + 2
        self.tick_fs = max(6, font_size - 2)

        self.sim_shape_ny, self.sim_shape_nx = sim_shape
        self.dt = dt
        self.dx = dx
        self.dynamic_z = bool(dynamic_z)
        self.zlim_user = zlim
        self.cmap_name = main_plot_cmap_name
        self.field_name_label = field_name_label

        w, h = output_video_size
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        self.fig.clf()

        gs = GridSpec(
            2, 3,
            figure=self.fig,
            width_ratios=[1.5, 48.5, 20],
            height_ratios=[1, 1],
            left=0.04, right=0.97, bottom=0.08, top=0.92,
            wspace=0.05, hspace=0.3
        )

        self.cbar_ax     = self.fig.add_subplot(gs[:, 0])
        self.ax_main     = self.fig.add_subplot(gs[:, 1], projection='3d')
        self.ax_velocity = self.fig.add_subplot(gs[0, 2])
        self.ax_spectrum = self.fig.add_subplot(gs[1, 2])

        if (self.sim_shape_nx > 0) and (self.sim_shape_ny > 0):
            # Generate x,y
            x_coords = np.arange(self.sim_shape_nx)*self.dx
            y_coords = np.arange(self.sim_shape_ny)*self.dx
            self.X, self.Y = np.meshgrid(x_coords, y_coords)
            # domain max extents
            self.domain_max_x = x_coords[-1]
            self.domain_max_y = y_coords[-1]
            self.cx, self.cy = x_coords.mean(), y_coords.mean()
        else:
            self.X, self.Y = np.array([[]]), np.array([[]])
            self.domain_max_x, self.domain_max_y = 1.0, 1.0
            self.cx, self.cy = 0.5, 0.5

        self.ax_main.set_xlabel("X (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_ylabel("Y (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_zlabel(self.field_name_label, fontsize=self.label_fs, labelpad=10)

        if not self.dynamic_z and self.zlim_user:
            self.ax_main.set_zlim(*self.zlim_user)

        self.vel_line,  = self.ax_velocity.plot([], [], '-b', label='Ring Avg. Rate')
        self.spec_line, = self.ax_spectrum.plot([], [], '-r', label='Ring Avg. Mag.')

        self.ax_velocity.set_xlabel("Time (s)", fontsize=self.label_fs)
        self.ax_velocity.set_ylabel(f"Ring Avg Rate ({self.field_name_label}/s)", fontsize=self.label_fs)
        self.ax_spectrum.set_xlabel("Frequency (Hz)", fontsize=self.label_fs)
        self.ax_spectrum.set_ylabel(f"Ring Avg ({self.field_name_label})", fontsize=self.label_fs)

        for a in [self.ax_velocity, self.ax_spectrum]:
            a.legend(fontsize=self.tick_fs)
            a.grid(True, linestyle=':', alpha=0.7)

        self.cbar = None
        target_cells_plot = 5000
        self.surf_stride = 1
        total_cells = self.sim_shape_nx*self.sim_shape_ny
        if total_cells > target_cells_plot:
            self.surf_stride = int(np.sqrt(total_cells/target_cells_plot))

        self.wire_stride = max(1, self.surf_stride*2)

        self.field_slice_to_visualise = None
        self.velocity_history = []
        self.amplitude_history_for_fft = []
        self.current_time = 0.0
        self.monitor_ring_radius = None

    def update(self, field_slice, velocity_history, amplitude_history, current_time, monitor_ring=None):
        # The sim internally stores field as (nx, ny). Here we expect field_slice as (ny, nx) => we transpose in code.
        # If you pass in the solver's shape (nx, ny) you may need to .T here.
        self.field_slice_to_visualise = field_slice.T  # transpose if needed
        self.velocity_history = velocity_history
        self.amplitude_history_for_fft = amplitude_history
        self.current_time = current_time
        self.monitor_ring_radius = monitor_ring

    def render_composite_frame(self):
        fld = self.field_slice_to_visualise
        if fld is None or fld.size == 0:
            logging.warning("No field slice data to render.")
            return

        self.ax_main.cla()
        self.ax_main.set_xlabel("X (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_ylabel("Y (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_zlabel(self.field_name_label, fontsize=self.label_fs, labelpad=10)

        if self.sim_shape_nx > 0 and self.sim_shape_ny > 0:
            aspect_z = 0.6*max(np.ptp(self.X), np.ptp(self.Y))
            if aspect_z < 1e-12:
                aspect_z = 1.0
            self.ax_main.set_box_aspect((np.ptp(self.X), np.ptp(self.Y), aspect_z))

        current_min_val, current_max_val = float(np.min(fld)), float(np.max(fld))
        if abs(current_max_val - current_min_val) < 1e-14:
            # Avoid degenerate range
            current_min_val -= 0.1
            current_max_val += 0.1

        if self.dynamic_z:
            self.ax_main.set_zlim(current_min_val, current_max_val)
        else:
            self.ax_main.set_zlim(*self.zlim_user)

        # prepare strides
        X_surf = self.X[::self.surf_stride, ::self.surf_stride]
        Y_surf = self.Y[::self.surf_stride, ::self.surf_stride]
        fld_surf = fld[::self.surf_stride, ::self.surf_stride]

        norm_surf = plt.Normalize(vmin=current_min_val, vmax=current_max_val)
        surf = self.ax_main.plot_surface(
            X_surf, Y_surf, fld_surf,
            cmap=self.cmap_name, norm=norm_surf,
            rstride=1, cstride=1,
            antialiased=False, linewidth=0, alpha=0.8
        )

        # wireframe
        X_wire = self.X[::self.wire_stride, ::self.wire_stride]
        Y_wire = self.Y[::self.wire_stride, ::self.wire_stride]
        fld_wire = fld[::self.wire_stride, ::self.wire_stride]
        self.ax_main.plot_wireframe(
            X_wire, Y_wire, fld_wire,
            color='gray', linewidth=0.3, alpha=0.3
        )

        # Monitoring ring
        if self.monitor_ring_radius and self.monitor_ring_radius > 0:
            theta_ring = np.linspace(0, 2*np.pi, 200)
            xs_ring = self.cx + self.monitor_ring_radius*np.cos(theta_ring)
            ys_ring = self.cy + self.monitor_ring_radius*np.sin(theta_ring)
            # Place the ring near the bottom of z-range, but slightly above it
            z_range = current_max_val - current_min_val
            if z_range < 1e-8:
                z_range = 0.1
            z_plot_val_ring = current_min_val + 0.05*z_range

            self.ax_main.plot(
                xs_ring, ys_ring,
                zs=z_plot_val_ring,
                linestyle=':',
                color='red', linewidth=1.5
            )

        self.ax_main.set_title(f"Field (t={self.current_time:.4e} s)", fontsize=self.title_fs)

        # colorbar
        if self.cbar is None:
            self.cbar = self.fig.colorbar(
                cm.ScalarMappable(norm=norm_surf, cmap=self.cmap_name),
                cax=self.cbar_ax
            )
            self.cbar_ax.set_ylabel(self.field_name_label, fontsize=self.label_fs)
            self.cbar_ax.tick_params(labelsize=self.tick_fs)
        else:
            self.cbar.mappable.set_norm(norm_surf)
            self.cbar.mappable.set_clim(current_min_val, current_max_val)

        # velocity (ring avg rate)
        if self.velocity_history:
            times, values = zip(*self.velocity_history)
            self.vel_line.set_data(times, values)
            self.ax_velocity.relim()
            self.ax_velocity.autoscale_view()

        self.ax_velocity.set_title("Ring Avg. Rate", fontsize=self.title_fs)

        # amplitude (ring avg) spectrum
        if len(self.amplitude_history_for_fft) > 1:
            signal_fft = np.asarray(self.amplitude_history_for_fft, dtype=float)
            if self.dt > 0:
                freq_fft = np.fft.rfftfreq(signal_fft.size, d=self.dt)
                mag_fft  = np.abs(np.fft.rfft(signal_fft))/signal_fft.size
                if len(freq_fft) > 1:
                    # skip DC
                    self.spec_line.set_data(freq_fft[1:], mag_fft[1:])
                else:
                    self.spec_line.set_data(freq_fft, mag_fft)
                self.ax_spectrum.relim()
                self.ax_spectrum.autoscale_view()

        self.ax_spectrum.set_title("Ring Avg. Spectrum", fontsize=self.title_fs)

        for ax_ in [self.ax_main, self.ax_velocity, self.ax_spectrum, self.cbar_ax]:
            ax_.tick_params(labelsize=self.tick_fs)

        self.fig.canvas.draw()
