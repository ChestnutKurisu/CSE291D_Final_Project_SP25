# wave_sim/visualizer.py
import numpy as np
import matplotlib
matplotlib.use("Agg") # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import logging


class WaveVisualizer:
    """Visualise the 2D wave slice together with ring-average metrics."""

    def __init__(self,
                 sim_shape,        # (nx, ny) tuple for the simulation grid
                 output_video_size,# (width, height) in pixels
                 dt,               # Timestep dt
                 dx,               # Grid spacing dx
                 main_plot_cmap_name="Spectral",
                 dynamic_z=True,   # Whether Z-axis limits adapt to data
                 zlim=(-0.25, 1.0),# Default Z-axis limits if dynamic_z is False
                 font_size=14,
                 field_name_label="Amplitude"): # Label for the Z-axis and colorbar

        plt.rcParams.update({'font.size': font_size, 'figure.max_open_warning': 0}) # Suppress max_open_warning
        self.label_fs = font_size
        self.title_fs = font_size + 2
        self.tick_fs = max(6, font_size - 2)

        self.sim_shape_ny, self.sim_shape_nx = sim_shape # Note: order (ny, nx) for numpy arrays
        self.dt, self.dx = dt, dx
        self.dynamic_z = bool(dynamic_z)
        self.zlim_user = zlim
        self.cmap_name = main_plot_cmap_name
        self.field_name_label = field_name_label

        w, h = output_video_size
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        # Clear figure if it exists to prevent overlap in interactive sessions
        self.fig.clf()


        gs = GridSpec(
            2, 3,
            figure=self.fig,
            width_ratios=[1.5, 48.5, 20], # Adjusted for colorbar text | c-bar | 3-D | two 2-D plots |
            height_ratios=[1, 1],
            left=0.04, right=0.97, bottom=0.09, top=0.92, # Adjusted margins
            wspace=0.05, hspace=0.3
        )

        self.cbar_ax     = self.fig.add_subplot(gs[:, 0])
        self.ax_main     = self.fig.add_subplot(gs[:, 1], projection='3d')
        self.ax_velocity = self.fig.add_subplot(gs[0, 2]) # Ring-average rate
        self.ax_spectrum = self.fig.add_subplot(gs[1, 2]) # Ring-average spectrum

        if self.sim_shape_nx > 0 and self.sim_shape_ny > 0:
            x_coords = np.arange(self.sim_shape_nx) * dx
            y_coords = np.arange(self.sim_shape_ny) * dx # Assuming dx=dy
            self.X, self.Y = np.meshgrid(x_coords, y_coords)
            self.domain_max_x = x_coords[-1]
            self.domain_max_y = y_coords[-1]
            self.cx, self.cy = x_coords.mean(), y_coords.mean() # Center of the domain
        else: # Handle empty sim_shape if it occurs
            self.X, self.Y = np.array([[]]), np.array([[]])
            self.domain_max_x, self.domain_max_y = 1.0, 1.0
            self.cx, self.cy = 0.5, 0.5

        self.ax_main.set_xlabel("X (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_ylabel("Y (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_zlabel(self.field_name_label, fontsize=self.label_fs, labelpad=10)

        if self.sim_shape_nx > 0 and self.sim_shape_ny > 0:
            aspect_z = 0.6 * max(np.ptp(self.X), np.ptp(self.Y)) if max(np.ptp(self.X), np.ptp(self.Y)) > 0 else 1.0
            self.ax_main.set_box_aspect((np.ptp(self.X), np.ptp(self.Y), aspect_z))

        if not self.dynamic_z and self.zlim_user:
            self.ax_main.set_zlim(*self.zlim_user)

        self.vel_line,  = self.ax_velocity.plot([], [], '-b', label='Avg. Field Rate')
        self.spec_line, = self.ax_spectrum.plot([], [], '-r', label='Avg. Field Mag.')
        self.ax_velocity.set_xlabel("Time (s)", fontsize=self.label_fs)
        self.ax_velocity.set_ylabel(f"Ring Avg. Rate ({self.field_name_label}/s)", fontsize=self.label_fs)
        self.ax_spectrum.set_xlabel("Frequency (Hz)", fontsize=self.label_fs)
        self.ax_spectrum.set_ylabel(f"Ring Avg. Mag. ({self.field_name_label})", fontsize=self.label_fs)

        for ax_ in [self.ax_velocity, self.ax_spectrum]:
            ax_.legend(fontsize=self.tick_fs)
            ax_.grid(True, linestyle=':', alpha=0.7)

        self.cbar = None
        target_cells_plot = 5000 # For plot_surface stride
        self.surf_stride = 1
        if self.sim_shape_nx * self.sim_shape_ny > target_cells_plot:
             self.surf_stride = max(1, int(np.sqrt((self.sim_shape_nx * self.sim_shape_ny) / target_cells_plot)))
        self.wire_stride = max(1, self.surf_stride * 2) # Coarser for wireframe

        self.field_slice_to_visualise = None
        self.velocity_history = []
        self.amplitude_history_for_fft = []
        self.current_time = 0.0
        self.monitor_ring_radius = None

    def update(self, field_slice, velocity_history, amplitude_history, current_time, monitor_ring=None):
        self.field_slice_to_visualise = field_slice.T # Transpose for meshgrid convention if needed
        self.velocity_history = velocity_history
        self.amplitude_history_for_fft = amplitude_history # This is ring-avg field, not its rate
        self.current_time = current_time
        self.monitor_ring_radius = monitor_ring

    def render_composite_frame(self):
        fld = self.field_slice_to_visualise
        if fld is None or self.X is None or self.Y is None:
            logging.warning("Visualizer: field_slice or meshgrid not available for rendering.")
            return

        # --- 3D Main Plot ---
        self.ax_main.cla() # Clear previous frame elements
        self.ax_main.set_xlabel("X (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_ylabel("Y (m)", fontsize=self.label_fs, labelpad=10)
        self.ax_main.set_zlabel(self.field_name_label, fontsize=self.label_fs, labelpad=10)

        if self.sim_shape_nx > 0 and self.sim_shape_ny > 0:
            aspect_z = 0.6 * max(np.ptp(self.X), np.ptp(self.Y)) if max(np.ptp(self.X), np.ptp(self.Y)) > 0 else 1.0
            self.ax_main.set_box_aspect((np.ptp(self.X), np.ptp(self.Y), aspect_z))

        current_min_val, current_max_val = np.min(fld), np.max(fld)
        if current_min_val == current_max_val: # Avoid singular limits
            current_min_val -= 0.1
            current_max_val += 0.1
            if current_min_val == 0 and current_max_val == 0: # if fld is all zero
                 current_min_val, current_max_val = self.zlim_user if self.zlim_user else (-0.1, 0.1)


        if self.dynamic_z:
            self.ax_main.set_zlim(current_min_val, current_max_val)
            current_zlim_for_ring = (current_min_val, current_max_val)
        else:
            self.ax_main.set_zlim(*self.zlim_user)
            current_zlim_for_ring = self.zlim_user

        # Ensure X, Y, fld have compatible shapes for plot_surface after striding
        X_surf, Y_surf = self.X[::self.surf_stride, ::self.surf_stride], self.Y[::self.surf_stride, ::self.surf_stride]
        fld_surf = fld[::self.surf_stride, ::self.surf_stride]

        norm_surf = plt.Normalize(vmin=current_min_val, vmax=current_max_val)
        self.ax_main.plot_surface(
            X_surf, Y_surf, fld_surf,
            cmap=self.cmap_name, norm=norm_surf,
            rstride=1, cstride=1, antialiased=False, linewidth=0, alpha=0.9
        )

        X_wire, Y_wire = self.X[::self.wire_stride, ::self.wire_stride], self.Y[::self.wire_stride, ::self.wire_stride]
        fld_wire = fld[::self.wire_stride, ::self.wire_stride]
        self.ax_main.plot_wireframe(
            X_wire, Y_wire, fld_wire,
            color='grey', linewidth=0.3, alpha=0.4
        )

        # Monitoring ring
        if self.monitor_ring_radius and self.monitor_ring_radius > 0:
            theta_ring = np.linspace(0, 2 * np.pi, 100)
            xs_ring = self.cx + self.monitor_ring_radius * np.cos(theta_ring)
            ys_ring = self.cy + self.monitor_ring_radius * np.sin(theta_ring)
            # Plot ring slightly above the mid-point of z-axis for visibility
            z_plot_val_ring = (
                    current_zlim_for_ring[1]
                    - 0.65 * (current_zlim_for_ring[1] - current_zlim_for_ring[0])
            )

            self.ax_main.plot(xs_ring, ys_ring, z_plot_val_ring, ':', color='red', linewidth=1.5, zorder=100, clip_on=False)

        self.ax_main.set_title(f"Field (t={self.current_time:.3e} s)", fontsize=self.title_fs)

        # --- Colorbar ---
        if self.cbar is None:
            self.cbar = self.fig.colorbar(
                cm.ScalarMappable(norm=norm_surf, cmap=self.cmap_name),
                cax=self.cbar_ax
            )
            self.cbar_ax.set_ylabel(self.field_name_label, fontsize=self.label_fs)
            self.cbar_ax.tick_params(labelsize=self.tick_fs)
        else:
            self.cbar.mappable.set_norm(norm_surf)
            self.cbar.mappable.set_clim(current_min_val, current_max_val) # Update clim for existing colorbar


        # --- Velocity (Ring-Average Rate) Plot ---
        if self.velocity_history:
            times, values = zip(*self.velocity_history)
            self.vel_line.set_data(times, values)
            self.ax_velocity.relim()
            self.ax_velocity.autoscale_view(True, True, True)
        self.ax_velocity.set_title(f"Ring Avg. Rate (r={self.monitor_ring_radius or 0:.2f}m)", fontsize=self.title_fs)

        # --- Spectrum (Ring-Average Amplitude) Plot ---
        if len(self.amplitude_history_for_fft) > 1:
            signal_fft = np.asarray(self.amplitude_history_for_fft, dtype=float).ravel()
            if len(signal_fft) > 1 and self.dt > 0:
                freq_fft = np.fft.rfftfreq(signal_fft.size, d=self.dt)
                mag_fft = np.abs(np.fft.rfft(signal_fft)) / signal_fft.size # Normalize

                # Avoid plotting DC component and handle nyquist if needed
                if len(freq_fft) > 1:
                    self.spec_line.set_data(freq_fft[1:], mag_fft[1:])
                elif len(freq_fft) == 1 : # Only DC
                     self.spec_line.set_data(freq_fft, mag_fft)

                self.ax_spectrum.relim()
                self.ax_spectrum.autoscale_view(True, True, True)
        self.ax_spectrum.set_title(f"Ring Avg. Spectrum", fontsize=self.title_fs)

        for ax_ in [self.ax_main, self.ax_velocity, self.ax_spectrum, self.cbar_ax]:
            ax_.tick_params(labelsize=self.tick_fs)

        self.fig.canvas.draw()
