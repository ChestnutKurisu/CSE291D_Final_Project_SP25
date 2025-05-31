import os
import numpy as np
import logging
from datetime import datetime
from matplotlib import animation
from tqdm import tqdm
from .visualizer import WaveVisualizer

# --- Utility ---
def ricker_wavelet(t, f0, t0):
    """Generate a Ricker wavelet."""
    arg = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)

# --- Base Solver Class ---
class WaveSolver:
    def __init__(self, L, dx, dt, steps, ring_radius, wave_type_name, field_plot_label):
        self.L = L
        self.dx = dx
        self.dy = dx # Assuming dx=dy
        self.dt = dt
        self.steps = steps
        self.ring_radius = ring_radius
        self.wave_type_name = wave_type_name
        self.field_plot_label = field_plot_label

        self.x = np.arange(0, self.L + self.dx, self.dx)
        self.y = np.arange(0, self.L + self.dy, self.dy)
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        self.current_step = 0
        self.current_time = 0.0

        # For ring average metrics
        self.dist_from_center = np.sqrt((self.xx - self.L / 2)**2 + (self.yy - self.L / 2)**2)
        ring_thick = 3 * self.dx
        self.ring_mask = np.abs(self.dist_from_center - self.ring_radius) <= ring_thick / 2
        if not np.any(self.ring_mask):
            logging.warning("Ring mask is empty. Ring-average metrics will be zero.")
        self.prev_ring_avg_field = 0.0
        self.velocity_hist = []
        self.amp_hist = []

    def initialize_fields(self):
        raise NotImplementedError

    def advance_one_step(self):
        raise NotImplementedError

    def get_field_to_visualize(self):
        raise NotImplementedError

    def calculate_energy(self):
        raise NotImplementedError

    def _log_step_info(self):
        field_to_viz = self.get_field_to_visualize()
        if np.any(self.ring_mask):
            current_ring_avg_field = np.mean(field_to_viz[self.ring_mask])
            ring_avg_rate = (current_ring_avg_field - self.prev_ring_avg_field) / self.dt if self.dt > 0 else 0.0
        else:
            current_ring_avg_field = 0.0
            ring_avg_rate = 0.0

        self.prev_ring_avg_field = current_ring_avg_field
        energy = self.calculate_energy()

        self.velocity_hist.append((self.current_time, ring_avg_rate))
        self.amp_hist.append(current_ring_avg_field)

        logging.info(
            f"step={self.current_step} t={self.current_time:.3e} s, "
            f"ring_avg_field={current_ring_avg_field:.6e}, "
            f"ring_avg_rate={ring_avg_rate:.6e}, "
            f"total_energy_proxy={energy:.6e}"
        )

# --- Scalar Wave Solver ---
class ScalarWaveSolver(WaveSolver):
    def __init__(self, L, dx, steps, ring_radius, c_wave, wave_type_name, field_plot_label):
        self.c_wave = c_wave
        if self.c_wave <= 0:
            raise ValueError(f"Wave speed must be positive. Got {self.c_wave}")

        # CFL condition for 2D scalar wave equation (5-point stencil)
        # dt <= dx / (c * sqrt(2))
        # Using a safety factor (e.g., 0.9 for Courant number related to 1D, so 0.9/sqrt(2) for 2D)
        cfl_safety_factor = 0.7 # This is S_ Courant number S = c*dt/dx. Max S for 2D is 1/sqrt(2) ~ 0.707
        dt = cfl_safety_factor * dx / (self.c_wave * np.sqrt(2)) if self.c_wave > 0 else 0.01

        super().__init__(L, dx, dt, steps, ring_radius, wave_type_name, field_plot_label)
        self.S_sq = (self.c_wave * self.dt / self.dx)**2
        self.f = np.zeros((self.nx, self.ny, 3)) # u_previous, u_current, u_next
        self.initialize_fields()
        logging.info(
            f"Initialized ScalarWaveSolver: type='{self.wave_type_name}', c={self.c_wave:.2f} m/s, "
            f"dt={self.dt:.3e} s, dx={self.dx:.3e} m, S^2={self.S_sq:.3f}, CFL check (S*sqrt(2)): {self.c_wave * self.dt / self.dx * np.sqrt(2):.3f} (must be <=1)"
        )


    def initialize_fields(self):
        # Initial condition: Gaussian pulse
        width = 0.05 * self.L
        self.f[:, :, 0] = np.exp(-((self.xx - self.L / 2)**2 + (self.yy - self.L / 2)**2) / width**2)

        # Second step (u^1) using u_t(0)=0 approximation
        lap_f0 = np.zeros_like(self.f[:,:,0])
        lap_f0[1:-1, 1:-1] = (
            (self.f[:-2, 1:-1, 0] + self.f[2:, 1:-1, 0] - 2 * self.f[1:-1, 1:-1, 0]) / self.dx**2 +
            (self.f[1:-1, :-2, 0] + self.f[1:-1, 2:, 0] - 2 * self.f[1:-1, 1:-1, 0]) / self.dy**2
        )
        self.f[1:-1, 1:-1, 1] = self.f[1:-1, 1:-1, 0] + 0.5 * self.S_sq * self.dx**2 * lap_f0[1:-1, 1:-1] # Corrected S_sq usage


    def advance_one_step(self):
        lap_f1 = np.zeros_like(self.f[:,:,1])
        lap_f1[1:-1, 1:-1] = (
            (self.f[:-2, 1:-1, 1] + self.f[2:, 1:-1, 1] - 2 * self.f[1:-1, 1:-1, 1]) / self.dx**2 +
            (self.f[1:-1, :-2, 1] + self.f[1:-1, 2:, 1] - 2 * self.f[1:-1, 1:-1, 1]) / self.dy**2
        )

        # u_next = 2*u_current - u_previous + (c*dt)^2 * Laplacian(u_current)
        # S_sq = (c*dt/dx)^2 so (c*dt)^2 * Laplacian = S_sq * dx^2 * Laplacian
        self.f[1:-1, 1:-1, 2] = (
            2 * self.f[1:-1, 1:-1, 1] - self.f[1:-1, 1:-1, 0] +
            self.S_sq * self.dx**2 * lap_f1[1:-1, 1:-1] # Corrected S_sq usage
        )

        self.f[:, :, 0] = self.f[:, :, 1].copy()
        self.f[:, :, 1] = self.f[:, :, 2].copy()

        self.current_step += 1
        self.current_time = self.current_step * self.dt

    def get_field_to_visualize(self):
        return self.f[:, :, 1] # u_current

    def calculate_energy(self):
        # Energy proxy: integral of field squared
        return np.sum(self.f[:, :, 1]**2 * self.dx * self.dy)

# --- Elastic Wave Solver (2D P-SV, collocated grid, velocity-stress) ---
class ElasticWaveSolver2D(WaveSolver):
    def __init__(self, L, dx, steps, ring_radius, rho, lame_lambda, lame_mu,
                 source_x_fract, source_y_fract, source_freq, source_type):

        self.rho = rho
        self.lame_lambda = lame_lambda
        self.lame_mu = lame_mu

        if self.rho <= 0: raise ValueError("Density (rho) must be positive.")
        if self.lame_mu <=0: raise ValueError("Shear modulus (mu) must be positive.")
        # (lambda + 2*mu) must also be positive, usually lambda >= 0 for most materials
        if self.lame_lambda + 2*self.lame_mu <= 0: raise ValueError("lambda + 2*mu must be positive.")

        self.vp = np.sqrt((self.lame_lambda + 2 * self.lame_mu) / self.rho)
        self.vs = np.sqrt(self.lame_mu / self.rho)

        # CFL condition for 2D elastic wave on collocated grid
        # dt <= dx / (Vp * sqrt(2))
        cfl_safety_factor = 0.7 # Courant number related safety
        dt = cfl_safety_factor * dx / (self.vp * np.sqrt(2)) if self.vp > 0 else 0.01

        super().__init__(L, dx, dt, steps, ring_radius, "Elastic P-SV", "|u| (Displacement)")

        # Source parameters
        self.source_ix = int(source_x_fract * (self.nx -1))
        self.source_jy = int(source_y_fract * (self.ny -1))
        self.source_freq = source_freq
        self.source_type = source_type
        self.source_delay = 1.0 / self.source_freq # Ricker delay for peak

        # Fields: velocities (vx, vy), stresses (sxx, syy, sxy), displacements (ux, uy)
        # Velocities at n-1/2, stresses & displacements at n
        self.vx = np.zeros((self.nx, self.ny))
        self.vy = np.zeros((self.nx, self.ny))
        self.sxx = np.zeros((self.nx, self.ny))
        self.syy = np.zeros((self.nx, self.ny))
        self.sxy = np.zeros((self.nx, self.ny))
        self.ux = np.zeros((self.nx, self.ny))
        self.uy = np.zeros((self.nx, self.ny))

        self.initialize_fields()
        logging.info(
            f"Initialized ElasticWaveSolver2D: rho={self.rho:.2f}, lambda={self.lame_lambda:.2f}, mu={self.lame_mu:.2f} -> "
            f"Vp={self.vp:.2f} m/s, Vs={self.vs:.2f} m/s, "
            f"dt={self.dt:.3e} s, dx={self.dx:.3e} m, "
            f"CFL check (Vp*dt/dx*sqrt(2)): {self.vp * self.dt / self.dx * np.sqrt(2):.3f} (must be <=1)"
        )
        logging.info(f"Elastic source: type='{self.source_type}', freq={self.source_freq:.1f} Hz at ({self.source_ix*self.dx:.2f}m, {self.source_jy*self.dy:.2f}m)")

    def initialize_fields(self):
        # All fields start at zero
        pass

    def _calculate_spatial_derivatives(self, field):
        # Central differences for interior points, one-sided at boundary (np.gradient behavior)
        # Or, for fixed boundaries, we only care about derivatives for interior updates.
        # Example: df/dx at (i,j) = (field[i+1,j] - field[i-1,j]) / (2*dx)
        # For explicit loops, we'd slice. For whole array ops:

        # Using np.gradient is convenient but applies specific boundary conditions.
        # For strict Dirichlet, better to slice and ensure boundaries are not changed.
        # Let's do explicit slicing for interior.

        df_dy = np.zeros_like(field)
        df_dx = np.zeros_like(field)

        # df_dx
        df_dx[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * self.dx)
        # df_dy
        df_dy[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * self.dy)

        # Simple boundary for derivatives (effectively zero gradient for these calculations if needed)
        # or let them be, as updates are only for interior.
        df_dx[0, :] = (field[1, :] - field[0, :]) / self.dx # Forward
        df_dx[-1, :] = (field[-1, :] - field[-2, :]) / self.dx # Backward
        df_dy[:, 0] = (field[:, 1] - field[:, 0]) / self.dy # Forward
        df_dy[:, -1] = (field[:, -1] - field[:, -2]) / self.dy # Backward

        return df_dx, df_dy

    def advance_one_step(self):
        # Calculate spatial derivatives of current stresses (sigma^n)
        dsxx_dx, dsxx_dy = self._calculate_spatial_derivatives(self.sxx) # dsxx_dy not used directly for vx
        dsyy_dx, dsyy_dy = self._calculate_spatial_derivatives(self.syy) # dsyy_dx not used directly for vy
        dsxy_dx, dsxy_dy = self._calculate_spatial_derivatives(self.sxy)

        # 1. Update velocities from n-1/2 to n+1/2
        # Interior points only. Boundaries (vx, vy = 0) are fixed.
        vx_new = self.vx.copy()
        vy_new = self.vy.copy()

        vx_new[1:-1, 1:-1] = self.vx[1:-1, 1:-1] + (self.dt / self.rho) * \
                             (dsxx_dx[1:-1, 1:-1] + dsxy_dy[1:-1, 1:-1])
        vy_new[1:-1, 1:-1] = self.vy[1:-1, 1:-1] + (self.dt / self.rho) * \
                             (dsxy_dx[1:-1, 1:-1] + dsyy_dy[1:-1, 1:-1])
        self.vx, self.vy = vx_new, vy_new


        # Calculate spatial derivatives of new velocities (v^{n+1/2})
        dvx_dx, dvx_dy = self._calculate_spatial_derivatives(self.vx)
        dvy_dx, dvy_dy = self._calculate_spatial_derivatives(self.vy)

        # 2. Update stresses from n to n+1
        # Interior points only.
        sxx_new = self.sxx.copy()
        syy_new = self.syy.copy()
        sxy_new = self.sxy.copy()

        sxx_new[1:-1, 1:-1] = self.sxx[1:-1, 1:-1] + self.dt * \
                              ((self.lame_lambda + 2 * self.lame_mu) * dvx_dx[1:-1, 1:-1] + \
                               self.lame_lambda * dvy_dy[1:-1, 1:-1])
        syy_new[1:-1, 1:-1] = self.syy[1:-1, 1:-1] + self.dt * \
                              (self.lame_lambda * dvx_dx[1:-1, 1:-1] + \
                               (self.lame_lambda + 2 * self.lame_mu) * dvy_dy[1:-1, 1:-1])
        sxy_new[1:-1, 1:-1] = self.sxy[1:-1, 1:-1] + self.dt * self.lame_mu * \
                              (dvx_dy[1:-1, 1:-1] + dvy_dx[1:-1, 1:-1])

        self.sxx, self.syy, self.sxy = sxx_new, syy_new, sxy_new

        # Add source term (Ricker wavelet) - applied to stress rates, so effectively add to stress
        # This is a common way to inject an explosive source.
        # Source is active for a certain duration. Let's make it time-dependent.
        # Time t_n for stress source application (center of dt interval for v^{n+1/2})
        time_for_source = self.current_time # or self.current_time + self.dt/2
        src_val = ricker_wavelet(time_for_source, self.source_freq, self.source_delay)

        # Scale source amplitude (heuristic, may need tuning)
        # Based on mu * dt^2 / (rho * dx^2) or similar scaling factor of simulation
        src_amp_scale = self.lame_mu * 1e-1 # Heuristic scaling for visibility

        if self.source_type == "explosive":
            self.sxx[self.source_ix, self.source_jy] += src_val * src_amp_scale
            self.syy[self.source_ix, self.source_jy] += src_val * src_amp_scale
        # For force source, it would be added to velocity update step, typically dv/dt = ... + F/rho
        # Here, simplified by adding to stress components that would be generated by such force.
        elif self.source_type == "force_x": # Creates shear primarily
             self.sxy[self.source_ix, self.source_jy] += src_val * src_amp_scale
        elif self.source_type == "force_y": # Creates shear primarily
             self.sxy[self.source_ix, self.source_jy] += src_val * src_amp_scale


        # 3. Update displacements from n to n+1 using v^{n+1/2}
        self.ux[1:-1, 1:-1] += self.dt * self.vx[1:-1, 1:-1]
        self.uy[1:-1, 1:-1] += self.dt * self.vy[1:-1, 1:-1]

        self.current_step += 1
        self.current_time = self.current_step * self.dt


    def get_field_to_visualize(self):
        return np.sqrt(self.ux**2 + self.uy**2)

    def calculate_energy(self):
        # Elastic energy: 0.5 * rho * v^2 (kinetic) + 0.5 * strain : stress (potential)
        # Simplified: integral of displacement magnitude squared or velocity magnitude squared
        # For proxy: kinetic energy based on v^{n+1/2}
        kinetic_energy = 0.5 * self.rho * (self.vx**2 + self.vy**2)
        # Potential energy is more complex with stress/strain.
        # Using displacement magnitude for simplicity, though not strictly energy.
        # Using sum of squares of velocities as a proxy for instantaneous energy
        return np.sum(self.rho * (self.vx**2 + self.vy**2) * self.dx * self.dy)


# --- Main Simulation Runner ---
def run_simulation(
    out_path="wave_2d.mp4",
    steps=100,
    ring_radius=0.15, # Fractional of L/2
    log_interval=1,
    wave_type="acoustic",
    # Scalar wave params
    c_acoustic=1.0,
    vp_scalar=2.0,
    vs_scalar=1.0,
    # Elastic wave params
    rho=1.0,
    lame_lambda=1.0,
    lame_mu=1.0,
    source_x_fract=0.5,
    source_y_fract=0.5,
    source_freq=5.0,
    source_type_elastic="explosive",
):
    L_domain = 2.0  # Domain length [m]
    dx_grid = 0.01 # Grid spacing [m]

    # Logging setup
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, f"sim_{wave_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Remove existing handlers to avoid duplicate logs if run multiple times in same session
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Starting simulation: wave_type='{wave_type}', steps={steps}, output='{out_path}'")

    actual_ring_radius = ring_radius * (L_domain / 2.0)

    solver: WaveSolver
    if wave_type == "acoustic":
        solver = ScalarWaveSolver(L_domain, dx_grid, steps, actual_ring_radius, c_acoustic, "Acoustic", "Amplitude")
    elif wave_type == "P":
        solver = ScalarWaveSolver(L_domain, dx_grid, steps, actual_ring_radius, vp_scalar, "P-wave Potential", r"$\Phi$ (P-potential)")
    elif wave_type == "S_SH":
        solver = ScalarWaveSolver(L_domain, dx_grid, steps, actual_ring_radius, vs_scalar, "S-wave (SH)", r"$u_z$ (SH)")
    elif wave_type == "S_SV_potential":
        solver = ScalarWaveSolver(L_domain, dx_grid, steps, actual_ring_radius, vs_scalar, "S-wave (SV Potential)", r"$\Psi_z$ (SV-potential)")
    elif wave_type == "elastic":
        solver = ElasticWaveSolver2D(L_domain, dx_grid, steps, actual_ring_radius, rho, lame_lambda, lame_mu,
                                     source_x_fract, source_y_fract, source_freq, source_type_elastic)
    else:
        raise ValueError(f"Unknown wave_type: {wave_type}")

    viz = WaveVisualizer(
        sim_shape=(solver.nx, solver.ny),
        output_video_size=(1280, 720), # Smaller for faster testing
        dt=solver.dt,
        dx=solver.dx,
        main_plot_cmap_name="Spectral",
        dynamic_z=False, # Keep z-lims fixed for better perception of wave growth/decay
        zlim=(-0.5, 1.0) if wave_type != "elastic" else (-0.01, 0.05), # Adjust zlim for displacement
        font_size=12,
        field_name_label=solver.field_plot_label,
    )

    # Determine initial zlim for elastic waves based on expected max displacement or early steps
    if wave_type == "elastic" and hasattr(solver, 'source_amp_scale'): # A bit of a hacky check
        # Estimate max displacement roughly. This is very heuristic.
        # max_u_est = solver.source_amp_scale * solver.dt**2 / (solver.rho * solver.dx) * 10 # Very rough
        # A few steps to find a good zlim might be better if dynamic_z is False
        # For now, use a pre-set or let the first frame set it if dynamic_z were True
        pass # zlim is preset, might need tuning based on source strength

    writer = animation.FFMpegWriter(fps=25, bitrate=5000)
    with writer.saving(viz.fig, out_path, dpi=150):
        for k_step in tqdm(range(solver.steps), desc=f"Simulating {solver.wave_type_name}"):
            solver.advance_one_step()

            # if k_step % log_interval == 0:
            #     solver._log_step_info()

            field_to_plot = solver.get_field_to_visualize()
            viz.update(field_to_plot, solver.velocity_hist, solver.amp_hist, solver.current_time, monitor_ring=actual_ring_radius)
            viz.render_composite_frame()
            writer.grab_frame()

    logging.info(f"Simulation finished. Video saved to {out_path}")
    return out_path
