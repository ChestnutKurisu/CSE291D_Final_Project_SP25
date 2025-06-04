import os
import numpy as np
import logging
from datetime import datetime
from matplotlib import animation
from tqdm import tqdm
from .visualizer import WaveVisualizer


def ricker_wavelet(t, f0, t0):
    """Generate a Ricker wavelet for time t, peak freq f0, and time delay t0."""
    arg = (np.pi * f0 * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


class WaveSolver:
    """
    Base class for wave solvers. Handles domain setup, ring averaging,
    optional absorbing boundaries, and logging of metrics.
    """

    def __init__(self, L, dx, dt, steps, ring_radius, wave_type_name, field_plot_label,
                 absorb_width_fract=0.0, absorb_strength=0.0,
                 log_interval=10):
        self.L = L
        self.dx = dx
        self.dy = dx
        self.dt = dt
        self.steps = steps
        self.log_interval = log_interval

        self.wave_type_name = wave_type_name
        self.field_plot_label = field_plot_label

        self.x = np.arange(0, self.L + self.dx, self.dx)
        self.y = np.arange(0, self.L + self.dy, self.dy)
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')  # shape (nx, ny)

        self.current_step = 0
        self.current_time = 0.0

        # Set up ring averaging:
        # Mask for points near ring_radius from center
        self.dist_from_center = np.sqrt((self.xx - self.L / 2) ** 2 +
                                        (self.yy - self.L / 2) ** 2)
        ring_thick = 3 * self.dx
        self.ring_mask = np.abs(self.dist_from_center - ring_radius) <= ring_thick / 2
        self.prev_ring_avg_field = 0.0
        self.velocity_hist = []
        self.amp_hist = []
        # Check ring once:
        self.has_ring = np.any(self.ring_mask)
        if not self.has_ring:
            logging.warning(
                "Ring mask is empty. No ring metrics will be meaningful. "
                "Check ring_radius and domain size L."
            )

        # Absorbing boundary (damping) field
        self.damping_field = np.ones((self.nx, self.ny), dtype=float)
        if absorb_width_fract > 0 and absorb_strength > 0:
            self.absorb_width_cells_x = int(absorb_width_fract * self.nx)
            self.absorb_width_cells_y = int(absorb_width_fract * self.ny)

            damping_profile_x = np.ones(self.nx, dtype=float)
            for i in range(self.absorb_width_cells_x):
                # Exponential ramp
                coef = np.exp(-absorb_strength * (
                    (self.absorb_width_cells_x - i - 1) / self.absorb_width_cells_x
                ) ** 2)
                damping_profile_x[i] *= coef
                damping_profile_x[-(i + 1)] *= coef

            damping_profile_y = np.ones(self.ny, dtype=float)
            for j in range(self.absorb_width_cells_y):
                coef = np.exp(-absorb_strength * (
                    (self.absorb_width_cells_y - j - 1) / self.absorb_width_cells_y
                ) ** 2)
                damping_profile_y[j] *= coef
                damping_profile_y[-(j + 1)] *= coef

            # Outer product
            self.damping_field = np.outer(damping_profile_x, damping_profile_y)

            logging.info(
                f"Absorbing layer: width_x={self.absorb_width_cells_x}, "
                f"width_y={self.absorb_width_cells_y}, strength={absorb_strength}"
            )
        else:
            logging.info("No absorbing boundary layer (width or strength is zero).")

    def initialize_fields(self):
        raise NotImplementedError

    def advance_one_step(self):
        raise NotImplementedError

    def get_field_to_visualize(self):
        raise NotImplementedError

    def calculate_energy(self):
        raise NotImplementedError

    def _log_step_info(self):
        """
        Logs ring-average amplitude, rate of change, and total energy
        at the current time step.
        """
        field = self.get_field_to_visualize()
        if self.has_ring:
            current_ring_avg_field = np.mean(field[self.ring_mask])
        else:
            current_ring_avg_field = 0.0

        ring_avg_rate = 0.0
        if self.dt > 0 and self.current_step > 0:
            ring_avg_rate = ((current_ring_avg_field - self.prev_ring_avg_field)
                             / self.dt)
        self.prev_ring_avg_field = current_ring_avg_field

        energy = self.calculate_energy()

        self.velocity_hist.append((self.current_time, ring_avg_rate))
        self.amp_hist.append(current_ring_avg_field)

        logging.info(
            f"step={self.current_step} time={self.current_time:.4e}s "
            f"ring_avg={current_ring_avg_field:.6e} ring_rate={ring_avg_rate:.6e} "
            f"energy={energy:.6e}"
        )


class ScalarWaveSolver(WaveSolver):
    """
    Simple 2D scalar wave solver using the standard 3-level explicit
    second-order scheme for the acoustic wave equation.
    """

    def __init__(self, L, dx, steps, ring_radius, c_wave,
                 wave_type_name, field_plot_label,
                 absorb_width_fract=0.0, absorb_strength=0.0,
                 log_interval=10):

        if c_wave <= 0:
            raise ValueError(f"Wave speed must be positive. Got {c_wave}")

        cfl_safety_factor = 0.7
        dt = cfl_safety_factor * dx / (c_wave * np.sqrt(2))

        super().__init__(L, dx, dt, steps, ring_radius, wave_type_name,
                         field_plot_label, absorb_width_fract, absorb_strength,
                         log_interval=log_interval)

        self.c_wave = c_wave
        self.S_sq = (self.c_wave * self.dt / self.dx) ** 2

        # f[..., 0] = u_{n-1}, f[..., 1] = u_n, f[..., 2] = u_{n+1}
        self.f = np.zeros((self.nx, self.ny, 3), dtype=float)

        self.initialize_fields()

        logging.info(
            f"Initialized ScalarWaveSolver '{self.wave_type_name}' with c={self.c_wave:.3f} m/s, "
            f"dt={self.dt:.3e}s, dx={self.dx:.3e}m, steps={steps}"
        )
        cfl_value = self.c_wave * self.dt / self.dx * np.sqrt(2)
        if cfl_value >= 1.0:
            logging.warning("CFL condition possibly violated (cfl_value >= 1).")

    def initialize_fields(self):
        # Gaussian initial condition at t=0
        width = 0.05 * self.L
        self.f[:, :, 0] = np.exp(
            -((self.xx - self.L / 2)**2 + (self.yy - self.L / 2)**2) / width**2
        )

        # For the second time step, we use a standard half-step approximation:
        lap_f0 = np.zeros_like(self.f[:, :, 0])
        # Vectorized Laplacian
        lap_f0[1:-1, 1:-1] = (
            (self.f[:-2, 1:-1, 0] + self.f[2:, 1:-1, 0]
             - 2.0*self.f[1:-1, 1:-1, 0]) / self.dx**2
          + (self.f[1:-1, :-2, 0] + self.f[1:-1, 2:, 0]
             - 2.0*self.f[1:-1, 1:-1, 0]) / self.dy**2
        )

        self.f[1:-1, 1:-1, 1] = \
            self.f[1:-1, 1:-1, 0] + 0.5 * self.S_sq * self.dx**2 * lap_f0[1:-1, 1:-1]

    def advance_one_step(self):
        # Vectorized Laplacian of current wavefield f[..., 1]
        lap_f1 = np.zeros_like(self.f[:, :, 1])
        lap_f1[1:-1, 1:-1] = (
            (self.f[:-2, 1:-1, 1] + self.f[2:, 1:-1, 1]
             - 2.0*self.f[1:-1, 1:-1, 1]) / self.dx**2
          + (self.f[1:-1, :-2, 1] + self.f[1:-1, 2:, 1]
             - 2.0*self.f[1:-1, 1:-1, 1]) / self.dy**2
        )

        self.f[1:-1, 1:-1, 2] = (
            2 * self.f[1:-1, 1:-1, 1] - self.f[1:-1, 1:-1, 0]
            + self.S_sq * self.dx**2 * lap_f1[1:-1, 1:-1]
        )

        # Apply damping:
        self.f[..., 2] *= self.damping_field

        # Shift time levels
        self.f[..., 0] = self.f[..., 1]
        self.f[..., 1] = self.f[..., 2]

        self.current_step += 1
        self.current_time = self.current_step * self.dt

    def get_field_to_visualize(self):
        return self.f[..., 1]  # The current wavefield

    def calculate_energy(self):
        # Use integral of u^2 as a potential-energy proxy
        return np.sum(self.f[..., 1]**2) * (self.dx * self.dy)


class ElasticWaveSolver2D(WaveSolver):
    """
    2D elastic wave solver (velocity-stress, collocated). Includes
    optional absorbing boundary and an explosive or directional source.
    """

    def __init__(self, L, dx, steps, ring_radius,
                 rho, lame_lambda, lame_mu,
                 source_x_fract, source_y_fract, source_freq, source_type,
                 absorb_width_fract, absorb_strength,
                 elastic_source_amplitude=1.0,  # <--- NEW amplitude scale
                 log_interval=10):

        self.rho = rho
        self.lame_lambda = lame_lambda
        self.lame_mu = lame_mu

        if self.rho <= 0.0:
            raise ValueError("Density (rho) must be > 0.")
        if self.lame_mu < 0.0:
            raise ValueError("Shear modulus (mu) must be >= 0.")
        if self.lame_lambda + 2*self.lame_mu <= 0:
            raise ValueError("lambda + 2*mu must be positive for real Vp.")

        self.vp = np.sqrt((self.lame_lambda + 2*self.lame_mu) / self.rho)
        self.vs = np.sqrt(self.lame_mu / self.rho) if self.lame_mu > 0 else 0.0

        cfl_safety_factor = 0.7
        if self.vp > 0:
            dt = cfl_safety_factor * dx / (self.vp * np.sqrt(2))
        else:
            dt = 0.001

        super().__init__(L, dx, dt, steps, ring_radius,
                         wave_type_name="Elastic P-SV",
                         field_plot_label="|u| Displacement",
                         absorb_width_fract=absorb_width_fract,
                         absorb_strength=absorb_strength,
                         log_interval=log_interval)

        # Setup source
        self.source_ix = int(source_x_fract*(self.nx-1))
        self.source_jy = int(source_y_fract*(self.ny-1))
        self.source_freq = source_freq
        self.source_type = source_type
        self.source_delay = 1.0 / self.source_freq if self.source_freq > 0 else 0.0

        # Scale factor for stress injection
        # This helps the wave to have a visible amplitude when
        # λ, μ, or ρ are large. Adjust from CLI if needed.
        self.src_amp_scale = elastic_source_amplitude

        # Allocate fields
        self.vx = np.zeros((self.nx, self.ny))
        self.vy = np.zeros((self.nx, self.ny))
        self.sxx = np.zeros((self.nx, self.ny))
        self.syy = np.zeros((self.nx, self.ny))
        self.sxy = np.zeros((self.nx, self.ny))
        self.ux = np.zeros((self.nx, self.ny))
        self.uy = np.zeros((self.nx, self.ny))

        self.initialize_fields()

        logging.info(
            f"Initialized ElasticWaveSolver2D: rho={self.rho:g}, "
            f"lambda={self.lame_lambda:g}, mu={self.lame_mu:g} -> vp={self.vp:.3f}, vs={self.vs:.3f}, "
            f"dt={self.dt:.3e}, steps={steps}, source_amp={self.src_amp_scale:.3g}"
        )
        cfl_val = self.vp*self.dt/self.dx*np.sqrt(2)
        if cfl_val >= 1.0:
            logging.warning("CFL condition possibly violated (Vp dt/dx * sqrt(2) >= 1).")

    def initialize_fields(self):
        # Start all fields at zero
        pass

    def advance_one_step(self):
        # 1) compute derivatives of stress
        dsxx_dx = (self.sxx[2:, 1:-1] - self.sxx[:-2, 1:-1])/(2*self.dx)
        dsyy_dy = (self.syy[1:-1, 2:] - self.syy[1:-1, :-2])/(2*self.dy)
        dsxy_dx = (self.sxy[2:, 1:-1] - self.sxy[:-2, 1:-1])/(2*self.dx)
        dsxy_dy = (self.sxy[1:-1, 2:] - self.sxy[1:-1, :-2])/(2*self.dy)

        # update vx, vy in interior
        self.vx[1:-1, 1:-1] += (self.dt / self.rho)*(
            dsxx_dx + dsxy_dy
        )
        self.vy[1:-1, 1:-1] += (self.dt / self.rho)*(
            dsxy_dx + dsyy_dy
        )

        # apply damping to velocities
        self.vx *= self.damping_field
        self.vy *= self.damping_field

        # 2) compute derivatives of velocity
        dvx_dx = (self.vx[2:, 1:-1] - self.vx[:-2, 1:-1])/(2*self.dx)
        dvy_dy = (self.vy[1:-1, 2:] - self.vy[1:-1, :-2])/(2*self.dy)
        dvx_dy = (self.vx[1:-1, 2:] - self.vx[1:-1, :-2])/(2*self.dy)
        dvy_dx = (self.vy[2:, 1:-1] - self.vy[:-2, 1:-1])/(2*self.dx)

        # update stresses
        self.sxx[1:-1, 1:-1] += self.dt * (
            (self.lame_lambda + 2*self.lame_mu)*dvx_dx + self.lame_lambda*dvy_dy
        )
        self.syy[1:-1, 1:-1] += self.dt * (
            self.lame_lambda*dvx_dx + (self.lame_lambda + 2*self.lame_mu)*dvy_dy
        )
        self.sxy[1:-1, 1:-1] += self.dt * self.lame_mu * (dvx_dy + dvy_dx)

        # source injection
        if self.source_freq > 0:
            t_now = self.current_time
            src_val = ricker_wavelet(t_now, self.source_freq, self.source_delay)
            # Multiply by our amplitude scale
            src_val *= self.src_amp_scale
            i, j = self.source_ix, self.source_jy
            if self.source_type == "explosive":
                self.sxx[i, j] += src_val
                self.syy[i, j] += src_val
            elif self.source_type == "force_x":
                self.sxy[i, j] += src_val
            elif self.source_type == "force_y":
                self.sxy[i, j] += src_val

        # apply damping to stresses
        self.sxx *= self.damping_field
        self.syy *= self.damping_field
        self.sxy *= self.damping_field

        # 3) update displacements from velocities
        self.ux[1:-1, 1:-1] += self.dt * self.vx[1:-1, 1:-1]
        self.uy[1:-1, 1:-1] += self.dt * self.vy[1:-1, 1:-1]

        self.current_step += 1
        self.current_time = self.current_step * self.dt

    def get_field_to_visualize(self):
        # Return displacement magnitude
        return np.sqrt(self.ux**2 + self.uy**2)

    def calculate_energy(self):
        # Kinetic energy
        ke_density = 0.5*self.rho*(self.vx**2 + self.vy**2)
        total_ke = np.sum(ke_density)*self.dx*self.dy

        # Strain energy
        # exx = d(ux)/dx, eyy = d(uy)/dy, exy = 0.5(d(ux)/dy + d(uy)/dx)
        # We approximate with central differences:
        exx = (self.ux[2:, 1:-1] - self.ux[:-2, 1:-1])/(2*self.dx)
        eyy = (self.uy[1:-1, 2:] - self.uy[1:-1, :-2])/(2*self.dy)
        dux_dy = (self.ux[1:-1, 2:] - self.ux[1:-1, :-2])/(2*self.dy)
        duy_dx = (self.uy[2:, 1:-1] - self.uy[:-2, 1:-1])/(2*self.dx)
        exy_mid = 0.5*(dux_dy + duy_dx)

        # Potential energy density = 0.5[ λ(exx+eyy)^2 + 2μ(exx^2+eyy^2+2exy^2) ]
        # We'll store it in an array of shape (nx-2, ny-2)
        exx_eyy_sum = exx + eyy
        pe_density = 0.5*(
            self.lame_lambda*(exx_eyy_sum**2)
            + 2*self.lame_mu*(exx**2 + eyy**2 + 2*exy_mid**2)
        )
        total_pe = np.sum(pe_density)*self.dx*self.dy

        return total_ke + total_pe


class ElasticPotentialSolver(WaveSolver):
    """
    2D elastic solver using phi, psi potentials. Shown here for completeness.
    ...
    (unchanged or omitted if you do not need it.)
    """
    # Omitted here for brevity. Keep if needed.


def run_simulation(
        out_path="wave_2d.mp4",
        steps=100,
        ring_radius=0.15,  # fraction of (L/2)
        log_interval=10,
        wave_type="acoustic",
        # scalar wave params
        c_acoustic=1.0,
        vp_scalar=2.0,
        vs_scalar=1.0,
        # elastic wave params
        rho=1.0,
        lame_lambda=1.0,
        lame_mu=1.0,
        source_x_fract=0.5,
        source_y_fract=0.5,
        source_freq=5.0,
        source_type_elastic="explosive",
        source_potential_type="P",
        absorb_width_fract=0.1,
        absorb_strength=20.0,  # reduced default a bit
        # Additional amplitude scale for elastic source:
        elastic_source_amplitude=1.0,
):
    """
    Main driver that instantiates the solver, runs steps, and writes an MP4.
    """
    L_domain = 2.0
    dx_grid = 0.01

    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(
        logs_dir, f"sim_{wave_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    for hdlr in logging.root.handlers[:]:
        logging.root.removeHandler(hdlr)
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    logging.info(
        f"Starting simulation: wave_type='{wave_type}' steps={steps}, out='{out_path}'"
    )

    actual_ring_radius = ring_radius*(L_domain/2.0)

    # pick the solver
    if wave_type == "acoustic":
        solver = ScalarWaveSolver(
            L=L_domain, dx=dx_grid, steps=steps, ring_radius=actual_ring_radius,
            c_wave=c_acoustic,
            wave_type_name="Acoustic",
            field_plot_label="Amplitude",
            absorb_width_fract=0.0,
            absorb_strength=0.0,
            log_interval=log_interval
        )
    elif wave_type == "P":
        solver = ScalarWaveSolver(
            L=L_domain, dx=dx_grid, steps=steps, ring_radius=actual_ring_radius,
            c_wave=vp_scalar,
            wave_type_name="P-wave (Scalar)",
            field_plot_label="P-potential",
            absorb_width_fract=0.0,
            absorb_strength=0.0,
            log_interval=log_interval
        )
    elif wave_type == "S_SH":
        solver = ScalarWaveSolver(
            L=L_domain, dx=dx_grid, steps=steps, ring_radius=actual_ring_radius,
            c_wave=vs_scalar,
            wave_type_name="S-wave (SH)",
            field_plot_label="SH field",
            absorb_width_fract=0.0,
            absorb_strength=0.0,
            log_interval=log_interval
        )
    elif wave_type == "S_SV_potential":
        solver = ScalarWaveSolver(
            L=L_domain, dx=dx_grid, steps=steps, ring_radius=actual_ring_radius,
            c_wave=vs_scalar,
            wave_type_name="S-wave (SV-Potential)",
            field_plot_label="SV potential",
            absorb_width_fract=0.0,
            absorb_strength=0.0,
            log_interval=log_interval
        )
    elif wave_type == "elastic":
        solver = ElasticWaveSolver2D(
            L=L_domain, dx=dx_grid, steps=steps, ring_radius=actual_ring_radius,
            rho=rho, lame_lambda=lame_lambda, lame_mu=lame_mu,
            source_x_fract=source_x_fract, source_y_fract=source_y_fract,
            source_freq=source_freq, source_type=source_type_elastic,
            absorb_width_fract=absorb_width_fract,
            absorb_strength=absorb_strength,
            elastic_source_amplitude=elastic_source_amplitude,
            log_interval=log_interval
        )
    elif wave_type == "elastic_potentials":
        # You may have your own "ElasticPotentialSolver" class
        # included here.  For brevity we skip it.
        raise NotImplementedError("elastic_potentials not fully shown here.")
    else:
        raise ValueError(f"Unknown wave_type={wave_type}")

    # Setup the visualizer
    # For elastic waves, amplitude can vary widely, so let dynamic_z handle it.
    if wave_type in ["elastic", "elastic_potentials"]:
        vis_dynamic_z = True
        vis_zlim = None
    else:
        # For a small initial pulse in scalar waves, fix a range:
        vis_dynamic_z = False
        vis_zlim = (-0.5, 1.0)

    viz = WaveVisualizer(
        sim_shape=(solver.ny, solver.nx),  # note the order for the visualizer
        output_video_size=(1920, 1080),
        dt=solver.dt,
        dx=solver.dx,
        main_plot_cmap_name="Spectral",
        dynamic_z=vis_dynamic_z,
        zlim=vis_zlim,
        font_size=12,
        field_name_label=solver.field_plot_label
    )

    writer = animation.FFMpegWriter(fps=25, bitrate=3000)

    with writer.saving(viz.fig, out_path, dpi=120):
        for _ in tqdm(range(solver.steps), desc=f"Simulating {solver.wave_type_name}"):
            solver.advance_one_step()

            # Log info only every log_interval steps
            if solver.current_step % solver.log_interval == 0:
                solver._log_step_info()

            field_for_viz = solver.get_field_to_visualize()
            viz.update(field_for_viz, solver.velocity_hist, solver.amp_hist,
                       solver.current_time, monitor_ring=actual_ring_radius)
            viz.render_composite_frame()
            writer.grab_frame()

    logging.info(f"Simulation finished. Video saved to {out_path}")
    logging.getLogger().removeHandler(console_handler)
    console_handler.close()
    return out_path
