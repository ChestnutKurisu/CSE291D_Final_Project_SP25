"""Capillary wave example using the consolidated solver."""

import numpy as np

from wave_sim import CapillaryWave
from wave_sim.animation_utils import generate_1d_animation, DEFAULT_OUTPUT_DIR_1D
from wave_sim.initial_conditions import gaussian_1d


def eta0(x):
    L = x[-1] + (x[1] - x[0])
    return gaussian_1d(x, center=L/4, sigma=L/20)


def deta0_dt(x):
    return np.zeros_like(x)

DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR_1D


def generate_animation(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    out_name: str = "capillary_wave.mp4",
    steps: int | None = None,
) -> str:
    L_domain = 2.0
    Nx_points = 400
    T_duration = 2.0
    sigma_val = 0.072
    rho_val = 1.0
    dx_val = L_domain / Nx_points
    dt_val = 0.5 * dx_val * np.sqrt(rho_val / sigma_val)

    sim = CapillaryWave(L=L_domain, Nx=Nx_points, sigma=sigma_val, rho=rho_val, dt=dt_val, T=T_duration)
    sim.initial_conditions(eta0, deta0_dt)

    return generate_1d_animation(
        solver=sim,
        out_name=out_name,
        plot_variable_name="eta_now",
        title_prefix="Capillary Wave",
        y_label="Surface displacement",
        output_dir=output_dir,
        y_lims=(-0.5, 0.5),
        total_steps=steps,
    )


if __name__ == "__main__":
    path = generate_animation()
