# TODO: https://aistudio.google.com/prompts/1y2hhaGn7JiM9G8NkcJiIQn18W8vgPPhW

import argparse
from wave_sim.simulation import run_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2D wave simulations.")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps.")
    parser.add_argument("--output", default="wave_2d.mp4", help="Output video path.")
    parser.add_argument("--log_interval", type=int, default=10, help="Interval (in steps) between log entries.")
    parser.add_argument(
        "--wave_type",
        type=str,
        default="acoustic",
        choices=["acoustic", "P", "S_SH", "S_SV_potential", "elastic"],
        help="Type of wave to simulate."
    )
    # Scalar wave parameters
    parser.add_argument("--c_acoustic", type=float, default=1.0, help="Wave speed for acoustic simulation.")
    parser.add_argument("--vp_scalar", type=float, default=2.0, help="P-wave velocity for scalar P-potential simulation.")
    parser.add_argument("--vs_scalar", type=float, default=1.0, help="S-wave velocity for scalar S-potential/SH simulation.")

    # Elastic wave parameters
    parser.add_argument("--rho", type=float, default=1.0, help="Density for elastic simulation (kg/m^3).")
    parser.add_argument("--lame_lambda", type=float, default=1.0, help="Lamé's first parameter for elastic simulation (Pa).")
    parser.add_argument("--lame_mu", type=float, default=1.0, help="Lamé's second parameter (shear modulus) for elastic simulation (Pa).")
    parser.add_argument("--source_x_fract", type=float, default=0.5, help="Fractional x-position of elastic source (0 to 1).")
    parser.add_argument("--source_y_fract", type=float, default=0.5, help="Fractional y-position of elastic source (0 to 1).")
    parser.add_argument("--source_freq", type=float, default=5.0, help="Peak frequency of Ricker wavelet source (Hz).")
    parser.add_argument("--source_type_elastic", type=str, default="explosive", choices=["explosive", "force_x", "force_y"], help="Type of source for elastic waves.")


    args = parser.parse_args()

    # For elastic waves, vp and vs are derived from rho, lambda, mu
    # For scalar P and S waves, vp_scalar and vs_scalar are used directly.
    # Ensure vp and vs for elastic are not confused with scalar vp/vs.
    # The simulation module will handle this distinction.

    out_path = run_simulation(
        out_path=args.output,
        steps=args.steps,
        log_interval=args.log_interval,
        wave_type=args.wave_type,
        c_acoustic=args.c_acoustic,
        vp_scalar=args.vp_scalar,
        vs_scalar=args.vs_scalar,
        rho=args.rho,
        lame_lambda=args.lame_lambda,
        lame_mu=args.lame_mu,
        source_x_fract=args.source_x_fract,
        source_y_fract=args.source_y_fract,
        source_freq=args.source_freq,
        source_type_elastic=args.source_type_elastic,
    )
    print(f"Simulation complete. Output saved to: {out_path}")