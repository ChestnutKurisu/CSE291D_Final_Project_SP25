import argparse
from wave_sim.simulation import run_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2D wave simulations.")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of simulation steps.")
    parser.add_argument("--output", default="wave_2d.mp4",
                        help="Output video path.")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Interval (in steps) between log entries.")
    parser.add_argument("--wave_type", type=str, default="acoustic",
                        choices=["acoustic", "P", "S_SH", "S_SV_potential",
                                 "elastic", "elastic_potentials"],
                        help="Type of wave to simulate.")

    # Scalar wave parameters
    parser.add_argument("--c_acoustic", type=float, default=1.0,
                        help="Wave speed for acoustic simulation (m/s).")
    parser.add_argument("--vp_scalar", type=float, default=2.0,
                        help="P-wave velocity for scalar P simulation (m/s).")
    parser.add_argument("--vs_scalar", type=float, default=1.0,
                        help="S-wave velocity for scalar S/SH simulation (m/s).")

    # Elastic wave parameters
    parser.add_argument("--rho", type=float, default=1.0,
                        help="Density for elastic simulation (kg/m^3).")
    parser.add_argument("--lame_lambda", type=float, default=1.0,
                        help="Lamé's first parameter λ (Pa).")
    parser.add_argument("--lame_mu", type=float, default=1.0,
                        help="Lamé's second parameter μ (Pa).")

    parser.add_argument("--source_x_fract", type=float, default=0.5,
                        help="Fractional x-position of the source (0 to 1).")
    parser.add_argument("--source_y_fract", type=float, default=0.5,
                        help="Fractional y-position of the source (0 to 1).")
    parser.add_argument("--source_freq", type=float, default=5.0,
                        help="Peak frequency of Ricker wavelet source (Hz).")
    parser.add_argument("--source_type_elastic", type=str, default="explosive",
                        choices=["explosive", "force_x", "force_y"],
                        help="Source type for 'elastic' waves.")
    parser.add_argument("--source_potential_type", type=str, default="P",
                        choices=["P", "S", "both"],
                        help="Which potential(s) to drive for 'elastic_potentials' type.")

    # Let users set amplitude scaling for the elastic source if needed:
    parser.add_argument("--elastic_source_amplitude", type=float, default=1.0,
                        help="Additional amplitude scale for the elastic source stress injection.")

    # Absorbing boundary parameters
    parser.add_argument("--absorb_width_fract", type=float, default=0.1,
                        help="Fractional width of absorbing layer.")
    parser.add_argument("--absorb_strength", type=float, default=40.0,
                        help="Strength of damping in the absorbing layer.")

    args = parser.parse_args()

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
        source_potential_type=args.source_potential_type,
        absorb_width_fract=args.absorb_width_fract,
        absorb_strength=args.absorb_strength,
        # Pass in the user amplitude for the elastic solver:
        elastic_source_amplitude=args.elastic_source_amplitude,
    )
    print(f"Simulation complete. Output saved to: {out_path}")
