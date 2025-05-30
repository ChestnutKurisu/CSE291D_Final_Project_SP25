from wave_sim.simulation import run_simulation
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the wave simulation")
    parser.add_argument("--steps", type=int, default=20, help="Number of simulation steps")
    parser.add_argument("--output", default="wave_2d.mp4", help="Output video path")
    parser.add_argument("--log_interval", type=int, default=1,
                        help="Interval (in steps) between log entries")

    parser.add_argument(
        "--wave_type",
        type=str,
        default="acoustic",
        choices=["acoustic", "P", "S_SH", "S_SV_potential", "elastic"],
        help=(
            "Type of wave to simulate: 'acoustic', 'P' for P-wave potential, "
            "'S_SH' for shear horizontal displacement, or 'S_SV_potential' "
            "for shear vertical wave potential."
        ),
    )
    parser.add_argument(
        "--c_acoustic", type=float, default=1.0,
        help="Wave speed for the basic acoustic simulation"
    )
    parser.add_argument(
        "--vp", type=float, default=2.0,
        help="P-wave velocity when simulating P-waves"
    )
    parser.add_argument(
        "--vs", type=float, default=1.0,
        help="S-wave velocity when simulating S-waves"
    )
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--lambda_lame", type=float, default=1.0)
    parser.add_argument("--mu_lame", type=float, default=1.0)
    parser.add_argument("--f0", type=float, default=25.0)

    args = parser.parse_args()

    out = run_simulation(
        out_path=args.output,
        steps=args.steps,
        log_interval=args.log_interval,
        wave_type=args.wave_type,
        c_acoustic=args.c_acoustic,
        rho=args.rho,
        lame_lambda=args.lambda_lame,
        lame_mu=args.mu_lame,
        f0=args.f0,
        vp=args.vp,
        vs=args.vs,
    )
    print(f"Saved -> {out}")
