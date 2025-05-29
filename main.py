from wave_sim import run_simulation, solve_p_wave, solve_s_wave
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the wave simulation")
    parser.add_argument("--steps", type=int, default=19,
                        help="Number of simulation steps")
    parser.add_argument("--output", default="wave_2d.mp4",
                        help="Output video path (baseline only)")
    parser.add_argument(
        "--wave", choices=["baseline", "p", "s"], default="baseline",
        help="Select which wave solver to run")
    args = parser.parse_args()

    if args.wave == "baseline":
        out = run_simulation(out_path=args.output, steps=args.steps)
        print(f"Saved -> {out}")
    elif args.wave == "p":
        snapshots, field = solve_p_wave(nt=args.steps)
        print("P-wave simulation complete. Final amplitude:", field[field.shape[0] // 2, field.shape[1] // 2])
    else:  # args.wave == "s"
        snapshots, field = solve_s_wave(nt=args.steps)
        print("S-wave simulation complete. Final amplitude:", field[field.shape[0] // 2, field.shape[1] // 2])
