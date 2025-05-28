from wave_sim.simulation import run_simulation
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the wave simulation")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration via CuPy")
    parser.add_argument("--steps", type=int, default=199, help="Number of simulation steps")
    parser.add_argument("--output", default="wave_2d.mp4", help="Output video path")
    args = parser.parse_args()

    out = run_simulation(out_path=args.output, steps=args.steps, use_gpu=args.gpu)
    print(f"Saved -> {out}")
