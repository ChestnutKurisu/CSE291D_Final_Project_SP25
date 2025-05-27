import os
from wave_sim import PWaveSimulation, SWaveSimulation

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def run_and_save(sim, name, steps=100):
    ani = sim.animate(steps=steps)
    path = f"{name}.mp4"
    ani.save(path)
    return path


def main():
    os.makedirs('output', exist_ok=True)
    p_sim = PWaveSimulation(grid_size=100)
    s_sim = SWaveSimulation(grid_size=100)
    files = [
        run_and_save(p_sim, os.path.join('output', 'p_wave')),
        run_and_save(s_sim, os.path.join('output', 's_wave')),
    ]
    print('Generated files:', files)


if __name__ == '__main__':
    main()
