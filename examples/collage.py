import os
from wave_sim import (
    SeismicPWaveSimulation,
    SeismicSWaveSimulation,
    InternalGravityWaveSimulation,
    ElectromagneticWaveSimulation,
)

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def run_and_save(sim, name, steps=100):
    ani = sim.animate(steps=steps)
    path = f"{name}.mp4"
    ani.save(path)
    return path


def main():
    os.makedirs('output', exist_ok=True)
    demos = [
        (SeismicPWaveSimulation(grid_size=100), 'p_wave'),
        (SeismicSWaveSimulation(grid_size=100), 's_wave'),
        (InternalGravityWaveSimulation(grid_size=100), 'gravity_wave'),
        (ElectromagneticWaveSimulation(grid_size=100), 'em_wave'),
    ]
    files = [run_and_save(sim, os.path.join('output', name)) for sim, name in demos]
    print('Generated files:', files)


if __name__ == '__main__':
    main()
