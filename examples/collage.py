import os
import inspect
import wave_sim


def run_and_save(sim_cls, name, steps=100):
    sim = sim_cls(grid_size=100)
    ani = sim.animate(steps=steps)
    path = f"{name}.mp4"
    ani.save(path)
    return path


def main():
    os.makedirs('output', exist_ok=True)
    wave_classes = [
        cls for name, cls in inspect.getmembers(wave_sim, inspect.isclass)
        if name.endswith('Simulation') and name != 'WaveSimulation'
    ]
    files = []
    for cls in sorted(wave_classes, key=lambda c: c.__name__):
        base_name = cls.__name__.replace('Simulation', '').lower()
        files.append(run_and_save(cls, os.path.join('output', base_name)))
    print('Generated files:', files)


if __name__ == '__main__':
    main()
