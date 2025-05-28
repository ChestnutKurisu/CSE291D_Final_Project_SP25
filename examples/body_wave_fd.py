"""High-resolution finite-difference body wave examples using ``wave_sim``."""

import matplotlib.pyplot as plt
import numpy as np

from wave_sim import PWaveSimulation, SWaveSimulation, SHWaveSimulation, SVWaveSimulation


def gaussian(X, Y, sigma=5.0):
    cx = X.shape[0] // 2
    cy = Y.shape[1] // 2
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


def run(sim_cls, title):
    sim = sim_cls(grid_size=512, boundary="absorbing", source_func=gaussian)
    frames = sim.simulate(steps=750)
    plt.figure()
    plt.imshow(frames[-1], cmap="seismic", origin="lower", aspect="auto")
    plt.title(title)
    plt.colorbar(label="Amplitude")


if __name__ == "__main__":
    run(PWaveSimulation, "P-Wave")
    run(SWaveSimulation, "S-Wave")
    run(SHWaveSimulation, "SH-Wave")
    run(SVWaveSimulation, "SV-Wave")
    plt.show()

