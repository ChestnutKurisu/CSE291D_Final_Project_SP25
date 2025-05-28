"""Rotating shallow water Kelvin wave using the unified solver."""

import matplotlib.pyplot as plt
import numpy as np

from wave_sim import KelvinWave


def eta0(y):
    return np.exp(-((y - 2.5) / 0.5) ** 2)


if __name__ == "__main__":
    sim = KelvinWave(L=10.0, Ny=800, T=10.0)
    sim.initial_conditions(eta0)
    sol = sim.solve()
    y = sim.y
    plt.figure(figsize=(8, 4))
    plt.plot(y, sol[-1], label="eta at final time")
    plt.xlabel("y")
    plt.ylabel("Surface perturbation")
    plt.title("Kelvin Wave - Final Surface Perturbation")
    plt.grid(True)
    plt.legend()
    plt.show()

