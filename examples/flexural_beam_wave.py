"""Flexural beam wave solved with the consolidated solver."""

import matplotlib.pyplot as plt
import numpy as np

from wave_sim import FlexuralBeamWave


def w0(x):
    L = x[-1]
    return np.exp(-100 * (x - L / 2) ** 2)


if __name__ == "__main__":
    sim = FlexuralBeamWave(L=2.0, Nx=801, T=5.0)
    sim.initial_conditions(w0)
    sol = sim.solve()
    plt.figure(figsize=(8, 4))
    plt.plot(sim.x, sol[-1], label="w at final time")
    plt.xlabel("x")
    plt.ylabel("Displacement")
    plt.title("Flexural Beam Wave - Final")
    plt.grid(True)
    plt.legend()
    plt.show()

