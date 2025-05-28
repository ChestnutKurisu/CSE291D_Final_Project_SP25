"""Alfv\u00e9n wave example using the consolidated solver."""

import matplotlib.pyplot as plt
import numpy as np

from wave_sim import AlfvenWave


def v0(x):
    L = x[-1]
    return np.sin(2 * np.pi * x / L)


if __name__ == "__main__":
    sim = AlfvenWave(L=2.0, Nx=800, T=2.0)
    sim.initial_conditions(v0)
    sol = sim.solve()
    plt.figure(figsize=(8, 4))
    plt.plot(sim.x, sol[-1], label="v_perp at final time")
    plt.xlabel("x")
    plt.ylabel("Transverse velocity")
    plt.title("Alfv\u00e9n Wave - Final State")
    plt.grid(True)
    plt.legend()
    plt.show()

