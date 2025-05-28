"""1-D internal gravity wave using the consolidated solver."""

import matplotlib.pyplot as plt
import numpy as np

from wave_sim import InternalGravityWave


def psi0(x):
    return np.exp(-100 * (x - 1.0) ** 2)


if __name__ == "__main__":
    sim = InternalGravityWave(L=2.0, Nx=800, T=3.0)
    sim.initial_conditions(psi0)
    sol = sim.solve()
    x = sim.x
    plt.figure(figsize=(8, 4))
    plt.plot(x, sol[-1], label="psi at final time")
    plt.xlabel("x")
    plt.ylabel("psi")
    plt.title("Internal Gravity Wave - Final State")
    plt.grid(True)
    plt.legend()
    plt.show()

