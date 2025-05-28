"""Linear Rossby planetary wave solved spectrally."""

import matplotlib.pyplot as plt
import numpy as np

from wave_sim import RossbyPlanetaryWave


def psi0(X, Y):
    Lx = X.max()
    Ly = Y.max()
    return np.exp(-((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2) / 0.2)


if __name__ == "__main__":
    sim = RossbyPlanetaryWave(Nx=256, Ny=256, T=3.0)
    sim.initial_conditions(psi0)
    sol = sim.solve()
    X, Y = np.meshgrid(
        np.linspace(0, sim.Lx, sim.Nx, endpoint=False),
        np.linspace(0, sim.Ly, sim.Ny, endpoint=False),
        indexing="ij",
    )
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, sol[-1], levels=20, cmap="RdBu_r")
    plt.colorbar(label="Streamfunction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rossby Planetary Wave - Final \u03c8")
    plt.show()

