import numpy as np
import matplotlib.pyplot as plt

# Internal Gravity Wave Solver
# ----------------------------
# Solves psi_tt = N^2 psi_xx on a 1-D grid using a simple
# finite difference scheme with Dirichlet boundaries.

Lx = 2.0
Nx = 400
Nfreq = 1.0

c = Nfreq

dx = Lx / (Nx - 1)
Tmax = 3.0
dt = 0.8 * dx / c
nt = int(Tmax / dt)

x = np.linspace(0, Lx, Nx)

psi_old = np.zeros(Nx)
psi = np.exp(-100 * (x - Lx / 2) ** 2)
psi_old[:] = psi
psi_new = np.zeros_like(psi)

for _ in range(nt):
    for i in range(1, Nx - 1):
        psi_new[i] = (
            2 * psi[i]
            - psi_old[i]
            + (c * dt / dx) ** 2 * (psi[i + 1] - 2 * psi[i] + psi[i - 1])
        )
    psi_new[0] = 0.0
    psi_new[-1] = 0.0
    psi_old, psi = psi, psi_new

plt.figure(figsize=(8, 4))
plt.plot(x, psi, label="psi at final time")
plt.xlabel("x")
plt.ylabel("psi")
plt.title("Internal Gravity Wave - Final State")
plt.grid(True)
plt.legend()
plt.show()
