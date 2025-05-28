import numpy as np
import matplotlib.pyplot as plt

# Alfv\u00e9n Wave Solver
# ---------------------
# Solves v_tt = v_A^2 v_xx along a uniform magnetic field.

Lx = 2.0
Nx = 200

B0 = 1.0
rho = 1.0
mu0 = 1.0

vA = B0 / np.sqrt(mu0 * rho)

dx = Lx / (Nx - 1)
Tmax = 2.0
dt = 0.8 * dx / vA
nt = int(Tmax / dt)

x = np.linspace(0, Lx, Nx)

v_old = np.sin(2 * np.pi * x / Lx)
v = v_old.copy()
v_new = np.zeros_like(v)

for _ in range(nt):
    for i in range(1, Nx - 1):
        v_new[i] = (
            2 * v[i]
            - v_old[i]
            + (vA * dt / dx) ** 2 * (v[i + 1] - 2 * v[i] + v[i - 1])
        )
    v_new[0] = 0.0
    v_new[-1] = 0.0
    v_old, v = v, v_new

plt.figure(figsize=(8, 4))
plt.plot(x, v, label="v_perp at final time")
plt.xlabel("x")
plt.ylabel("Transverse velocity")
plt.title("Alfv\u00e9n Wave - Final State")
plt.grid(True)
plt.legend()
plt.show()
