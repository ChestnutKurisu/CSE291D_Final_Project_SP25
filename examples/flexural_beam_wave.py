import numpy as np
import matplotlib.pyplot as plt

# Flexural Beam Wave Solver (Euler-Bernoulli)
# ------------------------------------------
# Solves w_tt + D w_xxxx = 0 for a thin beam using a
# simple explicit finite difference scheme.

Lx = 2.0
Nx = 201

D = 0.01

x = np.linspace(0, Lx, Nx)
dx = x[1] - x[0]

cfl = 0.2
dt = cfl * dx ** 2 / np.sqrt(D)
Tmax = 5.0
nt = int(Tmax / dt)

w_old = np.zeros(Nx)
w = np.exp(-100 * (x - Lx / 2) ** 2)
w_old[:] = w
w_new = np.zeros_like(w)

for _ in range(nt):
    for i in range(2, Nx - 2):
        w_xx = (w[i + 1] - 2 * w[i] + w[i - 1]) / dx ** 2
        w_xx_plus = (w[i + 2] - 2 * w[i + 1] + w[i]) / dx ** 2
        w_xx_minus = (w[i] - 2 * w[i - 1] + w[i - 2]) / dx ** 2
        w_xxxx = (w_xx_plus - 2 * w_xx + w_xx_minus) / dx ** 2
        w_new[i] = 2 * w[i] - w_old[i] + dt ** 2 * (-D * w_xxxx)
    w_new[0] = 0.0
    w_new[1] = 0.0
    w_new[-1] = 0.0
    w_new[-2] = 0.0
    w_old, w = w, w_new

plt.figure(figsize=(8, 4))
plt.plot(x, w, label="w at final time")
plt.xlabel("x")
plt.ylabel("Displacement")
plt.title("Flexural Beam Wave - Final")
plt.grid(True)
plt.legend()
plt.show()
