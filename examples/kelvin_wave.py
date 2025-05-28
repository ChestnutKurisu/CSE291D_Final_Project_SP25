import numpy as np
import matplotlib.pyplot as plt

# Kelvin Wave Solver (1-D along y with rotation)
# ---------------------------------------------
# Simple explicit finite difference scheme for the rotating shallow water
# equations demonstrating a boundary trapped Kelvin wave.

Ly = 10.0
Ny = 400

dy = Ly / (Ny - 1)

H = 1.0
f = 1.0
g = 9.81

c = np.sqrt(g * H)
Tmax = 10.0
dt = 0.4 * dy / c
nt = int(Tmax / dt)

y = np.linspace(0, Ly, Ny)

u = np.zeros(Ny)
v = np.zeros(Ny)
eta = np.exp(-((y - Ly / 4) / 0.5) ** 2)

u_new = np.zeros_like(u)
v_new = np.zeros_like(v)
eta_new = np.zeros_like(eta)

for _ in range(nt):
    for j in range(1, Ny - 1):
        du = -g * (eta[j + 1] - eta[j - 1]) / (2 * dy) + f * v[j]
        dv = -f * u[j]
        deta = -H * (u[j + 1] - u[j - 1]) / (2 * dy)
        u_new[j] = u[j] + dt * du
        v_new[j] = v[j] + dt * dv
        eta_new[j] = eta[j] + dt * deta

    u_new[0] = 0.0
    v_new[0] = 0.0
    eta_new[0] = eta[0]

    u_new[-1] = u[-1]
    v_new[-1] = v[-1]
    eta_new[-1] = eta[-1]

    u, v, eta = u_new.copy(), v_new.copy(), eta_new.copy()

plt.figure(figsize=(8, 4))
plt.plot(y, eta, label="eta at final time")
plt.xlabel("y")
plt.ylabel("Surface perturbation")
plt.title("Kelvin Wave - Final Surface Perturbation")
plt.grid(True)
plt.legend()
plt.show()
