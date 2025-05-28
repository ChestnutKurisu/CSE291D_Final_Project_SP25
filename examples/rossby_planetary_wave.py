import numpy as np
import matplotlib.pyplot as plt

# Rossby Planetary Wave Solver using a simple spectral method.
# -----------------------------------------------------------
# Integrates zeta_t + beta * psi_x = 0 with zeta = nabla^2 psi on a 2-D
# periodic domain using FFTs.

Nx = 128
Ny = 128
Lx = 2.0 * np.pi
Ly = 2.0 * np.pi

dx = Lx / Nx
dy = Ly / Ny

beta = 1.0
dt = 0.01
nt = 300

kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
kx2D, ky2D = np.meshgrid(kx, ky, indexing="ij")
k2 = kx2D ** 2 + ky2D ** 2
k2[0, 0] = 1e-14

x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

psi0 = np.exp(-((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2) / 0.2)
psi_hat = np.fft.fftn(psi0)
zeta_hat = -k2 * psi_hat

for _ in range(nt):
    psi_x_hat = 1j * kx2D * psi_hat
    zeta_hat = zeta_hat + dt * (-beta * psi_x_hat)
    psi_hat = -zeta_hat / k2

psi_final = np.fft.ifftn(psi_hat).real

plt.figure(figsize=(6, 5))
plt.contourf(X, Y, psi_final, levels=20, cmap="RdBu_r")
plt.colorbar(label="Streamfunction")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Rossby Planetary Wave - Final \u03c8")
plt.show()
