import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
import copy

# ── physical / numerical parameters ────────────────────────────────────────────
L   = 1.0          # domain length
dx  = 0.01         # spatial step
c   = 1.0          # wave speed
dt  = 0.707 * dx / c      # CFL-limited time step
nsteps = 199               # number of time steps

# ── grid and initial field ─────────────────────────────────────────────────────
x  = np.arange(0, L + dx, dx)
y  = np.arange(0, L + dx, dx)
xx, yy = np.meshgrid(x, y)

npts = len(x)
f = np.zeros((npts, npts, 3))

xc, w = 0.5, 0.05   # Gaussian centre and width
f[:, :, 0] = np.exp(-((xx - xc)**2 + (yy - xc)**2) / w**2)

# first “kick” for leap-frog
f[1:-1, 1:-1, 1] = f[1:-1, 1:-1, 0] + 0.5 * c**2 * (
    (f[:-2, 1:-1, 0] + f[2:, 1:-1, 0] - 2*f[1:-1, 1:-1, 0]) +
    (f[1:-1, :-2, 0] + f[1:-1, 2:, 0] - 2*f[1:-1, 1:-1, 0])
) * (dt / dx)**2

# ── figure / writer setup ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(6, 5))
ax  = fig.add_subplot(projection='3d')
surf = ax.plot_surface(xx, yy, f[:, :, 0],
                       rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
wire = ax.plot_wireframe(xx, yy, f[:, :, 0],
                         rstride=10, cstride=10, color='green')
ax.set_zlim(-0.25, 1.0)

writer = animation.FFMpegWriter(fps=30, bitrate=2400)

# ── time stepping + frame capture ──────────────────────────────────────────────
with writer.saving(fig, "wave_2d.mp4", dpi=150):
    for k in range(nsteps):
        # leap-frog update to f[:,:,2]
        f[1:-1, 1:-1, 2] = -f[1:-1, 1:-1, 0] + 2*f[1:-1, 1:-1, 1] + c**2 * (
            (f[:-2, 1:-1, 1] + f[2:, 1:-1, 1] - 2*f[1:-1, 1:-1, 1]) +
            (f[1:-1, :-2, 1] + f[1:-1, 2:, 1] - 2*f[1:-1, 1:-1, 1])
        ) * (dt / dx)**2

        # roll time planes
        f[:, :, 0], f[:, :, 1] = f[:, :, 1], f[:, :, 2]

        # update surface & wireframe
        surf.remove(); wire.remove()
        surf = ax.plot_surface(xx, yy, f[:, :, 1],
                               rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        wire = ax.plot_wireframe(xx, yy, f[:, :, 1],
                                 rstride=10, cstride=10, color='green')
        ax.set_title(f"t = {k*dt:.2f}")
        writer.grab_frame()

plt.close(fig)
print("Saved → wave_2d.mp4")
