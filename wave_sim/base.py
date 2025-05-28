import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

class WaveSimulation:
    """Simple 2D wave equation simulator.

    Parameters
    ----------
    grid_size : int, optional
        Number of cells in each spatial dimension.
    c : float, optional
        Wave propagation speed.
    dx : float, optional
        Spatial resolution.
    dt : float, optional
        Time step size.
    boundary : {"reflective", "periodic", "absorbing"}, optional
        Boundary condition applied at the domain edges.
    """

    def __init__(self, grid_size=100, c=1.0, dx=1.0, dt=0.1, boundary="reflective", backend="gpu"):
        self.n = grid_size
        self.c = c
        self.dx = dx
        self.dt = dt
        self.boundary = boundary

        if backend == "gpu" and cp is not None:
            self.xp = cp
        else:
            self.xp = np

        self.u_prev = self.xp.zeros((self.n, self.n))
        self.u_curr = self.xp.zeros((self.n, self.n))
        self.time = 0.0
        cfl = self.c * self.dt / self.dx
        if cfl > 1 / np.sqrt(2):
            warnings.warn(
                "CFL condition violated: c * dt / dx = {:.2f} > 1/sqrt(2)".format(cfl)
            )

    def step(self):
        xp = self.xp
        c2 = (self.c * self.dt / self.dx) ** 2
        laplacian = (
            xp.roll(self.u_curr, 1, axis=0)
            + xp.roll(self.u_curr, -1, axis=0)
            + xp.roll(self.u_curr, 1, axis=1)
            + xp.roll(self.u_curr, -1, axis=1)
            - 4 * self.u_curr
        )
        u_next = 2 * self.u_curr - self.u_prev + c2 * laplacian
        if self.boundary == "reflective":
            u_next[0, :] = 0
            u_next[-1, :] = 0
            u_next[:, 0] = 0
            u_next[:, -1] = 0
        elif self.boundary == "periodic":
            pass  # np.roll already provides periodic boundaries
        elif self.boundary == "absorbing":
            u_next[0, :] = u_next[1, :]
            u_next[-1, :] = u_next[-2, :]
            u_next[:, 0] = u_next[:, 1]
            u_next[:, -1] = u_next[:, -2]
        else:
            raise ValueError(f"Unknown boundary condition {self.boundary}")
        self.u_prev, self.u_curr = self.u_curr, u_next
        self.time += self.dt
        return u_next

    def initialize(self, source_pos=None, amplitude=1.0, source_func=None):
        """Set the initial displacement field.

        Parameters
        ----------
        source_pos : tuple of int, optional
            Coordinates of a point source, ignored if ``source_func`` is given.
        amplitude : float, optional
            Amplitude of the point source.
        source_func : callable, optional
            Function ``f(X, Y)`` returning an array of shape ``(n, n)`` with the
            initial displacement. ``X`` and ``Y`` are coordinate arrays in units
            of ``dx``.
        """
        xp = self.xp
        if source_func is not None:
            x = xp.arange(self.n) * self.dx
            y = xp.arange(self.n) * self.dx
            X, Y = xp.meshgrid(x, y, indexing="ij")
            self.u_curr = xp.asarray(source_func(cp.asnumpy(X) if xp is cp else X,
                                                 cp.asnumpy(Y) if xp is cp else Y))
            if xp is cp:
                self.u_curr = cp.asarray(self.u_curr)
        else:
            if source_pos is None:
                source_pos = (self.n // 2, self.n // 2)
            self.u_curr[source_pos] = amplitude

    def simulate(self, steps=100):
        frames = []
        for _ in range(steps):
            arr = self.step().copy()
            if self.xp is cp:
                arr = cp.asnumpy(arr)
            frames.append(arr)
        return frames

    def animate(self, steps=100, interval=30, cmap="viridis"):
        frames = self.simulate(steps)
        fig, ax = plt.subplots()
        im = ax.imshow(frames[0], cmap=cmap, vmin=-1, vmax=1)
        ax.set_axis_off()

        def update(i):
            im.set_data(frames[i])
            return (im,)

        ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)
        return ani
