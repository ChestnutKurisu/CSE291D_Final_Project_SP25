import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class WaveSimulation:
    """Simple 2D wave equation simulator."""

    def __init__(self, grid_size=100, c=1.0, dx=1.0, dt=0.1):
        self.n = grid_size
        self.c = c
        self.dx = dx
        self.dt = dt
        self.u_prev = np.zeros((self.n, self.n))
        self.u_curr = np.zeros((self.n, self.n))

    def step(self):
        c2 = (self.c * self.dt / self.dx) ** 2
        laplacian = (
            np.roll(self.u_curr, 1, axis=0)
            + np.roll(self.u_curr, -1, axis=0)
            + np.roll(self.u_curr, 1, axis=1)
            + np.roll(self.u_curr, -1, axis=1)
            - 4 * self.u_curr
        )
        u_next = 2 * self.u_curr - self.u_prev + c2 * laplacian
        # simple reflective boundary
        u_next[0, :] = 0
        u_next[-1, :] = 0
        u_next[:, 0] = 0
        u_next[:, -1] = 0
        self.u_prev, self.u_curr = self.u_curr, u_next
        return u_next

    def initialize(self, source_pos=None, amplitude=1.0):
        if source_pos is None:
            source_pos = (self.n // 2, self.n // 2)
        self.u_curr[source_pos] = amplitude

    def simulate(self, steps=100):
        frames = []
        for _ in range(steps):
            frames.append(self.step().copy())
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
