import cupy as cp
import numpy as np
from .base import SceneObject


class StaticDampening(SceneObject):
    """Static dampening mask, optionally with faded edges."""

    def __init__(self, dampening_field: np.ndarray, border_thickness: int = 0):
        h, w = dampening_field.shape
        self.d_field = cp.array(dampening_field, dtype=cp.float32)
        if border_thickness > 0:
            for i in range(border_thickness):
                fade = (i / border_thickness) ** 0.5
                self.d_field[i, :] = cp.minimum(self.d_field[i, :], fade)
                self.d_field[h - 1 - i, :] = cp.minimum(self.d_field[h - 1 - i, :], fade)
                self.d_field[:, i] = cp.minimum(self.d_field[:, i], fade)
                self.d_field[:, w - 1 - i] = cp.minimum(self.d_field[:, w - 1 - i], fade)

    def render(self, field: cp.ndarray, wave_speed_field: cp.ndarray, dampening_field: cp.ndarray):
        dampening_field[:] = self.d_field

    def update_field(self, field: cp.ndarray, t: float):
        pass
