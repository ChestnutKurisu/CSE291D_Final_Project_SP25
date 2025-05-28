import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

import cv2

from .simulator import SceneObject


class PointSource(SceneObject):
    def __init__(self, x, y, freq=0.1, amplitude=1.0):
        self.x = int(x)
        self.y = int(y)
        self.freq = freq
        self.amplitude = amplitude
        self.phase = 0.0

    def render(self, field, wave_speed_field, dampening_field):
        pass

    def update_field(self, field, t):
        if cp is not None and isinstance(field, cp.ndarray):
            field[self.y, self.x] += cp.sin(t * self.freq * 2 * cp.pi) * self.amplitude
        else:
            field[self.y, self.x] += np.sin(t * self.freq * 2 * np.pi) * self.amplitude

    def render_visualization(self, image):
        if 0 <= self.y < image.shape[0] and 0 <= self.x < image.shape[1]:
            cv2.circle(image, (self.x, self.y), 3, (50, 50, 50), -1)

class ConstantSpeed(SceneObject):
    def __init__(self, speed):
        self.speed = float(speed)

    def render(self, field, wave_speed_field, dampening_field):
        wave_speed_field[:] = self.speed

    def update_field(self, field, t):
        pass

    def render_visualization(self, image):
        pass
