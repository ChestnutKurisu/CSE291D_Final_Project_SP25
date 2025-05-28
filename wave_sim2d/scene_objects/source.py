import cupy as cp
from .base import SceneObject


class PointSource(SceneObject):
    """Sinusoidal point source at a grid coordinate."""

    def __init__(self, px: int, py: int, amplitude: float, freq: float, phase: float = 0.0, opacity: float = 0.0):
        self.px = px
        self.py = py
        self.amplitude = amplitude
        self.freq = freq
        self.phase = phase
        self.opacity = opacity

    def render(self, field: cp.ndarray, wave_speed_field: cp.ndarray, dampening_field: cp.ndarray):
        pass

    def update_field(self, field: cp.ndarray, t: float):
        val = self.amplitude * cp.sin(self.freq * t + self.phase)
        if self.opacity <= 0.0:
            field[self.py, self.px] = val
        else:
            field[self.py, self.px] = field[self.py, self.px] * self.opacity + val * (1.0 - self.opacity)
