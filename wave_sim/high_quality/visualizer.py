import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

import cv2

from .simulator import WaveSimulator2D

# Predefined colormaps from the reference
colormap_wave1 = np.array([
    [255, 255, 255], [254, 254, 253], [254, 253, 252], [253, 252, 250],
    [253, 250, 248], [252, 249, 246], [252, 248, 244], [251, 246, 242],
    [251, 245, 240], [250, 243, 237], [250, 242, 235], [249, 240, 232],
    [248, 238, 230], [248, 237, 227], [247, 235, 224], [247, 233, 221],
    [246, 231, 218], [245, 229, 215], [245, 227, 212], [244, 225, 209],
])


def get_colormap_lut(name="wave1", invert=False):
    if name == "wave1":
        colors = colormap_wave1 / 255.0
    else:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(name)
        colors = cmap(np.linspace(0, 1, 256))[:, :3]
    if invert:
        colors = 1.0 - colors
    return (colors * 255).astype(np.uint8)


class WaveVisualizer:
    def __init__(self, field_colormap=None, intensity_colormap=None):
        self.field_colormap = field_colormap
        self.intensity_colormap = intensity_colormap
        self.field = None
        self.intensity = None
        self.intensity_exp_average_factor = 0.98
        self.visualization_image = None

    def update(self, sim: WaveSimulator2D):
        xp = sim.xp
        self.field = sim.get_field()
        if self.intensity is None:
            self.intensity = xp.zeros_like(self.field)
        t = self.intensity_exp_average_factor
        self.intensity = self.intensity * t + (self.field ** 2) * (1.0 - t)
        self.visualization_image = sim.render_visualization()

    def render_intensity(self, brightness_scale=1.0, exp=0.5, overlay_visualization=True):
        xp = cp if cp is not None and isinstance(self.intensity, cp.ndarray) else np
        gray = (xp.clip((self.intensity ** exp) * brightness_scale, 0.0, 1.0) * 254.0).astype(np.uint8)
        img = self.intensity_colormap[gray.get()] if xp is cp else self.intensity_colormap[gray]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if overlay_visualization:
            img = cv2.add(img, self.visualization_image)
        return img

    def render_field(self, brightness_scale=1.0, overlay_visualization=True):
        xp = cp if cp is not None and isinstance(self.field, cp.ndarray) else np
        gray = (xp.clip(self.field * brightness_scale, -1.0, 1.0) * 127 + 127).astype(np.uint8)
        img = self.field_colormap[gray.get()] if xp is cp else self.field_colormap[gray]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if overlay_visualization:
            img = cv2.add(img, self.visualization_image)
        return img
