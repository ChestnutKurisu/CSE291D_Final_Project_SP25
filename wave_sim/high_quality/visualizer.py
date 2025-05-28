import numpy as np
try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

from ..backend import get_array_module

import cv2
import matplotlib.pyplot as plt  # Keep this for fallback colormaps

from .simulator import WaveSimulator2D
from .colormaps import (
    colormap_wave1,
    colormap_wave2,
    colormap_wave3,
    colormap_wave4,
    colormap_icefire,
)

# THE LOCAL DEFINITION OF colormap_wave1 THAT WAS HERE HAS BEEN REMOVED

__LUT_CACHE = {}

_PRESET_TABLE = {
    "wave1": colormap_wave1,
    "wave2": colormap_wave2,
    "wave3": colormap_wave3,
    "wave4": colormap_wave4,
    "icefire": colormap_icefire,
}


def get_colormap_lut(
    name: str = "wave1",
    invert: bool = False,
    black_level: float = 0.0,
    make_symmetric: bool = False,
    backend: str = "auto",
) -> np.ndarray:
    """Return a 256×3 uint8 colour-map lookup table on the requested backend."""

    global __LUT_CACHE
    xp = get_array_module(backend)
    key = (name, invert, black_level, make_symmetric, xp.__name__)
    if key in __LUT_CACHE:
        return __LUT_CACHE[key]

    if name in _PRESET_TABLE:
        # Preset tables are uint8; normalise in float for interpolation
        base = _PRESET_TABLE[name].astype(np.float32) / 255.0
        t = np.linspace(0, 1, base.shape[0])
        t256 = np.linspace(0, 1, 256)
        colors = np.vstack([np.interp(t256, t, base[:, ch]) for ch in range(3)]).T
    else:
        colors = plt.get_cmap(name)(np.linspace(0, 1, 256))[:, :3]

    if invert:
        colors = 1.0 - colors
    if make_symmetric:
        colors = np.vstack([colors[:128], colors[127::-1]])

    colors = np.clip(colors * (1.0 - black_level) + black_level, 0.0, 1.0)
    lut_np = (colors * 255).astype(np.uint8)
    lut = xp.asarray(lut_np)
    __LUT_CACHE[key] = lut
    return lut


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
        if xp is cp:
            img = xp.take(self.intensity_colormap, gray, axis=0)
            img = img.get()
        else:
            img = np.take(self.intensity_colormap, gray, axis=0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if overlay_visualization:
            img = cv2.add(img, self.visualization_image)
        return img

    def render_field(self, brightness_scale=1.0, overlay_visualization=True):
        xp = cp if cp is not None and isinstance(self.field, cp.ndarray) else np
        gray = (xp.clip(self.field * brightness_scale, -1.0, 1.0) * 127 + 127).astype(np.uint8)
        if xp is cp:
            img = xp.take(self.field_colormap, gray, axis=0)
            img = img.get()
        else:
            img = np.take(self.field_colormap, gray, axis=0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if overlay_visualization:
            img = cv2.add(img, self.visualization_image)
        return img
