import numpy as np
try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

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
) -> np.ndarray:
    """Return a 256×3 uint8 colour-map lookup table."""

    global __LUT_CACHE
    key = (name, invert, black_level, make_symmetric)
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
    lut = (colors * 255).astype(np.uint8)
    __LUT_CACHE[key] = lut
    return lut


class WaveVisualizer:
    def __init__(self, field_colormap=None, intensity_colormap=None):
        self.field_colormap = field_colormap
        self.intensity_colormap = intensity_colormap
        # GPU copies of colour maps for CuPy processing
        if cp is not None:
            self.field_colormap_gpu = cp.asarray(field_colormap) if field_colormap is not None else None
            self.intensity_colormap_gpu = cp.asarray(intensity_colormap) if intensity_colormap is not None else None
        else:
            self.field_colormap_gpu = None
            self.intensity_colormap_gpu = None

        self.field = None
        self.intensity = None
        self.intensity_exp_average_factor = 0.98
        self.visualization_image = None
        self.visualization_image_gpu = None

    def update(self, sim: WaveSimulator2D):
        xp = sim.xp
        self.field = sim.get_field()
        if self.intensity is None:
            self.intensity = xp.zeros_like(self.field)
        t = self.intensity_exp_average_factor
        self.intensity = self.intensity * t + (self.field ** 2) * (1.0 - t)
        self.visualization_image = sim.render_visualization()
        if cp is not None and isinstance(self.field, cp.ndarray):
            self.visualization_image_gpu = cp.asarray(self.visualization_image)
        else:
            self.visualization_image_gpu = None

    def render_intensity(self, brightness_scale=1.0, exp=0.5, overlay_visualization=True, output_size=None):
        xp = cp if cp is not None and isinstance(self.intensity, cp.ndarray) else np
        is_gpu = xp is cp
        gray = (xp.clip((self.intensity ** exp) * brightness_scale, 0.0, 1.0) * 254.0).astype(xp.uint8)
        if is_gpu:
            lut = self.intensity_colormap_gpu if self.intensity_colormap_gpu is not None else cp.asarray(self.intensity_colormap)
            img = lut[gray]
            img = img[..., ::-1]
            if output_size is not None:
                try:
                    import cupyx.scipy.ndimage as cnd
                    zoom_y = output_size[1] / img.shape[0]
                    zoom_x = output_size[0] / img.shape[1]
                    img = cnd.zoom(img, (zoom_y, zoom_x, 1), order=1).astype(cp.uint8)
                except Exception:
                    if cv2 is not None:
                        img = cp.asarray(cv2.resize(cp.asnumpy(img), output_size))
                    else:
                        from scipy.ndimage import zoom
                        zoom_y = output_size[1] / img.shape[0]
                        zoom_x = output_size[0] / img.shape[1]
                        img = cp.asarray(zoom(cp.asnumpy(img), (zoom_y, zoom_x, 1), order=1).astype(np.uint8))
            if overlay_visualization:
                if self.visualization_image_gpu is not None:
                    img = cp.clip(img.astype(cp.int16) + self.visualization_image_gpu.astype(cp.int16), 0, 255).astype(cp.uint8)
                else:
                    img_cpu = img.get()
                    if cv2 is not None:
                        img_cpu = cv2.add(img_cpu, self.visualization_image)
                    else:
                        img_cpu = np.clip(img_cpu.astype(np.int16) + self.visualization_image.astype(np.int16), 0, 255).astype(np.uint8)
                    return img_cpu
            return img.get()
        else:
            gray_cpu = gray
            img = self.intensity_colormap[gray_cpu]
            img = img[..., ::-1]
            if output_size is not None:
                if cv2 is not None:
                    img = cv2.resize(img, output_size)
                else:
                    from scipy.ndimage import zoom
                    zoom_y = output_size[1] / img.shape[0]
                    zoom_x = output_size[0] / img.shape[1]
                    img = zoom(img, (zoom_y, zoom_x, 1), order=1).astype(np.uint8)
            if overlay_visualization:
                if cv2 is not None:
                    img = cv2.add(img, self.visualization_image)
                else:
                    img = np.clip(img.astype(np.int16) + self.visualization_image.astype(np.int16), 0, 255).astype(np.uint8)
            return img

    def render_field(self, brightness_scale=1.0, overlay_visualization=True, output_size=None):
        xp = cp if cp is not None and isinstance(self.field, cp.ndarray) else np
        is_gpu = xp is cp
        gray = (xp.clip(self.field * brightness_scale, -1.0, 1.0) * 127 + 127).astype(xp.uint8)
        if is_gpu:
            lut = self.field_colormap_gpu if self.field_colormap_gpu is not None else cp.asarray(self.field_colormap)
            img = lut[gray]
            img = img[..., ::-1]
            if output_size is not None:
                try:
                    import cupyx.scipy.ndimage as cnd
                    zoom_y = output_size[1] / img.shape[0]
                    zoom_x = output_size[0] / img.shape[1]
                    img = cnd.zoom(img, (zoom_y, zoom_x, 1), order=1).astype(cp.uint8)
                except Exception:
                    if cv2 is not None:
                        img = cp.asarray(cv2.resize(cp.asnumpy(img), output_size))
                    else:
                        from scipy.ndimage import zoom
                        zoom_y = output_size[1] / img.shape[0]
                        zoom_x = output_size[0] / img.shape[1]
                        img = cp.asarray(zoom(cp.asnumpy(img), (zoom_y, zoom_x, 1), order=1).astype(np.uint8))
            if overlay_visualization:
                if self.visualization_image_gpu is not None:
                    img = cp.clip(img.astype(cp.int16) + self.visualization_image_gpu.astype(cp.int16), 0, 255).astype(cp.uint8)
                else:
                    img_cpu = img.get()
                    if cv2 is not None:
                        img_cpu = cv2.add(img_cpu, self.visualization_image)
                    else:
                        img_cpu = np.clip(img_cpu.astype(np.int16) + self.visualization_image.astype(np.int16), 0, 255).astype(np.uint8)
                    return img_cpu
            return img.get()
        else:
            gray_cpu = gray
            img = self.field_colormap[gray_cpu]
            img = img[..., ::-1]
            if output_size is not None:
                if cv2 is not None:
                    img = cv2.resize(img, output_size)
                else:
                    from scipy.ndimage import zoom
                    zoom_y = output_size[1] / img.shape[0]
                    zoom_x = output_size[0] / img.shape[1]
                    img = zoom(img, (zoom_y, zoom_x, 1), order=1).astype(np.uint8)
            if overlay_visualization:
                if cv2 is not None:
                    img = cv2.add(img, self.visualization_image)
                else:
                    img = np.clip(img.astype(np.int16) + self.visualization_image.astype(np.int16), 0, 255).astype(np.uint8)
            return img
