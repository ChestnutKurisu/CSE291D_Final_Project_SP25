import cupy as cp
import numpy as np
import cv2
import matplotlib.pyplot as plt


class WaveVisualizer:
    """Render wave field and intensity using color maps."""

    def __init__(self, field_colormap: np.ndarray = None, intensity_colormap: np.ndarray = None, intensity_decay: float = 0.98):
        self.field_colormap = field_colormap
        self.intensity_colormap = intensity_colormap
        self.intensity_decay = intensity_decay
        self.intensity = None
        self.field = None

    def update(self, wave_field_gpu: cp.ndarray):
        self.field = wave_field_gpu
        if self.intensity is None:
            self.intensity = cp.zeros_like(wave_field_gpu)
        self.intensity = self.intensity_decay * self.intensity + (1.0 - self.intensity_decay) * (self.field ** 2)

    def render_field(self, brightness_scale: float = 1.0, overlay: np.ndarray = None) -> np.ndarray:
        field_clamped = cp.clip(self.field * brightness_scale, -1.0, 1.0)
        idx = ((field_clamped + 1.0) * 127.5).astype(cp.uint8)
        if self.field_colormap is not None:
            color_lut = cp.asarray(self.field_colormap)
            colored = color_lut[idx]
        else:
            colored = cp.stack([idx, idx, idx], axis=-1)
        out = colored.get().reshape((self.field.shape[0], self.field.shape[1], 3))
        if overlay is not None:
            out = cv2.add(overlay, out.astype(np.uint8))
        return out.astype(np.uint8)

    def render_intensity(self, brightness_scale: float = 1.0, exponent: float = 0.5, overlay: np.ndarray = None) -> np.ndarray:
        i_val = self.intensity ** exponent
        i_val = i_val * brightness_scale
        i_val = cp.clip(i_val, 0.0, 1.0)
        idx = (i_val * 255.0).astype(cp.uint8)
        if self.intensity_colormap is not None:
            color_lut = cp.asarray(self.intensity_colormap)
            colored = color_lut[idx]
        else:
            colored = cp.stack([idx, idx, idx], axis=-1)
        out = colored.get().reshape((self.field.shape[0], self.field.shape[1], 3))
        if overlay is not None:
            out = cv2.add(overlay, out.astype(np.uint8))
        return out.astype(np.uint8)


def get_colormap_lut(name: str = "afmhot", size: int = 256, invert: bool = False, black_level: float = 0.0) -> np.ndarray:
    cmap = plt.get_cmap(name)
    arr = cmap(np.linspace(0, 1, size))[:, :3]
    if invert:
        arr = 1.0 - arr
    if black_level != 0.0:
        arr = arr * (1.0 - black_level) + black_level
    arr = (arr * 255).astype(np.uint8)
    return arr
