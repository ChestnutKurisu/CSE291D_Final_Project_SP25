import numpy as np

# Simple predefined colour maps used for visualisation

# Wave 2-4 are synthetic gradients; values are uint8

def _gradient(start, end, steps=20):
    t = np.linspace(0.0, 1.0, steps)[:, None]
    return np.round((1 - t) * np.array(start) + t * np.array(end)).astype(np.uint8)

colormap_wave2 = _gradient((0, 50, 150), (255, 255, 255))
colormap_wave3 = _gradient((150, 0, 50), (255, 255, 255))
colormap_wave4 = _gradient((50, 150, 0), (255, 255, 255))
colormap_icefire = _gradient((0, 0, 255), (255, 50, 0))
