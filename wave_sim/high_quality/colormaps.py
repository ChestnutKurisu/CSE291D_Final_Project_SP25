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

# Reference gradient used by some examples
colormap_wave1 = np.array([
        [255, 255, 255], [254, 254, 253], [254, 253, 252], [253, 252, 250],
        [253, 250, 248], [252, 249, 246], [252, 248, 244], [251, 246, 242],
        [251, 245, 240], [250, 243, 237], [250, 242, 235], [249, 240, 232],
        [248, 238, 230], [248, 237, 227], [247, 235, 224], [247, 233, 221],
        [246, 231, 218], [245, 229, 215], [245, 227, 212], [244, 225, 209],
    ], dtype=np.uint8)
