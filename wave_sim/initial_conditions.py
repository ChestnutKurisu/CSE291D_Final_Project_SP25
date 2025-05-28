import numpy as np


def gaussian_1d(x: np.ndarray, center: float, sigma: float = 1.0) -> np.ndarray:
    """Return 1-D Gaussian profile."""
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def gaussian_2d(X: np.ndarray, Y: np.ndarray, center=None, sigma: float = 5.0) -> np.ndarray:
    """Return 2-D Gaussian pulse."""
    if center is None:
        x0 = X.shape[0] // 2
        y0 = Y.shape[1] // 2
    else:
        x0, y0 = center
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))

