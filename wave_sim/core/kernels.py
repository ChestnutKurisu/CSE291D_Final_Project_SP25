import numpy as np

LAPLACIAN_KERNEL = np.array(
    [[0.066, 0.184, 0.066],
     [0.184, -1.0, 0.184],
     [0.066, 0.184, 0.066]],
    dtype=np.float32,
)

def get_laplacian_kernel(xp=np):
    """Return laplacian kernel as array for the given array module."""
    return xp.asarray(LAPLACIAN_KERNEL)

