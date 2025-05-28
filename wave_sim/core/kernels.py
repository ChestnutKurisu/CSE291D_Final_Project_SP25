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


def finite_difference_laplacian(u, boundary="fill", xp=np, kernel=None):
    """Apply a 3x3 Laplacian stencil via ``xp.pad``/``xp.roll``.

    Parameters
    ----------
    u : array-like
        2-D field to operate on.
    boundary : {"fill", "wrap", "symm"}, optional
        How to treat values outside the domain. ``fill`` pads with zeros,
        ``wrap`` implements periodic boundaries and ``symm`` mirrors the
        array.
    xp : module, optional
        Array module (:mod:`numpy` or :mod:`cupy`).

    Returns
    -------
    array-like
        Laplacian of ``u`` with the same shape as ``u``.
    """

    if kernel is None:
        k = get_laplacian_kernel(xp)
    else:
        k = kernel

    if boundary == "wrap":
        # periodic boundaries allow an efficient roll-based stencil
        return (
            k[1, 1] * u
            + k[0, 1] * xp.roll(u, -1, axis=0)
            + k[2, 1] * xp.roll(u, 1, axis=0)
            + k[1, 0] * xp.roll(u, -1, axis=1)
            + k[1, 2] * xp.roll(u, 1, axis=1)
            + k[0, 0] * xp.roll(xp.roll(u, -1, axis=0), -1, axis=1)
            + k[0, 2] * xp.roll(xp.roll(u, -1, axis=0), 1, axis=1)
            + k[2, 0] * xp.roll(xp.roll(u, 1, axis=0), -1, axis=1)
            + k[2, 2] * xp.roll(xp.roll(u, 1, axis=0), 1, axis=1)
        )

    if boundary == "symm":
        pad_mode = "symmetric"
    else:
        pad_mode = "constant"

    up = xp.pad(u, 1, mode=pad_mode)
    return (
        k[0, 0] * up[:-2, :-2]
        + k[0, 1] * up[:-2, 1:-1]
        + k[0, 2] * up[:-2, 2:]
        + k[1, 0] * up[1:-1, :-2]
        + k[1, 1] * up[1:-1, 1:-1]
        + k[1, 2] * up[1:-1, 2:]
        + k[2, 0] * up[2:, :-2]
        + k[2, 1] * up[2:, 1:-1]
        + k[2, 2] * up[2:, 2:]
    )


