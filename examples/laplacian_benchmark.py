"""Benchmark Laplacian implementations.

This script compares the previous ``convolve2d`` based Laplacian with the
finite-difference version implemented using :func:`cp.pad`/`np.pad` and
``cp.roll``.
"""

import time
import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.signal as cusig
except Exception:  # pragma: no cover - optional dependency
    cp = None  # type: ignore
    cusig = None  # type: ignore

import scipy.signal as sig

from wave_sim.core.kernels import get_laplacian_kernel, finite_difference_laplacian


def _time_fn(fn, arr, steps=50):
    start = time.perf_counter()
    for _ in range(steps):
        fn(arr)
    return time.perf_counter() - start


def benchmark(n=512, steps=50, backend="gpu"):
    xp = cp if backend == "gpu" and cp is not None else np
    arr = xp.random.random((n, n)).astype(xp.float32)
    kernel = get_laplacian_kernel(xp)

    if xp is cp:
        def conv(a):
            return cusig.convolve2d(a, kernel, mode="same", boundary="fill")
    else:
        def conv(a):
            return sig.convolve2d(a, kernel, mode="same", boundary="fill")

    def fd(a):
        return finite_difference_laplacian(a, boundary="fill", xp=xp, kernel=kernel)

    if xp is cp:
        xp.cuda.Stream.null.synchronize()
    t_conv = _time_fn(conv, arr, steps)
    if xp is cp:
        xp.cuda.Stream.null.synchronize()
    t_fd = _time_fn(fd, arr, steps)
    if xp is cp:
        xp.cuda.Stream.null.synchronize()

    print(f"backend={backend} size={n} steps={steps}")
    print(f"  convolve2d: {t_conv:.4f}s")
    print(f"  finite diff: {t_fd:.4f}s")


if __name__ == "__main__":
    benchmark()
