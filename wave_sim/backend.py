import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None


def get_array_module(backend: str = "auto"):
    """Return array module depending on requested backend."""
    if backend == "gpu":
        if cp is None:
            raise ImportError("CuPy is not available for GPU backend")
        return cp
    if backend in {"cpu", "numpy"}:
        return np
    if backend == "auto":
        return cp if cp is not None else np
    raise ValueError(f"Unknown backend '{backend}'")

