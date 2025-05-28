import numpy as np

from .scene_objects.static_dampening import StaticDampening
from .scene_objects.static_refractive_index import StaticRefractiveIndex
from .scene_objects.source import PointSource


def wave_catalog_list():
    """Return list of (name, scene_constructor) tuples."""

    waves = []

    def wave_pwave(width, height):
        refr_index = np.ones((height, width), dtype=np.float32) * 1.0
        damp = np.ones((height, width), dtype=np.float32)
        d_obj = StaticDampening(damp, border_thickness=40)
        c_obj = StaticRefractiveIndex(refr_index)
        px, py = width // 2, height // 2
        src = PointSource(px, py, amplitude=0.2, freq=0.2, phase=0.0, opacity=0.0)
        return [d_obj, c_obj, src]

    waves.append(("PrimaryWave2D", wave_pwave))

    def wave_shwave(width, height):
        refr_index = np.ones((height, width), dtype=np.float32) * 1.2
        damp = np.ones((height, width), dtype=np.float32)
        d_obj = StaticDampening(damp, border_thickness=40)
        c_obj = StaticRefractiveIndex(refr_index)
        px, py = width // 2, height // 2
        src = PointSource(px, py, amplitude=0.3, freq=0.15)
        return [d_obj, c_obj, src]

    waves.append(("SHWave2D", wave_shwave))

    # Placeholder constructors for remaining wave types
    for i in range(18):
        def ctor(w, h, i=i):
            refr_index = np.ones((h, w), dtype=np.float32)
            damp = np.ones((h, w), dtype=np.float32)
            d_obj = StaticDampening(damp, border_thickness=40)
            c_obj = StaticRefractiveIndex(refr_index)
            src = PointSource(w // 2, h // 2, amplitude=0.2, freq=0.1 + 0.01 * i)
            return [d_obj, c_obj, src]
        waves.append((f"WaveType{i+3}", ctor))

    return waves
