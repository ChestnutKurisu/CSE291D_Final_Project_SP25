import numpy as np
from .scene_objects.static_dampening import StaticDampening
from .scene_objects.static_refractive_index import StaticRefractiveIndex
from .scene_objects.source import PointSource


def wave_catalog_list():
    """Return list of (name, scene_constructor) tuples for wave types."""
    waves = []

    def wave_primary(width, height):
        refr_index = np.ones((height, width), dtype=np.float32)
        damp = np.ones((height, width), dtype=np.float32)
        d_obj = StaticDampening(damp, border_thickness=40)
        c_obj = StaticRefractiveIndex(refr_index)
        src = PointSource(width // 2, height // 2, amplitude=0.2, freq=0.2)
        return [d_obj, c_obj, src]

    waves.append(("PrimaryWave2D", wave_primary))

    def wave_sh(width, height):
        refr_index = np.ones((height, width), dtype=np.float32) * 1.2
        damp = np.ones((height, width), dtype=np.float32)
        d_obj = StaticDampening(damp, border_thickness=40)
        c_obj = StaticRefractiveIndex(refr_index)
        src = PointSource(width // 2, height // 2, amplitude=0.3, freq=0.15)
        return [d_obj, c_obj, src]

    waves.append(("SHWave2D", wave_sh))

    return waves
