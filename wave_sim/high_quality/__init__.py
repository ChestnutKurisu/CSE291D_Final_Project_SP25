"""High quality GPU-accelerated wave simulation utilities."""

from .simulator import WaveSimulator2D, SceneObject
from .visualizer import WaveVisualizer, get_colormap_lut
from .runner import simulate_wave
from .scene_objects import (
    PointSource,
    ConstantSpeed,
    StaticDampening,
    StaticRefractiveIndex,
    StaticImageScene,
    StrainRefractiveIndex,
    StaticRefractiveIndexPolygon,
)

__all__ = [
    "WaveSimulator2D",
    "SceneObject",
    "WaveVisualizer",
    "get_colormap_lut",
    "simulate_wave",
    "PointSource",
    "ConstantSpeed",
    "StaticDampening",
    "StaticRefractiveIndex",
    "StaticImageScene",
    "StrainRefractiveIndex",
    "StaticRefractiveIndexPolygon",
]
