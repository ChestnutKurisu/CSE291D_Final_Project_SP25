import numpy as np
import types
import sys
import pytest

# ensure wave_sim modules use fallback if cv2 is missing
if 'cv2' not in sys.modules:
    sys.modules['cv2'] = types.SimpleNamespace(circle=lambda *a, **k: None,
                                              fillPoly=lambda *a, **k: None,
                                              line=lambda *a, **k: None,
                                              getRotationMatrix2D=lambda *a, **k: np.eye(2,3),
                                              transform=lambda arr, mat: arr)

from wave_sim.high_quality.scene_objects import (
    PointSource,
    LineSource,
    ModulatorSmoothSquare,
    ModulatorDiscreteSignal,
)


def test_point_source_phase_and_modulator():
    f = np.zeros((3, 3))
    src = PointSource(1, 1, freq=1.0, amplitude=2.0, phase=np.pi / 2, amp_modulator=lambda t: 0.5)
    src.update_field(f, 0.0)
    assert f[1, 1] == pytest.approx(1.0)


def test_line_source_basic():
    f = np.zeros((2, 5))
    line = LineSource(0, 0, 4, 0, phase=np.pi/2)
    line.update_field(f, 0.0)
    assert np.count_nonzero(f[0]) > 0


def test_modulator_smooth_square_range():
    m = ModulatorSmoothSquare(frequency=1.0, phase=0.0, smoothness=0.5)
    vals = [m(t) for t in np.linspace(0, 2*np.pi, 10)]
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_modulator_discrete_signal_interp():
    m = ModulatorDiscreteSignal([0, 1, 2], [0, 1, 0])
    assert m(0.5) == pytest.approx(0.5)
    assert m(1.5) == pytest.approx(0.5)

