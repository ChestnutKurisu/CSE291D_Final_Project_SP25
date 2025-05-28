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
    GaussianBlobSource,
)
from wave_sim.high_quality import WaveSimulator2D, ConstantSpeed
from wave_sim.core.boundary import BoundaryCondition
from wave_sim.initial_conditions import gaussian_2d


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


def test_gaussian_blob_source_basic():
    f = np.zeros((16, 16))
    src = GaussianBlobSource(8, 8, sigma_px=2, freq=0.0, amplitude=1.0)
    src.update_field(f, 0.0)
    assert f[8, 8] > 0.0
    assert np.sum(f) > 0.0


def test_absorbing_sponge_reduces_energy():
    w, h = 32, 32
    X, Y = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
    init = gaussian_2d(X, Y, center=(2, h // 2), sigma=2.0)

    sim_ref = WaveSimulator2D(w, h, [ConstantSpeed(1.0)], initial_field=init,
                              backend="cpu", boundary=BoundaryCondition.REFLECTIVE)
    sim_abs = WaveSimulator2D(w, h, [ConstantSpeed(1.0)], initial_field=init,
                              backend="cpu", boundary=BoundaryCondition.ABSORBING,
                              sponge_thickness=8)
    for _ in range(10):
        sim_ref.update_field()
        sim_abs.update_field()
    energy_ref = float(np.sum(np.abs(sim_ref.get_field())))
    energy_abs = float(np.sum(np.abs(sim_abs.get_field())))
    assert energy_abs < energy_ref

