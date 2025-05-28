import numpy as np
import types
import sys

# minimal dummy cv2 to satisfy imports
if 'cv2' not in sys.modules:
    sys.modules['cv2'] = types.SimpleNamespace(circle=lambda *a, **k: None,
                                              fillPoly=lambda *a, **k: None,
                                              line=lambda *a, **k: None,
                                              getRotationMatrix2D=lambda *a, **k: np.eye(2,3),
                                              transform=lambda arr, mat: arr)

from wave_sim.wave_catalog import (
    RayleighWave,
    LoveWave,
    LambS0Mode,
    LambA0Mode,
    StoneleyWave,
    ScholteWave,
)


def _check_energy(solver, steps=3, rtol=1e-3):
    e0 = solver.energy()
    for _ in range(steps):
        solver.step()
    e1 = solver.energy()
    assert np.isclose(e0, e1, rtol=rtol)


def test_rayleigh_energy():
    _check_energy(RayleighWave())


def test_love_energy():
    _check_energy(LoveWave())


def test_lamb_s0_energy():
    _check_energy(LambS0Mode())


def test_lamb_a0_energy():
    _check_energy(LambA0Mode())


def test_stoneley_energy():
    _check_energy(StoneleyWave())


def test_scholte_energy():
    _check_energy(ScholteWave())
