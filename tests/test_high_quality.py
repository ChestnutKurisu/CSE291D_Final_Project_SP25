import pytest

from wave_sim.high_quality import (
    WaveSimulator2D,
    ConstantSpeed,
    PointSource,
    get_colormap_lut,
)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_simulator_backend(backend):
    try:
        sim = WaveSimulator2D(
            32, 32, [ConstantSpeed(1.0), PointSource(16, 16)], backend=backend
        )
    except Exception:
        pytest.skip(f"backend {backend} not available")
    for _ in range(3):
        sim.update_scene()
        sim.update_field()
    assert sim.get_field() is not None


def test_colormap_names():
    for name in ["wave1", "wave2", "wave3", "wave4", "icefire"]:
        lut = get_colormap_lut(name)
        assert lut.shape == (256, 3)
