"""Rayleigh surface wave example using the new solver."""

from wave_sim.wave_catalog import RayleighWave


def generate_animation(steps: int = 10):
    """Run a short demo returning final displacement field."""
    sim = RayleighWave()
    for _ in range(steps):
        sim.step()
    return sim.get_displacement()


if __name__ == "__main__":
    generate_animation()
