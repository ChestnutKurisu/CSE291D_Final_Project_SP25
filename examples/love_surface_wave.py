"""Love wave demo using the new solver."""

from wave_sim.wave_catalog import LoveWave


def generate_animation(steps: int = 10):
    """Run a short demo returning displacement."""
    sim = LoveWave()
    for _ in range(steps):
        sim.step()
    return sim.get_displacement()


if __name__ == "__main__":
    generate_animation()
