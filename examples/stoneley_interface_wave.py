"""Stoneley wave demonstration."""

from wave_sim.wave_catalog import StoneleyWave


def generate_animation(steps: int = 10):
    sim = StoneleyWave()
    for _ in range(steps):
        sim.step()
    return sim.get_displacement()


if __name__ == "__main__":
    generate_animation()
