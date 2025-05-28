"""Rayleigh surface wave example using the elastic solver."""

from wave_sim.elastic2d import rayleigh_surface_demo


def generate_animation(steps: int = 10):
    """Run a short demo returning final displacement field."""
    sim = rayleigh_surface_demo()
    for _ in range(steps):
        sim.update_field()
    return sim.get_displacement()


if __name__ == "__main__":
    generate_animation()
