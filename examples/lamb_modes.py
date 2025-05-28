"""Demonstrate Lamb S0 and A0 mode solvers."""

from wave_sim.wave_catalog import LambS0Mode, LambA0Mode


def run_s0(steps: int = 10):
    sim = LambS0Mode()
    for _ in range(steps):
        sim.step()
    return sim.get_displacement()


def run_a0(steps: int = 10):
    sim = LambA0Mode()
    for _ in range(steps):
        sim.step()
    return sim.get_displacement()


if __name__ == "__main__":
    run_s0()
    run_a0()
