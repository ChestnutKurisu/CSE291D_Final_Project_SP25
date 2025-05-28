from wave_sim.high_quality import simulate_wave, ConstantSpeed, LineSource, ModulatorSmoothSquare


def scene(resolution):
    w, h = resolution
    mod = ModulatorSmoothSquare(frequency=0.05, smoothness=0.3)
    objs = [
        ConstantSpeed(1.0),
        LineSource(w // 4, h // 2, 3 * w // 4, h // 2, freq=0.1, amplitude=5.0, amp_modulator=mod),
    ]
    return objs, w, h, None


if __name__ == "__main__":
    simulate_wave(scene, "line_source.mp4", steps=300, sim_steps_per_frame=2, resolution=(256, 256))
