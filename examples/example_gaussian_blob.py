from wave_sim.high_quality import simulate_wave, ConstantSpeed, GaussianBlobSource


def scene(resolution):
    w, h = resolution
    objs = [
        ConstantSpeed(1.0),
        GaussianBlobSource(w // 2, h // 2, sigma_px=8, freq=0.1, amplitude=5.0),
    ]
    return objs, w, h, None


if __name__ == "__main__":
    simulate_wave(scene, "gaussian_blob.mp4", steps=300, sim_steps_per_frame=2, resolution=(256, 256))
