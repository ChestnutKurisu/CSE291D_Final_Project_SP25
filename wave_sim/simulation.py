import os
import logging
from datetime import datetime

import numpy as np
from matplotlib import animation
from tqdm import tqdm
from .visualizer import WaveVisualizer
from .vector_elastic import simulate_elastic_wave
from .elastic_waves import simulate_elastic_potentials, ricker_wavelet


def run_simulation(
        out_path="wave_2d.mp4",
        steps=40,
        ring_radius=0.15,
        log_interval=1,
        wave_type="acoustic",
        c_acoustic=1.0,
        vp=2.0,
        vs=1.0,
        rho=1.0,
        lame_lambda=1.0,
        lame_mu=1.0,
        f0=25.0,
        absorb_width=10,
        absorb_strength=2.0,
):
    L = 2.0
    dx = 0.01

    if wave_type == "acoustic":
        sim_wave_speed = c_acoustic
        field_description = "Acoustic Amplitude"
    elif wave_type == "P":
        sim_wave_speed = vp
        field_description = r"P-wave Potential ($\Phi$)"
    elif wave_type == "S_SH":
        sim_wave_speed = vs
        field_description = r"SH-wave Disp. ($u_z$)"
    elif wave_type == "S_SV_potential":
        sim_wave_speed = vs
        field_description = r"SV-wave Potential ($\Psi_z$)"
    elif wave_type == "elastic":
        sim_wave_speed = max(vp, vs)
        field_description = "Displacement Magnitude"
    elif wave_type == "elastic_potentials":
        sim_wave_speed = max(vp, vs)
        field_description = "Displacement Magnitude"
    else:
        raise ValueError(f"Unknown wave_type: {wave_type}")

    if sim_wave_speed <= 0:
        raise ValueError(f"Wave speed must be positive. Got {sim_wave_speed} for {wave_type}.")

    cfl_factor = 0.7
    if wave_type in ("elastic", "elastic_potentials"):
        dt = cfl_factor * dx / (np.sqrt(2.0) * sim_wave_speed)
        if np.sqrt(2.0) * sim_wave_speed * dt / dx >= 1.0:
            raise ValueError("Unstable dt for elastic case")
    else:
        dt = cfl_factor * dx / sim_wave_speed
    nsteps = steps

    x = np.arange(0, L + dx, dx)
    y = np.arange(0, L + dx, dx)
    xx, yy = np.meshgrid(x, y)

    npts = len(x)

    damping = np.ones((npts, npts))
    for i in range(absorb_width):
        coef = np.exp(-absorb_strength * ((absorb_width - i) / absorb_width) ** 2)
        damping[i, :] *= coef
        damping[-i - 1, :] *= coef
        damping[:, i] *= coef
        damping[:, -i - 1] *= coef

    if wave_type == "elastic":
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(
            logs_dir,
            f"simulation_{wave_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(message)s")
        field_snaps, velocity_hist, amp_hist, dt = simulate_elastic_wave(
            steps=steps,
            dx=dx,
            L=L,
            vp=vp,
            vs=vs,
            rho=rho,
            lame_lambda=lame_lambda,
            lame_mu=lame_mu,
            dt=dt,
            f0=f0,
            ring_radius=ring_radius,
            absorb_width=absorb_width,
            absorb_strength=absorb_strength,
        )
        viz = WaveVisualizer(
            sim_shape=field_snaps[0].shape,
            output_video_size=(1920, 1080),
            dt=dt,
            dx=dx,
            main_plot_cmap_name="Spectral",
            dynamic_z=False,
            zlim=(-0.25, 1.0),
            font_size=16,
            field_name_label=field_description,
        )
        writer = animation.FFMpegWriter(fps=30, bitrate=8000)
        with writer.saving(viz.fig, out_path, dpi=100):
            for k, field in enumerate(field_snaps):
                tnow = k * dt
                vel = velocity_hist[k][1]
                amp = amp_hist[k]
                if k % log_interval == 0:
                    logging.info(
                        "step=%d t=%.3f ring_avg=%.6f ring_rate=%.6f energy=%.6f",
                        k,
                        tnow,
                        amp,
                        vel,
                        float(np.sum(field ** 2 * dx * dx)),
                    )
                viz.update(field, velocity_hist[: k + 1], amp_hist[: k + 1], tnow, monitor_ring=ring_radius)
                viz.render_composite_frame()
                writer.grab_frame()
        logging.info("Simulation completed. Output saved to %s", out_path)
        return out_path

    if wave_type == "elastic_potentials":
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(
            logs_dir,
            f"simulation_{wave_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(message)s")
        (ux, uz), snaps = simulate_elastic_potentials(
            nx=npts,
            nz=npts,
            dx=dx,
            dz=dx,
            vp=vp,
            vs=vs,
            dt=dt,
            nt=steps,
            f0=f0,
            source="both",
            absorb_width=absorb_width,
            absorb_strength=absorb_strength,
        )
        field_snaps = [np.sqrt(u_x**2 + u_z**2) for (u_x, u_z) in snaps]
        final_field = np.sqrt(ux**2 + uz**2)
        field_snaps.append(final_field)
        velocity_hist = []
        amp_hist = []
        prev_amp = 0.0
        ring_thick = 3 * dx
        dist = np.sqrt((xx - L/2)**2 + (yy - L/2)**2)
        ring_mask = np.abs(dist - ring_radius) <= ring_thick/2
        viz = WaveVisualizer(
            sim_shape=field_snaps[0].shape,
            output_video_size=(1920, 1080),
            dt=dt,
            dx=dx,
            main_plot_cmap_name="Spectral",
            dynamic_z=False,
            zlim=(-0.25, 1.0),
            font_size=16,
            field_name_label=field_description,
        )
        writer = animation.FFMpegWriter(fps=30, bitrate=8000)
        with writer.saving(viz.fig, out_path, dpi=100):
            for k, field in enumerate(field_snaps):
                tnow = k * dt
                if np.any(ring_mask):
                    amp = field[ring_mask].mean()
                    vel = (amp - prev_amp)/dt
                else:
                    amp = 0.0
                    vel = 0.0
                prev_amp = amp
                velocity_hist.append((tnow, vel))
                amp_hist.append(amp)
                if k % log_interval == 0:
                    logging.info(
                        "step=%d t=%.3f ring_avg=%.6f ring_rate=%.6f energy=%.6f",
                        k,
                        tnow,
                        amp,
                        vel,
                        float(np.sum(field**2 * dx * dx)),
                    )
                viz.update(field, velocity_hist, amp_hist, tnow, monitor_ring=ring_radius)
                viz.render_composite_frame()
                writer.grab_frame()
        logging.info("Simulation completed. Output saved to %s", out_path)
        return out_path

    f = np.zeros((npts, npts, 3))
    xc, w = L / 2, 0.05
    f[:, :, 0] = np.exp(-((xx - xc) ** 2 + (yy - xc) ** 2) / w ** 2)
    src_i = npts // 2
    src_j = npts // 2

    S_sq = (sim_wave_speed * dt / dx) ** 2

    laplacian_f0 = (
        (f[:-2, 1:-1, 0] + f[2:, 1:-1, 0] - 2 * f[1:-1, 1:-1, 0]) +
        (f[1:-1, :-2, 0] + f[1:-1, 2:, 0] - 2 * f[1:-1, 1:-1, 0])
    )
    f[1:-1, 1:-1, 1] = f[1:-1, 1:-1, 0] + 0.5 * S_sq * laplacian_f0

    viz = WaveVisualizer(
        sim_shape=f.shape[:2],
        output_video_size=(1920, 1080),
        dt=dt,
        dx=dx,
        main_plot_cmap_name="Spectral",
        dynamic_z=False,
        zlim=(-0.25, 1.0),
        font_size=16,
        field_name_label=field_description,
    )

    dist = np.sqrt((xx - xc) ** 2 + (yy - xc) ** 2)
    ring_thick = 3 * dx
    ring_mask = np.abs(dist - ring_radius) <= ring_thick / 2

    velocity_hist, amp_hist = [], []
    if np.any(ring_mask):
        prev_amp = f[:, :, 1][ring_mask].mean()
    else:
        prev_amp = 0.0
        logging.warning("Ring mask is empty. Ring-average metrics will be zero.")

    # ── logging setup ─────────────────────────────────────────────────
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(
        logs_dir,
        f"simulation_{wave_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    logging.info(
        "Simulation start: wave_type=%s, field_description='%s', speed=%.3f m/s, steps=%d, "
        "ring_radius=%.3f m, log_interval=%d, dt=%.3e s, dx=%.3e m, S_sq=%.3f",
        wave_type,
        field_description,
        sim_wave_speed,
        steps,
        ring_radius,
        log_interval,
        dt,
        dx,
        S_sq,
    )

    writer = animation.FFMpegWriter(fps=30, bitrate=8000)
    with writer.saving(viz.fig, out_path, dpi=100):
        for k in tqdm(range(nsteps), desc=f"Simulating {wave_type} ({field_description})"):
            laplacian_f1 = (
                (f[:-2, 1:-1, 1] + f[2:, 1:-1, 1] - 2 * f[1:-1, 1:-1, 1]) +
                (f[1:-1, :-2, 1] + f[1:-1, 2:, 1] - 2 * f[1:-1, 1:-1, 1])
            )
            f[1:-1, 1:-1, 2] = (
                -f[1:-1, 1:-1, 0]
                + 2 * f[1:-1, 1:-1, 1]
                + S_sq * laplacian_f1
            )
            f[:, :, 2] *= damping
            f[src_i, src_j, 2] += ricker_wavelet(k * dt, f0)
            f[:, :, 0] = f[:, :, 1].copy()
            f[:, :, 1] = f[:, :, 2].copy()

            if np.any(ring_mask):
                amp = f[:, :, 1][ring_mask].mean()
                vel = (amp - prev_amp) / dt if dt > 0 else 0.0
            else:
                amp = 0.0
                vel = 0.0
            prev_amp = amp
            energy = float(np.sum(f[:, :, 1] ** 2 * dx * dx))
            tnow = k * dt
            velocity_hist.append((tnow, vel))
            amp_hist.append(amp)

            if k % log_interval == 0:
                logging.info(
                    "step=%d t=%.3f s, ring_avg_field=%.6f, ring_avg_field_rate=%.6f, total_energy_proxy=%.6f",
                    k,
                    tnow,
                    amp,
                    vel,
                    energy,
                )

            viz.update(f[:, :, 1], velocity_hist, amp_hist, tnow, monitor_ring=ring_radius)
            viz.render_composite_frame()
            writer.grab_frame()

    logging.info("Simulation completed. Output saved to %s", out_path)

    return out_path
