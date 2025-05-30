import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import cg


def ricker_wavelet(t: float, f0: float) -> float:
    tau = 1.0 / f0
    term = np.pi * f0 * (t - tau)
    return (1.0 - 2.0 * term**2) * np.exp(-term**2)


# ---------------------------------------------------------------------------
# explicit velocity-stress solver
# ---------------------------------------------------------------------------
def simulate_elastic_wave(
    steps: int,
    dx: float,
    L: float,
    vp: float,
    vs: float,
    rho: float,
    lame_lambda: float,
    lame_mu: float,
    dt: float | None = None,
    f0: float = 25.0,
    ring_radius: float = 0.15,
    absorb_width: int = 10,
    absorb_strength: float = 2.0,
):
    npts = int(L / dx) + 1
    x = np.arange(npts) * dx
    y = np.arange(npts) * dx
    xx, yy = np.meshgrid(x, y)
    if dt is None:
        cfl = 0.6
        dt = cfl * dx / (np.sqrt(2.0) * max(vp, vs))
    if (np.sqrt(2.0) * max(vp, vs) * dt / dx) >= 1.0:
        raise ValueError("Unstable time step for elastic solver")

    vx = np.zeros((npts, npts))
    vy = np.zeros((npts, npts))
    sxx = np.zeros((npts, npts))
    syy = np.zeros((npts, npts))
    sxy = np.zeros((npts, npts))

    damping = np.ones((npts, npts))
    for i in range(absorb_width):
        coef = np.exp(-absorb_strength * ((absorb_width - i) / absorb_width) ** 2)
        damping[i, :] *= coef
        damping[-i - 1, :] *= coef
        damping[:, i] *= coef
        damping[:, -i - 1] *= coef

    ux = np.zeros((npts, npts))
    uy = np.zeros((npts, npts))

    src_i = npts // 2
    src_j = npts // 2

    dist = np.sqrt((xx - L / 2) ** 2 + (yy - L / 2) ** 2)
    ring_thick = 3 * dx
    ring_mask = np.abs(dist - ring_radius) <= ring_thick / 2
    velocity_hist: list[tuple[float, float]] = []
    amp_hist: list[float] = []
    field_snaps: list[np.ndarray] = []
    prev_amp = np.mean(np.sqrt(vx**2 + vy**2)[ring_mask]) if np.any(ring_mask) else 0.0

    for it in range(steps):
        dvx_dx = (vx[2:, 1:-1] - vx[:-2, 1:-1]) / (2 * dx)
        dvy_dy = (vy[1:-1, 2:] - vy[1:-1, :-2]) / (2 * dx)
        dvx_dy = (vx[1:-1, 2:] - vx[1:-1, :-2]) / (2 * dx)
        dvy_dx = (vy[2:, 1:-1] - vy[:-2, 1:-1]) / (2 * dx)

        sxx[1:-1, 1:-1] += ((lame_lambda + 2 * lame_mu) * dvx_dx + lame_lambda * dvy_dy) * dt
        syy[1:-1, 1:-1] += ((lame_lambda + 2 * lame_mu) * dvy_dy + lame_lambda * dvx_dx) * dt
        sxy[1:-1, 1:-1] += (lame_mu * (dvy_dx + dvx_dy)) * dt

        sxx *= damping
        syy *= damping
        sxy *= damping

        src_val = ricker_wavelet(it * dt, f0)
        sxx[src_i, src_j] += src_val
        syy[src_i, src_j] += src_val

        dsxx_dx = (sxx[2:, 1:-1] - sxx[:-2, 1:-1]) / (2 * dx)
        dsxy_dy = (sxy[1:-1, 2:] - sxy[1:-1, :-2]) / (2 * dx)
        dsyy_dy = (syy[1:-1, 2:] - syy[1:-1, :-2]) / (2 * dx)
        dsxy_dx = (sxy[2:, 1:-1] - sxy[:-2, 1:-1]) / (2 * dx)

        vx[1:-1, 1:-1] += dt / rho * (dsxx_dx + dsxy_dy)
        vy[1:-1, 1:-1] += dt / rho * (dsxy_dx + dsyy_dy)

        vx *= damping
        vy *= damping

        ux += vx * dt
        uy += vy * dt

        mag_v = np.sqrt(vx**2 + vy**2)
        mag_u = np.sqrt(ux**2 + uy**2)
        field_snaps.append(mag_u.copy())
        if np.any(ring_mask):
            amp = np.mean(mag_u[ring_mask])
            vel = (amp - prev_amp) / dt
        else:
            amp = 0.0
            vel = 0.0
        prev_amp = amp
        velocity_hist.append((it * dt, vel))
        amp_hist.append(amp)
    return (field_snaps, velocity_hist, amp_hist, dt)


# ---------------------------------------------------------------------------
# incremental potential (static solve)
# ---------------------------------------------------------------------------
def solve_incremental_elastic(
    force_x: np.ndarray,
    force_y: np.ndarray,
    dx: float,
    lame_lambda: float,
    lame_mu: float,
    fixed_mask: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 1000,
):
    ny, nx = force_x.shape
    idx = lambda i, j: i * nx + j

    free_mask = ~fixed_mask
    map_free = -np.ones((ny, nx), dtype=int)
    free_indices = np.argwhere(free_mask)
    for n_id, (i, j) in enumerate(free_indices):
        map_free[i, j] = n_id
    dof_f = len(free_indices) * 2

    H_ff = lil_matrix((dof_f, dof_f))
    b_f = np.zeros(dof_f)

    for i, j in free_indices:
        kf = map_free[i, j]
        row_u = 2 * kf
        row_v = row_u + 1
        diag = 4 * lame_mu + 2 * lame_lambda
        H_ff[row_u, row_u] = diag
        H_ff[row_v, row_v] = diag
        b_f[row_u] = force_x[i, j]
        b_f[row_v] = force_y[i, j]
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni = i + di
            nj = j + dj
            if 0 <= ni < ny and 0 <= nj < nx and free_mask[ni, nj]:
                kn = map_free[ni, nj]
                H_ff[row_u, 2 * kn] += -lame_mu
                H_ff[row_v, 2 * kn + 1] += -lame_mu

    H_ff = csr_matrix(H_ff)
    x0 = np.zeros(dof_f)
    sol_f, info = cg(H_ff, b_f, x0=x0, tol=tol, maxiter=max_iter)
    if info != 0:
        raise RuntimeError("CG did not converge")

    ux = np.zeros((ny, nx))
    uy = np.zeros((ny, nx))
    for i, j in free_indices:
        kf = map_free[i, j]
        ux[i, j] = sol_f[2 * kf]
        uy[i, j] = sol_f[2 * kf + 1]
    return ux, uy

