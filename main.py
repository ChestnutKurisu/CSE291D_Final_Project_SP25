import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import time
from dataclasses import dataclass, field, fields
from typing import List, Tuple, Optional, Any, Union
import os
import argparse
import sys


def register_plume_colormap() -> None:
    """
    Create the custom ‘plume’ colormap and add it to Matplotlib’s global
    colormap registry.  Works on every Matplotlib ≥3.2 (new API) and on
    older releases that still ship ``cm.register_cmap``.
    """
    plume_rgb = [
        "#001428",  # almost-black navy background
        "#06336a",  # deep blue
        "#135996",  # blue-cyan transition
        "#1b7cb4",  # cyan-teal
        "#2ca4c9",  # bright cyan
        "#6ad3d7",  # pale turquoise
        "#b9f0e2",  # mint / very light cyan
        "#fdfecb",  # soft yellow-white highlights
    ]
    cmap = LinearSegmentedColormap.from_list("plume", plume_rgb, N=256)

    #  Matplotlib ≥3.5 – preferred way
    if hasattr(mpl, "colormaps"):
        if "plume" not in mpl.colormaps:
            mpl.colormaps.register(cmap)

    # fallback for very old versions that still expose cm.register_cmap
    elif hasattr(mpl.cm, "register_cmap"):
        try:
            mpl.cm.register_cmap(name="plume", cmap=cmap)
        except ValueError:
            pass  # already registered


register_plume_colormap()

# --- Numba Configuration ---
_NUMBA_AVAILABLE = False
_NUMBA_JIT_OPTIONS = {'parallel': True, 'fastmath': True, 'cache': True, 'nogil': True}
try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
    print("Numba detected. CPU functions can be JIT-compiled if GPU is disabled.")
except ImportError:
    print("Numba not found. CPU functions will use standard NumPy if GPU is disabled.")

# --- GPU Configuration ---
try:
    import cupy as cp

    try:
        cp.array([1, 2, 3]).sum()  # Test if CuPy is functional
        DEFAULT_GPU_ENABLED = True
    except cp.cuda.runtime.CUDARuntimeError:
        print("CuPy imported but CUDA runtime error. Disabling GPU.")
        DEFAULT_GPU_ENABLED = False
    except Exception as e:
        print(f"CuPy imported but encountered an issue ({e}). Disabling GPU.")
        DEFAULT_GPU_ENABLED = False

    from cupy import ElementwiseKernel

    _lamb_oseen_kernel = ElementwiseKernel(
        'float32 r_sq, float32 core_a_sq',
        'float32 out',
        r'''
        const float eps = 1e-7f;
        float r_safe = (r_sq < eps) ? eps : r_sq;
        float val = (1.0f - expf(-r_safe / core_a_sq)) / r_safe;
        out = (r_sq < eps * 10.0f) ? (1.0f / core_a_sq) : val;
        ''',
        'lamb_oseen'
    )


    def _lamb_oseen_factor_xp(r_sq, core_a_sq, xp, _):
        if xp is cp:
            return _lamb_oseen_kernel(r_sq.astype(cp.float32), cp.float32(core_a_sq))
        else:
            # fall back to existing CPU version (NumPy or Numba)
            return _lamb_oseen_factor_cpu_numba(r_sq, core_a_sq, True, True)
except ImportError:
    DEFAULT_GPU_ENABLED = False


# --- Simulation Configuration Dataclass ---
@dataclass
class SimConfig:
    N_VORTICES: int = 20
    N_TRACERS: int = 500000
    DOMAIN_RADIUS: float = 1.0
    SIMULATION_TIME: float = 3.0
    DT: float = 0.002
    OUTPUT_FILENAME: str = "point_vortex_dynamics.mp4"
    PLOT_INTERVAL: int = 2
    DPI: int = 120

    VORTEX_CORE_A_SQ: float = 0.001
    TRACER_CORE_A_SQ: float = 0.0005

    BOUNDARY_WARN_THRESHOLD: float = 0.995

    GPU_ENABLED: bool = DEFAULT_GPU_ENABLED
    GPU_DEVICE_ID: Optional[int] = None
    RANDOM_SEED: Optional[int] = 42

    TRACER_PARTICLE_SIZE: float = 0.3
    TRACER_ALPHA: float = 0.4
    TRACER_CMAP: str = "jet"
    TRACER_COLORING_MODE: str = "group"
    NUM_TRACER_GROUPS: int = 3

    TRACER_GLOW_LAYERS: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.10, 0.10),
            (0.05, 0.05)
        ]
    )

    VORTEX_MARKER_SIZE_BASE: float = 10
    VORTEX_MARKER_SIZE_SCALE: float = 20  # Typo in original, kept for consistency
    VORTEX_COLOR_POS: str = '#FFFF00'
    VORTEX_COLOR_NEG: str = '#FF00FF'

    FIGURE_BG_COLOR: str = '#080808'
    AXES_BG_COLOR: str = '#101010'

    FPS: int = 30
    FFMPEG_CODEC: str = "libx264"
    FFMPEG_PRESET: str = "ultrafast"
    FFMPEG_CRF: int = 23
    FFMPEG_CQ: int = 19
    FFMPEG_THREADS: int = 0  # 0 means auto for ffmpeg, or we can set a better default

    # Performance knobs
    USE_FLOAT32_CPU: bool = True  # Use float32 on CPU path (Numba or NumPy)
    ANIM_TRACERS_MAX: int = 5_000_000  # Max tracers to render in animation
    ANIM_GLOW_MAX: int = 2_000_000  # Max tracers to have glow effect

    # Derived fields
    xp: Any = field(init=False)
    rng: Any = field(init=False)
    float_type: Any = field(init=False)
    _anim_idx: Optional[np.ndarray] = field(init=False, repr=False, default=None)

    _lamb_oseen_factor_func: Any = field(init=False, repr=False)
    _get_velocities_induced_by_vortices_func: Any = field(init=False, repr=False)
    get_vortex_velocities_func: Any = field(init=False, repr=False)

    def __post_init__(self):
        if self.GPU_ENABLED:
            if self.GPU_DEVICE_ID is not None:
                try:
                    print(f"Attempting to use GPU device ID: {self.GPU_DEVICE_ID}")
                    cp.cuda.Device(self.GPU_DEVICE_ID).use()
                except Exception as e:
                    print(f"Failed to set GPU device ID {self.GPU_DEVICE_ID}: {e}. Using default GPU.")
            self.xp = cp
            self.float_type = cp.float32
            print(f"CuPy active on CUDA device {cp.cuda.runtime.getDevice()}. Using GPU acceleration with float32.")

            self._lamb_oseen_factor_func = _lamb_oseen_factor_xp
            self._get_velocities_induced_by_vortices_func = _get_velocities_induced_by_vortices_xp
            self.get_vortex_velocities_func = get_vortex_velocities_xp

        else:  # CPU Path
            self.xp = np
            self.float_type = np.float32 if self.USE_FLOAT32_CPU else np.float64
            prec_str = 'float32' if self.USE_FLOAT32_CPU else 'float64'
            if _NUMBA_AVAILABLE:
                print(
                    f"CuPy not available or disabled. Running on CPU with NumPy and Numba JIT ({prec_str}). First computation step may be slow due to JIT compilation.")
                self._lamb_oseen_factor_func = _lamb_oseen_factor_cpu_numba
                self._get_velocities_induced_by_vortices_func = _get_velocities_induced_by_vortices_cpu_numba
                self.get_vortex_velocities_func = get_vortex_velocities_cpu_numba
            else:
                print(
                    f"CuPy not available or disabled. Numba not found. Running on CPU with standard NumPy ({prec_str}). This will be slow.")
                self._lamb_oseen_factor_func = _lamb_oseen_factor_xp
                self._get_velocities_induced_by_vortices_func = _get_velocities_induced_by_vortices_xp
                self.get_vortex_velocities_func = get_vortex_velocities_xp

        if self.RANDOM_SEED is not None:
            # Use NumPy for seeding the subsampling index regardless of backend for consistency
            np_rng_for_sampling = np.random.default_rng(self.RANDOM_SEED)
            self.rng = self.xp.random.default_rng(self.RANDOM_SEED)
        else:
            np_rng_for_sampling = np.random.default_rng()
            self.rng = self.xp.random.default_rng()

        # Tracer sub-sampling index for animation
        if self.N_TRACERS > self.ANIM_TRACERS_MAX and self.ANIM_TRACERS_MAX > 0:
            self._anim_idx = np_rng_for_sampling.choice(
                self.N_TRACERS, self.ANIM_TRACERS_MAX, replace=False
            ).astype(np.int32)  # int32 is fine for indices
            print(f"Animation will render a subsample of {self.ANIM_TRACERS_MAX}/{self.N_TRACERS} tracers.")
        elif self.ANIM_TRACERS_MAX <= 0 and self.N_TRACERS > 0:
            print("Warning: ANIM_TRACERS_MAX <= 0, no tracers will be rendered in animation.")
            self._anim_idx = np.array([], dtype=np.int32)  # Empty selection
        else:  # Render all tracers or N_TRACERS is 0
            self._anim_idx = None

        if self.N_TRACERS > 750000 and self.xp == np and not _NUMBA_AVAILABLE:
            print(f"Warning: N_TRACERS ({self.N_TRACERS}) is high for CPU without Numba. Simulation will be very slow.")

        valid_coloring_modes = ["group", "scalar", "speed"]
        if self.TRACER_COLORING_MODE not in valid_coloring_modes:
            raise ValueError(
                f"Invalid TRACER_COLORING_MODE: {self.TRACER_COLORING_MODE}. Must be one of {valid_coloring_modes}.")

        if not isinstance(self.TRACER_GLOW_LAYERS, list) or \
                not all(isinstance(item, tuple) and len(item) == 2 and
                        isinstance(item[0], (int, float)) and isinstance(item[1], (int, float))
                        for item in self.TRACER_GLOW_LAYERS):
            raise ValueError("TRACER_GLOW_LAYERS must be a list of (float, float) tuples.")


# --- Lamb-Oseen Factor Implementations ---
def _lamb_oseen_factor_xp(r_sq, core_a_sq, xp, float_type):
    epsilon = 1e-12 if float_type == xp.float64 else (1e-7 if float_type == xp.float32 else 1e-7)
    r_sq_safe = xp.where(r_sq < epsilon, epsilon, r_sq)
    val = (1.0 - xp.exp(-r_sq_safe / core_a_sq)) / r_sq_safe
    limit_val = 1.0 / core_a_sq
    return xp.where(r_sq < epsilon * 10, limit_val, val)


if _NUMBA_AVAILABLE:
    @njit(**_NUMBA_JIT_OPTIONS)
    def _lamb_oseen_factor_cpu_numba_impl(r_sq, core_a_sq, epsilon_val, epsilon_factor_10_val):
        out_val = np.empty_like(r_sq)
        limit_val = 1.0 / core_a_sq
        # Loop for prange if r_sq is 2D, or rely on Numba vectorization for 1D/elementwise
        if r_sq.ndim == 1:
            for i in prange(r_sq.shape[0]):
                r_sq_i = r_sq[i]
                if r_sq_i < epsilon_val:
                    r_sq_safe_i = epsilon_val
                else:
                    r_sq_safe_i = r_sq_i

                val_i = (1.0 - np.exp(-r_sq_safe_i / core_a_sq)) / r_sq_safe_i

                if r_sq_i < epsilon_factor_10_val:
                    out_val[i] = limit_val
                else:
                    out_val[i] = val_i
            return out_val
        elif r_sq.ndim == 2:  # Common case for M x N interactions
            for i in prange(r_sq.shape[0]):
                for j in range(r_sq.shape[1]):  # Inner loop not parallel for typical prange
                    r_sq_ij = r_sq[i, j]
                    if r_sq_ij < epsilon_val:
                        r_sq_safe_ij = epsilon_val
                    else:
                        r_sq_safe_ij = r_sq_ij

                    val_ij = (1.0 - np.exp(-r_sq_safe_ij / core_a_sq)) / r_sq_safe_ij

                    if r_sq_ij < epsilon_factor_10_val:
                        out_val[i, j] = limit_val
                    else:
                        out_val[i, j] = val_ij
            return out_val
        else:  # Fallback for other dimensions, though not expected here
            r_sq_safe = np.where(r_sq < epsilon_val, epsilon_val, r_sq)
            val = (1.0 - np.exp(-r_sq_safe / core_a_sq)) / r_sq_safe
            return np.where(r_sq < epsilon_factor_10_val, limit_val, val)


    def _lamb_oseen_factor_cpu_numba(r_sq, core_a_sq, xp_is_numpy, float_type_is_np_float_type):
        # xp_is_numpy and float_type_is_np_float_type are for signature compatibility with _xp version.
        # Numba path implies xp=np. The actual float type is config.float_type.
        current_float_type = r_sq.dtype  # Get dtype from input array
        epsilon = 1e-12 if current_float_type == np.float64 else (1e-7 if current_float_type == np.float32 else 1e-7)
        return _lamb_oseen_factor_cpu_numba_impl(r_sq, core_a_sq, epsilon, epsilon * 10)


# --- Velocity Calculation Implementations ---
def _get_velocities_induced_by_vortices_xp(target_positions, vortex_positions, vortex_strengths,
                                           core_a_sq, config: SimConfig, total_vortex_strength_for_bg_flow):
    xp = config.xp
    M = target_positions.shape[0]
    N = vortex_positions.shape[0]

    if N == 0 or M == 0:
        return xp.zeros_like(target_positions)

    velocities = xp.zeros_like(target_positions)

    norm_sq_all_vortices = xp.sum(vortex_positions ** 2, axis=1)
    epsilon_norm_sq = 1e-9 if config.float_type == xp.float64 else (1e-6 if config.float_type == xp.float32 else 1e-6)
    norm_sq_all_vortices_safe = xp.where(norm_sq_all_vortices < epsilon_norm_sq, epsilon_norm_sq, norm_sq_all_vortices)

    img_v_positions = (config.DOMAIN_RADIUS ** 2 / norm_sq_all_vortices_safe[:, xp.newaxis]) * vortex_positions
    img_v_strengths = -vortex_strengths

    target_pos_exp = target_positions[:, xp.newaxis, :]

    v_pos_exp = vortex_positions[xp.newaxis, :, :]
    diff_real = target_pos_exp - v_pos_exp
    r_sq_real = xp.sum(diff_real ** 2, axis=2)

    interaction_factor_real = config._lamb_oseen_factor_func(r_sq_real, core_a_sq, xp, config.float_type)
    coeff_real = vortex_strengths[xp.newaxis, :] / (2 * xp.pi)
    term_real = coeff_real * interaction_factor_real

    velocities[:, 0] += xp.sum(-term_real * diff_real[:, :, 1], axis=1)
    velocities[:, 1] += xp.sum(term_real * diff_real[:, :, 0], axis=1)

    img_v_pos_exp = img_v_positions[xp.newaxis, :, :]
    diff_img = target_pos_exp - img_v_pos_exp
    r_sq_img = xp.sum(diff_img ** 2, axis=2)

    interaction_factor_img = config._lamb_oseen_factor_func(r_sq_img, core_a_sq, xp, config.float_type)
    coeff_img = img_v_strengths[xp.newaxis, :] / (2 * xp.pi)
    term_img = coeff_img * interaction_factor_img

    velocities[:, 0] += xp.sum(-term_img * diff_img[:, :, 1], axis=1)
    velocities[:, 1] += xp.sum(term_img * diff_img[:, :, 0], axis=1)

    if xp.abs(total_vortex_strength_for_bg_flow) > epsilon_norm_sq:
        K_bg = total_vortex_strength_for_bg_flow / (2 * xp.pi * config.DOMAIN_RADIUS ** 2)
        velocities[:, 0] += -K_bg * target_positions[:, 1]
        velocities[:, 1] += K_bg * target_positions[:, 0]

    return velocities


if _NUMBA_AVAILABLE:
    @njit(**_NUMBA_JIT_OPTIONS)
    def _get_velocities_induced_by_vortices_cpu_numba_impl(
            target_positions, vortex_positions, vortex_strengths,
            core_a_sq, total_vortex_strength_for_bg_flow,
            domain_radius, domain_radius_sq, pi_val,
            lo_epsilon, lo_epsilon_factor_10,
            epsilon_norm_sq_val
    ):
        M = target_positions.shape[0]
        N = vortex_positions.shape[0]
        velocities = np.zeros_like(target_positions)

        if N == 0 or M == 0: return velocities

        norm_sq_all_vortices = np.sum(vortex_positions ** 2, axis=1)
        img_v_positions_np = np.empty_like(vortex_positions)
        for i in prange(N):
            n_sq_safe = norm_sq_all_vortices[i]
            if n_sq_safe < epsilon_norm_sq_val: n_sq_safe = epsilon_norm_sq_val
            factor = domain_radius_sq / n_sq_safe
            img_v_positions_np[i, 0] = factor * vortex_positions[i, 0]
            img_v_positions_np[i, 1] = factor * vortex_positions[i, 1]
        img_v_strengths = -vortex_strengths

        # Using prange for the outer loop over target particles (M)
        for i in prange(M):
            vel_x_i = 0.0
            vel_y_i = 0.0
            tx_i = target_positions[i, 0]
            ty_i = target_positions[i, 1]

            for j in range(N):  # Inner loop over source vortices (N)
                # Real vortex
                dx_r = tx_i - vortex_positions[j, 0]
                dy_r = ty_i - vortex_positions[j, 1]
                r_sq_r_ij = dx_r * dx_r + dy_r * dy_r

                # Scalar Lamb-Oseen call
                r_sq_r_arr = np.array([r_sq_r_ij], dtype=target_positions.dtype)  # Temp array for LO func
                interaction_factor_real_ij = \
                _lamb_oseen_factor_cpu_numba_impl(r_sq_r_arr, core_a_sq, lo_epsilon, lo_epsilon_factor_10)[0]

                coeff_real_j = vortex_strengths[j] / (2 * pi_val)
                term_real_ij = coeff_real_j * interaction_factor_real_ij
                vel_x_i += -term_real_ij * dy_r
                vel_y_i += term_real_ij * dx_r

                # Image vortex
                dx_i = tx_i - img_v_positions_np[j, 0]
                dy_i = ty_i - img_v_positions_np[j, 1]
                r_sq_i_ij = dx_i * dx_i + dy_i * dy_i

                r_sq_i_arr = np.array([r_sq_i_ij], dtype=target_positions.dtype)  # Temp array for LO func
                interaction_factor_img_ij = \
                _lamb_oseen_factor_cpu_numba_impl(r_sq_i_arr, core_a_sq, lo_epsilon, lo_epsilon_factor_10)[0]

                coeff_img_j = img_v_strengths[j] / (2 * pi_val)
                term_img_ij = coeff_img_j * interaction_factor_img_ij
                vel_x_i += -term_img_ij * dy_i
                vel_y_i += term_img_ij * dx_i

            velocities[i, 0] = vel_x_i
            velocities[i, 1] = vel_y_i

        if np.abs(total_vortex_strength_for_bg_flow) > epsilon_norm_sq_val:
            K_bg = total_vortex_strength_for_bg_flow / (2 * pi_val * domain_radius_sq)
            for i in prange(M):  # Can also be vectorized outside if preferred
                velocities[i, 0] += -K_bg * target_positions[i, 1]
                velocities[i, 1] += K_bg * target_positions[i, 0]
        return velocities


    def _get_velocities_induced_by_vortices_cpu_numba(target_positions, vortex_positions, vortex_strengths,
                                                      core_a_sq, config: SimConfig, total_vortex_strength_for_bg_flow):
        current_float_type = config.float_type
        lo_epsilon = 1e-12 if current_float_type == np.float64 else (1e-7 if current_float_type == np.float32 else 1e-7)
        epsilon_norm_sq = 1e-9 if current_float_type == np.float64 else (
            1e-6 if current_float_type == np.float32 else 1e-6)

        return _get_velocities_induced_by_vortices_cpu_numba_impl(
            target_positions, vortex_positions, vortex_strengths,
            core_a_sq, total_vortex_strength_for_bg_flow,
            config.DOMAIN_RADIUS, config.DOMAIN_RADIUS ** 2, np.pi,
            lo_epsilon, lo_epsilon * 10,
            epsilon_norm_sq
        )


def get_vortex_velocities_xp(v_positions, v_strengths, config: SimConfig, total_vortex_strength):
    xp = config.xp
    N = v_positions.shape[0]
    if N == 0: return xp.zeros_like(v_positions)

    v_pos_i = v_positions[:, xp.newaxis, :]
    v_pos_j = v_positions[xp.newaxis, :, :]

    diff_real = v_pos_i - v_pos_j
    r_sq_real = xp.sum(diff_real ** 2, axis=2)
    identity_mask = xp.eye(N, dtype=bool)

    interaction_factor_real = config._lamb_oseen_factor_func(r_sq_real, config.VORTEX_CORE_A_SQ, xp, config.float_type)
    coeff_real = v_strengths[xp.newaxis, :] / (2 * xp.pi)
    term_real_masked = coeff_real * interaction_factor_real
    term_real_masked = xp.where(identity_mask, 0.0, term_real_masked)

    velocities = xp.zeros_like(v_positions)
    velocities[:, 0] = xp.sum(-term_real_masked * diff_real[:, :, 1], axis=1)
    velocities[:, 1] = xp.sum(term_real_masked * diff_real[:, :, 0], axis=1)

    norm_sq_all = xp.sum(v_positions ** 2, axis=1)
    eps_norm_sq = 1e-9 if config.float_type == xp.float64 else (1e-6 if config.float_type == xp.float32 else 1e-6)
    norm_sq_safe = xp.where(norm_sq_all < eps_norm_sq, eps_norm_sq, norm_sq_all)

    img_pos_j_denom = norm_sq_safe[xp.newaxis, :, xp.newaxis]
    # Handle cases where denom is zero even after safegaurd (e.g. if eps_norm_sq is zero, which it isn't here)
    # This mainly guards against user error setting eps_norm_sq too low or zero.
    img_pos_j_denom = xp.where(img_pos_j_denom == 0, eps_norm_sq, img_pos_j_denom)

    img_pos_j = ((config.DOMAIN_RADIUS ** 2) / img_pos_j_denom) * v_pos_j
    img_str_j = -v_strengths[xp.newaxis, :]

    diff_img = v_pos_i - img_pos_j
    r_sq_img = xp.sum(diff_img ** 2, axis=2)

    interaction_factor_img = config._lamb_oseen_factor_func(r_sq_img, config.VORTEX_CORE_A_SQ, xp, config.float_type)
    coeff_img = img_str_j / (2 * xp.pi)
    term_img = coeff_img * interaction_factor_img

    velocities[:, 0] += xp.sum(-term_img * diff_img[:, :, 1], axis=1)
    velocities[:, 1] += xp.sum(term_img * diff_img[:, :, 0], axis=1)

    if xp.abs(total_vortex_strength) > eps_norm_sq:
        K_bg = total_vortex_strength / (2 * xp.pi * config.DOMAIN_RADIUS ** 2)
        velocities[:, 0] += -K_bg * v_positions[:, 1]
        velocities[:, 1] += K_bg * v_positions[:, 0]
    return velocities


if _NUMBA_AVAILABLE:
    @njit(**_NUMBA_JIT_OPTIONS)
    def get_vortex_velocities_cpu_numba_impl(
            v_positions, v_strengths, total_vortex_strength,
            vortex_core_a_sq, domain_radius, domain_radius_sq, pi_val,
            lo_epsilon, lo_epsilon_factor_10,
            epsilon_norm_sq_val
    ):
        N = v_positions.shape[0]
        if N == 0: return np.zeros_like(v_positions)
        velocities = np.zeros_like(v_positions)

        # For Numba, N*N can be slow with Python-level broadcasting.
        # Explicit loops are often better for Numba's prange.
        for i in prange(N):  # Loop over each vortex `i` whose velocity we are calculating
            v_pos_i_vec = v_positions[i, :]
            vel_x_i = 0.0
            vel_y_i = 0.0

            # 1. Interaction with other *real* vortices (j != i)
            for j in range(N):
                if i == j: continue  # Skip self-interaction

                v_pos_j_vec = v_positions[j, :]
                diff_real_x = v_pos_i_vec[0] - v_pos_j_vec[0]
                diff_real_y = v_pos_i_vec[1] - v_pos_j_vec[1]
                r_sq_real_ij = diff_real_x ** 2 + diff_real_y ** 2

                r_sq_real_arr = np.array([r_sq_real_ij], dtype=v_positions.dtype)
                interaction_factor_real_ij = \
                _lamb_oseen_factor_cpu_numba_impl(r_sq_real_arr, vortex_core_a_sq, lo_epsilon, lo_epsilon_factor_10)[0]

                coeff_real_j = v_strengths[j] / (2 * pi_val)
                term_real_ij = coeff_real_j * interaction_factor_real_ij

                vel_x_i += -term_real_ij * diff_real_y
                vel_y_i += term_real_ij * diff_real_x

            # 2. Interaction with *image* vortices (including self-image, so loop all j)
            for j in range(N):
                v_pos_j_vec = v_positions[j, :]
                norm_sq_j = v_pos_j_vec[0] ** 2 + v_pos_j_vec[1] ** 2
                norm_sq_safe_j = norm_sq_j if norm_sq_j >= epsilon_norm_sq_val else epsilon_norm_sq_val
                if norm_sq_safe_j == 0: norm_sq_safe_j = epsilon_norm_sq_val  # Extra safety

                img_factor = domain_radius_sq / norm_sq_safe_j
                img_pos_j_x = img_factor * v_pos_j_vec[0]
                img_pos_j_y = img_factor * v_pos_j_vec[1]
                img_str_j = -v_strengths[j]

                diff_img_x = v_pos_i_vec[0] - img_pos_j_x
                diff_img_y = v_pos_i_vec[1] - img_pos_j_y
                r_sq_img_ij = diff_img_x ** 2 + diff_img_y ** 2

                r_sq_img_arr = np.array([r_sq_img_ij], dtype=v_positions.dtype)
                interaction_factor_img_ij = \
                _lamb_oseen_factor_cpu_numba_impl(r_sq_img_arr, vortex_core_a_sq, lo_epsilon, lo_epsilon_factor_10)[0]

                coeff_img_j = img_str_j / (2 * pi_val)
                term_img_ij = coeff_img_j * interaction_factor_img_ij

                vel_x_i += -term_img_ij * diff_img_y
                vel_y_i += term_img_ij * diff_img_x

            velocities[i, 0] = vel_x_i
            velocities[i, 1] = vel_y_i

        if np.abs(total_vortex_strength) > epsilon_norm_sq_val:
            K_bg = total_vortex_strength / (2 * pi_val * domain_radius_sq)
            for i in prange(N):  # Can also be vectorized
                velocities[i, 0] += -K_bg * v_positions[i, 1]
                velocities[i, 1] += K_bg * v_positions[i, 0]
        return velocities


    def get_vortex_velocities_cpu_numba(v_positions, v_strengths, config: SimConfig, total_vortex_strength):
        current_float_type = config.float_type
        lo_epsilon = 1e-12 if current_float_type == np.float64 else (1e-7 if current_float_type == np.float32 else 1e-7)
        epsilon_norm_sq = 1e-9 if current_float_type == np.float64 else (
            1e-6 if current_float_type == np.float32 else 1e-6)
        return get_vortex_velocities_cpu_numba_impl(
            v_positions, v_strengths, total_vortex_strength,
            config.VORTEX_CORE_A_SQ, config.DOMAIN_RADIUS, config.DOMAIN_RADIUS ** 2, np.pi,
            lo_epsilon, lo_epsilon * 10,
            epsilon_norm_sq
        )


# --- Core Simulation Logic (Initialization and Step) ---
def initialize_vortices(config: SimConfig):
    # (Original initialize_vortices function remains unchanged, uses config.xp and config.rng)
    xp = config.xp
    positions = xp.zeros((config.N_VORTICES, 2), dtype=config.float_type)
    strengths = xp.zeros(config.N_VORTICES, dtype=config.float_type)

    if config.N_VORTICES == 0:
        return positions, strengths

    if config.N_VORTICES >= 4:
        s = 0.4 * config.DOMAIN_RADIUS
        positions[0] = xp.array([-s, s], dtype=config.float_type);
        strengths[0] = 1.5
        positions[1] = xp.array([s, s], dtype=config.float_type);
        strengths[1] = -1.5
        positions[2] = xp.array([-s, -s], dtype=config.float_type);
        strengths[2] = -1.5
        positions[3] = xp.array([s, -s], dtype=config.float_type);
        strengths[3] = 1.5

        if config.N_VORTICES > 4:
            num_remaining = config.N_VORTICES - 4
            radii = config.rng.uniform(0.1, 0.7, num_remaining).astype(config.float_type) * config.DOMAIN_RADIUS
            angles = config.rng.uniform(0, 2 * xp.pi, num_remaining).astype(config.float_type)
            positions[4:, 0] = radii * xp.cos(angles)
            positions[4:, 1] = radii * xp.sin(angles)

            rand_strengths_choices = xp.array([-0.75, 0.75, -0.5, 0.5], dtype=config.float_type)
            chosen_indices = config.rng.integers(0, len(rand_strengths_choices), num_remaining)
            rand_strengths_base = rand_strengths_choices[chosen_indices]
            strengths[4:] = rand_strengths_base * config.rng.uniform(0.5, 1.0, num_remaining).astype(config.float_type)
    else:
        radii = config.rng.uniform(0.1, 0.7, config.N_VORTICES).astype(config.float_type) * config.DOMAIN_RADIUS
        angles = config.rng.uniform(0, 2 * xp.pi, config.N_VORTICES).astype(config.float_type)
        positions[:, 0] = radii * xp.cos(angles)
        positions[:, 1] = radii * xp.sin(angles)

        rand_strengths_choices = xp.array([-1.0, 1.0], dtype=config.float_type)
        chosen_indices = config.rng.integers(0, len(rand_strengths_choices), config.N_VORTICES)
        base_s = rand_strengths_choices[chosen_indices]
        strengths[:] = base_s * config.rng.uniform(0.5, 1.5, config.N_VORTICES).astype(config.float_type)

    total_initial_strength = xp.sum(strengths)
    print(f"Total initial vortex strength: {total_initial_strength:.3e}")
    return positions, strengths


def initialize_tracers(config: SimConfig):
    # (Original initialize_tracers function remains unchanged, uses config.xp and config.rng)
    xp = config.xp
    tracer_pos = xp.zeros((config.N_TRACERS, 2), dtype=config.float_type)
    tracer_scalar_values = xp.zeros(config.N_TRACERS, dtype=config.float_type)

    if config.N_TRACERS == 0:
        return tracer_pos, tracer_scalar_values

    if config.TRACER_COLORING_MODE == "group":
        print(
            f"Initializing tracers with 'group' coloring: {config.NUM_TRACER_GROUPS} groups, cmap '{config.TRACER_CMAP}'.")
        num_groups = max(1, config.NUM_TRACER_GROUPS)
        tracers_per_group = config.N_TRACERS // num_groups
        patch_radius_base = config.DOMAIN_RADIUS * 0.25
        patch_center_dist = config.DOMAIN_RADIUS * 0.45
        current_idx = 0
        for i in range(num_groups):
            num_in_patch = tracers_per_group if i < num_groups - 1 else config.N_TRACERS - current_idx
            if num_in_patch <= 0: continue

            angle_offset = (2 * xp.pi / num_groups) * i
            center_x = patch_center_dist * xp.cos(angle_offset)
            center_y = patch_center_dist * xp.sin(angle_offset)
            r_sqrt_uniform = config.rng.uniform(0, 1, num_in_patch).astype(config.float_type)
            r = patch_radius_base * xp.sqrt(r_sqrt_uniform)
            theta = config.rng.uniform(0, 2 * xp.pi, num_in_patch).astype(config.float_type)

            start, end = current_idx, current_idx + num_in_patch
            tracer_pos[start:end, 0] = center_x + r * xp.cos(theta)
            tracer_pos[start:end, 1] = center_y + r * xp.sin(theta)
            tracer_scalar_values[start:end] = (i + 0.5) / num_groups  # Ensure float division
            current_idx += num_in_patch

    elif config.TRACER_COLORING_MODE == "scalar":
        max_r_init = config.DOMAIN_RADIUS * 0.70
        r_sqrt_uniform = config.rng.uniform(0, 1, config.N_TRACERS).astype(config.float_type)
        radii = max_r_init * xp.sqrt(r_sqrt_uniform)
        angles = config.rng.uniform(0, 2 * xp.pi, config.N_TRACERS).astype(config.float_type)
        tracer_pos[:, 0] = radii * xp.cos(angles)
        tracer_pos[:, 1] = radii * xp.sin(angles)
        # high scalar in the core, fading outward
        tracer_scalar_values[:] = 1.0 - (radii / max_r_init)

    elif config.TRACER_COLORING_MODE == "speed":
        print("Initializing tracers for 'speed' coloring mode.")
        max_r_init = config.DOMAIN_RADIUS * 0.7
        r_sqrt_uniform = config.rng.uniform(0, 1, config.N_TRACERS).astype(config.float_type)
        radii = max_r_init * xp.sqrt(r_sqrt_uniform)
        angles = config.rng.uniform(0, 2 * xp.pi, config.N_TRACERS).astype(config.float_type)
        tracer_pos[:, 0] = radii * xp.cos(angles)
        tracer_pos[:, 1] = radii * xp.sin(angles)
        tracer_scalar_values[:] = 0.0

    dist_sq = xp.sum(tracer_pos ** 2, axis=1)
    if config.N_TRACERS > 0:
        initial_max_dist_sq = xp.max(dist_sq)
        if initial_max_dist_sq > config.DOMAIN_RADIUS ** 2 * 1.000001:
            initial_max_dist = xp.sqrt(initial_max_dist_sq)
            scale_factor = (config.DOMAIN_RADIUS * 0.99) / initial_max_dist
            tracer_pos *= scale_factor
            print("Warning: Some initial tracer positions were outside the domain and have been scaled in.")

    return tracer_pos, tracer_scalar_values


def get_tracer_velocities(t_positions, v_positions, v_strengths, config: SimConfig, total_vortex_strength):
    return config._get_velocities_induced_by_vortices_func(
        t_positions, v_positions, v_strengths,
        config.TRACER_CORE_A_SQ, config, total_vortex_strength
    )


def rk4_step_system(v_pos, t_pos, v_str, total_v_str, config: SimConfig):
    dt = config.DT

    k1_v = config.get_vortex_velocities_func(v_pos, v_str, config, total_v_str)
    k1_t = get_tracer_velocities(t_pos, v_pos, v_str, config, total_v_str)

    v_pos_k2_arg = v_pos + 0.5 * dt * k1_v
    t_pos_k2_arg = t_pos + 0.5 * dt * k1_t
    k2_v = config.get_vortex_velocities_func(v_pos_k2_arg, v_str, config, total_v_str)
    k2_t = get_tracer_velocities(t_pos_k2_arg, v_pos_k2_arg, v_str, config, total_v_str)

    v_pos_k3_arg = v_pos + 0.5 * dt * k2_v
    t_pos_k3_arg = t_pos + 0.5 * dt * k2_t
    k3_v = config.get_vortex_velocities_func(v_pos_k3_arg, v_str, config, total_v_str)
    k3_t = get_tracer_velocities(t_pos_k3_arg, v_pos_k3_arg, v_str, config, total_v_str)

    v_pos_k4_arg = v_pos + dt * k3_v
    t_pos_k4_arg = t_pos + dt * k3_t
    k4_v = config.get_vortex_velocities_func(v_pos_k4_arg, v_str, config, total_v_str)
    k4_t = get_tracer_velocities(t_pos_k4_arg, v_pos_k4_arg, v_str, config, total_v_str)

    new_v_pos = v_pos + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    new_t_pos = t_pos + (dt / 6.0) * (k1_t + 2 * k2_t + 2 * k3_t + k4_t)

    return new_v_pos, new_t_pos


def enforce_boundaries(positions, config: SimConfig, current_sim_time: float, is_vortex=False):
    # (Original enforce_boundaries function remains unchanged)
    xp = config.xp
    if positions.shape[0] == 0: return positions

    norm_sq = xp.sum(positions ** 2, axis=1)

    if is_vortex:
        boundary_check_radius_sq = (config.DOMAIN_RADIUS * config.BOUNDARY_WARN_THRESHOLD) ** 2
        escaped_mask = norm_sq > boundary_check_radius_sq
        if xp.any(escaped_mask):
            current_dist_escaped = xp.sqrt(norm_sq[escaped_mask])
            max_dist_escaped = xp.max(current_dist_escaped)
            print(
                f"WARNING: {xp.sum(escaped_mask)} vortices near/past boundary threshold ({config.DOMAIN_RADIUS * config.BOUNDARY_WARN_THRESHOLD:.2f}) at t={current_sim_time:.3f}. Max dist: {max_dist_escaped:.3f}.")

            truly_outside_mask = norm_sq > config.DOMAIN_RADIUS ** 2
            if xp.any(truly_outside_mask):
                indices_truly_outside = xp.where(truly_outside_mask)[0]
                current_dist_truly_outside = xp.sqrt(norm_sq[indices_truly_outside])
                positions[indices_truly_outside] *= (
                            config.DOMAIN_RADIUS * 0.999 / current_dist_truly_outside[:, xp.newaxis])
    else:  # Tracers
        truly_outside_mask = norm_sq > config.DOMAIN_RADIUS ** 2
        if xp.any(truly_outside_mask):
            indices_truly_outside = xp.where(truly_outside_mask)[0]
            current_dist_truly_outside = xp.sqrt(norm_sq[indices_truly_outside])
            positions[indices_truly_outside] *= (
                        config.DOMAIN_RADIUS * 0.9999 / current_dist_truly_outside[:, xp.newaxis])
    return positions


def calculate_angular_impulse(v_positions, v_strengths, xp):
    # (Original function unchanged)
    if v_positions.shape[0] == 0: return xp.array(0.0, dtype=v_positions.dtype if hasattr(v_positions,
                                                                                          'dtype') else np.float64)
    r_sq = xp.sum(v_positions ** 2, axis=1)
    return xp.sum(v_strengths * r_sq)


def calculate_linear_impulse(v_positions, v_strengths, xp):
    # (Original function unchanged)
    default_dtype = v_positions.dtype if hasattr(v_positions, 'dtype') else np.float64
    if v_positions.shape[0] == 0:
        return xp.array(0.0, dtype=default_dtype), xp.array(0.0, dtype=default_dtype)
    P_x = xp.sum(v_strengths * v_positions[:, 1])
    P_y = -xp.sum(v_strengths * v_positions[:, 0])
    return P_x, P_y


def run_simulation(config: SimConfig):
    xp = config.xp

    vortex_pos, vortex_strengths = initialize_vortices(config)
    tracer_pos, tracer_scalar_values = initialize_tracers(config)

    num_steps = int(config.SIMULATION_TIME / config.DT) if config.DT > 0 else 0
    if num_steps <= 0 and config.SIMULATION_TIME > 0:
        print(
            f"Warning: SIMULATION_TIME ({config.SIMULATION_TIME}) or DT ({config.DT}) may result in zero steps. Adjust PLOT_INTERVAL or sim params.")
        num_steps = 0 if config.PLOT_INTERVAL > 0 else 1

    history_pack_template = {
        "tracer_pos": [], "vortex_pos": [], "times": [],
        "Lz": [], "Px": [], "Py": []
    }

    # Store subsampled tracer_scalar_values for animation
    if config._anim_idx is not None and tracer_scalar_values.shape[0] > 0:
        history_pack_template["tracer_scalar_values"] = tracer_scalar_values[config._anim_idx].copy()
    elif tracer_scalar_values.shape[
        0] > 0 and config.ANIM_TRACERS_MAX > 0:  # Render all if not subsampling and ANIM_TRACERS_MAX allows
        history_pack_template["tracer_scalar_values"] = tracer_scalar_values.copy()
    else:  # No tracers or ANIM_TRACERS_MAX is 0
        history_pack_template["tracer_scalar_values"] = xp.array([], dtype=config.float_type)

    history_pack_template["vortex_strengths"] = vortex_strengths

    total_vortex_strength = xp.sum(vortex_strengths)
    initial_Lz = calculate_angular_impulse(vortex_pos, vortex_strengths, xp)
    initial_Px, initial_Py = calculate_linear_impulse(vortex_pos, vortex_strengths, xp)

    history_pack_template["initial_Lz"] = initial_Lz
    history_pack_template["Lz_denom"] = initial_Lz if xp.abs(initial_Lz) > 1e-9 else xp.array(1.0,
                                                                                              dtype=config.float_type)
    history_pack_template["initial_Px"] = initial_Px
    history_pack_template["Px_denom"] = initial_Px if xp.abs(initial_Px) > 1e-9 else xp.array(1.0,
                                                                                              dtype=config.float_type)
    history_pack_template["initial_Py"] = initial_Py
    history_pack_template["Py_denom"] = initial_Py if xp.abs(initial_Py) > 1e-9 else xp.array(1.0,
                                                                                              dtype=config.float_type)

    current_sim_time_val = 0.0

    start_sim_time_wc = time.time()
    for step in range(num_steps + 1):
        if step % config.PLOT_INTERVAL == 0:
            # Store subsampled tracer positions
            if config._anim_idx is not None and tracer_pos.shape[0] > 0:
                history_pack_template["tracer_pos"].append(tracer_pos[config._anim_idx].copy())
            elif tracer_pos.shape[0] > 0 and config.ANIM_TRACERS_MAX > 0:
                history_pack_template["tracer_pos"].append(tracer_pos.copy())
            elif not history_pack_template["tracer_pos"]:  # Append empty if first frame and no tracers to animate
                history_pack_template["tracer_pos"].append(xp.empty((0, 2), dtype=config.float_type))

            history_pack_template["vortex_pos"].append(vortex_pos.copy())
            history_pack_template["times"].append(current_sim_time_val)

            Lz = calculate_angular_impulse(vortex_pos, vortex_strengths, xp)
            Px, Py = calculate_linear_impulse(vortex_pos, vortex_strengths, xp)
            history_pack_template["Lz"].append(Lz)
            history_pack_template["Px"].append(Px)
            history_pack_template["Py"].append(Py)

        if step == num_steps: break

        vortex_pos, tracer_pos = rk4_step_system(vortex_pos, tracer_pos, vortex_strengths, total_vortex_strength,
                                                 config)

        vortex_pos = enforce_boundaries(vortex_pos, config, current_sim_time_val, is_vortex=True)
        tracer_pos = enforce_boundaries(tracer_pos, config, current_sim_time_val, is_vortex=False)

        current_sim_time_val += config.DT

        if step > 0 and history_pack_template["Lz"] and (step % max(1, num_steps // 20) == 0 or step == num_steps - 1):
            Lz_curr = history_pack_template["Lz"][-1]
            rel_Lz_error = xp.abs((Lz_curr - history_pack_template["initial_Lz"]) / history_pack_template["Lz_denom"])
            print(f"Step {step}/{num_steps}, Sim Time: {current_sim_time_val:.2f}s, Rel. ΔLz/Lz₀: {rel_Lz_error:.2e}")

    end_sim_time_wc = time.time()
    print(f"Simulation finished in {end_sim_time_wc - start_sim_time_wc:.2f} seconds (wall clock).")

    # Convert CuPy arrays to NumPy
    if config.GPU_ENABLED:
        for key, val in history_pack_template.items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], cp.ndarray):
                history_pack_template[key] = [cp.asnumpy(arr) for arr in val]
            elif isinstance(val, cp.ndarray):
                history_pack_template[key] = cp.asnumpy(val)
            elif hasattr(val, 'get'):
                history_pack_template[key] = np.array(val.get())

    scalar_keys = ["initial_Lz", "Lz_denom", "initial_Px", "Px_denom", "initial_Py", "Py_denom"]
    for key in scalar_keys:
        if key in history_pack_template:
            current_val = history_pack_template[key]
            if isinstance(current_val, (cp.ndarray, np.ndarray)) and current_val.ndim == 0:
                history_pack_template[key] = current_val.item()
            elif hasattr(current_val, 'get'):
                history_pack_template[key] = current_val.get()

    history_pack_template["times"] = np.array(history_pack_template["times"], dtype=np.float64)
    for key in ["Lz", "Px", "Py"]:
        if history_pack_template[key]:
            history_pack_template[key] = np.array(history_pack_template[key],
                                                  dtype=np.float64)  # Ensure consistent dtype for plotting
        else:
            history_pack_template[key] = np.array([], dtype=np.float64)

    # Pre-calculate tracer speeds for animation if mode is "speed"
    # Ensure this happens after converting tracer_pos to list of NumPy arrays
    if config.TRACER_COLORING_MODE == "speed" and \
            (config.N_TRACERS > 0 and config.ANIM_TRACERS_MAX > 0) and \
            len(history_pack_template.get("tracer_pos", [])) > 1:
        print("Pre-calculating tracer speeds for animation normalization...")

        if not history_pack_template["tracer_pos"] or not isinstance(history_pack_template["tracer_pos"][0],
                                                                     np.ndarray):
            print("Warning: Tracer position history is not in the expected format for speed calculation.")
            history_pack_template["precalculated_tracer_speeds_norm"] = None
            history_pack_template["global_max_speed_for_anim"] = 1.0
        else:
            try:
                all_tracer_pos_np = np.stack(history_pack_template["tracer_pos"])  # (frames, tracers_shown, 2)
                times_np = history_pack_template["times"]  # Already a NumPy array

                num_hist_frames, num_tracers_shown = all_tracer_pos_np.shape[0], all_tracer_pos_np.shape[1]
                all_tracer_speeds_norm = np.zeros((num_hist_frames, num_tracers_shown), dtype=np.float32)
                peak_speed_val = 0.0

                if num_hist_frames > 1 and num_tracers_shown > 0:
                    displacements = np.diff(all_tracer_pos_np, axis=0)  # (frames-1, tracers_shown, 2)
                    dt_frames = np.diff(times_np)  # (frames-1)

                    speeds_mag = np.zeros((num_hist_frames - 1, num_tracers_shown), dtype=np.float32)
                    valid_dt_mask = dt_frames > 1e-9  # Avoid division by zero

                    if np.any(valid_dt_mask):
                        # Ensure dt_frames[valid_dt_mask, np.newaxis] broadcasts correctly
                        # speeds_mag needs to be indexed correctly if not all dt are valid.
                        # Iterate for simplicity if sparse valid_dt:
                        for f_idx in range(num_hist_frames - 1):
                            if valid_dt_mask[f_idx]:
                                speeds_mag[f_idx, :] = np.linalg.norm(displacements[f_idx], axis=-1) / dt_frames[f_idx]

                    if speeds_mag.size > 0: peak_speed_val = np.max(speeds_mag)

                history_pack_template["global_max_speed_for_anim"] = max(peak_speed_val, 1e-8)
                print(
                    f"Peak tracer speed for animation normalization: {history_pack_template['global_max_speed_for_anim']:.3e}")

                if num_hist_frames > 1 and num_tracers_shown > 0 and speeds_mag.size > 0:
                    all_tracer_speeds_norm[1:] = np.clip(
                        speeds_mag / history_pack_template["global_max_speed_for_anim"], 0.0, 1.0)

                history_pack_template["precalculated_tracer_speeds_norm"] = all_tracer_speeds_norm
            except ValueError as e:
                print(
                    f"Warning: Could not stack tracer positions for speed pre-calculation: {e}. Speed coloring might be affected.")
                history_pack_template["precalculated_tracer_speeds_norm"] = None
                history_pack_template["global_max_speed_for_anim"] = 1.0
            except Exception as e_gen:
                print(f"An unexpected error occurred during speed pre-calculation: {e_gen}")
                history_pack_template["precalculated_tracer_speeds_norm"] = None
                history_pack_template["global_max_speed_for_anim"] = 1.0
    else:  # Conditions for speed pre-calculation not met
        history_pack_template["precalculated_tracer_speeds_norm"] = None
        history_pack_template["global_max_speed_for_anim"] = 1.0

    return history_pack_template


# --- Animation Logic ---
def animate(data_pack, config: SimConfig):
    num_frames = len(data_pack.get("tracer_pos", []))
    if config.N_VORTICES > 0:
        num_frames_vortex = len(data_pack.get("vortex_pos", []))
        if num_frames == 0 and num_frames_vortex > 0:
            num_frames = num_frames_vortex
        elif num_frames_vortex == 0 and num_frames > 0:
            pass  # Use tracer frame count
        elif num_frames_vortex != num_frames and num_frames_vortex > 0 and num_frames > 0:
            print(
                f"Warning: Mismatch in tracer ({num_frames}) and vortex ({num_frames_vortex}) frame counts. Using smaller: {min(num_frames, num_frames_vortex)}")
            num_frames = min(num_frames, num_frames_vortex)

    if num_frames == 0:
        print("No history data to animate. Check SIMULATION_TIME, DT, and PLOT_INTERVAL.")
        return

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[12, 1, 1, 0.5], hspace=0.3)

    ax_main = fig.add_subplot(gs[0])
    ax_Lz = fig.add_subplot(gs[1])
    ax_P = fig.add_subplot(gs[2])
    ax_info = fig.add_subplot(gs[3]);
    ax_info.axis('off')

    fig.patch.set_facecolor(config.FIGURE_BG_COLOR)
    ax_main.set_facecolor(config.AXES_BG_COLOR)
    ax_main.set_aspect('equal')
    ax_main.set_xlim(-config.DOMAIN_RADIUS * 1.02, config.DOMAIN_RADIUS * 1.02)
    ax_main.set_ylim(-config.DOMAIN_RADIUS * 1.02, config.DOMAIN_RADIUS * 1.02)

    # Updated title to reflect shown tracers
    num_tracers_shown_in_anim = 0
    if data_pack["tracer_pos"] and len(data_pack["tracer_pos"][0]) > 0:
        num_tracers_shown_in_anim = data_pack["tracer_pos"][0].shape[0]

    title_str = (f"2D Point Vortex Dynamics ({config.N_VORTICES}V, {num_tracers_shown_in_anim}T, "
                 f"Mode: {config.TRACER_COLORING_MODE})")
    ax_main.set_title(title_str, color='white', fontsize=14)
    ax_main.set_xticks([]);
    ax_main.set_yticks([])
    for spine in ax_main.spines.values(): spine.set_edgecolor('gray')

    domain_circle = plt.Circle((0, 0), config.DOMAIN_RADIUS, color='gray', fill=False, ls='-', lw=0.8, alpha=0.6)
    ax_main.add_artist(domain_circle)
    tracer_cmap = plt.cm.get_cmap(config.TRACER_CMAP)

    initial_tracer_positions_np = np.empty((0, 2), dtype=config.float_type)  # Use config.float_type which is resolved
    if num_tracers_shown_in_anim > 0 and data_pack["tracer_pos"]:
        initial_tracer_positions_np = np.asarray(data_pack["tracer_pos"][0])

    # global_max_speed calculation is now done in run_simulation and stored in data_pack

    initial_tracer_scalar_values_np = np.asarray(
        data_pack.get("tracer_scalar_values", np.empty(0, dtype=config.float_type)))
    if initial_tracer_scalar_values_np.ndim == 0 and num_tracers_shown_in_anim > 0:
        initial_tracer_scalar_values_np = np.full(num_tracers_shown_in_anim, initial_tracer_scalar_values_np.item(),
                                                  dtype=config.float_type)

    scatter_layers = []
    if num_tracers_shown_in_anim > 0 and initial_tracer_positions_np.shape[0] > 0:
        norm_scalar_values = initial_tracer_scalar_values_np
        if config.TRACER_COLORING_MODE != "speed" and norm_scalar_values.size > 0:
            min_val, max_val = np.min(norm_scalar_values), np.max(norm_scalar_values)
            if not (0 <= min_val <= 1 and 0 <= max_val <= 1 and (max_val - min_val > 1e-6 or min_val == max_val)):
                if max_val > min_val + 1e-6:  # Add epsilon for robustness
                    norm_scalar_values = (norm_scalar_values - min_val) / (max_val - min_val)
                else:
                    norm_scalar_values = np.full_like(norm_scalar_values, 0.5)

        scatter_colors_mapped = tracer_cmap(norm_scalar_values if norm_scalar_values.size > 0 else 0.5)

        # Conditional glow layers
        draw_glow = num_tracers_shown_in_anim <= config.ANIM_GLOW_MAX
        if draw_glow:
            for size_mult, alpha_mult in reversed(config.TRACER_GLOW_LAYERS):
                glow_scatter = ax_main.scatter([], [],
                                               s=config.TRACER_PARTICLE_SIZE * size_mult,
                                               marker='o', edgecolors='none',
                                               alpha=config.TRACER_ALPHA * alpha_mult, zorder=1)
                scatter_layers.append(glow_scatter)

        main_scatter = ax_main.scatter([], [],
                                       s=config.TRACER_PARTICLE_SIZE,
                                       marker='o', edgecolors='none', alpha=config.TRACER_ALPHA, zorder=2)
        scatter_layers.append(main_scatter)  # Always add main scatter layer

        for scat_layer in scatter_layers:
            scat_layer.set_offsets(initial_tracer_positions_np)
            scat_layer.set_facecolors(scatter_colors_mapped)

    vortex_scatter = None
    if config.N_VORTICES > 0 and data_pack["vortex_pos"] and data_pack["vortex_pos"][0].shape[0] > 0:
        v_strengths_np = np.asarray(data_pack["vortex_strengths"])
        vortex_colors = [config.VORTEX_COLOR_POS if s > 0 else config.VORTEX_COLOR_NEG for s in v_strengths_np]

        max_abs_strength = np.max(np.abs(v_strengths_np)) if len(v_strengths_np) > 0 else 1.0
        if max_abs_strength < 1e-9: max_abs_strength = 1.0
        vortex_sizes = config.VORTEX_MARKER_SIZE_SCALE * np.abs(
            v_strengths_np) / max_abs_strength + config.VORTEX_MARKER_SIZE_BASE

        initial_vortex_positions_np = np.asarray(data_pack["vortex_pos"][0])
        vortex_scatter = ax_main.scatter(initial_vortex_positions_np[:, 0], initial_vortex_positions_np[:, 1],
                                         s=vortex_sizes, c=vortex_colors,
                                         edgecolors='black', linewidths=0.5, zorder=10, alpha=0.9)
    if vortex_scatter is None:
        vortex_scatter = ax_main.scatter([], [], s=[], c=[])  # Dummy

    time_text = ax_main.text(0.02, 0.95, '', transform=ax_main.transAxes, color='white', fontsize=10)

    ax_Lz.set_xlim(0, max(config.SIMULATION_TIME,
                          data_pack["times"][-1] if data_pack["times"].size > 0 else config.SIMULATION_TIME))
    Lz_hist_np = data_pack.get("Lz", np.array([]))
    Lz0_np = data_pack.get("initial_Lz", 0.0)
    Lz_denom_np = data_pack.get("Lz_denom", 1.0)

    if abs(Lz_denom_np) > 1e-9 and Lz_hist_np.size > 0:
        rel_err_Lz = (Lz_hist_np - Lz0_np) / Lz_denom_np
        max_abs_rel_err = np.max(np.abs(rel_err_Lz)) if rel_err_Lz.size > 0 else 1e-5
        plot_Lz_y_limit = max(1e-5, max_abs_rel_err * 1.2)
        ax_Lz.set_ylim(-plot_Lz_y_limit, plot_Lz_y_limit)
        ax_Lz.set_ylabel("Rel. ΔLz/Lz₀", color='lightgray', fontsize=9)
    else:
        ax_Lz.set_ylabel("Lz (abs)", color='lightgray', fontsize=9)
        if Lz_hist_np.size > 0:
            min_Lz, max_Lz = np.min(Lz_hist_np), np.max(Lz_hist_np)
            margin = (max_Lz - min_Lz) * 0.1 + 1e-5
            ax_Lz.set_ylim(min_Lz - margin, max_Lz + margin)
        else:
            ax_Lz.set_ylim(-1, 1)
    ax_Lz.tick_params(axis='both', colors='lightgray', labelsize=8);
    ax_Lz.set_facecolor(config.AXES_BG_COLOR);
    [s.set_edgecolor('gray') for s in ax_Lz.spines.values()]
    line_Lz, = ax_Lz.plot([], [], lw=1.2, color='#FFD700')

    ax_P.set_xlim(0, max(config.SIMULATION_TIME,
                         data_pack["times"][-1] if data_pack["times"].size > 0 else config.SIMULATION_TIME))
    ax_P.set_ylabel("Px, Py (abs)", color='lightgray', fontsize=9)
    ax_P.tick_params(axis='both', colors='lightgray', labelsize=8);
    ax_P.set_facecolor(config.AXES_BG_COLOR);
    [s.set_edgecolor('gray') for s in ax_P.spines.values()]
    line_Px, = ax_P.plot([], [], lw=1.2, color='#00FF00', label='Px')
    line_Py, = ax_P.plot([], [], lw=1.2, color='#FF00FF', label='Py')
    ax_P.legend(fontsize='x-small', facecolor=config.AXES_BG_COLOR, edgecolor='gray', labelcolor='lightgray',
                loc='upper right')
    all_P_values = np.concatenate((data_pack.get("Px", np.array([])), data_pack.get("Py", np.array([]))))
    if all_P_values.size > 0:
        min_P, max_P = np.min(all_P_values), np.max(all_P_values)
        margin = (max_P - min_P) * 0.1 + 1e-5 if (max_P - min_P) > 1e-9 else 1e-5
        ax_P.set_ylim(min_P - margin, max_P + margin)
    else:
        ax_P.set_ylim(-1, 1)

    gpu_info_text = f"GPU: {'ON' if config.GPU_ENABLED else 'OFF'} ({config.xp.__name__}{' Numba' if not config.GPU_ENABLED and _NUMBA_AVAILABLE else ''})"
    info_str = f"DT:{config.DT:.1e} SimT:{config.SIMULATION_TIME:.1f}s {gpu_info_text} Seed:{config.RANDOM_SEED}"
    ax_info.text(0.5, 0.5, info_str, color='lightgray', ha='center', va='center', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    update_elements = []
    if num_tracers_shown_in_anim > 0 and scatter_layers: update_elements.extend(scatter_layers)
    if config.N_VORTICES > 0: update_elements.append(vortex_scatter)  # vortex_scatter is now dummy if no vortices
    update_elements.extend([time_text, line_Lz, line_Px, line_Py])

    def update(frame_idx):
        if num_tracers_shown_in_anim > 0 and data_pack["tracer_pos"] and frame_idx < len(data_pack["tracer_pos"]):
            frame_tracer_pos = data_pack["tracer_pos"][frame_idx]
            if frame_tracer_pos.size > 0:
                if config.TRACER_COLORING_MODE == "speed":
                    precalc_speeds = data_pack.get("precalculated_tracer_speeds_norm")
                    if precalc_speeds is not None and frame_idx < precalc_speeds.shape[0]:
                        current_speed_norm = precalc_speeds[frame_idx]
                    else:
                        current_speed_norm = np.zeros(frame_tracer_pos.shape[0], dtype=np.float32)

                    current_scatter_colors = tracer_cmap(current_speed_norm)
                    for scat in scatter_layers:
                        scat.set_offsets(frame_tracer_pos)
                        scat.set_facecolors(current_scatter_colors)
                else:
                    for scat in scatter_layers:
                        scat.set_offsets(frame_tracer_pos)

        if config.N_VORTICES > 0 and data_pack["vortex_pos"] and frame_idx < len(data_pack["vortex_pos"]):
            frame_vortex_pos = data_pack["vortex_pos"][frame_idx]
            if frame_vortex_pos.shape[0] > 0:
                vortex_scatter.set_offsets(frame_vortex_pos)

        current_times_np = data_pack["times"][:frame_idx + 1]
        if current_times_np.size > 0:
            time_text.set_text(f"Time: {current_times_np[-1]:.2f}s")

        current_Lz_values = Lz_hist_np[:frame_idx + 1]
        if current_Lz_values.size > 0:
            if abs(Lz_denom_np) > 1e-9:
                line_Lz.set_data(current_times_np, (current_Lz_values - Lz0_np) / Lz_denom_np)
            else:
                line_Lz.set_data(current_times_np, current_Lz_values)

        current_Px_values = data_pack.get("Px", np.array([]))[:frame_idx + 1]
        current_Py_values = data_pack.get("Py", np.array([]))[:frame_idx + 1]
        if current_Px_values.size > 0: line_Px.set_data(current_times_np, current_Px_values)
        if current_Py_values.size > 0: line_Py.set_data(current_times_np, current_Py_values)

        if frame_idx > 0 and frame_idx % 50 == 0: print(f"Animating frame {frame_idx + 1}/{num_frames}")
        return update_elements

    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  interval=max(1, 1000 // config.FPS), blit=True)

    print(f"Saving animation to {config.OUTPUT_FILENAME} (FFmpeg: {config.FFMPEG_CODEC}, {config.FPS}fps)...")
    start_save_time = time.time()

    writer_extra_args = ["-pix_fmt", "yuv420p", "-preset", config.FFMPEG_PRESET]
    if config.FFMPEG_CODEC in {"h264_nvenc", "hevc_nvenc"}:
        writer_extra_args.extend(["-rc", "vbr", "-cq", str(config.FFMPEG_CQ)])
    else:
        writer_extra_args.extend(["-crf", str(config.FFMPEG_CRF)])

    # Set FFmpeg threads
    actual_ffmpeg_threads = config.FFMPEG_THREADS
    if actual_ffmpeg_threads == 0:  # If 0, pick a sensible default
        try:
            cpu_count = os.cpu_count()
            if cpu_count:
                actual_ffmpeg_threads = cpu_count
            else:  # Fallback if os.cpu_count() is None
                actual_ffmpeg_threads = 4
        except NotImplementedError:
            actual_ffmpeg_threads = 4  # Fallback
        print(f"FFMPEG_THREADS was 0, automatically set to {actual_ffmpeg_threads}")

    writer_extra_args.extend(["-threads", str(actual_ffmpeg_threads)])

    writer = animation.FFMpegWriter(fps=config.FPS, codec=config.FFMPEG_CODEC,
                                    metadata=dict(artist='VortexSim'),
                                    extra_args=writer_extra_args)
    try:
        ani.save(config.OUTPUT_FILENAME, writer=writer, dpi=config.DPI)
        end_save_time = time.time()
        print(f"Animation saved in {end_save_time - start_save_time:.2f} seconds.")
    except FileNotFoundError:
        print("ERROR: FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH.")
    except Exception as e:
        print(f"ERROR: Failed to save animation: {e}")
        print(
            f"FFmpeg command arguments used (approximate): ffmpeg -y -fps {config.FPS} -i pipe:0 {' '.join(writer_extra_args)} {config.OUTPUT_FILENAME}")

    plt.close(fig)


# --- Argument Parsing and Main Execution ---
def parse_glow_layers_arg(arg_string: str) -> List[Tuple[float, float]]:
    layers = []
    if not arg_string.strip(): return []
    try:
        pairs = arg_string.split(';')
        for pair_str in pairs:
            if not pair_str.strip(): continue
            values = pair_str.split(',')
            if len(values) != 2:
                raise ValueError("Each layer must have two float values (size_mult, alpha_mult).")
            layers.append((float(values[0].strip()), float(values[1].strip())))
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid format for --tracer-glow-layers: '{arg_string}'. "
            f"Expected 'size1,alpha1;size2,alpha2;...'. Error: {e}"
        )
    return layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2D Point Vortex Dynamics Simulation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    config_defaults = SimConfig()
    use_boolean_optional_action = hasattr(argparse, 'BooleanOptionalAction')

    for f_info in fields(SimConfig):
        if not f_info.init: continue

        arg_name = f"--{f_info.name.lower().replace('_', '-')}"
        default_val = getattr(config_defaults, f_info.name)

        kwargs = {"dest": f_info.name, "default": default_val}

        current_help_default_str = str(default_val)
        if f_info.name == "TRACER_GLOW_LAYERS" and isinstance(default_val, list):
            current_help_default_str = ';'.join([f'{s},{a}' for s, a in default_val]) if default_val else "None"

        kwargs["help"] = f"(Default: {current_help_default_str})"

        actual_type = f_info.type
        is_optional = False
        if hasattr(actual_type, '__origin__') and actual_type.__origin__ is Union:
            type_args = [t for t in actual_type.__args__ if t is not type(None)]
            if len(type_args) == 1 and type(None) in actual_type.__args__:
                actual_type = type_args[0]
                is_optional = True

        if actual_type == bool:
            if use_boolean_optional_action:
                kwargs["action"] = argparse.BooleanOptionalAction
            else:
                kwargs["type"] = lambda x: str(x).lower() == 'true'
        elif f_info.name == "TRACER_GLOW_LAYERS":
            kwargs["type"] = parse_glow_layers_arg
        elif is_optional and actual_type == int:
            # Handles Optional[int] like RANDOM_SEED, GPU_DEVICE_ID
            def optional_int_type(x):
                if str(x).lower() in ['none', 'null', '']: return None
                return int(x)


            kwargs["type"] = optional_int_type
        elif actual_type in (int, float, str):
            kwargs["type"] = actual_type
        else:
            # This path should ideally not be taken for SimConfig fields meant to be CLI configurable.
            # For List[Tuple[...]] it's handled by parse_glow_layers_arg.
            # Other complex types would need custom parsers or not be CLI args.
            print(
                f"Note: Argument for '{f_info.name}' of type {f_info.type} uses string parsing by default if not handled.")
            kwargs["type"] = str

        parser.add_argument(arg_name, **kwargs)

    args = parser.parse_args()

    try:
        sim_config_from_args = SimConfig(**vars(args))
    except Exception as e:
        print(f"Error creating simulation configuration from arguments: {e}")
        sys.exit(1)

    print(f"--- Configuration ---")
    for key, value in vars(sim_config_from_args).items():
        if key in ["xp", "rng", "float_type", "_anim_idx",
                   "_lamb_oseen_factor_func", "_get_velocities_induced_by_vortices_func",
                   "get_vortex_velocities_func"]: continue
        print(f"  {key}: {value}")
    print(f"  Derived xp: {sim_config_from_args.xp.__name__}")
    print(f"  Derived float_type: {sim_config_from_args.float_type}")
    print(f"---------------------")

    simulation_data_package = run_simulation(sim_config_from_args)

    # Check if there are frames to animate based on tracer_pos list content
    # (which now stores subsampled data or empty arrays if ANIM_TRACERS_MAX is 0)
    can_animate = False
    if simulation_data_package["times"].size > 1:
        if simulation_data_package.get("tracer_pos") and \
                len(simulation_data_package["tracer_pos"]) > 0 and \
                simulation_data_package["tracer_pos"][0].shape[0] > 0:  # Check if first frame has tracers
            can_animate = True
        elif sim_config_from_args.N_VORTICES > 0 and \
                simulation_data_package.get("vortex_pos") and \
                len(simulation_data_package["vortex_pos"]) > 0 and \
                simulation_data_package["vortex_pos"][0].shape[0] > 0:  # Or if vortices exist to animate
            can_animate = True

    if can_animate:
        print("Starting animation process...")
        animate(simulation_data_package, sim_config_from_args)
    else:
        num_frames_generated = simulation_data_package["times"].size
        min_frames_needed = 2
        print(f"Simulation generated {num_frames_generated} frame(s) of data.")
        if not (simulation_data_package.get("tracer_pos") and \
                len(simulation_data_package["tracer_pos"]) > 0 and \
                simulation_data_package["tracer_pos"][0].shape[0] > 0) and \
                not (sim_config_from_args.N_VORTICES > 0 and \
                     simulation_data_package.get("vortex_pos") and \
                     len(simulation_data_package["vortex_pos"]) > 0 and \
                     simulation_data_package["vortex_pos"][0].shape[0] > 0):
            print(
                "No tracers or vortices were configured to be shown in the animation (check ANIM_TRACERS_MAX or N_VORTICES).")

        if sim_config_from_args.DT > 0 and num_frames_generated < min_frames_needed:
            expected_steps = sim_config_from_args.SIMULATION_TIME / sim_config_from_args.DT
            if sim_config_from_args.PLOT_INTERVAL > expected_steps and expected_steps > 0:
                print(f"Warning: PLOT_INTERVAL ({sim_config_from_args.PLOT_INTERVAL}) is higher than "
                      f"total simulation steps ({expected_steps:.0f}). Not enough data points saved.")
        elif sim_config_from_args.DT <= 0:
            print(f"Warning: DT is {sim_config_from_args.DT}, which may lead to no simulation steps.")

        if num_frames_generated <= 1:
            print(f"Not enough data points to animate (at least {min_frames_needed} needed based on PLOT_INTERVAL).")

    print("Script finished.")
