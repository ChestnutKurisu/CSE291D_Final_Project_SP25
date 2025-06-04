import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional
import os

# --- Matplotlib Styling ---
# plt.style.use('seaborn-v0_8-darkgrid') # A good base style
plt.rcParams['font.family'] = 'serif'
# Try to use Times or a common serif font
try:
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
except: # Fallback if Times New Roman is not found
    pass
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

plt.style.use('seaborn-v0_8-whitegrid')      # light theme
plt.rcParams['figure.facecolor'] = 'white'   # ensure white figures
plt.rcParams['axes.facecolor']   = 'white'   # ensure white axes
plt.rcParams['savefig.facecolor']= 'white'   # exported files stay white

# --- Custom Colormap Registration (from your main script) ---
def register_plume_colormap() -> None:
    plume_rgb = [
        "#001428", "#06336a", "#135996", "#1b7cb4",
        "#2ca4c9", "#6ad3d7", "#b9f0e2", "#fdfecb",
    ]
    cmap = LinearSegmentedColormap.from_list("plume", plume_rgb, N=256)
    if hasattr(plt.colormaps, "register"):
        if "plume" not in plt.colormaps:
            plt.colormaps.register(cmap)
    elif hasattr(plt.cm, "register_cmap"):
        try:
            plt.cm.register_cmap(name="plume", cmap=cmap)
        except ValueError:
            pass
register_plume_colormap()

# --- Simplified SimConfig for Report Generation ---
@dataclass
class ReportSimConfig:
    N_VORTICES: int = 20
    N_TRACERS: int = 1_300_000 # Reduced for static initial plot
    DOMAIN_RADIUS: float = 1.0
    SIMULATION_TIME: float = 10.0 # Short simulation for impulse plots
    DT: float = 0.002
    PLOT_INTERVAL: int = 1 # Store every step for smooth impulse plots

    VORTEX_CORE_A_SQ: float = 0.001
    TRACER_CORE_A_SQ: float = 0.0005 # Used for initial_config plot if it were dynamic

    RANDOM_SEED: Optional[int] = 42
    NUM_TRACER_GROUPS: int = 5
    TRACER_COLORING_MODE: str = "group" # For initial_config plot

    # Fixed to NumPy for this script
    xp: Any = np
    float_type: Any = np.float32
    rng: Any = field(init=False)

    # For direct calls, not dynamic dispatch like in main sim
    # These would be set if we were using the full __post_init__ logic
    _lamb_oseen_factor_func: Any = field(init=False)
    _get_velocities_induced_by_vortices_func: Any = field(init=False)
    get_vortex_velocities_func: Any = field(init=False)

    def __post_init__(self):
        if self.RANDOM_SEED is not None:
            self.rng = np.random.default_rng(self.RANDOM_SEED)
        else:
            self.rng = np.random.default_rng()
        
        # Point to the NumPy versions of functions directly
        self._lamb_oseen_factor_func = _lamb_oseen_factor_np
        self._get_velocities_induced_by_vortices_func = _get_velocities_induced_by_vortices_np
        self.get_vortex_velocities_func = get_vortex_velocities_np

# --- Core Physics/Math Functions (NumPy versions adapted from your main script) ---

def _lamb_oseen_factor_np(r_sq, core_a_sq, xp_dummy, float_type_dummy): # xp and float_type are fixed to np
    epsilon = 1e-7 # For float32
    r_sq_safe = np.where(r_sq < epsilon, epsilon, r_sq)
    val = (1.0 - np.exp(-r_sq_safe / core_a_sq)) / r_sq_safe
    limit_val = 1.0 / core_a_sq
    return np.where(r_sq < epsilon * 10, limit_val, val)

def _get_velocities_induced_by_vortices_np(target_positions, vortex_positions, vortex_strengths,
                                           core_a_sq, config: ReportSimConfig, total_vortex_strength_for_bg_flow):
    xp = np # Explicitly NumPy
    M = target_positions.shape[0]
    N = vortex_positions.shape[0]

    if N == 0 or M == 0:
        return xp.zeros_like(target_positions)

    velocities = xp.zeros_like(target_positions, dtype=config.float_type)
    
    norm_sq_all_vortices = xp.sum(vortex_positions**2, axis=1)
    epsilon_norm_sq = 1e-6 # For float32
    norm_sq_all_vortices_safe = xp.where(norm_sq_all_vortices < epsilon_norm_sq, epsilon_norm_sq, norm_sq_all_vortices)
    
    img_v_positions = (config.DOMAIN_RADIUS**2 / norm_sq_all_vortices_safe[:, xp.newaxis]) * vortex_positions
    img_v_strengths = -vortex_strengths
    
    target_pos_exp = target_positions[:, xp.newaxis, :]
    
    v_pos_exp = vortex_positions[xp.newaxis, :, :]
    diff_real = target_pos_exp - v_pos_exp
    r_sq_real = xp.sum(diff_real**2, axis=2)
    
    interaction_factor_real = config._lamb_oseen_factor_func(r_sq_real, core_a_sq, xp, config.float_type)
    coeff_real = vortex_strengths[xp.newaxis, :] / (2 * xp.pi)
    term_real = coeff_real * interaction_factor_real

    velocities[:, 0] += xp.sum(-term_real * diff_real[:, :, 1], axis=1)
    velocities[:, 1] += xp.sum( term_real * diff_real[:, :, 0], axis=1)

    img_v_pos_exp = img_v_positions[xp.newaxis, :, :]
    diff_img = target_pos_exp - img_v_pos_exp
    r_sq_img = xp.sum(diff_img**2, axis=2)
    
    interaction_factor_img = config._lamb_oseen_factor_func(r_sq_img, core_a_sq, xp, config.float_type)
    coeff_img = img_v_strengths[xp.newaxis, :] / (2 * xp.pi)
    term_img = coeff_img * interaction_factor_img

    velocities[:, 0] += xp.sum(-term_img * diff_img[:, :, 1], axis=1)
    velocities[:, 1] += xp.sum( term_img * diff_img[:, :, 0], axis=1)
    
    if xp.abs(total_vortex_strength_for_bg_flow) > epsilon_norm_sq:
        K_bg = total_vortex_strength_for_bg_flow / (2 * xp.pi * config.DOMAIN_RADIUS**2)
        velocities[:, 0] += -K_bg * target_positions[:, 1]
        velocities[:, 1] +=  K_bg * target_positions[:, 0]
        
    return velocities

def get_vortex_velocities_np(v_positions, v_strengths, config: ReportSimConfig, total_vortex_strength):
    xp = np # Explicitly NumPy
    N = v_positions.shape[0]
    if N == 0: return xp.zeros_like(v_positions)

    v_pos_i = v_positions[:, xp.newaxis, :]
    v_pos_j = v_positions[xp.newaxis, :, :]

    diff_real = v_pos_i - v_pos_j
    r_sq_real = xp.sum(diff_real**2, axis=2)
    identity_mask = xp.eye(N, dtype=bool)

    interaction_factor_real = config._lamb_oseen_factor_func(r_sq_real, config.VORTEX_CORE_A_SQ, xp, config.float_type)
    coeff_real = v_strengths[xp.newaxis, :] / (2 * xp.pi)
    term_real_masked = coeff_real * interaction_factor_real
    term_real_masked = xp.where(identity_mask, 0.0, term_real_masked)

    velocities = xp.zeros_like(v_positions, dtype=config.float_type)
    velocities[:, 0] = xp.sum(-term_real_masked * diff_real[:, :, 1], axis=1)
    velocities[:, 1] = xp.sum( term_real_masked * diff_real[:, :, 0], axis=1)

    norm_sq_all = xp.sum(v_positions**2, axis=1)
    eps_norm_sq = 1e-6 # For float32
    norm_sq_safe = xp.where(norm_sq_all < eps_norm_sq, eps_norm_sq, norm_sq_all)
    
    img_pos_j_denom = norm_sq_safe[xp.newaxis, :, xp.newaxis]
    img_pos_j_denom = xp.where(img_pos_j_denom == 0, eps_norm_sq, img_pos_j_denom)

    img_pos_j = ((config.DOMAIN_RADIUS ** 2) / img_pos_j_denom) * v_pos_j
    img_str_j = -v_strengths[xp.newaxis, :]

    diff_img = v_pos_i - img_pos_j
    r_sq_img = xp.sum(diff_img**2, axis=2)

    interaction_factor_img = config._lamb_oseen_factor_func(r_sq_img, config.VORTEX_CORE_A_SQ, xp, config.float_type)
    coeff_img = img_str_j / (2 * xp.pi)
    term_img = coeff_img * interaction_factor_img

    velocities[:, 0] += xp.sum(-term_img * diff_img[:, :, 1], axis=1)
    velocities[:, 1] += xp.sum( term_img * diff_img[:, :, 0], axis=1)

    if xp.abs(total_vortex_strength) > eps_norm_sq:
        K_bg = total_vortex_strength / (2 * xp.pi * config.DOMAIN_RADIUS**2)
        velocities[:, 0] += -K_bg * v_positions[:, 1]
        velocities[:, 1] +=  K_bg * v_positions[:, 0]
    return velocities

def initialize_vortices(config: ReportSimConfig):
    xp = config.xp
    positions = xp.zeros((config.N_VORTICES, 2), dtype=config.float_type)
    strengths = xp.zeros(config.N_VORTICES, dtype=config.float_type)
    if config.N_VORTICES == 0: return positions, strengths

    if config.N_VORTICES >= 4:
        s = 0.4 * config.DOMAIN_RADIUS
        positions[0] = xp.array([-s,  s], dtype=config.float_type); strengths[0] = 1.5
        positions[1] = xp.array([ s,  s], dtype=config.float_type); strengths[1] = -1.5
        positions[2] = xp.array([-s, -s], dtype=config.float_type); strengths[2] = -1.5
        positions[3] = xp.array([ s, -s], dtype=config.float_type); strengths[3] = 1.5
        if config.N_VORTICES > 4:
            num_remaining = config.N_VORTICES - 4
            radii = config.rng.uniform(0.1, 0.7, num_remaining).astype(config.float_type) * config.DOMAIN_RADIUS
            angles = config.rng.uniform(0, 2 * xp.pi, num_remaining).astype(config.float_type)
            positions[4:, 0] = radii * xp.cos(angles)
            positions[4:, 1] = radii * xp.sin(angles)
            rand_strengths_choices = xp.array([-0.75, 0.75, -0.5, 0.5], dtype=config.float_type)
            chosen_indices = config.rng.integers(0, len(rand_strengths_choices), num_remaining)
            strengths[4:] = rand_strengths_choices[chosen_indices] * config.rng.uniform(0.5, 1.0, num_remaining).astype(config.float_type)
    else: # Fewer than 4 vortices
        radii = config.rng.uniform(0.1, 0.7, config.N_VORTICES).astype(config.float_type) * config.DOMAIN_RADIUS
        angles = config.rng.uniform(0, 2 * xp.pi, config.N_VORTICES).astype(config.float_type)
        positions[:, 0] = radii * xp.cos(angles)
        positions[:, 1] = radii * xp.sin(angles)
        rand_strengths_choices = xp.array([-1.0, 1.0], dtype=config.float_type)
        chosen_indices = config.rng.integers(0, len(rand_strengths_choices), config.N_VORTICES)
        strengths[:] = rand_strengths_choices[chosen_indices] * config.rng.uniform(0.5, 1.5, config.N_VORTICES).astype(config.float_type)
    return positions, strengths

def initialize_tracers(config: ReportSimConfig):
    xp = config.xp
    tracer_pos = xp.zeros((config.N_TRACERS, 2), dtype=config.float_type)
    tracer_scalar_values = xp.zeros(config.N_TRACERS, dtype=config.float_type)
    if config.N_TRACERS == 0: return tracer_pos, tracer_scalar_values

    if config.TRACER_COLORING_MODE == "group":
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
            r_vals = patch_radius_base * xp.sqrt(r_sqrt_uniform) # r_vals, not r
            theta = config.rng.uniform(0, 2 * xp.pi, num_in_patch).astype(config.float_type)
            start, end = current_idx, current_idx + num_in_patch
            tracer_pos[start:end, 0] = center_x + r_vals * xp.cos(theta)
            tracer_pos[start:end, 1] = center_y + r_vals * xp.sin(theta)
            tracer_scalar_values[start:end] = (i + 0.5) / num_groups
            current_idx += num_in_patch
    # Other modes not strictly needed for these specific figures, but good to have
    elif config.TRACER_COLORING_MODE == "scalar":
        max_r_init = config.DOMAIN_RADIUS * 0.70
        r_sqrt_uniform = config.rng.uniform(0, 1, config.N_TRACERS).astype(config.float_type)
        radii  = max_r_init * xp.sqrt(r_sqrt_uniform)
        angles = config.rng.uniform(0, 2*xp.pi, config.N_TRACERS).astype(config.float_type)
        tracer_pos[:, 0] = radii * xp.cos(angles)
        tracer_pos[:, 1] = radii * xp.sin(angles)
        tracer_scalar_values[:] = 1.0 - (radii / max_r_init) # high scalar in the core
    
    dist_sq = xp.sum(tracer_pos**2, axis=1)
    if config.N_TRACERS > 0:
        initial_max_dist_sq = xp.max(dist_sq) if dist_sq.size > 0 else 0
        if initial_max_dist_sq > config.DOMAIN_RADIUS**2 * 1.000001: 
            initial_max_dist = xp.sqrt(initial_max_dist_sq)
            scale_factor = (config.DOMAIN_RADIUS * 0.99) / initial_max_dist 
            tracer_pos *= scale_factor
    return tracer_pos, tracer_scalar_values

def rk4_step_vortices_only(v_pos, v_str, total_v_str, config: ReportSimConfig):
    dt = config.DT
    # k1_v = get_vortex_velocities_np(v_pos, v_str, config, total_v_str)
    k1_v = config.get_vortex_velocities_func(v_pos, v_str, config, total_v_str) # Use func from config
    
    v_pos_k2_arg = v_pos + 0.5 * dt * k1_v
    k2_v = config.get_vortex_velocities_func(v_pos_k2_arg, v_str, config, total_v_str)
    
    v_pos_k3_arg = v_pos + 0.5 * dt * k2_v
    k3_v = config.get_vortex_velocities_func(v_pos_k3_arg, v_str, config, total_v_str)
    
    v_pos_k4_arg = v_pos + dt * k3_v
    k4_v = config.get_vortex_velocities_func(v_pos_k4_arg, v_str, config, total_v_str)
    
    new_v_pos = v_pos + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    return new_v_pos

def enforce_boundaries(positions, config: ReportSimConfig):
    xp = config.xp
    if positions.shape[0] == 0: return positions
    norm_sq = xp.sum(positions**2, axis=1)
    
    # Simplified for vortices, which is what impulse plots use
    truly_outside_mask = norm_sq > config.DOMAIN_RADIUS**2
    if xp.any(truly_outside_mask):
        indices_truly_outside = xp.where(truly_outside_mask)[0]
        current_dist_truly_outside = xp.sqrt(norm_sq[indices_truly_outside])
        # Ensure division is safe for particles exactly at origin (though unlikely for vortices after a step)
        safe_dist = np.where(current_dist_truly_outside == 0, 1e-9, current_dist_truly_outside)
        positions[indices_truly_outside] *= (config.DOMAIN_RADIUS * 0.999 / safe_dist[:, xp.newaxis])
    return positions

def calculate_angular_impulse(v_positions, v_strengths, xp):
    if v_positions.shape[0] == 0: return xp.array(0.0, dtype=v_positions.dtype)
    r_sq = xp.sum(v_positions**2, axis=1)
    return xp.sum(v_strengths * r_sq)

def calculate_linear_impulse(v_positions, v_strengths, xp):
    if v_positions.shape[0] == 0: 
        return xp.array(0.0, dtype=v_positions.dtype), xp.array(0.0, dtype=v_positions.dtype)
    P_x = xp.sum(v_strengths * v_positions[:, 1])
    P_y = -xp.sum(v_strengths * v_positions[:, 0])
    return P_x, P_y

# --- Plotting Functions ---

def plot_vortex_image_schematic(config: ReportSimConfig, filename="vortex_image_schematic.png"):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_aspect('equal')
    ax.set_xlim(-config.DOMAIN_RADIUS * 1.8, config.DOMAIN_RADIUS * 1.8)
    ax.set_ylim(-config.DOMAIN_RADIUS * 1.8, config.DOMAIN_RADIUS * 1.8)
    ax.axis('off') # Turn off axis lines and ticks

    # Domain
    domain_circle = mpatches.Circle((0, 0), config.DOMAIN_RADIUS, color='gray', fill=False, ls='-', lw=1.5, zorder=1)
    ax.add_artist(domain_circle)
    ax.plot(0,0, 'k+', ms=5) # Origin marker
    ax.text(config.DOMAIN_RADIUS * 1.05, 0.05, r'$R_D$', fontsize=14, color='gray')

    # Real Vortex
    v_pos_k = np.array([config.DOMAIN_RADIUS * 0.5, config.DOMAIN_RADIUS * 0.3])
    v_strength_k = 1.0
    ax.plot(v_pos_k[0], v_pos_k[1], 'bo', ms=10, label=r'Real Vortex $\Gamma_k$', zorder=5)
    ax.text(v_pos_k[0] + 0.05, v_pos_k[1] + 0.05, r'$\mathbf{x}_k, \Gamma_k$', fontsize=14, color='blue')
    ax.annotate("", xy=v_pos_k, xytext=(0,0), arrowprops=dict(arrowstyle="->", color='blue', lw=1))


    # Image Vortex
    norm_sq_k = np.sum(v_pos_k**2)
    img_pos_k = (config.DOMAIN_RADIUS**2 / norm_sq_k) * v_pos_k
    img_strength_k = -v_strength_k
    ax.plot(img_pos_k[0], img_pos_k[1], 'ro', ms=10, label=r'Image Vortex $\Gamma_k^\prime = -\Gamma_k$', zorder=5)
    ax.text(img_pos_k[0] + 0.05, img_pos_k[1] + 0.05, r"$\mathbf{x}_k^\prime, \Gamma_k^\prime$", fontsize=14, color='red')
    ax.annotate("", xy=img_pos_k, xytext=(0,0), arrowprops=dict(arrowstyle="->", color='red', lw=1))
    
    # Line connecting origin, real, image
    ax.plot([0, img_pos_k[0]], [0, img_pos_k[1]], 'k--', lw=0.8, alpha=0.7, zorder=0)

    ax.set_title("Method of Images for a Single Vortex\nin a Circular Domain", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved schematic to {filename}")
    plt.close(fig)

def plot_initial_configuration(config: ReportSimConfig,
                               filename="initial_config_grouped_tracers.png"):
    v_pos, v_str = initialize_vortices(config)
    t_pos, t_scalars = initialize_tracers(config)

    fig_bg_color  = 'white'
    axes_bg_color = 'white'
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=fig_bg_color)
    ax.set_facecolor(axes_bg_color)
    ax.set_aspect('equal')
    ax.set_xlim(-config.DOMAIN_RADIUS * 1.02, config.DOMAIN_RADIUS * 1.02)
    ax.set_ylim(-config.DOMAIN_RADIUS * 1.02, config.DOMAIN_RADIUS * 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_edgecolor('gray')

    domain_circle = mpatches.Circle((0, 0), config.DOMAIN_RADIUS, color='gray', fill=False, ls='-', lw=0.8, alpha=0.6)
    ax.add_artist(domain_circle)

    # Tracers
    tracer_cmap_name = 'jet' # As per user's group mode example
    tracer_cmap = plt.cm.get_cmap(tracer_cmap_name)
    if t_pos.shape[0] > 0:
        # Normalize scalar values if not already in [0,1] (group mode should be)
        min_s, max_s = np.min(t_scalars), np.max(t_scalars)
        norm_scalars = t_scalars
        if not (0 <= min_s <=1 and 0 <= max_s <= 1):
            if max_s > min_s:
                norm_scalars = (t_scalars - min_s) / (max_s - min_s)
            else:
                norm_scalars = np.full_like(t_scalars, 0.5)

        ax.scatter(t_pos[:, 0], t_pos[:, 1], s=0.5, c=norm_scalars, cmap=tracer_cmap, alpha=0.6, edgecolors='none')

    # Vortices
    vortex_color_pos = '#FFFF00' # Yellow
    vortex_color_neg = '#FF00FF' # Magenta
    if v_pos.shape[0] > 0:
        v_colors = [vortex_color_pos if s > 0 else vortex_color_neg for s in v_str]
        max_abs_str = np.max(np.abs(v_str)) if len(v_str) > 0 else 1.0
        if max_abs_str < 1e-9: max_abs_str = 1.0
        v_sizes = 20 * np.abs(v_str) / max_abs_str + 10 # From SimConfig VORTEX_MARKER_SIZE_SCALE/BASE
        ax.scatter(v_pos[:, 0], v_pos[:, 1], s=v_sizes, c=v_colors, edgecolors='black', linewidths=0.5, zorder=10)

    title = f"Initial Configuration ({config.N_VORTICES} Vortices, {config.N_TRACERS} Tracers)\nMode: {config.TRACER_COLORING_MODE}, Groups: {config.NUM_TRACER_GROUPS}"
    # ax.set_title(title, color='white', fontsize=14)
    ax.set_title(title, color='black', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved initial configuration to {filename}")
    plt.close(fig)

def plot_impulse_evolution(config: ReportSimConfig,
                           angular_filename="angular_impulse_plot.png",
                           linear_filename="linear_impulse_plot.png"):

    fig_bg_color  = 'white'
    axes_bg_color = 'white'
    text_color    = 'black'
    
    v_pos, v_strengths = initialize_vortices(config)
    total_vortex_strength = np.sum(v_strengths)

    num_steps = int(config.SIMULATION_TIME / config.DT)
    
    history = {
        "times": np.zeros(num_steps // config.PLOT_INTERVAL + 1, dtype=config.float_type),
        "Lz": np.zeros(num_steps // config.PLOT_INTERVAL + 1, dtype=config.float_type),
        "Px": np.zeros(num_steps // config.PLOT_INTERVAL + 1, dtype=config.float_type),
        "Py": np.zeros(num_steps // config.PLOT_INTERVAL + 1, dtype=config.float_type)
    }
    
    current_sim_time = 0.0
    plot_idx = 0

    initial_Lz = calculate_angular_impulse(v_pos, v_strengths, config.xp)
    initial_Px, initial_Py = calculate_linear_impulse(v_pos, v_strengths, config.xp)
    
    Lz_denom = initial_Lz if np.abs(initial_Lz) > 1e-9 else 1.0

    for step in range(num_steps + 1):
        if step % config.PLOT_INTERVAL == 0:
            history["times"][plot_idx] = current_sim_time
            history["Lz"][plot_idx] = calculate_angular_impulse(v_pos, v_strengths, config.xp)
            Px_curr, Py_curr = calculate_linear_impulse(v_pos, v_strengths, config.xp)
            history["Px"][plot_idx] = Px_curr
            history["Py"][plot_idx] = Py_curr
            plot_idx += 1

        if step == num_steps: break

        v_pos = rk4_step_vortices_only(v_pos, v_strengths, total_vortex_strength, config)
        v_pos = enforce_boundaries(v_pos, config) # Enforce for vortices
        current_sim_time += config.DT

    times = history["times"][:plot_idx]
    Lz_hist = history["Lz"][:plot_idx]
    Px_hist = history["Px"][:plot_idx]
    Py_hist = history["Py"][:plot_idx]

    # fig_bg_color = '#080808'
    # axes_bg_color = '#101010'
    # text_color = 'lightgray'

    # Angular Impulse Plot
    fig_Lz, ax_Lz = plt.subplots(figsize=(8, 4), facecolor=fig_bg_color)
    ax_Lz.set_facecolor(axes_bg_color)
    
    rel_Lz_error = (Lz_hist - initial_Lz) / Lz_denom
    ax_Lz.plot(times, rel_Lz_error, lw=1.5, color='#FFD700') # Gold
    
    max_abs_rel_err = np.max(np.abs(rel_Lz_error)) if rel_Lz_error.size > 0 else 1e-5
    plot_Lz_y_limit = max(1e-5, max_abs_rel_err * 1.5)
    ax_Lz.set_ylim(-plot_Lz_y_limit, plot_Lz_y_limit)
    ax_Lz.set_xlabel("Time (s)", color=text_color)
    ax_Lz.set_ylabel(r"Relative $\Delta L_z / L_{z0}$", color=text_color)
    ax_Lz.set_title("Evolution of Relative Angular Impulse", color='white')
    ax_Lz.tick_params(axis='both', colors=text_color)
    ax_Lz.grid(True, linestyle='--', alpha=0.3)
    for spine in ax_Lz.spines.values(): spine.set_edgecolor('gray')
    
    plt.tight_layout()
    plt.savefig(angular_filename, dpi=150, facecolor=fig_Lz.get_facecolor())
    print(f"Saved angular impulse plot to {angular_filename}")
    plt.close(fig_Lz)

    # Linear Impulse Plot
    fig_P, ax_P = plt.subplots(figsize=(8, 4), facecolor=fig_bg_color)
    ax_P.set_facecolor(axes_bg_color)

    ax_P.plot(times, Px_hist, lw=1.5, color='#00FF00', label=r'$P_x(t)$') # Green
    ax_P.plot(times, Py_hist, lw=1.5, color='#FF00FF', label=r'$P_y(t)$') # Magenta
    
    all_P_values = np.concatenate((Px_hist, Py_hist))
    if all_P_values.size > 0:
        min_P, max_P = np.min(all_P_values), np.max(all_P_values)
        margin = (max_P - min_P) * 0.1 + 1e-5 if (max_P - min_P) > 1e-9 else 1e-5
        ax_P.set_ylim(min_P - margin, max_P + margin)
    else:
        ax_P.set_ylim(-1, 1)

    ax_P.set_xlabel("Time (s)", color=text_color)
    ax_P.set_ylabel("Linear Impulse Components (abs)", color=text_color)
    ax_P.set_title("Evolution of Linear Impulse Components", color='white')
    ax_P.tick_params(axis='both', colors=text_color)
    ax_P.grid(True, linestyle='--', alpha=0.3)
    for spine in ax_P.spines.values(): spine.set_edgecolor('gray')
    legend = ax_P.legend(facecolor='white',
                     edgecolor='black',
                     labelcolor='black')
    # legend = ax_P.legend(facecolor=axes_bg_color, edgecolor='gray', labelcolor=text_color)
    
    plt.tight_layout()
    plt.savefig(linear_filename, dpi=150, facecolor=fig_P.get_facecolor())
    print(f"Saved linear impulse plot to {linear_filename}")
    plt.close(fig_P)

    # Store params for markdown
    return {
        "SIMULATION_TIME": config.SIMULATION_TIME,
        "DT": config.DT,
        "N_VORTICES": config.N_VORTICES,
        "VORTEX_CORE_A_SQ": config.VORTEX_CORE_A_SQ,
        "initial_Lz": initial_Lz,
        "final_Lz_rel_error": rel_Lz_error[-1] if rel_Lz_error.size > 0 else 0,
        "initial_Px": initial_Px,
        "initial_Py": initial_Py,
        "final_Px": Px_hist[-1] if Px_hist.size > 0 else 0,
        "final_Py": Py_hist[-1] if Py_hist.size > 0 else 0,
    }

def generate_markdown_summary(params_impulse_plot, filename="report_figures_summary.md"):
    content = f"""# Summary for Report Figures Generation

This document summarizes parameters used for generating some figures in the report.

## Vortex Image Schematic
- Purely illustrative, based on `DOMAIN_RADIUS = {params_impulse_plot['DOMAIN_RADIUS']}`.
- Vortex at `(0.5 * R_D, 0.3 * R_D)`.

## Initial Configuration Plot (`initial_config_grouped_tracers.png`)
- `N_VORTICES`: {params_impulse_plot['N_VORTICES_init_config']}
- `N_TRACERS`: {params_impulse_plot['N_TRACERS_init_config']} (may be reduced from main sim for faster static plot)
- `DOMAIN_RADIUS`: {params_impulse_plot['DOMAIN_RADIUS']}
- `TRACER_COLORING_MODE`: `{params_impulse_plot['TRACER_COLORING_MODE_init_config']}`
- `NUM_TRACER_GROUPS`: {params_impulse_plot['NUM_TRACER_GROUPS_init_config']}
- `RANDOM_SEED`: {params_impulse_plot['RANDOM_SEED']}

## Impulse Evolution Plots (`*_impulse_plot.png`)
Parameters for the short simulation run to generate these plots:
| Parameter             | Value                |
|-----------------------|----------------------|
| `SIMULATION_TIME`     | {params_impulse_plot['SIMULATION_TIME']:.2f} s              |
| `DT`                  | {params_impulse_plot['DT']:.4f}                |
| `N_VORTICES`          | {params_impulse_plot['N_VORTICES']}                |
| `VORTEX_CORE_A_SQ`    | {params_impulse_plot['VORTEX_CORE_A_SQ']:.4f}           |
| `DOMAIN_RADIUS`       | {params_impulse_plot['DOMAIN_RADIUS']:.1f}                |
| `RANDOM_SEED`         | {params_impulse_plot['RANDOM_SEED']}                |
| Initial $L_z$         | {params_impulse_plot['initial_Lz']:.4e}          |
| Final Rel. $\\Delta L_z / L_{{z0}}$ | {params_impulse_plot['final_Lz_rel_error']:.2e}   |
| Initial $P_x$         | {params_impulse_plot['initial_Px']:.4e}          |
| Initial $P_y$         | {params_impulse_plot['initial_Py']:.4e}          |
| Final $P_x$           | {params_impulse_plot['final_Px']:.4e}            |
| Final $P_y$           | {params_impulse_plot['final_Py']:.4e}            |

**Note:** Impulse conservation depends on the specific vortex configuration (e.g. total strength) and numerical factors. These plots are representative of a typical short run.
"""
    with open(filename, 'w') as f:
        f.write(content)
    print(f"Saved markdown summary to {filename}")


if __name__ == "__main__":
    # --- Configuration for Schematic and Initial Plot ---
    # These params are mainly for the `initial_config` plot and schematic.
    # Impulse plots use slightly different settings defined below.
    report_config_general = ReportSimConfig(
        N_VORTICES=20,
        N_TRACERS=1_300_000, # Reduced for faster static plot generation
        DOMAIN_RADIUS=1.0,
        RANDOM_SEED=42,
        TRACER_COLORING_MODE="group",
        NUM_TRACER_GROUPS=5
    )
    
    # 1. Generate Vortex Image Schematic
    plot_vortex_image_schematic(report_config_general)

    # 2. Generate Initial Configuration Plot
    plot_initial_configuration(report_config_general)

    # --- Configuration for Impulse Plots (short simulation) ---
    report_config_impulse = ReportSimConfig(
        N_VORTICES=20, 
        DOMAIN_RADIUS=1.0,
        SIMULATION_TIME=10.0, # Short sim
        DT=0.002,
        PLOT_INTERVAL=1,     # Store every step for smooth curves
        VORTEX_CORE_A_SQ=0.001,
        RANDOM_SEED=42 # Same seed for consistency if vortices are involved
        # N_TRACERS, TRACER_CORE_A_SQ are not used for vortex-only impulse calculation
    )
    # Run short simulation for impulse data
    impulse_sim_results = plot_impulse_evolution(report_config_impulse)

    # 4. Generate Markdown Summary
    # Combine params from different configs for the summary table
    summary_params = {
        **vars(report_config_impulse), # Base from impulse config
        "N_VORTICES_init_config": report_config_general.N_VORTICES,
        "N_TRACERS_init_config": report_config_general.N_TRACERS,
        "TRACER_COLORING_MODE_init_config": report_config_general.TRACER_COLORING_MODE,
        "NUM_TRACER_GROUPS_init_config": report_config_general.NUM_TRACER_GROUPS,
        **impulse_sim_results # Add results from the impulse sim run
    }
    generate_markdown_summary(summary_params)

    print("\nAll report figures and summary generated.")