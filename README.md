# CSE 291 (SP 2025) Final Project: 2D Point Vortex Dynamics Simulation

**Author:** Param Somane

## Table of Contents
1.  [Overview](#overview)  
2.  [Example Animations](#example-animations)  
3.  [Features](#features)  
4.  [System Model](#system-model)  
5.  [Numerical Implementation](#numerical-implementation)  
6.  [Dependencies](#dependencies)  
7.  [Directory Structure](#directory-structure)  
8.  [How to Run](#how-to-run)  
    *   [Main Simulation](#main-simulation)  
    *   [Generating Report Figures](#generating-report-figures)  
9.  [Simulation Parameters](#simulation-parameters)  
10. [Output](#output)  
11. [Key Figures for Report](#key-figures-for-report)  
12. [Acknowledgments](#acknowledgments)

## Overview

This project implements a numerical simulation for two-dimensional point vortex dynamics within a bounded circular domain. The system models the motion of $N_v$ point vortices, regularized using the Lamb-Oseen model to prevent singularities and provide a finite core structure. These vortices interact with each other and influence the motion of $N_t$ passive tracer particles, which are advected by the flow.

The no-penetration boundary condition at the circular domain wall is enforced using the classical method of images. The equations of motion for both vortices and tracers form a system of ordinary differential equations (ODEs), which are integrated numerically using an explicit fourth-order Runge-Kutta (RK4) scheme.

The implementation is designed for performance and flexibility, supporting:
*   **CPU-based computation:** Leveraging NumPy for array operations and Numba for Just-In-Time (JIT) compilation of performance-critical loops.
*   **GPU acceleration:** Utilizing CuPy for massively parallel computations on NVIDIA GPUs.

The simulation's behavior is highly configurable through command-line arguments. The primary output is an MP4 animation visualizing the complex advection of tracers by the vortex flow, along with diagnostic plots of conserved quantities like linear and angular impulse to assess simulation fidelity.

This project fulfills the requirements for a continuum mechanical simulation, specifically focusing on a 2D fluid dynamics problem.

## Example Animations

The following animations demonstrate the simulation capabilities. They were generated with settings similar to those found in the `animation/` directory, involving a moderate number of vortices ($N_v=20$) and a large number of tracers ($N_t \approx 1.3 \times 10^6$).

1.  **Scalar Plume Advection (using `plume` colormap and glow effects):**
    This video shows tracers initialized with a scalar value based on their initial positions, advected by the vortices. It uses the custom 'plume' colormap and glow effects for enhanced visuals.
    [Watch Video: Scalar Plume](https://github.com/user-attachments/assets/03d3e51f-e83b-40c6-88af-02c2e2ff9d8f)
    <video src="https://github.com/user-attachments/assets/03d3e51f-e83b-40c6-88af-02c2e2ff9d8f"></video>

2.  **Grouped Tracer Advection (using `jet` colormap):**
    This video shows tracers initialized in several distinct groups, each with a different color. It illustrates how different regions of fluid are stretched and mixed by the vortex flow.
    [Watch Video: Grouped Tracers](https://github.com/user-attachments/assets/1cbf5575-aa51-412e-8bba-9535f1c960b8)
    <br>(If the video doesn't embed below, please use the link above to view.)
    <video src="https://github.com/user-attachments/assets/1cbf5575-aa51-412e-8bba-9535f1c960b8"></video>

*(Note: The example videos `animation/pv_scalar_plume_1.3M.mp4` and `animation/pv_group_1.3M.mp4` serve as references for the kind of output this simulation can produce.)*

## Features

*   **2D Point Vortex Dynamics:** Simulates the interaction of multiple point vortices.
*   **Lamb-Oseen Regularization:** Uses the Lamb-Oseen model for vortex cores to avoid singularities and allow for more realistic close-range interactions. Separate core sizes for vortex-vortex ($a_v^2$) and vortex-tracer ($a_t^2$) interactions.
*   **Passive Tracer Advection:** Tracks a large number of passive tracer particles to visualize the flow field.
*   **Circular Domain with Method of Images:** Enforces no-penetration boundary conditions on a circular domain using image vortices and a background correction flow for non-zero total circulation.
*   **RK4 Time Integration:** Employs a fourth-order Runge-Kutta scheme for accurate time evolution of the system.
*   **Dual Backend Support (CPU/GPU):**
    *   CPU: NumPy with Numba JIT compilation for accelerated loops.
    *   GPU: CuPy for CUDA-based acceleration on NVIDIA GPUs, including custom `ElementwiseKernel` for the Lamb-Oseen factor.
*   **Configurable Simulation Parameters:** Extensive set of parameters controllable via command-line arguments (see `python main.py --help`).
*   **Animation Output:** Generates MP4 video using Matplotlib and FFmpeg, showing particle dynamics and diagnostic plots.
*   **Multiple Tracer Coloring Modes:**
    *   `group`: Tracers colored by initial group assignment.
    *   `scalar`: Tracers colored by an initial scalar field (e.g., radial gradient).
    *   `speed`: Tracers colored by their instantaneous speed.
*   **Tracer Glow Effects:** Optional multi-layered glow effects for enhanced visualization of tracers.
*   **Conservation Monitoring:** Tracks and plots angular and linear impulse to assess simulation accuracy.
*   **Static Plot Generation:** Includes a separate script (`generate_plots.py`) to produce key figures for the technical report.

## System Model

The simulation is set in a 2D circular domain of radius $R$.
*   **Vortices:** $N_v$ vortices, each with position $r_k(t)$ and constant circulation strength $\Gamma_k$.
*   **Tracers:** $N_t$ passive tracers, each with position $x_j(t)$, advected by the flow.

The velocity field of a Lamb-Oseen vortex with strength $\Gamma$ and squared core radius $a^2$ at $r=(x,y)$ relative to the vortex is:

$$ u(x,y; \Gamma, a^2) = \frac{\Gamma}{2\pi} \frac{1 - e^{-(x^2+y^2)/a^2}}{x^2+y^2} \begin{pmatrix} -y \\ \\ x \end{pmatrix}$$

The method of images is used for the circular boundary: for each vortex $(r_k, \Gamma_k)$, an image vortex $(r_k', \Gamma_k')$ is placed at $r_k' = (R^2/\|r_k\|^2) r_k$ with strength $\Gamma_k' = -\Gamma_k$. A background rotational flow is added if $\sum \Gamma_k \neq 0$.

The equations of motion for vortex $i$ and tracer $l$ are

$$ \frac{dr_i}{dt} = \sum_{\substack{j = 1 \\ j \neq i}}^{N_v} u\bigl(r_i - r_j;\,\Gamma_j,a_v^{2}\bigr)+\sum_{j = 1}^{N_v} u\bigl(r_i - r_j^{\prime};\,\Gamma_j^{\prime},a_v^{2}\bigr)+u_{\text{bg}}(r_i), $$

$$ \frac{dx_l}{dt} = \sum_{j = 1}^{N_v} u\bigl(x_l - r_j;\,\Gamma_j,a_t^{2}\bigr)+ \sum_{j = 1}^{N_v} u\bigl(x_l - r_j^{\prime};\,\Gamma_j^{\prime},a_t^{2}\bigr)+ u_{\text{bg}}(x_l). $$

For full mathematical details, please refer to the accompanying LaTeX report in the `tex/` directory.

## Numerical Implementation

*   **Time Integration:** An explicit fourth-order Runge-Kutta (RK4) scheme integrates the ODEs for vortex and tracer positions.
*   **Boundary Enforcement:** After each RK4 step, particles found outside the domain are projected back just inside the boundary.
*   **Computational Backend:**
    *   The code uses a common array module interface `xp`, which points to `numpy` (CPU) or `cupy` (GPU).
    *   **CPU Path:** Uses NumPy for vectorized operations. Numba's `@njit` and `prange` are used to JIT-compile and parallelize performance-critical loops in functions like `_get_velocities_induced_by_vortices_cpu_numba_impl` and `get_vortex_velocities_cpu_numba_impl`.
    *   **GPU Path:** Uses CuPy for GPU-accelerated computations. A custom CuPy `ElementwiseKernel` (`_lamb_oseen_kernel`) efficiently calculates the Lamb-Oseen factor. Velocity summations are expressed using CuPy's vectorized array operations.
*   **Initialization:** Vortices and tracers are initialized based on `SimConfig` parameters, with options for deterministic and random placements, and various tracer coloring strategies.

## Dependencies

*   Python 3.10+
*   NumPy (`numpy`)
*   Matplotlib (`matplotlib`)
*   CuPy (`cupy`, optional, for GPU acceleration, requires NVIDIA GPU and CUDA toolkit)
*   Numba (`numba`, optional, for CPU JIT acceleration)
*   FFmpeg (must be installed and in system PATH for saving animations)

You can typically install the Python packages using pip:
```bash
pip install numpy matplotlib numba cupy-cudaXX # Replace XX with your CUDA version, e.g., cupy-cuda118 or cupy-cuda12x
```
If you don't have a compatible NVIDIA GPU or don't want to use GPU acceleration, you can omit `cupy`. The simulation will fall back to the CPU path. Numba is highly recommended for better performance on the CPU path.

## Directory Structure

```
CSE291D_Final_Project_SP25/
├── .git/                        # Git repository files
├── animation/                   # Output animations and example videos
│   ├── pv_group_1.3M.mp4        # Example animation (linked below)
│   └── pv_scalar_plume_1.3M.mp4 # Example animation (linked below)
├── tex/                         # LaTeX report source and related assets
│   ├── figs/                    # Figures generated by generate_plots.py
│   │   ├── angular_impulse_plot.png
│   │   ├── initial_config_grouped_tracers.png
│   │   ├── linear_impulse_plot.png
│   │   └── vortex_image_schematic.png
│   ├── report.tex
│   ├── Final_Project_Report.pdf
│   └── report_figures_summary.md
├── .gitignore
├── generate_plots.py            # Script to generate static plots for the report
├── main.py                      # Main simulation script
└── README.md                    # Project overview and instructions
```

## How to Run

### Main Simulation

The main simulation is run using `main.py`. You can customize various parameters via command-line arguments.

To see all available options and their default values:
```bash
python main.py --help
```

**Example Commands:**

1.  **Run on GPU (if available):**
    ```bash
    python main.py --n-vortices 20 --n-tracers 500000 --simulation-time 10.0 \
                   --output-filename animation_gpu.mp4 --gpu-enabled \
                   --tracer-coloring-mode group --num-tracer-groups 5
    ```

2.  **Run on CPU (using Numba for acceleration if available):**
    ```bash
    python main.py --n-vortices 10 --n-tracers 100000 --simulation-time 5.0 \
                   --output-filename animation_cpu.mp4 --no-gpu-enabled \
                   --tracer-coloring-mode scalar --random-seed 42
    ```
    (Note: `--no-gpu-enabled` explicitly disables GPU even if CuPy is available. If CuPy is not installed, it defaults to CPU.)

3.  **Run with specific tracer glow effects (example):**
    ```bash
    python main.py --n-tracers 100000 --tracer-glow-layers "0.1,0.1;0.05,0.05" \
                   --output-filename glow_effect.mp4
    ```

### Generating Report Figures

The script `generate_plots.py` is used to create static figures suitable for inclusion in a report, such as the vortex image schematic, initial particle configuration, and impulse conservation plots. It uses a simplified simulation setup internally for the impulse plots.

To run it:
```bash
python generate_plots.py
```
This will generate several PNG files (e.g., `vortex_image_schematic.png`, `initial_config_grouped_tracers.png`, `angular_impulse_plot.png`, `linear_impulse_plot.png`) and a markdown summary (`report_figures_summary.md`) in the project's root directory.

## Simulation Parameters

The simulation behavior is controlled by the `SimConfig` dataclass in `main.py`. Most parameters can be set via command-line arguments. Some key parameters include:

*   `--n-vortices`: Number of vortices.
*   `--n-tracers`: Number of tracer particles.
*   `--domain-radius`: Radius of the circular domain.
*   `--simulation-time`: Total physical time to simulate.
*   `--dt`: Time step for the RK4 integrator.
*   `--output-filename`: Name of the output MP4 animation file.
*   `--plot-interval`: Save data for animation every N steps.
*   `--dpi`: Dots Per Inch for the output animation.
*   `--vortex-core-a-sq`: Squared Lamb-Oseen core radius for vortex-vortex interactions.
*   `--tracer-core-a-sq`: Squared Lamb-Oseen core radius for vortex-tracer interactions.
*   `--gpu-enabled` / `--no-gpu-enabled`: Toggle GPU acceleration.
*   `--random-seed`: Seed for random number generation (for reproducible runs).
*   `--tracer-particle-size`, `--tracer-alpha`, `--tracer-cmap`: Visual properties for tracers.
*   `--tracer-coloring-mode`: How tracers are colored (`group`, `scalar`, `speed`).
*   `--num-tracer-groups`: Number of groups if `tracer-coloring-mode` is `group`.
*   `--tracer-glow-layers`: Defines glow effect layers, e.g., `"size_mult1,alpha_mult1;size_mult2,alpha_mult2"`.
*   `--anim-tracers-max`: Maximum number of tracers to render in animation (subsamples if $N_T$ is larger).

Refer to `python main.py --help` for a complete list and descriptions.

## Output

*   **Primary Output:** An MP4 animation file (e.g., `point_vortex_dynamics.mp4` or as specified by `--output-filename`). The animation shows:
    *   The main panel with vortices (larger markers, colored by strength sign) and tracers (small dots, colored by chosen mode) moving in the circular domain.
    *   Time evolution display.
    *   Subplots showing the history of angular impulse ($L_z$) and linear impulses ($P_x, P_y$).
    *   Simulation configuration summary text (DT, Sim Time, Backend, Seed).
*   **Static Figures (from `generate_plots.py`):**
    *   `vortex_image_schematic.png`: Illustrates the method of images.
    *   `initial_config_grouped_tracers.png`: Shows an example initial setup of particles.
    *   `angular_impulse_plot.png` & `linear_impulse_plot.png`: Plots showing conservation of these quantities from a short test simulation.
    *   `report_figures_summary.md`: A markdown file summarizing parameters used for generating these static plots.

## Key Figures for Report

The `generate_plots.py` script creates several figures useful for understanding the simulation setup and verifying its correctness. These are saved in the project root directory.

| Figure                                 | Description                                                                                                                                                              |
| :------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `vortex_image_schematic.png`           | Illustrates the method of images for a single vortex in the circular domain.                                                                                           |
| `initial_config_grouped_tracers.png` | Shows an example initial configuration of vortices and tracers, typically for the 'group' coloring mode.                                                                 |
| `angular_impulse_plot.png`           | Plots the relative change in angular impulse ($L_z$) over time for a test simulation, demonstrating its conservation (or near-conservation).                               |
| `linear_impulse_plot.png`            | Plots the components of linear impulse ($P_x, P_y$) over time for a test simulation, demonstrating their conservation (or near-conservation).                              |

A summary of parameters used to generate these specific figures can be found in `report_figures_summary.md`.

## Acknowledgments

OpenAI's GPT-4 model was utilized as an assistant in structuring and drafting portions of this README file and the accompanying LaTeX report, based on the provided Python source code and project guidelines. The author of the code (Param Somane) actively reviewed, and is expected to test and adjust, the generated text, equations, and figures to ensure accuracy and reflect their complete understanding and implementation.