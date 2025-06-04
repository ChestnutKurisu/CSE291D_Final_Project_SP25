# Summary for Report Figures Generation

This document summarizes parameters used for generating some figures in the report.

## Vortex Image Schematic
- Purely illustrative, based on `DOMAIN_RADIUS = 1.0`.
- Vortex at `(0.5 * R_D, 0.3 * R_D)`.

## Initial Configuration Plot (`initial_config_grouped_tracers.png`)
- `N_VORTICES`: 20
- `N_TRACERS`: 1300000 (may be reduced from main sim for faster static plot)
- `DOMAIN_RADIUS`: 1.0
- `TRACER_COLORING_MODE`: `group`
- `NUM_TRACER_GROUPS`: 5
- `RANDOM_SEED`: 42

## Impulse Evolution Plots (`*_impulse_plot.png`)
Parameters for the short simulation run to generate these plots:
| Parameter             | Value                |
|-----------------------|----------------------|
| `SIMULATION_TIME`     | 10.00 s              |
| `DT`                  | 0.0020                |
| `N_VORTICES`          | 20                |
| `VORTEX_CORE_A_SQ`    | 0.0010           |
| `DOMAIN_RADIUS`       | 1.0                |
| `RANDOM_SEED`         | 42                |
| Initial $L_z$         | 7.5522e-01          |
| Final Rel. $\Delta L_z / L_{z0}$ | -4.19e-04   |
| Initial $P_x$         | 7.6063e-01          |
| Initial $P_y$         | -1.1508e-01          |
| Final $P_x$           | -1.2660e+00            |
| Final $P_y$           | 2.8035e-01            |

**Note:** Impulse conservation depends on the specific vortex configuration (e.g. total strength) and numerical factors. These plots are representative of a typical short run.
