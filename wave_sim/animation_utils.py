import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

DEFAULT_OUTPUT_DIR_1D = "output_1d_animations_individual"

def generate_1d_animation(
    solver,
    out_name: str,
    plot_variable_name: str,
    title_prefix: str,
    y_label: str,
    output_dir: str = DEFAULT_OUTPUT_DIR_1D,
    x_label: str = "Position",
    y_lims: tuple | None = None,
    fps: int = 30,
    total_steps: int | None = None,
    figure_size: tuple = (8, 4),
) -> str:
    """Generate a simple 1-D matplotlib animation from a solver."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, out_name)

    writer = imageio.get_writer(out_path, fps=fps)
    fig, ax = plt.subplots(figsize=figure_size)

    if hasattr(solver, "x"):
        coords = solver.x
    elif hasattr(solver, "r"):
        coords = solver.r
    elif hasattr(solver, "y"):
        coords = solver.y
    else:
        raise AttributeError("Solver does not provide spatial coordinates")

    nsteps = getattr(solver, "nt", None)
    if nsteps is None:
        if hasattr(solver, "T") and hasattr(solver, "dt"):
            nsteps = int(solver.T / solver.dt)
        else:
            nsteps = 0
    if total_steps is not None:
        nsteps = min(total_steps, nsteps)

    for step in range(nsteps):
        solver.step()
        current_time = getattr(solver, "t", (step + 1) * solver.dt)
        ax.clear()
        data = getattr(solver, plot_variable_name)
        if isinstance(data, np.ndarray) and data.ndim > 1:
            data = data[0]
        ax.plot(coords, data)
        if y_lims is not None:
            ax.set_ylim(*y_lims)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title_prefix}, Time: {current_time:.3f}s")
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(img)

    writer.close()
    plt.close(fig)
    return out_path
