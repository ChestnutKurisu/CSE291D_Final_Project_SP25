# 2-D Wave Simulation

This project contains a Python implementation of 2‑D wave solvers for various wave types, including acoustic (scalar) and elastic waves. The program generates an MP4 animation of the wave propagation and logs diagnostic data.

## Governing Equations

### Scalar Waves
The solver models the classic second‑order wave equation for a field $u(x,y,t)$ with constant wave speed $c$:

$$
\frac{\partial^2 u}{\partial t^2} = c^2\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right).
$$

Initial conditions typically consist of a compact Gaussian pulse, and the simulation assumes zero displacement at the boundaries unless an absorbing layer is specified and implemented by the solver.

### Elastic Waves
For elastic waves, the system solves the 2D elastic wave equations. Two main formulations are available:
1.  **Velocity-Stress Formulation:** Solves the coupled first-order equations for velocity components ($v_x, v_y$) and stress tensor components ($\sigma_{xx}, \sigma_{yy}, \sigma_{xy}$).
    $$
    \rho \frac{\partial v_i}{\partial t} = \sum_j \frac{\partial \sigma_{ij}}{\partial x_j} + f_i
    $$
    $$
    \frac{\partial \sigma_{ij}}{\partial t} = C_{ijkl} \frac{\partial v_k}{\partial x_l}
    $$
    (summation implied, $C_{ijkl}$ is the stiffness tensor from Lamé parameters $\lambda, \mu$).
2.  **Scalar Potential Formulation:** Solves for P-wave potential ($\phi$) and S-wave potential ($\psi$):
    $$
    \frac{\partial^2 \phi}{\partial t^2} = v_P^2 \nabla^2 \phi
    $$
    $$
    \frac{\partial^2 \psi}{\partial t^2} = v_S^2 \nabla^2 \psi
    $$
    Displacements are recovered as $u_x = \frac{\partial\phi}{\partial x} - \frac{\partial\psi}{\partial y}$ and $u_z = \frac{\partial\phi}{\partial z} + \frac{\partial\psi}{\partial x}$ (using z for y here as per common seismology notation).

## Numerical Method

A uniform Cartesian grid is used with spacing $\Delta x = \Delta y$. Spatial derivatives are approximated with second‑order central differences.

### Scalar Wave Time Integration
Time integration follows a standard explicit three‑level scheme:
$$
 u^{n+1}_{i,j} = 2u^{n}_{i,j} - u^{n-1}_{i,j} + S^2\,\Delta x^2 \left(\nabla^2 u\right)^{n}_{i,j},
$$
where $S = c\,\Delta t/\Delta x$ is the Courant number and $\left(\nabla^2 u\right)^{n}_{i,j}$ is the discrete Laplacian. To satisfy the Courant–Friedrichs–Lewy (CFL) stability condition for the 2D 5-point stencil, the time step $\Delta t$ is chosen such that $S \le \frac{1}{\sqrt{2}}$. In `simulation.py`, $\Delta t = \text{safety\_factor} \cdot \frac{\Delta x}{c\sqrt{2}}$, with `safety_factor = 0.7`.

### Elastic Wave Time Integration
*   The **velocity-stress** formulation uses an explicit leapfrog scheme, staggered in time.
*   The **scalar potential** formulation for elastic waves uses a similar explicit scheme as the scalar acoustic waves for each potential.

An optional absorbing boundary layer (exponential damping) can be applied to reduce reflections from domain edges for elastic solvers.

## Running the Simulation

Install the Python dependencies:
```bash
pip install -r requirements.txt
