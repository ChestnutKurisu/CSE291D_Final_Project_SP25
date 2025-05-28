"""Analytical dispersion and wave speed solvers.

This module implements numerical solvers for various classical waves in
elastic and acoustic media.  Each solver is based on standard characteristic
relations from elastodynamics and returns either the wave speed or allowed
phase velocities.
"""

import numpy as np
from scipy.optimize import brentq


def rayleigh_wave_speed(alpha: float, beta: float) -> float:
    """Return the Rayleigh surface wave speed for a half-space.

    Parameters
    ----------
    alpha : float
        Longitudinal (P-wave) speed.
    beta : float
        Shear (S-wave) speed.

    Returns
    -------
    float
        The Rayleigh wave speed ``c_R``.
    """

    def characteristic(c):
        term1 = (2.0 - (c ** 2 / beta ** 2)) ** 2
        term2 = 4.0 * np.sqrt(1.0 - (c ** 2 / alpha ** 2)) * np.sqrt(
            1.0 - (c ** 2 / beta ** 2)
        )
        return term1 - term2

    c_lower = 0.9 * beta
    c_upper = beta * 0.9999
    return brentq(characteristic, c_lower, c_upper)


def love_wave_dispersion(
    freq: float,
    beta1: float,
    beta2: float,
    h: float,
    n_modes: int = 1,
    c_guess_bounds=(None, None),
):
    """Return phase velocities satisfying the Love wave dispersion equation.

    Parameters
    ----------
    freq : float
        Angular frequency ``\omega`` in rad/s.
    beta1 : float
        Shear speed in the top layer.
    beta2 : float
        Shear speed in the half-space (beta2 > beta1).
    h : float
        Layer thickness (m).
    n_modes : int, optional
        Number of modes/branches to return.
    c_guess_bounds : tuple, optional
        Optional search interval ``(c_min, c_max)``.

    Returns
    -------
    list of float
        Phase velocities for the lowest ``n_modes`` solutions.
    """
    omega = freq
    if c_guess_bounds[0] is None:
        c_guess_bounds = (beta1 + 1e-3, beta2 - 1e-3)

    def love_equation(c):
        if c >= beta2 or c <= beta1:
            return 1e6
        k = omega / c
        root1 = beta1 ** 2 / c ** 2 - 1
        root2 = beta2 ** 2 / c ** 2 - 1
        if root1 < 0 or root2 < 0:
            return 1e6
        lhs = np.tan(k * h * np.sqrt(root1))
        rhs = np.sqrt(root2) / np.sqrt(1 - beta1 ** 2 / c ** 2)
        return lhs - rhs

    c_min, c_max = c_guess_bounds
    c_vals = np.linspace(c_min, c_max, 200)
    sign_vals = np.sign([love_equation(cv) for cv in c_vals])

    roots = []
    for i in range(len(c_vals) - 1):
        if sign_vals[i] != sign_vals[i + 1]:
            try:
                root = brentq(love_equation, c_vals[i], c_vals[i + 1])
                roots.append(root)
            except ValueError:
                pass
    roots = sorted(list(set(np.round(roots, 5))))
    return roots[:n_modes]


def lamb_s0_mode(
    freq: float,
    alpha: float,
    beta: float,
    thickness: float,
    c_bounds=(None, None),
):
    """Return the S0 Lamb mode phase velocity for a plate of thickness ``2h``.

    Parameters
    ----------
    freq : float
        Angular frequency ``\omega`` in rad/s.
    alpha : float
        P-wave speed in the plate.
    beta : float
        S-wave speed in the plate.
    thickness : float
        Half thickness ``h`` of the plate.
    c_bounds : tuple, optional
        Optional search interval.

    Returns
    -------
    float or None
        The phase velocity if a root is found, otherwise ``None``.
    """
    omega = freq
    if c_bounds[0] is None:
        c_min = 0.9 * beta
        c_max = alpha * 1.5
    else:
        c_min, c_max = c_bounds

    def dispersion(c):
        k = omega / c
        p2 = k ** 2 - (omega / alpha) ** 2
        q2 = k ** 2 - (omega / beta) ** 2
        p = np.sqrt(abs(p2)) * (1 if p2 >= 0 else 1j)
        q = np.sqrt(abs(q2)) * (1 if q2 >= 0 else 1j)
        lhs = np.tan(p * thickness) * np.tan(q * thickness)
        rhs = (4 * k ** 2 * p * q) / ((k ** 2 - q ** 2) ** 2)
        return lhs - rhs

    c_vals = np.linspace(c_min, c_max, 200)
    f_vals = [dispersion(cv) for cv in c_vals]
    signs = np.sign(f_vals)

    for i in range(len(c_vals) - 1):
        if signs[i] != signs[i + 1]:
            try:
                return brentq(dispersion, c_vals[i], c_vals[i + 1])
            except Exception:
                continue
    return None


def lamb_a0_mode(
    freq: float,
    alpha: float,
    beta: float,
    thickness: float,
    c_bounds=(None, None),
):
    """Return the A0 Lamb mode phase velocity for a plate of thickness ``2h``."""
    omega = freq
    if c_bounds[0] is None:
        c_min = 0.1 * beta
        c_max = beta * 0.99
    else:
        c_min, c_max = c_bounds

    def dispersion(c):
        k = omega / c
        p2 = k ** 2 - (omega / alpha) ** 2
        q2 = k ** 2 - (omega / beta) ** 2
        p = np.sqrt(abs(p2)) * (1 if p2 >= 0 else 1j)
        q = np.sqrt(abs(q2)) * (1 if q2 >= 0 else 1j)
        lhs = np.tan(p * thickness) * np.tan(q * thickness)
        rhs = -(4 * k ** 2 * p * q) / ((k ** 2 - q ** 2) ** 2)
        return lhs - rhs

    c_vals = np.linspace(c_min, c_max, 200)
    f_vals = [dispersion(cv) for cv in c_vals]
    signs = np.sign(f_vals)

    for i in range(len(c_vals) - 1):
        if signs[i] != signs[i + 1]:
            try:
                return brentq(dispersion, c_vals[i], c_vals[i + 1])
            except Exception:
                continue
    return None


def stoneley_wave_speed(
    alpha1: float,
    beta1: float,
    rho1: float,
    alpha2: float,
    beta2: float,
    rho2: float,
):
    """Return the Stoneley wave speed for two contacting solids."""

    def characteristic(c):
        if c >= min(alpha1, alpha2):
            return 1e6
        try:
            kappa1 = np.sqrt(1.0 / beta1 ** 2 - 1.0 / c ** 2)
            kappa2 = np.sqrt(1.0 / beta2 ** 2 - 1.0 / c ** 2)
            eta1 = np.sqrt(1.0 / alpha1 ** 2 - 1.0 / c ** 2)
            eta2 = np.sqrt(1.0 / alpha2 ** 2 - 1.0 / c ** 2)
        except ValueError:
            return 1e6
        term1 = (rho1 * kappa1 + rho2 * kappa2) * (rho1 * eta1 + rho2 * eta2)
        term2 = np.sqrt(rho1 * rho2) * ((kappa1 + kappa2) * (eta1 + eta2))
        return term1 - term2

    c_lower = 0.99 * min(beta1, beta2)
    c_upper = 0.99 * min(alpha1, alpha2)
    if characteristic(c_lower) * characteristic(c_upper) > 0:
        return None
    return brentq(characteristic, c_lower, c_upper)


def scholte_wave_speed(
    alpha_s: float,
    beta_s: float,
    rho_s: float,
    c_f: float,
    rho_f: float,
):
    """Return the Scholte wave speed for a solid-fluid interface."""

    def characteristic(c):
        if c >= min(c_f, beta_s):
            return 1e6
        try:
            kappa_s = np.sqrt(1.0 / beta_s ** 2 - 1.0 / c ** 2)
            eta_s = np.sqrt(1.0 / alpha_s ** 2 - 1.0 / c ** 2)
            kappa_f = np.sqrt(1.0 / c_f ** 2 - 1.0 / c ** 2)
        except ValueError:
            return 1e6
        term1 = (rho_s * kappa_s + rho_f * kappa_f) * (rho_s * eta_s)
        term2 = np.sqrt(rho_s * rho_f) * (kappa_s + kappa_f) * eta_s
        return term1 - term2

    c_lower = 0.1 * min(c_f, beta_s)
    c_upper = 0.99 * min(c_f, beta_s)
    if characteristic(c_lower) * characteristic(c_upper) > 0:
        return None
    return brentq(characteristic, c_lower, c_upper)
