"""Dipole magnetic-field model and the current-to-field actuation matrix.

Each coil is a magnetic dipole (see :mod:`selective_em.coils`). The symbolic
field expression and its spatial derivatives are built once with ``sympy`` at
import time and turned into fast numpy callables.

Unlike the original ``lib.py``, every public function takes an explicit
``coils`` array (rows of ``[m0, m1, m2, r0_0, r0_1, r0_2]``) instead of relying
on a module-global coil list, so a single import can serve any coil subset.

A "target point" is a dict::

    {'X': , 'Y': , 'Z': ,                        # position (m), required
     'Bx': , 'By': , 'Bz': ,                      # field components
     'Bx_dx': , 'Bx_dy': , 'Bx_dz': ,             # field gradients
     'By_dy': , 'By_dz': }

Each field/gradient key is set to ``True`` to constrain that component or
``None`` to ignore it (see :func:`extract_map_i2b`). Only the position keys are
required.
"""

import numpy as np
import sympy as sp

from .coils import MU0, DEFAULT_COILS

# --- Symbolic dipole field, built once and lambdified to numpy ---------------
m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z = sp.symbols(
    "m0 m1 m2 r0_0 r0_1 r0_2 X Y Z")

_mu0 = 4 * sp.pi * 1e-7

# Displacement from the coil to the field point.
dx = X - r0_0
dy = Y - r0_1
dz = Z - r0_2

# Distance (small epsilon avoids division by zero).
r = sp.sqrt(dx**2 + dy**2 + dz**2) + 1e-9

dot_product = m0 * dx + m1 * dy + m2 * dz

model_Bx = (_mu0 / (4 * sp.pi)) * (3 * dx * dot_product / r**5 - m0 / r**3)
model_By = (_mu0 / (4 * sp.pi)) * (3 * dy * dot_product / r**5 - m1 / r**3)
model_Bz = (_mu0 / (4 * sp.pi)) * (3 * dz * dot_product / r**5 - m2 / r**3)

_args = (m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z)
dipole_model_Bx = sp.lambdify(_args, model_Bx, "numpy")
dipole_model_By = sp.lambdify(_args, model_By, "numpy")
dipole_model_Bz = sp.lambdify(_args, model_Bz, "numpy")

# Partial derivatives of the field components.
dipole_model_Bx_dx = sp.lambdify(_args, sp.diff(model_Bx, X), "numpy")
dipole_model_Bx_dy = sp.lambdify(_args, sp.diff(model_Bx, Y), "numpy")
dipole_model_Bx_dz = sp.lambdify(_args, sp.diff(model_Bx, Z), "numpy")
dipole_model_By_dy = sp.lambdify(_args, sp.diff(model_By, Y), "numpy")
dipole_model_By_dz = sp.lambdify(_args, sp.diff(model_By, Z), "numpy")
dipole_model_Bz_dz = sp.lambdify(_args, sp.diff(model_Bz, Z), "numpy")


def calculate_b_and_derivatives(currents, x, y, z, coils=DEFAULT_COILS):
    """Superpose the field and its derivatives from every coil at (x, y, z).

    Returns ``[Bx, By, Bz, Bx_dx, Bx_dy, Bx_dz, By_dy, By_dz]``.
    ``currents`` must have one entry per row of ``coils``.
    """
    coils = np.asarray(coils)
    totals = np.zeros(8)
    for i in range(len(coils)):
        cm0, cm1, cm2, cr0, cr1, cr2 = coils[i]
        contrib = np.array([
            dipole_model_Bx(cm0, cm1, cm2, cr0, cr1, cr2, x, y, z),
            dipole_model_By(cm0, cm1, cm2, cr0, cr1, cr2, x, y, z),
            dipole_model_Bz(cm0, cm1, cm2, cr0, cr1, cr2, x, y, z),
            dipole_model_Bx_dx(cm0, cm1, cm2, cr0, cr1, cr2, x, y, z),
            dipole_model_Bx_dy(cm0, cm1, cm2, cr0, cr1, cr2, x, y, z),
            dipole_model_Bx_dz(cm0, cm1, cm2, cr0, cr1, cr2, x, y, z),
            dipole_model_By_dy(cm0, cm1, cm2, cr0, cr1, cr2, x, y, z),
            dipole_model_By_dz(cm0, cm1, cm2, cr0, cr1, cr2, x, y, z),
        ])
        totals += contrib * currents[i]
    return totals


def map_i2b(target_points, coils=DEFAULT_COILS):
    """Actuation matrix A mapping the current vector to field + gradients.

    Produces 8 rows per target point (Bx, By, Bz and the 5 gradients) and one
    column per coil. This is the matrix A(p) of the paper.
    """
    coils = np.asarray(coils)
    num_coils = len(coils)
    A = np.zeros((8 * len(target_points), num_coils))
    for i, tp in enumerate(target_points):
        for j in range(num_coils):
            currents = np.zeros(num_coils)
            currents[j] = 1
            A[8 * i:8 * (i + 1), j] = calculate_b_and_derivatives(
                currents, tp["X"], tp["Y"], tp["Z"], coils)
    return A


def extract_map_i2b(target_points):
    """Row-selection matrix keeping only the constrained field components.

    A component is kept when its key on the target dict is not ``None``.
    """
    n = 8 * len(target_points)
    row_selection = np.zeros((n, n))
    keys = ["Bx", "By", "Bz", "Bx_dx", "Bx_dy", "Bx_dz", "By_dy", "By_dz"]
    for i, tp in enumerate(target_points):
        for offset, key in enumerate(keys):
            if tp.get(key) is not None:
                row_selection[8 * i + offset, 8 * i + offset] = 1
    return row_selection[np.sum(row_selection, axis=1) != 0]


def map_i2b_2d(target_points, coils=DEFAULT_COILS):
    """Planar (2D) actuation matrix: 5 rows per point (Bx, By, Bx_dx, Bx_dy, By_dy)."""
    coils = np.asarray(coils)
    num_coils = len(coils)
    A = np.zeros((5 * len(target_points), num_coils))
    for i, tp in enumerate(target_points):
        for j in range(num_coils):
            currents = np.zeros(num_coils)
            currents[j] = 1
            bx, by, bz, bx_dx, bx_dy, bx_dz, by_dy, by_dz = \
                calculate_b_and_derivatives(currents, tp["X"], tp["Y"], tp["Z"], coils)
            A[5 * i:5 * (i + 1), j] = [bx, by, bx_dx, bx_dy, by_dy]
    return A


def extract_map_i2b_2d(target_points):
    """Row-selection matrix for the planar (2D) actuation matrix."""
    n = 5 * len(target_points)
    row_selection = np.zeros((n, n))
    keys = ["Bx", "By", "Bx_dx", "Bx_dy", "By_dy"]
    for i, tp in enumerate(target_points):
        for offset, key in enumerate(keys):
            if tp.get(key) is not None:
                row_selection[5 * i + offset, 5 * i + offset] = 1
    return row_selection[np.sum(row_selection, axis=1) != 0]
