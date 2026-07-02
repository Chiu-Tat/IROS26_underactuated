"""Selectivity checks: Omni-Actuation Region, Influence Region, and the theorem.

Implements the paper's selectivity test. A motor at ``actuated_position`` can be
selectively rotated iff:

    1. it lies in its own **Omni-Actuation Region (OAR)** -- the MFW encloses its
       minimum actuation circle (MSD >= b_min), and
    2. no *other* motor lies in the actuated motor's **Influence Region (IR)** --
       i.e. while the actuated motor's field is shaped (CFW shrunk to radius
       ``b_min``), no other motor's field reaches its own minimum actuation circle.

All helpers accept an explicit ``coils`` array (see :mod:`selective_em.coils`).
"""

import numpy as np

from .coils import DEFAULT_COILS
from .field_model import map_i2b, extract_map_i2b
from .workspace import get_mfw, get_cfw_polytope, transform_and_extract_facets


def _single_target(position):
    """Build a 2D field target dict (Bx, By constrained) at a 3D position."""
    return [{
        "X": position[0], "Y": position[1], "Z": position[2],
        "Bx": True, "By": True, "Bz": None,
        "Bx_dx": None, "Bx_dy": None, "Bx_dz": None,
        "By_dy": None, "By_dz": None,
    }]


def in_oar(position, b_min, i_min, i_max, coils=DEFAULT_COILS):
    """True if ``position`` is inside its OAR (MFW encloses the b_min circle)."""
    G, K = get_mfw(_single_target(position), i_min, i_max, coils)
    return bool((K >= b_min).all())


def is_selectively_actuable(actuated_position, other_positions, b_min_actuated,
                            other_b_mins, i_min, i_max, coils=DEFAULT_COILS,
                            m=1000):
    """Selectivity theorem: actuated motor in its OAR and no other in its IR.

    ``b_min_actuated`` is the actuated motor's minimum actuation radius (also the
    CFW-shrinking radius); ``other_b_mins`` are the radii of the other motors.
    Returns ``True`` when the actuated motor can be selectively rotated.
    """
    if not in_oar(actuated_position, b_min_actuated, i_min, i_max, coils):
        return False
    return not any(
        _in_influence_region(other, actuated_position, i_max, b_min_actuated,
                             b_min_other, coils, m)
        for other, b_min_other in zip(other_positions, other_b_mins)
    )


def _in_influence_region(target_position, actuated_position, i_max, r_actuated,
                         b_min_target, coils, m):
    """True if ``target_position`` is inside the actuated motor's IR.

    The actuated motor's CFW is the ellipsoid ``i^T Q i <= r_actuated^2``
    intersected with the current box; its MFW is transformed onto the target and
    the target is "influenced" when the target's MSD reaches ``b_min_target``.
    """
    actuated_tp = _single_target(actuated_position)
    A = extract_map_i2b(actuated_tp) @ map_i2b(actuated_tp, coils)
    Q = A.T @ A
    hull, _, _, _ = get_cfw_polytope(Q, r_actuated ** 2, i_max, m,
                                     plot_verbose=False)
    G = hull.equations[:, :-1]
    k = -hull.equations[:, -1]

    target_tp = _single_target(target_position)
    A_target = extract_map_i2b(target_tp) @ map_i2b(target_tp, coils)
    _, d_target = transform_and_extract_facets(A_target, G, k)
    return float(np.min(d_target)) >= b_min_target


def point_in_polytope(point, N, d):
    """True if a spatial ``point`` lies inside the polytope ``N . point <= d``.

    Used for testing membership in a spatial OAR/IR map (H-representation over
    positions rather than fields).
    """
    return bool(np.all(N @ np.asarray(point).T <= d))
