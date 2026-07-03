"""Magnetic-Feasible Workspace (MFW) and Current-Feasible Workspace (CFW) geometry.

Maps directly onto the paper:

    * :func:`hyperplane_shifting_method` -- H-representation of the reachable-field
      zonotope for box current limits.
    * :func:`get_mfw` -- Magnetic-Feasible Workspace (MFW) at target points.
    * :func:`minimal_supporting_distance` -- MSD, the smallest ``d`` in ``N . B <= d``.
    * :func:`get_cfw_polytope` -- Algorithm 1: polytope H-representation of the CFW
      (ellipsoid intersect box) via surface sampling + projection + convex hull.
    * :func:`transform_and_extract_facets` -- Algorithm 2: MFW facets from a general
      CFW by mapping its vertices through the actuation matrix.
"""

from itertools import combinations

import numpy as np
from scipy.linalg import null_space
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

from .coils import DEFAULT_COILS
from .field_model import map_i2b, extract_map_i2b


def minimal_supporting_distance(d):
    """MSD: the smallest supporting distance in an MFW H-representation ``N . B <= d``."""
    return float(np.min(d))


def remove_duplicate_rows(N, d):
    """Remove duplicate ``(N, d)`` rows jointly, preserving original order."""
    if N.shape[0] != d.shape[0]:
        raise ValueError("Both matrices must have the same number of rows")
    combined = np.hstack((N, d))
    _, indices = np.unique(combined, axis=0, return_index=True)
    sorted_indices = np.sort(indices)
    return N[sorted_indices], d[sorted_indices]


def hyperplane_shifting_method(A, i_min, i_max):
    """H-representation ``(N, d_vec)`` of the zonotope A * i with i in [i_min, i_max].

    Generalized hyperplane-shifting method (Skuric et al.); ``A`` is the
    actuation matrix, ``i_min``/``i_max`` the per-coil current bounds (scalars).
    """
    def create_permutation_matrix(A):
        d = np.shape(A)[0]          # output-space dimension
        n = np.shape(A)[1]          # number of coils
        return np.asarray(list(combinations(np.arange(n), d - 1)))

    def create_field_combination_matrix(n):
        nums = np.arange(2 ** n)
        return ((nums.reshape(-1, 1) & (2 ** np.arange(n))) != 0).astype(int)

    dI = i_max - i_min
    M = create_permutation_matrix(A)
    nb_comb = np.shape(M)[0]

    d = np.shape(A)[0]
    n_coils = np.shape(A)[1]

    N = np.zeros((2 * nb_comb, d))
    d_vec = np.zeros((2 * nb_comb, 1))
    bmin = np.matmul(A, i_min * np.ones((n_coils, 1)))

    for i in range(nb_comb):
        # Step 1: initial hyperplane orthogonal to the selected unitary fields.
        W = A[:, M[i, :]]
        Wns = null_space(np.transpose(W))
        v = Wns[:, 0]

        # Step 2: normalize.
        temp = v / np.linalg.norm(v)
        n = temp.reshape((-1, 1))

        # Step 3: projections of the remaining columns.
        lj_arr = np.zeros((n_coils - (d - 1), 1))
        k = 0
        for j in range(n_coils):
            if j not in M[i, :]:
                lj = np.dot(np.transpose(A[:, j]), n)
                lj_arr[k, 0] = lj if np.isscalar(lj) else lj[0]
                k += 1

        C = create_field_combination_matrix(n_coils - (d - 1))
        h = np.matmul(C, dI * lj_arr)
        hp = np.max(h)
        hm = np.min(h)

        # Step 4: hyperplane supports.
        pp = hp * n + bmin
        pm = hm * n + bmin

        # Step 5: build the +/- hyperplane pair.
        N[i, :] = n.T
        N[i + nb_comb, :] = -n.T
        d_vec[i, :] = np.dot(n.T, pp)
        d_vec[i + nb_comb, :] = np.dot(-n.T, pm)
    return N, d_vec


def get_mfw(target_points, i_min, i_max, coils=DEFAULT_COILS):
    """Magnetic-Feasible Workspace at ``target_points`` for box current limits.

    Returns ``(G, k)`` such that the reachable field set is ``{B : G . B <= k}``.
    """
    A = extract_map_i2b(target_points) @ map_i2b(target_points, coils)
    G, k = hyperplane_shifting_method(A, i_min, i_max)
    k = k.reshape(-1, 1)
    return G, k


def transform_and_extract_facets(A, G, K):
    """Algorithm 2: MFW facets from a CFW polytope ``{i : G . i <= K}``.

    Enumerate CFW vertices, map them through ``A`` (``w = A . v``), and take the
    convex hull. Returns the MFW H-representation ``(N, d_vec)``.
    """
    from pypoman import compute_polytope_vertices  # heavy (cddlib); old path only
    vertices = np.array(compute_polytope_vertices(G, K))
    transformed_vertices = np.dot(vertices, A.T)
    hull = ConvexHull(transformed_vertices)
    N = -hull.equations[:, :-1]
    d_vec = -hull.equations[:, -1].reshape(-1, 1)
    return N, d_vec


def get_cfw_polytope(Q, r, i_abs_max, m, plot_verbose=False,
                     regularization_factor=1e-10):
    """Algorithm 1: polytope H-representation of the Current-Feasible Workspace.

    The CFW is the intersection of the ellipsoid ``i^T Q i <= r`` with the box
    ``-i_abs_max <= i_k <= i_abs_max``. Since that intersection is not itself a
    polytope, sample ``m`` points on the ellipsoid surface, project points that
    fall outside the box back onto it, and take the convex hull.

    Returns ``(hull, h_representation, remaining_points, deleted_points)`` where
    ``h_representation`` is ``{'A': ..., 'b': ...}`` for ``A i <= b``.
    """
    n = Q.shape[0]
    i_max = i_abs_max

    eigenvalues = np.linalg.eigvals(Q)
    min_eigenvalue = np.min(eigenvalues)
    is_positive_definite = min_eigenvalue > 1e-12

    if not is_positive_definite:
        # Regularize an indefinite/degenerate Q so the ellipsoid is well-defined.
        regularization_needed = max(regularization_factor,
                                    abs(min_eigenvalue) + 1e-8)
        Q = Q + regularization_needed * np.eye(n)

    eigenvals, eigenvecs = np.linalg.eigh(Q)
    eigenvals = np.maximum(eigenvals, 1e-12)
    sqrt_lambda = np.sqrt(eigenvals)

    # --- Generate m points on the unit sphere in current space ---
    if m == 0:
        points_on_unit_sphere = np.empty((0, n))
    elif n == 2:
        theta = np.linspace(0, 2 * np.pi, m, endpoint=False)
        points_on_unit_sphere = np.column_stack([np.cos(theta), np.sin(theta)])
    elif n == 3:
        indices = np.arange(0, m, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / m)
        theta = np.pi * (1 + np.sqrt(5)) * indices
        points_on_unit_sphere = np.column_stack([
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ])
    else:
        random_points = np.random.normal(size=(m, n))
        norm = np.linalg.norm(random_points, axis=1, keepdims=True)
        points_on_unit_sphere = np.zeros_like(random_points)
        non_zero = (norm > 1e-15).flatten()
        points_on_unit_sphere[non_zero] = random_points[non_zero] / norm[non_zero]
        if np.any(~non_zero):
            canonical = np.zeros(n)
            canonical[0] = 1.0
            points_on_unit_sphere[~non_zero] = canonical

    # --- Scale onto the ellipsoid surface ---
    semi_axes = np.sqrt(r) / sqrt_lambda
    ellipsoid_coords_principal = points_on_unit_sphere * semi_axes.reshape(1, -1)
    points_on_ellipsoid = ellipsoid_coords_principal @ eigenvecs.T

    # --- Project points outside the box inward onto the box ---
    original_points = points_on_ellipsoid.copy()
    max_abs_coord = np.max(np.abs(original_points), axis=1)
    outside_mask = max_abs_coord > i_max

    points_inside_bounds = original_points.copy()
    if np.any(outside_mask):
        outside_points = original_points[outside_mask]
        scaling_factors = i_max / max_abs_coord[outside_mask].reshape(-1, 1)
        points_inside_bounds[outside_mask] = outside_points * scaling_factors

    deleted_points = original_points[outside_mask]
    remaining_points = points_inside_bounds

    if remaining_points.shape[0] < n + 1:
        return None, None, remaining_points, deleted_points

    hull = ConvexHull(remaining_points)
    a_matrix = hull.equations[:, :-1]
    b_vector = -hull.equations[:, -1]
    h_representation = {"A": a_matrix, "b": b_vector}

    if plot_verbose:
        # Lazy import avoids a circular dependency with visualization.
        from . import visualization
        if n == 2:
            visualization.plot_cfw_2d(hull, remaining_points, deleted_points,
                                      i_max, Q, r)
        elif n == 3:
            visualization.plot_cfw_3d(hull, remaining_points, deleted_points,
                                      i_max, Q, r)

    return hull, h_representation, remaining_points, deleted_points


# ---------------------------------------------------------------------------
# Support-function MFW/CFW (improved replacement for Algorithms 1 & 2).
#
# The CFW is the *slab* {i : ||A_act i|| <= r} intersected with the current box
# {||i||_inf <= i_max}. We never build it as a polytope. Instead we evaluate the
# support function of the target MFW directly:
#
#     h_MFW(u) = max_{i in CFW} u . (A_target i) = h_CFW(A_target^T u),
#     h_CFW(c) = max { c . i : ||A_act i|| <= r,  ||i||_inf <= i_max },
#
# a small convex QCQP in the current vector. Sampling K unit directions u in the
# 2-D field plane yields, per direction, a supporting half-plane (outer bound)
# and a genuine MFW point A_target i* (inner bound), giving a deterministic,
# two-sided, provably-conservative bracket on the MSD:
#
#     inradius(conv{A_target i*_k}) <= MSD <= min_k h(u_k).
#
# This is deterministic for any coil count, needs no Q-regularisation and no
# vertex enumeration (no cddlib), and supports early-exit selectivity tests.
# ---------------------------------------------------------------------------


def cfw_support(c, Q, r, i_max, x0=None):
    """Support value ``h_CFW(c) = max c.i`` over the CFW, with its maximiser.

    The CFW is ``{i : i^T Q i <= r^2, ||i||_inf <= i_max}`` where ``Q = A_act^T
    A_act`` (rank-2, so the ``i^T Q i <= r^2`` set is an unbounded slab that the
    box makes bounded). Solved as a small convex QCQP with SLSQP; ``x0`` warm-
    starts the solve (``i = 0`` is always feasible). Returns ``(value, i_star)``.
    """
    c = np.asarray(c, dtype=float)
    n = Q.shape[0]
    r2 = float(r) ** 2

    # Closed-form shortcut: if the box optimum already satisfies the slab, the
    # quadratic constraint is inactive and no solve is needed.
    i_box = float(i_max) * np.sign(c)
    if float(i_box @ Q @ i_box) <= r2:
        return float(c @ i_box), i_box

    if x0 is None:
        x0 = np.zeros(n)
    res = minimize(
        lambda i: -float(c @ i), x0, jac=lambda i: -c, method="SLSQP",
        bounds=[(-i_max, i_max)] * n,
        constraints=[{
            "type": "ineq",
            "fun": lambda i: r2 - float(i @ Q @ i),
            "jac": lambda i: -2.0 * (Q @ i),
        }],
        options={"ftol": 1e-12, "maxiter": 200},
    )
    return -float(res.fun), res.x


def mfw_support(A_target, A_act, r, i_max, n_dirs=180):
    """MFW of ``A_target . CFW`` via support sampling (replaces Algorithms 1+2).

    ``A_act`` shapes the CFW (slab radius ``r`` plus current box ``i_max``);
    ``A_target`` maps it onto the motor being tested. Returns a dict with:

        ``N, d``          -- outer H-representation ``N B <= d`` (unit rows),
        ``inner_points``  -- genuine MFW points (inner approximation),
        ``msd``           -- ``min_k h(u_k)`` (deterministic support MSD),
        ``msd_lower``     -- inradius of the inner hull (guaranteed lower bound),
        ``msd_upper``     -- ``msd`` (guaranteed upper bound).

    ``msd_lower <= true MSD <= msd_upper`` always, and both converge to the true
    MSD as ``n_dirs`` grows (gap ~ O(1/n_dirs^2)).
    """
    Q = A_act.T @ A_act
    thetas = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    U = np.column_stack([np.cos(thetas), np.sin(thetas)])   # (K, 2) unit dirs
    H = np.empty(n_dirs)
    inner = np.empty((n_dirs, 2))
    x0 = None
    for k in range(n_dirs):
        h, i_star = cfw_support(A_target.T @ U[k], Q, r, i_max, x0)
        x0 = i_star                          # warm-start the next direction
        H[k] = h
        inner[k] = A_target @ i_star
    msd_upper = float(np.min(H))             # outer polygon inradius

    # Inner (inscribed) polygon inradius: a guaranteed lower bound on the MSD.
    msd_lower = 0.0
    if np.ptp(inner, axis=0).min() > 0:
        try:
            hull = ConvexHull(inner)
            # scipy hull eqs are unit-normalised: n.x + b <= 0 inside, so the
            # distance from the origin to each facet is -b (>0 if origin inside).
            msd_lower = float(np.min(-hull.equations[:, -1]))
        except Exception:
            msd_lower = 0.0

    return {
        "N": U, "d": H.reshape(-1, 1), "inner_points": inner,
        "msd": msd_upper, "msd_lower": max(msd_lower, 0.0), "msd_upper": msd_upper,
    }


def mfw_encloses_circle(A_target, A_act, r, i_max, rho, n_dirs=180):
    """Fast selectivity test: does ``A_target . CFW`` enclose the radius-``rho``
    circle? Evaluates the support MSD with early-exit -- the first direction with
    ``h(u_k) < rho`` proves ``MSD < rho`` and returns ``False`` immediately.
    """
    Q = A_act.T @ A_act
    thetas = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    x0 = None
    for theta in thetas:
        u = np.array([np.cos(theta), np.sin(theta)])
        h, x0 = cfw_support(A_target.T @ u, Q, r, i_max, x0)
        if h < rho:                          # MSD <= h < rho: cannot enclose
            return False
    return True
