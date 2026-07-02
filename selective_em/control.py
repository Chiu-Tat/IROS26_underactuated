"""Per-step current solving for tracking a rotating magnetic field.

Shared building blocks used by the calibration and experiment runners:

    * :func:`rotating_field_target` -- rotating-field target at a time instant.
    * :func:`precompute_field_matrices` -- the (Bx, By) field matrix and target
      vector for a set of target points.
    * :func:`unconstrained_objective` / :func:`unconstrained_gradient` -- penalty
      method for ``min ||i||^2`` subject to ``A i = b`` and the current box.
    * :func:`plot_optimization_results` -- the 6-panel diagnostic figure.

The paper's per-step controller solves ``min ||i||^2  s.t.  A(p) i = b`` (track
the target field) together with the selectivity constraint ``N_I i <= d_I``
(shrink the CFW to minimize the influence region). The equality/box/selectivity
constraints are assembled by the experiment runner using these helpers and
:func:`selective_em.workspace.get_cfw_polytope`.
"""

import numpy as np

from .coils import DEFAULT_COILS
from .field_model import dipole_model_Bx, dipole_model_By


def rotating_field_target(time_value, position, field_amplitude, angular_velocity):
    """Rotating-field target point at ``time_value`` (Bx, By on a circle)."""
    bx = field_amplitude * np.cos(angular_velocity * time_value)
    by = field_amplitude * np.sin(angular_velocity * time_value)
    x, y, z = position
    return [{"X": x, "Y": y, "Z": z, "Bx": bx, "By": by}]


def precompute_field_matrices(target_points, coils=DEFAULT_COILS):
    """Field matrix ``A`` (Bx rows stacked over By rows) and target vector.

    ``A`` has ``2 * n_points`` rows and one column per coil; the target vector
    stacks all Bx targets followed by all By targets. Target dicts here carry
    numeric ``Bx``/``By`` values (not the True/None flags used for workspace
    analysis).
    """
    coils = np.asarray(coils)
    n_coils = len(coils)
    n_points = len(target_points)

    target_coords = np.array([[p["X"], p["Y"], p["Z"]] for p in target_points])
    target_fields = np.array([[p["Bx"], p["By"]] for p in target_points])

    Bx_matrix = np.zeros((n_points, n_coils))
    By_matrix = np.zeros((n_points, n_coils))

    x_vec = target_coords[:, 0]
    y_vec = target_coords[:, 1]
    z_vec = target_coords[:, 2]
    for i in range(n_coils):
        m0, m1, m2, r0_0, r0_1, r0_2 = coils[i]
        Bx_matrix[:, i] = dipole_model_Bx(m0, m1, m2, r0_0, r0_1, r0_2,
                                          x_vec, y_vec, z_vec)
        By_matrix[:, i] = dipole_model_By(m0, m1, m2, r0_0, r0_1, r0_2,
                                          x_vec, y_vec, z_vec)

    A_matrix = np.vstack([Bx_matrix, By_matrix])
    target_vector = np.hstack([target_fields[:, 0], target_fields[:, 1]])
    return A_matrix, target_vector


def unconstrained_objective(currents, A_matrix, target_vector, current_limit,
                            penalty_weight=1e15):
    """Penalty-method objective: ``||i||^2`` + field-tracking + current-box penalties."""
    original_obj = np.sum(currents ** 2)
    predicted_fields = A_matrix @ currents
    field_error = np.sum((predicted_fields - target_vector) ** 2)
    current_violation = np.maximum(0, np.abs(currents) - current_limit)
    current_penalty = np.sum(current_violation ** 2)
    return original_obj + penalty_weight * (field_error + current_penalty)


def unconstrained_gradient(currents, A_matrix, target_vector, current_limit,
                           penalty_weight=1e15):
    """Analytical gradient of :func:`unconstrained_objective`."""
    grad_original = 2 * currents
    field_residual = A_matrix @ currents - target_vector
    grad_field = 2 * penalty_weight * (A_matrix.T @ field_residual)
    current_violation = np.maximum(0, np.abs(currents) - current_limit)
    grad_current = 2 * penalty_weight * current_violation * np.sign(currents)
    return grad_original + grad_field + grad_current


def plot_optimization_results(results_data, field_amplitude, angular_velocity,
                              current_limit, n_coils):
    """Six-panel diagnostic figure for a time-series current optimization."""
    import matplotlib.pyplot as plt

    times = np.array(results_data["time"])
    currents_array = np.array(results_data["currents"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(times, currents_array * 1000)  # mA
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Current (mA)")
    axes[0, 0].set_title("Coil Currents Over Time")
    axes[0, 0].grid(True)
    axes[0, 0].legend([f"Coil {i+1}" for i in range(n_coils)],
                      bbox_to_anchor=(1.05, 1), loc="upper left")

    target_bx = field_amplitude * np.cos(angular_velocity * times) * 1000
    target_by = field_amplitude * np.sin(angular_velocity * times) * 1000
    axes[0, 1].plot(times, target_bx, "r-", label="Target Bx")
    axes[0, 1].plot(times, target_by, "b-", label="Target By")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Magnetic Field (mT)")
    axes[0, 1].set_title("Target Rotating Magnetic Field")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[0, 2].plot(times, results_data["max_field_error"], "r-", label="Max Error")
    axes[0, 2].plot(times, results_data["rms_field_error"], "b-", label="RMS Error")
    axes[0, 2].set_xlabel("Time (s)")
    axes[0, 2].set_ylabel("Field Error (T)")
    axes[0, 2].set_title("Magnetic Field Errors")
    axes[0, 2].set_yscale("log")
    axes[0, 2].grid(True)
    axes[0, 2].legend()

    axes[1, 0].plot(times, results_data["optimization_time"], "g-")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Optimization Time (s)")
    axes[1, 0].set_title("Optimization Time per Step")
    axes[1, 0].grid(True)

    axes[1, 1].plot(times, np.array(results_data["max_current"]), "b-", label="Max Current")
    axes[1, 1].axhline(y=current_limit, color="r", linestyle="--",
                       label=f"Current Limit ({current_limit}A)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Current (A)")
    axes[1, 1].set_title("Maximum Current vs Limit")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    axes[1, 2].plot(target_bx, target_by, "k-", linewidth=2, label="Target Trajectory")
    axes[1, 2].set_xlabel("Bx (mT)")
    axes[1, 2].set_ylabel("By (mT)")
    axes[1, 2].set_title("Magnetic Field Trajectory")
    axes[1, 2].grid(True)
    axes[1, 2].axis("equal")
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()
