"""Selective per-step current solve (paper's experiment controller).

At each control step solves ``min ||i||^2`` subject to the field-tracking
equality ``A i = b``, the current box, and the selectivity constraint
``N_I i <= d_I`` (the shrunk CFW that minimizes the influence region). Uses the
3-coil experiment configuration (coils 2, 3, 7). Saves per-step currents to
``data/coil_currents_<amp>mT.csv``.

    python scripts/run_field_optimization.py
"""

import sys
import pathlib
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from selective_em import (
    EXPERIMENT_COILS, map_i2b, extract_map_i2b,
)
from selective_em.control import (
    rotating_field_target, precompute_field_matrices, plot_optimization_results,
)

# --- Control parameters ---
CURRENT_LIMIT = 17
FIELD_AMPLITUDE = 0.057            # Tesla
ANGULAR_VELOCITY = 2 * np.pi / 10  # rad/s
SIMULATION_DURATION = 10.0         # seconds
CONTROL_FREQUENCY = 20             # Hz
CONTROL_PERIOD = 1.0 / CONTROL_FREQUENCY
TARGET_POSITION = (0.0, -0.03, 0.0)
CFW_RADIUS = 0.057                 # field radius used to shrink the CFW
COILS = EXPERIMENT_COILS
N_COILS = len(COILS)
ACTIVE_COIL_NUMBERS = [2, 3, 7]    # 1-based coil labels for the CSV columns


def selectivity_constraint():
    """Exact CFW-shrinking selectivity constraint ``i^T Q i <= r^2`` at the target.

    The selective constraint is exactly ``||A(p) i|| <= CFW_RADIUS`` -- a single
    convex quadratic that SLSQP handles directly, replacing the many-facet
    polytope approximation ``N i <= d`` (Algorithm 1) with no loss of accuracy.
    """
    target_points = [{
        "X": TARGET_POSITION[0], "Y": TARGET_POSITION[1], "Z": TARGET_POSITION[2],
        "Bx": True, "By": True, "Bz": None,
        "Bx_dx": None, "Bx_dy": None, "Bx_dz": None, "By_dy": None, "By_dz": None,
    }]
    A = extract_map_i2b(target_points) @ map_i2b(target_points, COILS)
    return A.T @ A                         # Q; the constraint is i^T Q i <= r^2


def run_time_series_optimization():
    """Constrained (SLSQP) selective solve at every control step."""
    print("Computing selectivity constraint...")
    Q = selectivity_constraint()
    r2 = CFW_RADIUS ** 2
    print(f"Exact quadratic selectivity constraint i^T Q i <= {r2:.3e}")

    time_points = np.arange(0, SIMULATION_DURATION + CONTROL_PERIOD, CONTROL_PERIOD)
    results = {k: [] for k in ("time", "currents", "optimization_time", "success",
                               "objective_value", "max_field_error",
                               "rms_field_error", "max_current",
                               "convex_constraint_violations")}
    currents_guess = np.ones(N_COILS) * 0.1

    for i, t in enumerate(time_points):
        if i % 10 == 0:
            print(f"  t={t:.2f}s ({i + 1}/{len(time_points)})")
        target_points = rotating_field_target(t, TARGET_POSITION,
                                               FIELD_AMPLITUDE, ANGULAR_VELOCITY)
        A_matrix, target_vector = precompute_field_matrices(target_points, COILS)

        constraints = [
            {"type": "eq", "fun": lambda x, A=A_matrix, b=target_vector: A @ x - b},
            {"type": "ineq", "fun": lambda x: CURRENT_LIMIT - np.abs(x)},
            {"type": "ineq", "fun": lambda x, Q=Q, r2=r2: r2 - x @ Q @ x,
             "jac": lambda x, Q=Q: -2.0 * (Q @ x)},
        ]
        t0 = time.time()
        result = minimize(lambda x: np.sum(x ** 2), currents_guess, method="SLSQP",
                          constraints=constraints,
                          options={"ftol": 1e-12, "disp": False, "maxiter": 1000})
        dt = time.time() - t0

        field_errors = A_matrix @ result.x - target_vector
        results["time"].append(t)
        results["currents"].append(result.x.copy())
        results["optimization_time"].append(dt)
        results["success"].append(result.success)
        results["objective_value"].append(result.fun)
        results["max_field_error"].append(np.max(np.abs(field_errors)))
        results["rms_field_error"].append(np.sqrt(np.mean(field_errors ** 2)))
        results["max_current"].append(np.max(np.abs(result.x)))
        results["convex_constraint_violations"].append(
            max(0.0, float(result.x @ Q @ result.x) - r2))
        if result.success:
            currents_guess = result.x

    print(f"Success rate: {np.mean(results['success']):.1%}; "
          f"max current: {np.max(results['max_current']):.2f} A; "
          f"max constraint violation: {np.max(results['convex_constraint_violations']):.2e}")
    return results


if __name__ == "__main__":
    results = run_time_series_optimization()
    plot_optimization_results(results, FIELD_AMPLITUDE, ANGULAR_VELOCITY,
                              CURRENT_LIMIT, N_COILS)

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    times = np.array(results["time"])
    currents_array = np.array(results["currents"])  # (n_times, N_COILS)
    currents_df = pd.DataFrame({"Time(s)": times})
    for i in range(10):
        currents_df[f"Current_Coil_{i + 1}(A)"] = 0.0
    for col_idx, coil_num in enumerate(ACTIVE_COIL_NUMBERS):
        currents_df[f"Current_Coil_{coil_num}(A)"] = currents_array[:, col_idx]
    out = data_dir / f"coil_currents_{FIELD_AMPLITUDE * 1000:.0f}mT.csv"
    currents_df.to_csv(out, index=False)
    print(f"Currents saved to {out}")
