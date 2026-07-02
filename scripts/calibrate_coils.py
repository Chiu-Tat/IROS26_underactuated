"""Track a rotating field with an unconstrained (penalty-method) current solve.

Solves ``min ||i||^2`` with soft penalties for field tracking and the current
box, at each control step over the simulation. Uses all 10 coils. Saves the
per-step currents to ``data/coil_currents_<amp>mT.csv`` and shows diagnostics.

    python scripts/calibrate_coils.py
"""

import sys
import pathlib
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from selective_em import FULL_ARRAY
from selective_em.control import (
    rotating_field_target, precompute_field_matrices,
    unconstrained_objective, unconstrained_gradient, plot_optimization_results,
)

# --- Control parameters ---
CURRENT_LIMIT = 15
FIELD_AMPLITUDE = 0.057            # Tesla
ANGULAR_VELOCITY = 2 * np.pi / 10  # rad/s
SIMULATION_DURATION = 30.0         # seconds
CONTROL_FREQUENCY = 20             # Hz
CONTROL_PERIOD = 1.0 / CONTROL_FREQUENCY
TARGET_POSITION = (0.0, -0.03, 0.0)
COILS = FULL_ARRAY
N_COILS = len(COILS)


def run_time_series_optimization():
    """Unconstrained penalty-method solve at every control step."""
    time_points = np.arange(0, SIMULATION_DURATION + CONTROL_PERIOD, CONTROL_PERIOD)
    print(f"Optimizing at {CONTROL_FREQUENCY} Hz for {SIMULATION_DURATION:.1f}s "
          f"({len(time_points)} steps)")

    results = {k: [] for k in ("time", "currents", "optimization_time", "success",
                               "objective_value", "max_field_error",
                               "rms_field_error", "max_current")}
    currents_guess = np.ones(N_COILS) * 0.1

    for i, t in enumerate(time_points):
        if i % 10 == 0:
            print(f"  t={t:.2f}s ({i + 1}/{len(time_points)})")
        target_points = rotating_field_target(t, TARGET_POSITION,
                                               FIELD_AMPLITUDE, ANGULAR_VELOCITY)
        A_matrix, target_vector = precompute_field_matrices(target_points, COILS)

        t0 = time.time()
        result = minimize(
            lambda x: unconstrained_objective(x, A_matrix, target_vector, CURRENT_LIMIT),
            currents_guess, method="L-BFGS-B",
            jac=lambda x: unconstrained_gradient(x, A_matrix, target_vector, CURRENT_LIMIT),
            options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 1000},
        )
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
        if result.success:
            currents_guess = result.x

    print(f"Success rate: {np.mean(results['success']):.1%}; "
          f"max current: {np.max(results['max_current']):.2f} A")
    return results


if __name__ == "__main__":
    results = run_time_series_optimization()
    plot_optimization_results(results, FIELD_AMPLITUDE, ANGULAR_VELOCITY,
                              CURRENT_LIMIT, N_COILS)

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    times = np.array(results["time"])
    currents_array = np.array(results["currents"])
    currents_df = pd.DataFrame({"Time(s)": times})
    for i in range(N_COILS):
        currents_df[f"Current_Coil_{i + 1}(A)"] = currents_array[:, i]
    out = data_dir / f"coil_currents_{FIELD_AMPLITUDE * 1000:.0f}mT.csv"
    currents_df.to_csv(out, index=False)
    print(f"Currents saved to {out}")
