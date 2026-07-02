"""Plot Omni-Actuation Region (OAR) maps.

``--mode single`` sweeps a single target over a spatial plane and marks where the
MFW encloses the minimum actuation circle (MSD >= B_amp). ``--mode multi`` sweeps
the two joint angles and checks the 3-motor multi-target MFW against per-motor
radii (the coupled feasibility used by the arm).

    python scripts/plot_oar_map.py --mode single
    python scripts/plot_oar_map.py --mode multi
"""

import argparse
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from selective_em import get_mfw, robot_arm_kinematics, WORKSPACE_COILS


def plot_single(i_min=-15, i_max=15, b_amp=0.03, coils=WORKSPACE_COILS):
    """Single-target OAR over the z = -0.04 plane."""
    xx = np.linspace(-0.08, 0.08, 50)
    zz = np.linspace(-0.08, 0.08, 50)
    X, Z = np.meshgrid(zz, xx)

    P = np.zeros_like(X)
    for i in range(len(xx)):
        for j in range(len(zz)):
            target_points = [{
                "X": X[i, j], "Y": Z[i, j], "Z": -0.04,
                "Bx": True, "By": True, "Bz": None,
                "Bx_dx": None, "Bx_dy": None, "Bx_dz": None,
                "By_dy": None, "By_dz": None,
            }]
            _, K = get_mfw(target_points, i_min, i_max, coils)
            P[i, j] = 1 if (K >= b_amp).all() else -1

    plt.scatter(X[P == 1], Z[P == 1], c="green", s=5, label="In workspace")
    plt.scatter(X[P == -1], Z[P == -1], c="salmon", s=5, label="Out workspace")
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("Z")
    plt.legend()
    plt.grid()
    plt.gca().set_facecolor("lightgray")
    plt.show()


def plot_multi(i_min=-15, i_max=15, radii=(0.015, 0.015, 0.015),
               coils=WORKSPACE_COILS):
    """Multi-target (3-motor) OAR over the (alpha, beta) joint grid."""
    r1, r2, r3 = radii
    xx = np.linspace(-90, 90, 20)
    yy = np.linspace(-90, 90, 20)
    X, Y = np.meshgrid(xx, yy)

    P = np.zeros_like(X)
    for i in range(len(xx)):
        for j in range(len(yy)):
            x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(X[i, j], Y[i, j])
            target_points = [
                {"X": x1, "Y": y1, "Z": z1, "Bx": True, "By": True, "Bz": None,
                 "Bx_dx": None, "Bx_dy": None, "Bx_dz": None, "By_dy": None, "By_dz": None},
                {"X": x2, "Y": y2, "Z": z2, "Bx": True, "By": True, "Bz": None,
                 "Bx_dx": None, "Bx_dy": None, "Bx_dz": None, "By_dy": None, "By_dz": None},
                {"X": x3, "Y": y3, "Z": z3, "Bx": True, "By": True, "Bz": None,
                 "Bx_dx": None, "Bx_dy": None, "Bx_dz": None, "By_dy": None, "By_dz": None},
            ]
            N, d = get_mfw(target_points, i_min, i_max, coils)
            r_values = []
            for k in range(N.shape[0]):
                denom = (r1 * np.sqrt(N[k, 0] ** 2 + N[k, 1] ** 2)
                         + r2 * np.sqrt(N[k, 2] ** 2 + N[k, 3] ** 2)
                         + r3 * np.sqrt(N[k, 4] ** 2 + N[k, 5] ** 2))
                r_values.append(d[k] - denom)
            rr = min(r_values) if r_values else np.nan
            P[i, j] = 1 if rr >= 0 else -1

    plt.scatter(X[P == 1], Y[P == 1], c="green", s=5, label="In workspace")
    plt.scatter(X[P == -1], Y[P == -1], c="salmon", s=5, label="Out workspace")
    plt.gca().set_aspect("equal")
    plt.xlabel(r"$\alpha$ (degrees)")
    plt.ylabel(r"$\beta$ (degrees)")
    plt.legend()
    plt.grid()
    plt.gca().set_facecolor("lightgray")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    args = parser.parse_args()
    if args.mode == "single":
        plot_single()
    else:
        plot_multi()
