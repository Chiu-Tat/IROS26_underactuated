"""Spatial actuation/influence map: sweep a target over a plane and test MSD >= B_amp.

Marks, for a fixed coil configuration and current box, the spatial region where
a single target's MFW encloses the minimum actuation circle. Useful for
visualizing how far a shaped field "reaches" (its influence region).

    python scripts/plot_influence_region.py
"""

import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from selective_em import get_mfw, minimal_supporting_distance, WORKSPACE_COILS


def main(i_max=17, b_amp=0.055, z=-0.05, coils=WORKSPACE_COILS):
    i_min = -i_max
    xx = np.linspace(-0.03, 0.03, 50)
    yy = np.linspace(-0.06, 0.0, 50)
    X, Y = np.meshgrid(xx, yy)

    R = 0.06
    circle_mask = (X ** 2 + Y ** 2) <= R ** 2
    P = np.full_like(X, np.nan)

    for i in range(len(xx)):
        for j in range(len(yy)):
            if circle_mask[i, j]:
                target_points = [{
                    "X": X[i, j], "Y": Y[i, j], "Z": z,
                    "Bx": True, "By": True, "Bz": None,
                    "Bx_dx": None, "Bx_dy": None, "Bx_dz": None,
                    "By_dy": None, "By_dz": None,
                }]
                _, K = get_mfw(target_points, i_min, i_max, coils)
                P[i, j] = 1 if minimal_supporting_distance(K) >= b_amp else -1

    valid = ~np.isnan(P)
    plt.scatter(X[(P == 1) & valid], Y[(P == 1) & valid], c="green", s=10,
                label="In workspace")
    plt.scatter(X[(P == -1) & valid], Y[(P == -1) & valid], c="salmon", s=10,
                label="Out workspace")

    theta = np.linspace(0, 2 * np.pi, 100)
    for radius in (R, R + 0.01):
        plt.plot(radius * np.cos(theta), radius * np.sin(theta), "k--",
                 linewidth=1, alpha=0.5)

    plt.gca().set_aspect("equal")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend(loc="upper right")
    plt.xlim(-0.03, 0.03)
    plt.ylim(-0.06, 0.0)
    plt.grid()
    plt.gca().set_facecolor("lightgray")
    plt.show()


if __name__ == "__main__":
    main()
