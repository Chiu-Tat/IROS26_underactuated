"""Selectivity figure: shrink one motor's CFW and show every motor's MFW.

For a chosen joint configuration, build the actuated motor's Current-Feasible
Workspace (Algorithm 1), then map it (Algorithm 2) onto each of the three motors
and plot the resulting Magnetic-Feasible Workspace against that motor's minimum
actuation circle. Only the actuated motor's MFW should enclose its circle
(paper Fig. 10).

    python scripts/plot_cfw_mfw.py
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from selective_em import (
    map_i2b, extract_map_i2b, get_cfw_polytope,
    transform_and_extract_facets, robot_arm_kinematics, WORKSPACE_COILS,
)
from selective_em.visualization import plot_polytope_2d


def _target(x, y, z):
    return [{
        "X": x, "Y": y, "Z": z, "Bx": True, "By": True, "Bz": None,
        "Bx_dx": None, "Bx_dy": None, "Bx_dz": None, "By_dy": None, "By_dz": None,
    }]


def main(alpha=50, beta=-30, i_abs_max=15, m=2000, radii=(0.02, 0.03, 0.02),
         coils=WORKSPACE_COILS):
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(alpha, beta)
    motors = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]

    # Shrink the CFW using the actuated motor (motor 1) at radius r1.
    actuated = _target(*motors[0])
    A = extract_map_i2b(actuated) @ map_i2b(actuated, coils)
    Q = A.T @ A
    hull, _, _, _ = get_cfw_polytope(Q, radii[0] ** 2, i_abs_max, m,
                                     plot_verbose=True)
    G = hull.equations[:, :-1]
    k = -hull.equations[:, -1]

    # Map the shrunk CFW onto every motor and plot MFW vs. actuation circle.
    for idx, (pos, r) in enumerate(zip(motors, radii), start=1):
        tp = _target(*pos)
        A_i = extract_map_i2b(tp) @ map_i2b(tp, coils)
        N, d = transform_and_extract_facets(A_i, G, k)
        print(f"Motor {idx}: MSD (min d) = {float(np.min(d)):.6e}  (circle r = {r})")
        plot_polytope_2d(N, d, radius=r)


if __name__ == "__main__":
    main()
