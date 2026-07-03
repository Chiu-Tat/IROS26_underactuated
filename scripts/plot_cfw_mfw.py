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
    map_i2b, extract_map_i2b, mfw_support,
    robot_arm_kinematics, WORKSPACE_COILS,
)
from selective_em.visualization import plot_mfw_2d


def _target(x, y, z):
    return [{
        "X": x, "Y": y, "Z": z, "Bx": True, "By": True, "Bz": None,
        "Bx_dx": None, "Bx_dy": None, "Bx_dz": None, "By_dy": None, "By_dz": None,
    }]


def main(alpha=50, beta=-30, i_abs_max=15, n_dirs=180, radii=(0.02, 0.03, 0.02),
         coils=WORKSPACE_COILS):
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(alpha, beta)
    motors = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]

    # The actuated motor (motor 1) shapes the CFW: slab ||A_act i|| <= r1 + box.
    actuated = _target(*motors[0])
    A_act = extract_map_i2b(actuated) @ map_i2b(actuated, coils)

    # Map the shrunk CFW onto every motor and plot MFW vs. actuation circle.
    for idx, (pos, r) in enumerate(zip(motors, radii), start=1):
        tp = _target(*pos)
        A_i = extract_map_i2b(tp) @ map_i2b(tp, coils)
        res = mfw_support(A_i, A_act, radii[0], i_abs_max, n_dirs=n_dirs)
        print(f"Motor {idx}: MSD = {res['msd']:.6e} "
              f"(bracket [{res['msd_lower']:.6e}, {res['msd_upper']:.6e}], "
              f"circle r = {r})")
        plot_mfw_2d(res["inner_points"], radius=r, msd=res["msd"])


if __name__ == "__main__":
    main()
