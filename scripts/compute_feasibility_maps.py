"""Per-motor selective-actuation feasibility map over the joint grid.

For the chosen motor, sweeps the (alpha, beta) grid and marks each configuration
as feasible when that motor is selectively actuable (in its own OAR and no other
motor in its influence region). Replaces the old Check_feasibility_p1/p2/p3
scripts, which differed only in the actuated motor.

    python scripts/compute_feasibility_maps.py --motor 1

Return codes per grid cell:  1 feasible,  -1 infeasible,  -10 tip outside sphere.
"""

import argparse
import sys
import pathlib
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from selective_em import robot_arm_kinematics, is_selectively_actuable, WORKSPACE_COILS

# Radii (minimum actuation circle) per motor, and the reachability mask radius.
RADII = (0.02, 0.03, 0.02)
RATIO = 0.92
TIP_RADIUS = 0.045


def check_single_point(args):
    """Worker: is `motor` selectively actuable at this (alpha, beta)?"""
    alpha, beta, i_min, i_max, motor = args
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(alpha, beta)
    if x3 ** 2 + y3 ** 2 > TIP_RADIUS ** 2:
        return -10

    positions = [np.array([x1, y1, z1]), np.array([x2, y2, z2]),
                 np.array([x3, y3, z3])]
    idx = motor - 1
    actuated = positions[idx]
    others = [positions[o] for o in range(3) if o != idx]

    # Actuated motor keeps its full radius; the others are scaled by RATIO.
    b_min_actuated = RADII[idx]
    other_b_mins = [RADII[o] * RATIO for o in range(3) if o != idx]

    try:
        ok = is_selectively_actuable(actuated, others, b_min_actuated,
                                     other_b_mins, i_min, i_max, WORKSPACE_COILS)
    except RuntimeError:
        return -1
    return 1 if ok else -1


def main(motor=1, i_min=-15, i_max=15, n=20, num_workers=10):
    xx = np.linspace(-90, 90, n)
    yy = np.linspace(-90, 90, n)
    X, Y = np.meshgrid(xx, yy)

    args_list = [(X[i, j], Y[i, j], i_min, i_max, motor)
                 for i in range(len(xx)) for j in range(len(yy))]
    print(f"Feasibility map for motor {motor}: {len(args_list)} points, "
          f"{num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        results = pool.map(check_single_point, args_list)
    P = np.array(results).reshape(X.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(X, Y, P, levels=[0, np.inf], colors=["lightgreen"], alpha=0.9)
    plt.contourf(X, Y, P, levels=[-10, -9], colors=["lightcoral"], alpha=0.9)
    plt.contourf(X, Y, P, levels=[-1, 0], colors=["lightgray"], alpha=0.9)
    plt.gca().set_aspect("equal")
    plt.xlabel(r"$\alpha$ (degrees)", fontsize=18)
    plt.ylabel(r"$\beta$ (degrees)", fontsize=18)
    ticks = [-90, -45, 0, 45, 90]
    plt.xticks(ticks, fontsize=18)
    plt.yticks(ticks, fontsize=18)
    plt.grid()
    plt.gca().set_facecolor("lightgray")
    plt.title(f"Motor {motor} selective-actuation feasibility")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    main(motor=args.motor)
