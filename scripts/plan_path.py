"""Plan a selective-actuation path across the joint grid (paper Algorithm 3).

Builds direction-dependent feasibility maps (motor 1 -> can change alpha; motor 2
-> can change beta), runs the dual-layer axial A*, prints/saves the keypoints,
and shows the feasibility maps with the planned path.

    python scripts/plan_path.py            # full pipeline (computes feasibility maps)
    python scripts/plan_path.py --demo     # fast synthetic-obstacle A* demo
"""

import argparse
import csv
import sys
import pathlib
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from selective_em import (
    robot_arm_kinematics, is_selectively_actuable, a_star, extract_keypoints,
    WORKSPACE_COILS,
)

DATA_DIR = ROOT / "data"

RADII = (0.02, 0.03, 0.02)
RATIO = 0.92
TIP_RADIUS = 0.04


def _feasible_worker(args):
    """Is `motor` selectively actuable at (alpha, beta)? (tip mask r = 0.04)."""
    alpha, beta, i_min, i_max, motor = args
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(alpha, beta)
    if x3 ** 2 + y3 ** 2 > TIP_RADIUS ** 2:
        return -10
    positions = [np.array([x1, y1, z1]), np.array([x2, y2, z2]),
                 np.array([x3, y3, z3])]
    idx = motor - 1
    others = [positions[o] for o in range(3) if o != idx]
    other_b_mins = [RADII[o] * RATIO for o in range(3) if o != idx]
    try:
        ok = is_selectively_actuable(positions[idx], others, RADII[idx],
                                     other_b_mins, i_min, i_max, WORKSPACE_COILS)
    except RuntimeError:
        return -1
    return 1 if ok else -1


def _feasibility_map(motor, X, Y, i_min, i_max, num_workers=10):
    args = [(X[i, j], Y[i, j], i_min, i_max, motor)
            for i in range(X.shape[0]) for j in range(X.shape[1])]
    print(f"Computing motor-{motor} feasibility ({len(args)} points)...")
    with Pool(processes=num_workers) as pool:
        results = pool.map(_feasible_worker, args)
    return np.array(results).reshape(X.shape)


def run_pipeline(i_min=-15, i_max=15, n=20,
                 start_angles=(55, -30), goal_angles=(-30, 15)):
    xx = np.linspace(-90, 90, n)
    yy = np.linspace(-90, 90, n)
    X, Y = np.meshgrid(xx, yy)
    rows, cols = X.shape

    P1 = _feasibility_map(1, X, Y, i_min, i_max)  # x-direction (alpha)
    P2 = _feasibility_map(2, X, Y, i_min, i_max)  # y-direction (beta)
    x_feasible = (P1 == 1)
    y_feasible = (P2 == 1)

    # Map start/goal angles to grid indices (row <-> beta, col <-> alpha).
    start = (int(np.argmin(np.abs(yy - start_angles[1]))),
             int(np.argmin(np.abs(xx - start_angles[0]))))
    goal = (int(np.argmin(np.abs(yy - goal_angles[1]))),
            int(np.argmin(np.abs(xx - goal_angles[0]))))
    print(f"Start {start_angles} -> grid {start}; Goal {goal_angles} -> grid {goal}")

    path = a_star(start, goal, x_feasible, y_feasible, (rows, cols))

    if path is None:
        print("No path found!")
    else:
        path_coords = [(xx[j], yy[i]) for (i, j) in path]
        keypoints = extract_keypoints(path_coords)
        print(f"Path found ({len(path)} steps, {len(keypoints)} keypoints):")
        for a, b in keypoints:
            print(f"  alpha={a:7.2f}, beta={b:7.2f}")
        DATA_DIR.mkdir(exist_ok=True)
        save_path = DATA_DIR / "path_result.csv"
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["alpha", "beta"])
            writer.writerows(keypoints)
        print(f"Keypoints saved to {save_path}")

    _plot_pipeline(X, Y, P1, P2, path, xx, yy, start, goal)


def _plot_pipeline(X, Y, P1, P2, path, xx, yy, start, goal):
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    for ax, P, title in [(axs[0], P1, r"(a) X-direction feasibility ($\alpha$)"),
                         (axs[1], P2, r"(b) Y-direction feasibility ($\beta$)")]:
        ax.contourf(X, Y, P, levels=[0, np.inf], colors=["lightgreen"], alpha=0.9)
        ax.contourf(X, Y, P, levels=[-10, -9], colors=["lightcoral"], alpha=0.9)
        ax.contourf(X, Y, P, levels=[-1, 0], colors=["lightgray"], alpha=0.9)
        ax.set_title(title, fontsize=16)

    for ax in (axs[2], axs[3]):
        if path is not None:
            ax.plot([xx[j] for (i, j) in path], [yy[i] for (i, j) in path],
                    "b-", linewidth=2.5, label="Path")
            ax.plot(xx[start[1]], yy[start[0]], "go", markersize=10, label="Start")
            ax.plot(xx[goal[1]], yy[goal[0]], "ro", markersize=10, label="Goal")
            ax.legend(fontsize=12)

    combined = np.zeros_like(P1, dtype=float)
    combined[(P1 == 1) & (P2 == 1)] = 2
    combined[(P1 == 1) ^ (P2 == 1)] = 1
    combined[(P1 != 1) & (P2 != 1)] = 0
    combined[(P1 == -10) | (P2 == -10)] = -1
    axs[3].contourf(X, Y, combined, levels=[-1.5, -0.5, 0.5, 1.5, 2.5],
                    colors=["lightcoral", "lightgray", "lightyellow", "lightgreen"],
                    alpha=0.9)

    ticks = [-90, -45, 0, 45, 90]
    for ax in axs:
        ax.set_xlabel(r"$\alpha$ (degrees)", fontsize=16)
        ax.set_ylabel(r"$\beta$ (degrees)", fontsize=16)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_facecolor("lightgray")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    plt.show()


def run_demo():
    """Synthetic-obstacle A* demo (translated from the original Planner.py)."""
    rows, cols = 10, 10
    # x_feasible[i, j] = True  -> horizontal (x) motion allowed from that cell.
    x_feasible = np.ones((rows, cols), dtype=bool)
    x_feasible[2, 3:7] = False   # a horizontal wall blocking x-motion
    x_feasible[7, 3:7] = False
    # y_feasible[i, j] = True  -> vertical (y) motion allowed from that cell.
    y_feasible = np.ones((rows, cols), dtype=bool)
    y_feasible[3:7, 2] = False   # a vertical wall blocking y-motion
    y_feasible[0:7, 5] = False
    y_feasible[4, 8:10] = False

    start, goal = (0, 0), (9, 9)
    path = a_star(start, goal, x_feasible, y_feasible, (rows, cols))
    if path is None:
        print("No path found!")
        return
    print("Path found:", path)

    fig, axs = plt.subplots(1, 3, figsize=(15, 7))
    axs[0].imshow(~x_feasible, cmap="binary", origin="upper")
    axs[1].imshow(~y_feasible, cmap="binary", origin="upper")
    obstacle_map = np.zeros((rows, cols))
    obstacle_map[~x_feasible] = 1
    obstacle_map[~y_feasible] = 1
    axs[2].imshow(obstacle_map, cmap="Reds", alpha=0.3, origin="upper")
    path_array = np.array(path)
    axs[2].plot(path_array[:, 1], path_array[:, 0], "b-", linewidth=2)
    axs[2].plot(start[1], start[0], "go", markersize=10)
    axs[2].plot(goal[1], goal[0], "ro", markersize=10)
    for ax in axs:
        ax.grid(True)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true",
                        help="run the fast synthetic-obstacle demo instead")
    args = parser.parse_args()
    if args.demo:
        run_demo()
    else:
        run_pipeline()
