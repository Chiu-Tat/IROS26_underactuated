import numpy as np
import heapq
import matplotlib.pyplot as plt
from multiprocessing import Pool
from Kinematics import robot_arm_kinematics
from Check_feasibility_one_point import if_in_OAW, if_in_IR

# ===================== Feasibility check workers =====================

def check_single_point_p1(args):
    """Worker: x/alpha-direction feasibility (actuate point 1)."""
    alpha, beta, I_min, I_max, r1, r2, r3 = args
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(alpha, beta)
    if x3**2 + y3**2 > 0.04**2:
        return -10
    position_1 = np.array([x1, y1, z1])
    position_2 = np.array([x2, y2, z2])
    position_3 = np.array([x3, y3, z3])
    if if_in_OAW(position_1, r1, I_min, I_max):
        try:
            if if_in_IR(position_1, position_2, position_3, I_max, r1, r2, r3):
                return 1
            else:
                return -1
        except RuntimeError:
            return -1
    else:
        return -1

def check_single_point_p2(args):
    """Worker: y/beta-direction feasibility (actuate point 2)."""
    alpha, beta, I_min, I_max, r1, r2, r3 = args
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(alpha, beta)
    if x3**2 + y3**2 > 0.04**2:
        return -10
    position_1 = np.array([x1, y1, z1])
    position_2 = np.array([x2, y2, z2])
    position_3 = np.array([x3, y3, z3])
    if if_in_OAW(position_2, r2, I_min, I_max):
        try:
            if if_in_IR(position_2, position_1, position_3, I_max, r2, r1, r3):
                return 1
            else:
                return -1
        except RuntimeError:
            return -1
    else:
        return -1

# ===================== A* path planner =====================

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def a_star(start, goal, x_feasible, y_feasible, grid_size, turn_penalty=0.1):
    """
    A* path planning on a grid with direction-dependent feasibility.
    x_feasible[i,j] = True  =>  can move left/right (change alpha) from (i,j)
    y_feasible[i,j] = True  =>  can move up/down   (change beta)  from (i,j)
    turn_penalty: extra cost added when the movement direction changes.
    """
    rows, cols = grid_size
    # (di, dj, direction_type): y = row change (beta), x = col change (alpha)
    directions = [(-1, 0, 'y'), (1, 0, 'y'), (0, -1, 'x'), (0, 1, 'x')]

    visited = np.zeros((rows, cols), dtype=bool)
    heap = []  # (f, g, (i,j), path, prev_dir)
    heapq.heappush(heap, (manhattan(start, goal), 0, start, [start], None))

    while heap:
        f, g, (i, j), path, prev_dir = heapq.heappop(heap)

        if (i, j) == goal:
            return path

        if visited[i, j]:
            continue
        visited[i, j] = True

        for di, dj, dir_type in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if dir_type == 'x' and not x_feasible[i, j]:
                    continue
                if dir_type == 'y' and not y_feasible[i, j]:
                    continue

                # Base step cost + penalty for changing direction
                step_cost = 1
                current_dir = (di, dj)
                if prev_dir is not None and current_dir != prev_dir:
                    step_cost += turn_penalty

                new_g = g + step_cost
                new_h = manhattan((ni, nj), goal)
                heapq.heappush(heap, (new_g + new_h, new_g, (ni, nj), path + [(ni, nj)], current_dir))

    return None  # no path found

# ===================== Main =====================

if __name__ == '__main__':
    # ---- Grid parameters ----
    # step = 5  # expansion length in degrees
    # xx = np.arange(-90, 90 + step, step)  # alpha values
    # yy = np.arange(-90, 90 + step, step)  # beta values
    xx = np.linspace(-90, 90, 20)  # alpha values
    yy = np.linspace(-90, 90, 20)  # beta values
    X, Y = np.meshgrid(xx, yy)
    rows, cols = X.shape

    # ---- Physical parameters ----
    I_min = -15
    I_max = 15

    r1 = 0.02   # Tesla
    r2 = 0.03
    r3 = 0.02

    ratio = 0.92

    num_workers = 10

    # ---- Compute x-direction feasibility (p1) ----
    args_p1 = [
        (X[i, j], Y[i, j], I_min, I_max, r1, r2 * ratio, r3 * ratio)
        for i in range(rows) for j in range(cols)
    ]
    print(f"Computing x-direction feasibility ({len(args_p1)} points, {num_workers} workers)...")
    with Pool(processes=num_workers) as pool:
        results_p1 = pool.map(check_single_point_p1, args_p1)
    P1 = np.array(results_p1).reshape(X.shape)

    # ---- Compute y-direction feasibility (p2) ----
    args_p2 = [
        (X[i, j], Y[i, j], I_min, I_max, r1 * ratio, r2, r3 * ratio)
        for i in range(rows) for j in range(cols)
    ]
    print(f"Computing y-direction feasibility ({len(args_p2)} points, {num_workers} workers)...")
    with Pool(processes=num_workers) as pool:
        results_p2 = pool.map(check_single_point_p2, args_p2)
    P2 = np.array(results_p2).reshape(X.shape)

    # ---- Convert to boolean feasibility maps ----
    x_feasible = (P1 == 1)
    y_feasible = (P2 == 1)

    # ---- Map start / goal to grid indices ----
    # Grid layout: row i ↔ beta = yy[i],  col j ↔ alpha = xx[j]
    start_alpha, start_beta = 55, -30
    goal_alpha,  goal_beta  = -30, 15

    start_col = int(np.argmin(np.abs(xx - start_alpha)))
    start_row = int(np.argmin(np.abs(yy - start_beta)))
    goal_col  = int(np.argmin(np.abs(xx - goal_alpha)))
    goal_row  = int(np.argmin(np.abs(yy - goal_beta)))

    start = (start_row, start_col)
    goal  = (goal_row,  goal_col)

    print(f"Grid size: {rows} x {cols}")
    print(f"Start: (alpha={start_alpha}, beta={start_beta}) -> grid {start}")
    print(f"Goal : (alpha={goal_alpha}, beta={goal_beta}) -> grid {goal}")

    # ---- Run A* ----
    path = a_star(start, goal, x_feasible, y_feasible, (rows, cols))

    if path is None:
        print("No path found!")
    else:
        print(f"Path found with {len(path)} steps!")
        path_coords = [(xx[j], yy[i]) for (i, j) in path]
        print("Path (alpha, beta):")
        for coord in path_coords:
            print(f"  alpha={coord[0]:7.2f}, beta={coord[1]:7.2f}")

        # Extract keypoints: start, turn points (direction changes), and end
        keypoints = [path_coords[0]]  # start
        for k in range(1, len(path_coords) - 1):
            prev_dir = (path_coords[k][0] - path_coords[k-1][0],
                        path_coords[k][1] - path_coords[k-1][1])
            next_dir = (path_coords[k+1][0] - path_coords[k][0],
                        path_coords[k+1][1] - path_coords[k][1])
            if prev_dir != next_dir:
                keypoints.append(path_coords[k])
        keypoints.append(path_coords[-1])  # end

        print(f"Keypoints ({len(keypoints)} of {len(path_coords)} total):")
        for coord in keypoints:
            print(f"  alpha={coord[0]:7.2f}, beta={coord[1]:7.2f}")

        # Save keypoints to CSV
        import csv
        save_path = 'e:/CUHK/Research/Reconfigurable surgical robot document/Modeling/Paper_ICRA2026/path_result.csv'
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['alpha', 'beta'])
            for alpha_val, beta_val in keypoints:
                writer.writerow([alpha_val, beta_val])
        print(f"Keypoints saved to {save_path}")

    # ===================== Visualization =====================
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # ---- (a) X-direction feasibility ----
    ax = axs[0]
    ax.contourf(X, Y, P1, levels=[0, np.inf],   colors=['lightgreen'], alpha=0.9)
    ax.contourf(X, Y, P1, levels=[-10, -9],      colors=['lightcoral'], alpha=0.9)
    ax.contourf(X, Y, P1, levels=[-1, 0],        colors=['lightgray'],  alpha=0.9)
    ax.set_xlabel(r'$\alpha$ (degrees)', fontsize=16)
    ax.set_ylabel(r'$\beta$ (degrees)',  fontsize=16)
    ax.set_title(r'(a) X-direction feasibility ($\alpha$)', fontsize=16)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_facecolor('lightgray')

    # ---- (b) Y-direction feasibility ----
    ax = axs[1]
    ax.contourf(X, Y, P2, levels=[0, np.inf],   colors=['lightgreen'], alpha=0.9)
    ax.contourf(X, Y, P2, levels=[-10, -9],      colors=['lightcoral'], alpha=0.9)
    ax.contourf(X, Y, P2, levels=[-1, 0],        colors=['lightgray'],  alpha=0.9)
    ax.set_xlabel(r'$\alpha$ (degrees)', fontsize=16)
    ax.set_ylabel(r'$\beta$ (degrees)',  fontsize=16)
    ax.set_title(r'(b) Y-direction feasibility ($\beta$)', fontsize=16)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_facecolor('lightgray')

    # ---- (c) Path planning result ----
    ax = axs[2]

    if path is not None:
        path_alpha = [xx[j] for (i, j) in path]
        path_beta  = [yy[i] for (i, j) in path]
        ax.plot(path_alpha, path_beta, 'b-', linewidth=2.5, label='Path')
        ax.plot(xx[start_col], yy[start_row], 'go', markersize=10, label='Start')
        ax.plot(xx[goal_col],  yy[goal_row],  'ro', markersize=10, label='Goal')
        ax.legend(fontsize=12)

    ax.set_xlabel(r'$\alpha$ (degrees)', fontsize=16)
    ax.set_ylabel(r'$\beta$ (degrees)',  fontsize=16)
    # ax.set_title('(c) Path planning result', fontsize=16)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_facecolor('lightgray')

    # ---- (d) Combined obstacles with path ----
    ax = axs[3]
    combined = np.zeros_like(P1, dtype=float)
    combined[(P1 == 1) & (P2 == 1)]       = 2    # both feasible
    combined[(P1 == 1) ^ (P2 == 1)]       = 1    # only one feasible
    combined[(P1 != 1) & (P2 != 1)]       = 0    # neither feasible
    combined[(P1 == -10) | (P2 == -10)]   = -1   # outside sphere

    ax.contourf(X, Y, combined,
                levels=[-1.5, -0.5, 0.5, 1.5, 2.5],
                colors=['lightcoral', 'lightgray', 'lightyellow', 'lightgreen'],
                alpha=0.9)

    if path is not None:
        path_alpha = [xx[j] for (i, j) in path]
        path_beta  = [yy[i] for (i, j) in path]
        ax.plot(path_alpha, path_beta, 'b-', linewidth=2.5, label='Path')
        ax.plot(xx[start_col], yy[start_row], 'go', markersize=10, label='Start')
        ax.plot(xx[goal_col],  yy[goal_row],  'ro', markersize=10, label='Goal')
        ax.legend(fontsize=12)

    ax.set_xlabel(r'$\alpha$ (degrees)', fontsize=16)
    ax.set_ylabel(r'$\beta$ (degrees)',  fontsize=16)
    # ax.set_title('(d) Combined feasibility + Path', fontsize=16)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_facecolor('lightgray')

    # Common tick settings
    ticks = [-90, -45, 0, 45, 90]
    for ax in axs:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=14)

    # plt.tight_layout()
    plt.show()
