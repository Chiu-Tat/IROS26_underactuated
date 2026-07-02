"""Dual-layer axial A* path planner (paper Algorithm 3).

Plans a path across the joint-configuration grid where each axis has its own
feasibility map: motion along one joint (axis) is only allowed from cells where
that joint's motor is selectively actuable. This enforces the "one motor at a
time" constraint while still reaching the goal configuration.
"""

import heapq


def manhattan(p1, p2):
    """Manhattan-distance heuristic on grid indices."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def a_star(start, goal, x_feasible, y_feasible, grid_size, turn_penalty=0.1):
    """A* on a grid with direction-dependent feasibility.

    ``x_feasible[i, j]`` -- may move left/right (change alpha / column) from (i, j).
    ``y_feasible[i, j]`` -- may move up/down (change beta / row) from (i, j).
    ``turn_penalty`` adds cost whenever the movement direction changes, favoring
    long single-motor segments. Returns the path as a list of ``(row, col)`` or
    ``None`` if unreachable.
    """
    rows, cols = grid_size
    # (di, dj, axis): 'y' changes the row (beta), 'x' changes the column (alpha).
    directions = [(-1, 0, "y"), (1, 0, "y"), (0, -1, "x"), (0, 1, "x")]

    visited = [[False] * cols for _ in range(rows)]
    heap = []  # (f, g, (i, j), path, prev_dir)
    heapq.heappush(heap, (manhattan(start, goal), 0, start, [start], None))

    while heap:
        f, g, (i, j), path, prev_dir = heapq.heappop(heap)

        if (i, j) == goal:
            return path

        if visited[i][j]:
            continue
        visited[i][j] = True

        for di, dj, axis in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if axis == "x" and not x_feasible[i, j]:
                    continue
                if axis == "y" and not y_feasible[i, j]:
                    continue

                step_cost = 1
                current_dir = (di, dj)
                if prev_dir is not None and current_dir != prev_dir:
                    step_cost += turn_penalty

                new_g = g + step_cost
                new_h = manhattan((ni, nj), goal)
                heapq.heappush(
                    heap,
                    (new_g + new_h, new_g, (ni, nj), path + [(ni, nj)], current_dir),
                )

    return None


def extract_keypoints(path_coords):
    """Reduce a dense path to start, direction-change turns, and end."""
    if not path_coords:
        return []
    keypoints = [path_coords[0]]
    for k in range(1, len(path_coords) - 1):
        prev_dir = (path_coords[k][0] - path_coords[k - 1][0],
                    path_coords[k][1] - path_coords[k - 1][1])
        next_dir = (path_coords[k + 1][0] - path_coords[k][0],
                    path_coords[k + 1][1] - path_coords[k][1])
        if prev_dir != next_dir:
            keypoints.append(path_coords[k])
    keypoints.append(path_coords[-1])
    return keypoints
