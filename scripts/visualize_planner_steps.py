"""Visualize the dual-layer axial A* exploration steps (produces figures/planner_process.png).

Self-contained A* with snapshot recording plus grid/exploration drawing helpers;
renders a 2x3 figure of the obstacle maps and search progression.

    python scripts/visualize_planner_steps.py
"""

import pathlib

import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch

FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent / "figures"

# Manhattan distance heuristic
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# A* with step recording for visualization
def a_star_with_steps(start, goal, x_feasible, y_feasible, grid_size, record_steps=None):
    """
    A* algorithm that records intermediate exploration states.
    record_steps: list of step counts at which to snapshot the state.
    Returns: (path, snapshots)
        snapshots: list of dicts with 'visited', 'frontier', 'current_path' at each recorded step.
    """
    rows, cols = grid_size
    directions = [(-1, 0, 'y'), (1, 0, 'y'), (0, -1, 'x'), (0, 1, 'x')]

    visited = np.zeros((rows, cols), dtype=bool)
    heap = []
    heapq.heappush(heap, (0 + manhattan(start, goal), 0, start, [start]))

    snapshots = []
    step_count = 0
    record_set = set(record_steps) if record_steps else set()

    while heap:
        f, g, (i, j), path = heapq.heappop(heap)

        if (i, j) == goal:
            # Record final state
            snapshots.append({
                'visited': visited.copy(),
                'frontier': set((pos) for _, _, pos, _ in heap),
                'current_path': path[:],
                'found': True
            })
            return path, snapshots

        if visited[i, j]:
            continue
        visited[i, j] = True
        step_count += 1

        # Record snapshot at specified steps
        if step_count in record_set:
            frontier_positions = set()
            for _, _, pos, _ in heap:
                frontier_positions.add(pos)
            snapshots.append({
                'visited': visited.copy(),
                'frontier': frontier_positions,
                'current_path': path[:],
                'found': False
            })

        for di, dj, dir_type in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if dir_type == 'x' and not x_feasible[i, j]:
                    continue
                if dir_type == 'y' and not y_feasible[i, j]:
                    continue

                new_g = g + 1
                new_h = manhattan((ni, nj), goal)
                new_path = path + [(ni, nj)]
                heapq.heappush(heap, (new_g + new_h, new_g, (ni, nj), new_path))

    return None, snapshots


def build_feasibility_map(x_feasible, y_feasible):
    """
    Build a combined feasibility map with categories:
      0 = neither feasible (obstacle)
      1 = only x feasible
      2 = only y feasible
      3 = both feasible
    """
    rows, cols = x_feasible.shape
    fmap = np.zeros((rows, cols), dtype=int)
    fmap[x_feasible & ~y_feasible] = 1
    fmap[~x_feasible & y_feasible] = 2
    fmap[x_feasible & y_feasible] = 3
    return fmap


def draw_grid_map(ax, cell_colors, rows, cols, title=None):
    """
    Draw a grid map with per-cell RGBA colors.
    cell_colors: (rows, cols, 4) RGBA array
    """
    ax.imshow(cell_colors, origin='upper', interpolation='nearest',
              extent=[-0.5, cols - 0.5, rows - 0.5, -0.5])

    # Draw grid lines
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color='#CCCCCC', linewidth=0.5)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color='#CCCCCC', linewidth=0.5)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)


def get_base_colors(x_feasible, y_feasible):
    """
    Generate RGBA color array based on feasibility maps.
      - Both feasible:    green
      - Only x feasible:  yellow
      - Only y feasible:  blue
      - Neither feasible: dark gray
    """
    rows, cols = x_feasible.shape
    colors = np.ones((rows, cols, 4))  # default white (RGBA=1,1,1,1)

    both = x_feasible & y_feasible
    only_x = x_feasible & ~y_feasible
    only_y = ~x_feasible & y_feasible
    neither = ~x_feasible & ~y_feasible

    # Green for both feasible
    colors[both] = mcolors.to_rgba('#81C784')
    # Yellow for only x feasible
    colors[only_x] = mcolors.to_rgba('#FFF176')
    # Blue for only y feasible
    colors[only_y] = mcolors.to_rgba('#64B5F6')
    # Dark gray for obstacles
    colors[neither] = mcolors.to_rgba('#757575')

    return colors


def get_single_feasibility_colors(feasible, color_true, color_false):
    """Generate RGBA color array for a single feasibility map."""
    rows, cols = feasible.shape
    colors = np.zeros((rows, cols, 4))
    colors[feasible] = mcolors.to_rgba(color_true)
    colors[~feasible] = mcolors.to_rgba(color_false)
    return colors


def draw_start_goal(ax, start, goal):
    """Draw start (green) and goal (red) markers."""
    ax.plot(start[1], start[0], 's', color='#2E7D32', markersize=10,
            markeredgecolor='black', markeredgewidth=1.0, zorder=5)
    ax.plot(goal[1], goal[0], 's', color='#D32F2F', markersize=10,
            markeredgecolor='black', markeredgewidth=1.0, zorder=5)


def draw_path(ax, path, color="#FF3300", linewidth=2.5):
    """Draw the path as a line."""
    if path and len(path) > 1:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 1], path_arr[:, 0], '-', color=color,
                linewidth=linewidth, solid_capstyle='round', zorder=4)


def draw_exploration(ax, base_colors, visited, frontier, rows, cols,
                     x_feasible=None, y_feasible=None):
    """
    Build exploration-aware colors:
      - Unexplored non-obstacle cells: white
      - Explored (visited/frontier) cells: show their feasibility color
      - Obstacle cells: always dark gray
    """
    colors = np.ones((rows, cols, 4))  # default white

    # Build obstacle mask (any cell not feasible in at least one direction)
    if x_feasible is not None and y_feasible is not None:
        obstacle = ~x_feasible | ~y_feasible
    else:
        obstacle = np.zeros((rows, cols), dtype=bool)

    # Unexplored obstacles: light gray; explored obstacles: reveal true color
    for r in range(rows):
        for c in range(cols):
            if obstacle[r, c]:
                if visited[r, c] or (r, c) in frontier:
                    # Show actual feasibility color (yellow/blue/dark gray)
                    colors[r, c] = base_colors[r, c]
                else:
                    colors[r, c] = mcolors.to_rgba('#D0D0D0')

    # Reveal feasibility color only for visited non-obstacle cells
    for r in range(rows):
        for c in range(cols):
            if visited[r, c] and not obstacle[r, c]:
                colors[r, c] = base_colors[r, c]

    # Reveal feasibility color for frontier cells
    for (r, c) in frontier:
        if 0 <= r < rows and 0 <= c < cols and not obstacle[r, c]:
            colors[r, c] = base_colors[r, c]

    return colors


def main():
    # Grid setup (same as Planner.py)
    rows, cols = 10, 10

    x_feasible = np.ones((rows, cols), dtype=bool)
    x_feasible[2, 3:7] = False
    x_feasible[7, 3:7] = False

    y_feasible = np.ones((rows, cols), dtype=bool)
    y_feasible[3:7, 2] = False
    y_feasible[0:7, 5] = False
    y_feasible[4, 8:10] = False

    start = (0, 0)
    goal = (9, 9)

    # --- Run A* with snapshots at various exploration stages ---
    # First, run the full A* to find total steps
    full_path, _ = a_star_with_steps(start, goal, x_feasible, y_feasible,
                                      (rows, cols), record_steps=set())

    # Run again, recording at intermediate steps to show progression
    # We pick 4 steps: early start, approaching obstacle, navigating around, nearly done
    record_at = [5, 13, 40, 50]
    path, snapshots = a_star_with_steps(start, goal, x_feasible, y_feasible,
                                         (rows, cols), record_steps=record_at)

    if path is None:
        print("No path found!")
        return

    print("Path found:", path)
    print(f"Recorded {len(snapshots)} snapshots (including final)")

    # --- Build base color maps ---
    base_colors = get_base_colors(x_feasible, y_feasible)
    x_colors = get_single_feasibility_colors(x_feasible, '#FFFFFF', '#757575')
    y_colors = get_single_feasibility_colors(y_feasible, '#FFFFFF', '#757575')

    # ========================  FIGURE: 2 rows x 3 columns  ========================
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), facecolor='white')
    for ax_row in axs:
        for ax_item in ax_row:
            ax_item.set_facecolor('white')

    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # ---- (a) X-feasible map ----
    ax = axs[0, 0]
    draw_grid_map(ax, x_colors, rows, cols)
    draw_start_goal(ax, start, goal)
    ax.set_title(r'(a) Obstacle map for $X$', fontsize=18, pad=6)

    # ---- (b) Y-feasible map ----
    ax = axs[0, 1]
    draw_grid_map(ax, y_colors, rows, cols)
    draw_start_goal(ax, start, goal)
    ax.set_title(r'(b) Obstacle map for $Y$', fontsize=18, pad=6)

    # Exploration step titles with descriptions
    exploration_info = [
        ('(c) Step 5 of expansion',
         'A* begins from start; frontier\nexpands toward the goal heuristically.'),
        ('(d) Step 13 of expansion',
         'Frontier reaches infeasible cells;\nblocked directions force rerouting.'),
        ('(e) Step 40 of expansion',
         'Search detours around obstacles;\nfrontier explores alternative corridors.'),
        ('(f) Path found',
         'Goal reached; optimal path (yellow)\nretraced through explored nodes.'),
    ]

    # Map subplot positions for the 4 exploration panels: (0,2), (1,0), (1,1), (1,2)
    exp_axes = [axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]]

    for idx, (ax, (title, desc)) in enumerate(zip(exp_axes, exploration_info)):
        # Pick the right snapshot
        if idx < len(snapshots) - 1:
            snap = snapshots[idx]
        else:
            snap = snapshots[-1]

        # Skip exploration overlay for the last subfigure (f)
        if idx == len(exploration_info) - 1:
            draw_grid_map(ax, base_colors, rows, cols)
        else:
            exp_colors = draw_exploration(ax, base_colors, snap['visited'],
                                          snap['frontier'], rows, cols,
                                          x_feasible, y_feasible)
            draw_grid_map(ax, exp_colors, rows, cols)

        # Draw current best path for intermediate steps, final path for last
        if idx == len(exploration_info) - 1:
            draw_path(ax, path, color="#FF0000", linewidth=3)
        else:
            draw_path(ax, snap['current_path'])

        draw_start_goal(ax, start, goal)
        ax.set_title(title, fontsize=18, pad=6)

        # Add description text box
        # ax.text(0.02, 0.02, desc, transform=ax.transAxes,
        #         fontsize=7.5, verticalalignment='bottom',
        #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
        #                   edgecolor='#999999', alpha=0.85))

    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "planner_process.png",
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Figure saved as planner_process.png")


if __name__ == "__main__":
    main()
