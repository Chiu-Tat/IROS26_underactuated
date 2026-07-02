"""Plotting helpers for polytopes (MFW) and Current-Feasible Workspaces (CFW)."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from pypoman import compute_polytope_vertices


def plot_polytope(N, d):
    """Plot a 3D polytope given by its H-representation ``N . x <= d``."""
    vertices = np.array(compute_polytope_vertices(N, d))
    hull = ConvexHull(vertices)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    faces = [vertices[simplex] for simplex in hull.simplices]
    face_collection = Poly3DCollection(faces, alpha=0.25, linewidths=0.5,
                                       edgecolors="black")
    face_collection.set_facecolor("lightgreen")
    ax.add_collection3d(face_collection)
    plt.show()


def plot_polytope_2d(N, d, color="lightblue", alpha=0.5, radius=0.03):
    """Plot a 2D polytope (H-representation) with the minimum actuation circle."""
    vertices = np.array(compute_polytope_vertices(N, d))
    hull = ConvexHull(vertices)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")

    x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    max_range = max(x_max - x_min, y_max - y_min)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    ax.set_xlim(x_center - max_range / 2 - 0.01, x_center + max_range / 2 + 0.01)
    ax.set_ylim(y_center - max_range / 2 - 0.01, y_center + max_range / 2 + 0.01)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    hull_vertices = vertices[hull.vertices]
    hull_vertices = hull_vertices[ConvexHull(hull_vertices).vertices]
    ax.fill(hull_vertices[:, 0], hull_vertices[:, 1], color=color, alpha=alpha,
            label="Feasible Region")

    for simplex in hull.simplices:
        ax.plot(vertices[simplex, 0], vertices[simplex, 1], "k-", linewidth=1.5)

    circle = plt.Circle((0, 0), radius, fill=False, ec="red", linewidth=2,
                        label="Minimal actuation circle")
    ax.add_patch(circle)
    ax.legend(loc="upper right", fontsize=15)
    plt.show()


def _ellipse_points(Q, r, dim):
    """Sample the ellipsoid ``i^T Q i = r`` surface for plotting (2D or 3D)."""
    eigenvals, eigenvecs = np.linalg.eigh(Q)
    eigenvals = np.maximum(eigenvals, 1e-12)
    semi_axes = np.sqrt(r) / np.sqrt(eigenvals)
    if dim == 2:
        theta = np.linspace(0, 2 * np.pi, 200)
        unit = np.column_stack([np.cos(theta), np.sin(theta)])
        return (unit * semi_axes.reshape(1, -1)) @ eigenvecs.T
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_unit = np.outer(np.cos(u), np.sin(v))
    y_unit = np.outer(np.sin(u), np.sin(v))
    z_unit = np.outer(np.ones_like(u), np.cos(v))
    unit_surface = np.stack([x_unit.ravel(), y_unit.ravel(), z_unit.ravel()], axis=1)
    surface = (unit_surface * semi_axes.reshape(1, -1)) @ eigenvecs.T
    return surface, x_unit.shape


def plot_cfw_2d(hull, remaining_points, deleted_points, i_abs_max, Q, r):
    """Plot a 2D Current-Feasible Workspace: ellipse + box + hull of feasible points."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("2D Feasible Current Space (Ellipse & Square)")
    ax.set_xlabel("I_1")
    ax.set_ylabel("I_2")
    ax.grid(True)
    ax.set_aspect("equal")

    if hull is not None:
        hull_vertices = remaining_points[hull.vertices]
        center = np.mean(hull_vertices, axis=0)
        angles = np.arctan2(hull_vertices[:, 1] - center[1],
                            hull_vertices[:, 0] - center[0])
        hull_vertices_sorted = hull_vertices[np.argsort(angles)]
        closed = np.vstack([hull_vertices_sorted, hull_vertices_sorted[0]])
        ax.fill(closed[:, 0], closed[:, 1], color="cyan", alpha=0.3,
                edgecolor="b", linewidth=2, label="Feasible Region (Hull)")

    i_min, i_max = -i_abs_max, i_abs_max
    square = np.array([[i_min, i_min], [i_max, i_min], [i_max, i_max],
                       [i_min, i_max], [i_min, i_min]])
    ax.plot(square[:, 0], square[:, 1], "k-", linewidth=2, alpha=0.7)

    ellipse_points = _ellipse_points(Q, r, dim=2)
    ax.plot(ellipse_points[:, 0], ellipse_points[:, 1], "g-", linewidth=2,
            alpha=0.7, label="Ellipse Constraint")

    if len(remaining_points) > 0:
        ax.scatter(remaining_points[:, 0], remaining_points[:, 1], c="blue",
                   s=15, alpha=0.6)
    if len(deleted_points) > 0:
        ax.scatter(deleted_points[:, 0], deleted_points[:, 1], c="red", s=15,
                   alpha=0.6, label="Projected Points")

    all_points = np.vstack([remaining_points, ellipse_points, square[:-1]])
    max_lim = min(np.max(np.abs(all_points)) * 1.2, i_abs_max * 1.1)
    ax.set_xlim([-max_lim, max_lim])
    ax.set_ylim([-max_lim, max_lim])
    ax.legend()
    plt.show()


def plot_cfw_3d(hull, remaining_points, deleted_points, i_abs_max, Q, r):
    """Plot a 3D Current-Feasible Workspace: cube + hull of feasible points."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Feasible Current Space (Ellipsoid & Cube)")
    ax.set_xlabel("I_1")
    ax.set_ylabel("I_2")
    ax.set_zlabel("I_3")

    if hull is not None:
        triangles = hull.points[hull.simplices]
        ax.add_collection3d(Poly3DCollection(triangles, facecolors="cyan",
                                             edgecolors="b", linewidths=0.5,
                                             alpha=0.3))

    i_min, i_max = -i_abs_max, i_abs_max
    cube_faces = [
        [[i_min, i_min, i_min], [i_min, i_max, i_min], [i_min, i_max, i_max], [i_min, i_min, i_max]],
        [[i_max, i_min, i_min], [i_max, i_max, i_min], [i_max, i_max, i_max], [i_max, i_min, i_max]],
        [[i_min, i_min, i_min], [i_max, i_min, i_min], [i_max, i_min, i_max], [i_min, i_min, i_max]],
        [[i_min, i_max, i_min], [i_max, i_max, i_min], [i_max, i_max, i_max], [i_min, i_max, i_max]],
        [[i_min, i_min, i_min], [i_max, i_min, i_min], [i_max, i_max, i_min], [i_min, i_max, i_min]],
        [[i_min, i_min, i_max], [i_max, i_min, i_max], [i_max, i_max, i_max], [i_min, i_max, i_max]],
    ]
    ax.add_collection3d(Poly3DCollection(cube_faces, facecolors="lightgray",
                                         edgecolors="k", linewidths=1, alpha=0.15))

    corners = np.array([[i, j, k] for i in [i_min, i_max]
                        for j in [i_min, i_max] for k in [i_min, i_max]])
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c="k", marker="x", s=50)

    surface, shape = _ellipse_points(Q, r, dim=3)

    handles = [
        mpatches.Patch(facecolor="cyan", edgecolor="b", alpha=0.3),
        Line2D([0], [0], color="g", lw=4, alpha=0.5),
        mpatches.Patch(facecolor="lightgray", edgecolor="k", alpha=0.15),
        Line2D([0], [0], marker="x", color="k", linestyle="None", markersize=7),
    ]
    labels = ["Feasible Region (Hull)", "Ellipsoid Constraint",
              "Cube Constraint Faces", "Cube Corners"]
    ax.legend(handles, labels)
    ax.grid(True)

    all_points = np.vstack([remaining_points, surface, corners])
    max_lim = min(np.max(np.abs(all_points)) * 1.2, i_abs_max * 1.1)
    ax.set_xlim([-max_lim, max_lim])
    ax.set_ylim([-max_lim, max_lim])
    ax.set_zlim([-max_lim, max_lim])
    plt.show()
