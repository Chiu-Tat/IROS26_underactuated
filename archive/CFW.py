import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import DBSCAN
# import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.lines import Line2D
import time

def _merge_points_by_voxel(points, voxel_size):
    """
    Merges points that fall into the same voxel by replacing them with their centroid.

    Args:
        points (np.array): Array of points (shape: num_points, n_dimensions).
        voxel_size (float): The size of each voxel. Points within a voxel of this
                             size will be merged.

    Returns:
        np.array: Array of merged points.
    """
    if points.shape[0] == 0:
        return np.array([])
    if voxel_size <= 1e-9: # Avoid division by zero or ineffective merging
        # print("Warning: voxel_size is too small or non-positive. Skipping merging.")
        return points

    # Discretize point coordinates to assign them to voxels
    # Adding a small epsilon to voxel_size in division can help with boundary conditions if needed,
    # but simple division usually works.
    discretized_points = np.floor(points / voxel_size)

    # Find unique voxel coordinates and the inverse map
    # unique_voxel_coords: the coordinates of each unique voxel that contains points
    # inverse_indices: an array mapping each original point to the index of its unique voxel
    unique_voxel_coords, inverse_indices = np.unique(discretized_points, axis=0, return_inverse=True)

    merged_points_list = []
    # Iterate over each unique voxel that contains points
    for i in range(len(unique_voxel_coords)):
        # Get all original points that fall into the current unique voxel
        points_in_this_voxel = points[inverse_indices == i]
        # Calculate the centroid of these points
        centroid = np.mean(points_in_this_voxel, axis=0)
        merged_points_list.append(centroid)

    if not merged_points_list:
         return np.array([]) # Should ideally not be reached if input points is not empty
        
    return np.array(merged_points_list)

def Get_CFW(I_abs_max, P_max, R_list, m, plot_verbose=True, voxel_size_for_merging=None):
    """
    Calculates the convex feasible workspace (CFW) for a set of coils.
    Assumes all coil resistances in R_list are identical and I_min = -I_abs_max.

    The function generates points on an n-sphere defined by coil currents
    and power limits. For points outside the current bounds [-I_abs_max, I_abs_max],
    it projects them inward toward the origin until they are within bounds.
    Then it computes the convex hull of all feasible points.

    Args:
        I_abs_max (float): Absolute maximum allowable current for any coil.
        P_max (float): Maximum total power dissipation (P_max = R * sum(I_i^2)).
        R_list (list): List of coil resistances. All elements must be identical and positive.
        m (int): Number of points to generate on the sphere.
        plot_verbose (bool): If True and n=2 or n=3, generates plots.

    Returns:
        tuple:
            - hull (scipy.spatial.ConvexHull object or None): The convex hull.
            - H_representation (dict or None): H-representation (Ax <= b).
            - remaining_points (np.array): Feasible points.
            - deleted_points (np.array): Points originally outside bounds.
    """
    I_min = -I_abs_max
    I_max = I_abs_max

    R_list_np = np.array(R_list)
    n = len(R_list_np)

    if n == 0:
        print("Error: Coil resistance array R_list cannot be empty.")
        return None, None, np.array([]), np.array([])

    R_value = R_list_np[0]
    if not np.all(R_list_np == R_value):
        print("Error: All resistances in R_list must be identical.")
        return None, None, np.array([]), np.array([])
    
    if R_value <= 0:
        print("Error: All coil resistances in R_list must be positive.")
        return None, None, np.array([]), np.array([])

    if P_max <= 0:
        print("Error: P_max must be positive.")
        return None, None, np.array([]), np.array([])

    # --- 1. Calculate m evenly distributed points on the sphere ---
    radius = np.sqrt(P_max / R_value)

    if m == 0:
        points_on_sphere_unit = np.empty((0, n) if n > 0 else (0,))
    elif n == 1:
        num_neg = m // 2
        points_on_sphere_unit = np.ones((m, 1))
        if num_neg > 0:
            points_on_sphere_unit[:num_neg, 0] = -1.0
    elif n == 2:
        theta = np.linspace(0, 2 * np.pi, m, endpoint=False)
        points_on_sphere_unit = np.column_stack((np.cos(theta), np.sin(theta)))
    elif n == 3:
        points_on_sphere_unit = np.zeros((m, 3))
        indices = np.arange(0, m, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / m)
        theta = np.pi * (1 + np.sqrt(5)) * indices
        points_on_sphere_unit[:, 0] = np.cos(theta) * np.sin(phi)
        points_on_sphere_unit[:, 1] = np.sin(theta) * np.sin(phi)
        points_on_sphere_unit[:, 2] = np.cos(phi)
    else:  # n > 3
        random_points = np.random.normal(size=(m, n))
        norm = np.linalg.norm(random_points, axis=1, keepdims=True)
        points_on_sphere_unit = np.zeros_like(random_points)
        non_zero_norm_mask = (norm > 1e-15).flatten()
        points_on_sphere_unit[non_zero_norm_mask] = random_points[non_zero_norm_mask] / norm[non_zero_norm_mask]
        zero_norm_mask = ~non_zero_norm_mask
        if np.any(zero_norm_mask):
            canonical_vector = np.zeros(n)
            if n > 0: canonical_vector[0] = 1.0
            points_on_sphere_unit[zero_norm_mask] = canonical_vector

    points_on_sphere_scaled = points_on_sphere_unit * radius
    
    # --- 2. Vectorized projection for points outside the hypercube ---
    original_points = points_on_sphere_scaled.copy()

    # Calculate the maximum absolute coordinate value for each point
    # np.abs() is element-wise, np.max(..., axis=1) finds the max in each row (point)
    max_abs_coord_per_point = np.max(np.abs(points_on_sphere_scaled), axis=1)

    # Identify points where their maximum absolute coordinate exceeds I_max
    # These are the points outside the hypercube defined by [-I_max, I_max] in each dimension
    originally_outside_mask = max_abs_coord_per_point > I_max

    points_inside_bounds = points_on_sphere_scaled.copy()

    # Process only the points that are actually outside
    if np.any(originally_outside_mask):
        # Get the subset of points that are outside
        outside_points_subset = points_on_sphere_scaled[originally_outside_mask]
        
        # Get the max_abs_coord_per_point for these specific outside points
        # Reshape to allow broadcasting for scaling: (num_outside_points,) -> (num_outside_points, 1)
        max_abs_for_scaling = max_abs_coord_per_point[originally_outside_mask].reshape(-1, 1)
        
        # Calculate scaling factors to bring the largest component of each outside point exactly to I_max
        # This factor will be < 1 since max_abs_for_scaling > I_max (assuming I_max > 0)
        # If I_max is 0, scaling_factors will correctly be 0, making the points zero.
        # We check for max_abs_for_scaling > 0 to avoid division by zero if I_max > 0 but a point somehow has max_abs 0 yet is 'outside' (should not happen if I_max > 0).
        # However, max_abs_for_scaling comes from points where max_abs_coord_per_point > I_max, so if I_max >= 0, max_abs_for_scaling must be > 0.
        scaling_factors = I_max / max_abs_for_scaling
        
        # Apply scaling to the subset of points by broadcasting the scaling_factor for each point
        points_inside_bounds[originally_outside_mask] = outside_points_subset * scaling_factors

    deleted_points = original_points[originally_outside_mask]
    remaining_points = points_inside_bounds

    # --- 3. Filter points that are too close to the hypercube boundary after projection ---
    # This step removes points that, after projection, lie very close to or on the
    # faces of the hypercube defined by [-I_max, I_max]. This can be useful if
    # such points are considered numerically problematic or undesirable for subsequent
    # convex hull computation.
    
    # Define a small tolerance for proximity. Points where any |coordinate| is
    # greater than (I_max - proximity_epsilon) will be removed.
    # This means if I_max - |coordinate| < proximity_epsilon, the point is removed.
    proximity_epsilon = 1e-20 # A small positive value

    if remaining_points.shape[0] > 0:
        # Calculate for each point if any of its coordinates are too close to I_max or -I_max
        # A coordinate is "too close" if its absolute value is within epsilon of I_max.
        # abs_coords = np.abs(remaining_points)
        # distance_to_boundary = I_max - abs_coords
        # is_too_close_mask_per_coord = distance_to_boundary < proximity_epsilon
        # is_point_too_close_mask = np.any(is_too_close_mask_per_coord, axis=1)
        
        # Simplified:
        is_point_too_close_mask = np.any((I_max - np.abs(remaining_points)) < proximity_epsilon, axis=1)

        # Keep only points that are NOT too close to any boundary
        points_to_keep_mask = ~is_point_too_close_mask
        
        # Potentially, log how many points are removed by this step
        # num_removed_by_proximity = np.sum(is_point_too_close_mask)
        # if num_removed_by_proximity > 0 and plot_verbose:
        # print(f"Filtered out {num_removed_by_proximity} additional points due to proximity to hypercube boundary.")

        remaining_points = remaining_points[points_to_keep_mask]
        # Note: These points removed due to proximity are not added to `deleted_points`,
        # as `deleted_points` tracks points *originally* outside the current bounds before projection.

    # Optional: Merge close points using voxel grid downsampling
    if voxel_size_for_merging is not None and voxel_size_for_merging > 0:
        num_points_before_merge = remaining_points.shape[0]
        remaining_points = _merge_points_by_voxel(remaining_points, voxel_size_for_merging)
        if plot_verbose: # Or a more specific verbose flag for this step
            print(f"Points merged from {num_points_before_merge} to {remaining_points.shape[0]} using voxel_size={voxel_size_for_merging}")

    if remaining_points.shape[0] < n + 1:
        print(f"Warning: Only {remaining_points.shape[0]} points available. Need at least {n+1} for {n}-D convex hull.")
        if plot_verbose and (n == 2 or n == 3) and remaining_points.shape[0] > 0:
             _plot_points_only(n, remaining_points, deleted_points, I_max, P_max, R_value)
        return None, None, remaining_points, deleted_points

    try:
        hull = ConvexHull(remaining_points)
    except Exception as e:
        print(f"Error computing ConvexHull: {e}")
        if plot_verbose and (n == 2 or n == 3):
             _plot_points_only(n, remaining_points, deleted_points, I_max, P_max, R_value)
        return None, None, remaining_points, deleted_points

    A_matrix = hull.equations[:, :-1]
    b_vector = -hull.equations[:, -1]
    H_representation = {'A': A_matrix, 'b': b_vector}

    if plot_verbose and (n == 2 or n == 3):
        if n == 2:
            _plot_2d_convex_hull(hull, remaining_points, deleted_points, I_max, P_max, radius)
        elif n == 3:
            _plot_3d_convex_hull(hull, remaining_points, deleted_points, I_max, P_max, R_value)

    return hull, H_representation, remaining_points, deleted_points

def _plot_2d_convex_hull(hull, remaining_points, deleted_points, I_abs_max, P_max, radius):
    I_min, I_max = -I_abs_max, I_abs_max
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_title(f'2D Convex Hull and Points (P_max={P_max}, Sphere)')
    ax.set_xlabel('I_1 (Current in Coil 1)')
    ax.set_ylabel('I_2 (Current in Coil 2)')

    if deleted_points.shape[0] > 0:
        ax.scatter(deleted_points[:, 0], deleted_points[:, 1], c='gray', alpha=0.5, 
                   label='Deleted Points (on sphere, out of bounds)')
    if remaining_points.shape[0] > 0:
        ax.scatter(remaining_points[:, 0], remaining_points[:, 1], c='blue', alpha=0.7, 
                   label='Remaining Points (within bounds)')

    if hull is not None:
        for simplex in hull.simplices:
            ax.plot(remaining_points[simplex, 0], remaining_points[simplex, 1], 'r-')

    ax.plot([I_min, I_max, I_max, I_min, I_min], [I_min, I_min, I_max, I_max, I_min], 
            'k--', label=f'Bounding Box [{-I_abs_max}, {I_abs_max}]')

    theta_sphere = np.linspace(0, 2 * np.pi, 200)
    sphere_x = radius * np.cos(theta_sphere)
    sphere_y = radius * np.sin(theta_sphere)
    ax.plot(sphere_x, sphere_y, 'g:', alpha=0.6, label=f'Sphere (P_max={P_max})')

    ax.legend()
    ax.grid(True)
    plt.show()

def _plot_3d_convex_hull(hull, remaining_points, deleted_points, I_abs_max, P_max, R_value):
    I_min, I_max = -I_abs_max, I_abs_max
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'3D Convex Hull and Points (P_max={P_max}, Sphere)')
    ax.set_xlabel('I_1 (Current in Coil 1)')
    ax.set_ylabel('I_2 (Current in Coil 2)')
    ax.set_zlabel('I_3 (Current in Coil 3)')

    # if deleted_points.shape[0] > 0:
    #     ax.scatter(deleted_points[:, 0], deleted_points[:, 1], deleted_points[:, 2], 
    #                c='gray', alpha=0.3, label='Deleted Points (on sphere, out of bounds)')
    # if remaining_points.shape[0] > 0:
    #     ax.scatter(remaining_points[:, 0], remaining_points[:, 1], remaining_points[:, 2], 
    #                c='blue', alpha=0.5, label='Remaining Points (within bounds)')

    if hull is not None:
        triangles = hull.points[hull.simplices]
        hull_collection = Poly3DCollection(triangles, facecolors='pink', edgecolors='r', 
                                          linewidths=0.5, alpha=0.3)
        ax.add_collection3d(hull_collection)

    corners = np.array([
        [I_min, I_min, I_min], [I_max, I_min, I_min], [I_min, I_max, I_min], [I_min, I_min, I_max],
        [I_max, I_max, I_min], [I_max, I_min, I_max], [I_min, I_max, I_max], [I_max, I_max, I_max]
    ])
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c='k', marker='x', s=50, label='Bounding Box Corners')
    
    radius = np.sqrt(P_max / R_value)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_unit = np.outer(np.cos(u), np.sin(v))
    y_unit = np.outer(np.sin(u), np.sin(v))
    z_unit = np.outer(np.ones(np.size(u)), np.cos(v))
    
    sphere_x = radius * x_unit
    sphere_y = radius * y_unit
    sphere_z = radius * z_unit
    
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='g', alpha=0.2, 
                   linewidth=0, antialiased=True) # Label handled by custom legend
    
    handles, labels = ax.get_legend_handles_labels()
    custom_lines = [Line2D([0], [0], color='g', lw=4, alpha=0.5)]
    custom_labels = ['Sphere']
    handles.extend(custom_lines)
    labels.extend(custom_labels)
    
    ax.legend(handles, labels)
    ax.grid(True)
    plt.show()

def _plot_points_only(n_dim, remaining_points, deleted_points, I_abs_max, P_max, R_value):
    I_min, I_max = -I_abs_max, I_abs_max
    radius = np.sqrt(P_max / R_value)
    fig = plt.figure(figsize=(10, 8))

    if n_dim == 2:
        ax = fig.add_subplot(111)
        ax.set_title(f'2D Points (P_max={P_max}, Sphere) - No Convex Hull')
        ax.set_xlabel('I_1')
        ax.set_ylabel('I_2')
        if deleted_points.shape[0] > 0:
            ax.scatter(deleted_points[:, 0], deleted_points[:, 1], c='gray', alpha=0.5, label='Deleted Points')
        if remaining_points.shape[0] > 0:
            ax.scatter(remaining_points[:, 0], remaining_points[:, 1], c='blue', alpha=0.7, label='Remaining Points')
        ax.plot([I_min, I_max, I_max, I_min, I_min], [I_min, I_min, I_max, I_max, I_min], 'k--', label='Bounding Box')
        theta_sphere = np.linspace(0, 2 * np.pi, 200)
        sphere_x = radius * np.cos(theta_sphere)
        sphere_y = radius * np.sin(theta_sphere)
        ax.plot(sphere_x, sphere_y, 'g:', alpha=0.6, label=f'Sphere (P_max={P_max})')

    elif n_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'3D Points (P_max={P_max}, Sphere) - No Convex Hull')
        ax.set_xlabel('I_1')
        ax.set_ylabel('I_2')
        ax.set_zlabel('I_3')
        if deleted_points.shape[0] > 0:
            ax.scatter(deleted_points[:, 0], deleted_points[:, 1], deleted_points[:, 2], c='gray', alpha=0.3, label='Deleted Points')
        if remaining_points.shape[0] > 0:
            ax.scatter(remaining_points[:, 0], remaining_points[:, 1], remaining_points[:, 2], c='blue', alpha=0.5, label='Remaining Points')
        
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x_unit = np.outer(np.cos(u), np.sin(v))
        y_unit = np.outer(np.sin(u), np.sin(v))
        z_unit = np.outer(np.ones(np.size(u)), np.cos(v))
        sphere_x = radius * x_unit
        sphere_y = radius * y_unit
        sphere_z = radius * z_unit
        ax.plot_surface(sphere_x, sphere_y, sphere_z, color='g', alpha=0.2, linewidth=0, antialiased=True)
        
        handles, labels = ax.get_legend_handles_labels()
        custom_lines = [Line2D([0], [0], color='g', lw=4, alpha=0.5)] # For sphere legend
        custom_labels = ['Sphere']
        handles.extend(custom_lines)
        labels.extend(custom_labels)
        ax.legend(handles, labels)


    ax.legend()
    ax.grid(True)
    plt.show()

def check_points_distribution(K_points, P_max_val, R_list_val, convex_hull, I_abs_max_val):
    if convex_hull is None:
        print("Convex hull is None. Cannot check points distribution.")
        return

    min_I, max_I = -I_abs_max_val, I_abs_max_val
    R_list_val_np = np.array(R_list_val)
    n_dim = len(R_list_val_np)

    if n_dim == 0:
        print("Error: Coil resistance array R_list_val cannot be empty.")
        return
    
    R_value_internal = R_list_val_np[0]
    if not np.all(R_list_val_np == R_value_internal):
        print("Error: All resistances in R_list_val must be identical.")
        return
    if P_max_val <= 0: # Check P_max_val before using it
        print("Error: P_max_val must be positive.")
        return
    if R_value_internal <= 0:
        print("Error: All elements in R_list_val must be positive.")
        return
    if K_points <= 0:
        print("Error: K_points must be positive.")
        return

    radius_val = np.sqrt(P_max_val / R_value_internal)

    Z_normal = np.random.normal(size=(K_points, n_dim))
    norm_Z = np.linalg.norm(Z_normal, axis=1, keepdims=True)
    unit_sphere_points = np.zeros_like(Z_normal)
    valid_norms_mask = norm_Z.flatten() > 1e-12
    unit_sphere_points[valid_norms_mask, :] = Z_normal[valid_norms_mask, :] / norm_Z[valid_norms_mask, :]
    u_random = np.random.uniform(size=(K_points, 1))
    radii_scale = u_random**(1.0/n_dim)
    points_in_unit_ball = unit_sphere_points * radii_scale
    generated_points = points_in_unit_ball * radius_val

    A_hull = convex_hull.equations[:, :-1]
    d_hull_constants = convex_hull.equations[:, -1]
    evaluations = (A_hull @ generated_points.T) + d_hull_constants[:, np.newaxis]
    is_inside_hull_mask = np.all(evaluations <= 1e-9, axis=0)
    M_count = np.sum(is_inside_hull_mask)

    inside_cube_mask = np.all(np.abs(generated_points) <= max_I, axis=1) # Simplified check
    N_count = np.sum(inside_cube_mask)

    print(f"\n--- Points Distribution Check ---")
    print(f"Generated {K_points} points uniformly inside the sphere.")
    print(f"Number of these points inside the convex hull (M): {M_count}")
    print(f"Number of these points inside the hypercube [{-max_I}, {max_I}]^n (N): {N_count}")

    if N_count > 0:
        ratio_M_N = M_count / N_count
        print(f"Ratio M/N: {ratio_M_N:.4f}")
    else:
        print("Ratio M/N: Undefined (N=0)")
    print("--- End Points Distribution Check ---")

# --- Example Usage ---
if __name__ == '__main__':

    start_time = time.time()

    # print("--- 2D Example ---")
    # I_abs_max_2d = 10
    # P_max_2d = 500
    # R_2d = [3.0, 3.0]
    # m_points_2d = 200
    # desired_voxel_size = 5

    # hull_2d, H_rep_2d, rem_pts_2d, del_pts_2d = Get_CFW(
    #     I_abs_max_2d, P_max_2d, R_2d, m_points_2d,
    #     plot_verbose=True,
    #     voxel_size_for_merging=desired_voxel_size
    # )

    # if hull_2d:
    #     print(f"2D Convex Hull computed with {len(hull_2d.vertices)} vertices.")
    #     print(f"Number of remaining points: {rem_pts_2d.shape[0]}")
    #     print(f"Number of deleted points: {del_pts_2d.shape[0]}")
    #     K_for_check = 1000
    #     print(f"\nCalling check_points_distribution with K={K_for_check} for the 2D case...")
    #     check_points_distribution(K_for_check, P_max_2d, R_2d, hull_2d, I_abs_max_2d)
    # else:
    #     print("2D Convex Hull could not be computed.")
    #     print(f"Number of remaining points: {rem_pts_2d.shape[0]}")
    #     print(f"Number of deleted points: {del_pts_2d.shape[0]}")

    print("\n--- Example 2: 3D case ---")
    I_abs_max_3d = 10
    P_max_3d = 550
    R_3d = [3.0, 3.0, 3.0]
    m_points_3d = 202
    desired_voxel_size = 0.1

    hull_3d, H_rep_3d, rem_pts_3d, del_pts_3d = Get_CFW(
        I_abs_max_3d, P_max_3d, R_3d, m_points_3d,
        plot_verbose=True,
        voxel_size_for_merging=desired_voxel_size
    )

    if hull_3d:
        print(f"3D Convex Hull computed with {len(hull_3d.vertices)} vertices and {len(hull_3d.simplices)} facets.")
        print(f"Number of remaining points: {rem_pts_3d.shape[0]}")
        print(f"Number of deleted points: {del_pts_3d.shape[0]}")
        K_for_check = 200 
        print(f"\nCalling check_points_distribution with K={K_for_check} for the 3D case...")
        check_points_distribution(K_for_check, P_max_3d, R_3d, hull_3d, I_abs_max_3d)
    else:
        print("3D Convex Hull could not be computed.")
        print(f"Number of remaining points: {rem_pts_3d.shape[0]}")
        print(f"Number of deleted points: {del_pts_3d.shape[0]}")

    # 6D 3000

    # print("\n--- Example 3: 7D case ---")
    # I_abs_max_7d = 10.0
    # P_max_7d = 1900
    # R_7d = [3.0] * 7
    # m_points_7d = 8000  # Reduced m for a potentially faster ConvexHull with a finer voxel grid
    # desired_voxel_size_7d = 2.0  # Smaller voxel size for better resolution

    # hull_7d, H_rep_7d, rem_pts_7d, del_pts_7d = Get_CFW(
    #     I_abs_max_7d, P_max_7d, R_7d, m_points_7d,
    #     plot_verbose=False,
    #     voxel_size_for_merging=desired_voxel_size_7d
    # )
    # if hull_7d:
    #     print(f"7D Convex Hull computed with {len(hull_7d.vertices)} vertices and {len(hull_7d.simplices)} facets.")
    #     print(f"Number of remaining points: {rem_pts_7d.shape[0]}")
    #     print(f"Number of deleted points: {del_pts_7d.shape[0]}")
    #     K_for_check = 500
    #     print(f"\nCalling check_points_distribution with K={K_for_check} for the 7D case...")
    #     check_points_distribution(K_for_check, P_max_7d, R_7d, hull_7d, I_abs_max_7d)
    # else:
    #     print("7D Convex Hull could not be computed.")
    #     print(f"Number of remaining points: {rem_pts_7d.shape[0]}")
    #     print(f"Number of deleted points: {del_pts_7d.shape[0]}")

    # print("\n--- Example 3: 10D case (no plotting) ---")
    # I_abs_max_10d = 10.0
    # P_max_10d = 2000
    # R_10d = [3.0] * 10
    # m_points_10d = 10000
    # desired_voxel_size_10d = 90

    # hull_10d, H_rep_10d, rem_pts_10d, del_pts_10d = Get_CFW(
    #     I_abs_max_10d, P_max_10d, R_10d, m_points_10d,
    #     plot_verbose=False,
    #     voxel_size_for_merging=desired_voxel_size_10d
    # )

    # if hull_10d:
    #     print(f"10D Convex Hull computed with {len(hull_10d.vertices)} vertices and {len(hull_10d.simplices)} facets.")
    #     print(f"Number of remaining points: {rem_pts_10d.shape[0]}")
    #     print(f"Number of deleted points: {del_pts_10d.shape[0]}")
    #     K_for_check = 500
    #     print(f"\nCalling check_points_distribution with K={K_for_check} for the 10D case...")
    #     check_points_distribution(K_for_check, P_max_10d, R_10d, hull_10d, I_abs_max_10d)
    # else:
    #     print("10D Convex Hull could not be computed.")
    #     print(f"Number of remaining points: {rem_pts_10d.shape[0]}")
    #     print(f"Number of deleted points: {del_pts_10d.shape[0]}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")