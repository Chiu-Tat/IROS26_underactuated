import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import math
from lib import Extract_Map_I2B, Map_I2B, Plot_Hull_H_2D, Transform_and_Extract_Facets, HyperPlaneShiftingMethod

def plot_figure_1_original_points(points_on_ellipsoid, Q, r, I_abs_max):
    """
    First figure: Show original ellipsoid points with color coding
    Green: inside cube, Red: outside cube
    """
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.set_title('Figure 1: Original Ellipsoid Points Classification')
    ax.set_xlabel('I_1')
    ax.set_ylabel('I_2')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Plot the ellipse with background fill
    eigenvals, eigenvecs = np.linalg.eigh(Q)
    eigenvals = np.maximum(eigenvals, 1e-12)
    sqrt_lambda = np.sqrt(eigenvals)
    semi_axes = np.sqrt(r) / sqrt_lambda
    
    theta = np.linspace(0, 2 * np.pi, 200)
    unit_circle = np.array([np.cos(theta), np.sin(theta)]).T
    ellipse_coords_principal = unit_circle * semi_axes.reshape(1, -1)
    ellipse_points = ellipse_coords_principal @ eigenvecs.T
    
    # Fill ellipse region first
    ax.fill(ellipse_points[:, 0], ellipse_points[:, 1], 
            color='lightgreen', alpha=0.2, label='Ellipse Constraint Region')
    ax.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'g-', linewidth=2)
    
    # Plot the cube with background fill
    I_min, I_max = -I_abs_max, I_abs_max
    square_corners = np.array([[I_min, I_min], [I_max, I_min], [I_max, I_max], [I_min, I_max], [I_min, I_min]])
    ax.fill(square_corners[:, 0], square_corners[:, 1], 
            color='lightblue', alpha=0.2, label='Cube Constraint Region')
    ax.plot(square_corners[:, 0], square_corners[:, 1], 'k-', linewidth=2)
    
    # Classify points: inside or outside cube
    max_abs_coord = np.max(np.abs(points_on_ellipsoid), axis=1)
    inside_cube = max_abs_coord <= I_abs_max
    outside_cube = ~inside_cube
    
    # Plot points with color coding
    if np.any(inside_cube):
        ax.scatter(points_on_ellipsoid[inside_cube, 0], points_on_ellipsoid[inside_cube, 1], 
                  c='green', s=30, alpha=0.7, label=f'Inside Cube')
    
    if np.any(outside_cube):
        ax.scatter(points_on_ellipsoid[outside_cube, 0], points_on_ellipsoid[outside_cube, 1], 
                  c='red', s=30, alpha=0.7, label=f'Outside Cube')
    
    # Calculate axis limits to show both cube and ellipse completely
    ellipse_range = np.max(np.abs(ellipse_points))
    cube_range = I_abs_max
    max_range = max(ellipse_range, cube_range) * 1.2
    
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    
    # Add axis lines through origin
    ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
    
    # Remove tick values
    ax.set_xticks([])
    ax.set_yticks([])
    
    # ax.legend()
    plt.tight_layout()
    plt.show()

def plot_figure_2_projection_trajectories(points_on_ellipsoid, points_inside_bounds, Q, r, I_abs_max):
    """
    Second figure: Show projection trajectories for points moved inside the cube
    """
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.set_title('Figure 2: Point Projection Trajectories')
    ax.set_xlabel('I_1')
    ax.set_ylabel('I_2')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Plot the ellipse with background fill
    eigenvals, eigenvecs = np.linalg.eigh(Q)
    eigenvals = np.maximum(eigenvals, 1e-12)
    sqrt_lambda = np.sqrt(eigenvals)
    semi_axes = np.sqrt(r) / sqrt_lambda
    
    theta = np.linspace(0, 2 * np.pi, 200)
    unit_circle = np.array([np.cos(theta), np.sin(theta)]).T
    ellipse_coords_principal = unit_circle * semi_axes.reshape(1, -1)
    ellipse_points = ellipse_coords_principal @ eigenvecs.T
    
    # Fill ellipse region first
    ax.fill(ellipse_points[:, 0], ellipse_points[:, 1], 
            color='lightgreen', alpha=0.2, label='Ellipse Constraint Region')
    ax.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'g-', linewidth=2, alpha=0.7)
    
    # Plot the cube with background fill
    I_min, I_max = -I_abs_max, I_abs_max
    square_corners = np.array([[I_min, I_min], [I_max, I_min], [I_max, I_max], [I_min, I_max], [I_min, I_min]])
    ax.fill(square_corners[:, 0], square_corners[:, 1], 
            color='lightblue', alpha=0.2, label='Cube Constraint Region')
    ax.plot(square_corners[:, 0], square_corners[:, 1], 'k-', linewidth=2)
    
    # Identify points that were moved
    max_abs_coord = np.max(np.abs(points_on_ellipsoid), axis=1)
    outside_mask = max_abs_coord > I_abs_max
    
    # Plot trajectories for moved points
    if np.any(outside_mask):
        original_outside = points_on_ellipsoid[outside_mask]
        projected_outside = points_inside_bounds[outside_mask]
        
        for i in range(len(original_outside)):
            # Draw trajectory line
            ax.plot([original_outside[i, 0], projected_outside[i, 0]], 
                   [original_outside[i, 1], projected_outside[i, 1]], 
                   'r--', alpha=0.6, linewidth=1)
        
        # Plot original points (red)
        ax.scatter(original_outside[:, 0], original_outside[:, 1], 
                  c='red', s=40, alpha=0.8, marker='o', label=f'Original Points')
        
        # Plot projected points (orange)
        ax.scatter(projected_outside[:, 0], projected_outside[:, 1], 
                  c='orange', s=40, alpha=0.8, marker='s', label=f'Projected Points')

    # Plot points that didn't need to be moved (green)
    inside_mask = ~outside_mask
    if np.any(inside_mask):
        ax.scatter(points_on_ellipsoid[inside_mask, 0], points_on_ellipsoid[inside_mask, 1], 
                  c='green', s=30, alpha=0.7, label=f'Unchanged Points')

    # Calculate axis limits to show both cube and ellipse completely
    ellipse_range = np.max(np.abs(ellipse_points))
    cube_range = I_abs_max
    max_range = max(ellipse_range, cube_range) * 1.2
    
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    
    # Add axis lines through origin
    ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
    
    # Remove tick values
    ax.set_xticks([])
    ax.set_yticks([])
    
    # ax.legend()
    plt.tight_layout()
    plt.show()

def plot_figure_3_convex_hull_comparison(hull, remaining_points, Q, r, I_abs_max):
    """
    Third figure: Show convex hull of valid points compared with cube and ellipse
    """
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.set_title('Figure 3: Convex Hull vs Original Constraints')
    ax.set_xlabel('I_1')
    ax.set_ylabel('I_2')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Plot the cube boundary with background fill
    I_min, I_max = -I_abs_max, I_abs_max
    square_corners = np.array([[I_min, I_min], [I_max, I_min], [I_max, I_max], [I_min, I_max], [I_min, I_min]])
    ax.fill(square_corners[:, 0], square_corners[:, 1], 
            color='lightblue', alpha=0.2, label='Cube Constraint Region')
    ax.plot(square_corners[:, 0], square_corners[:, 1], 'k-', linewidth=2)
    
    # Plot the ellipse with background fill
    eigenvals, eigenvecs = np.linalg.eigh(Q)
    eigenvals = np.maximum(eigenvals, 1e-12)
    sqrt_lambda = np.sqrt(eigenvals)
    semi_axes = np.sqrt(r) / sqrt_lambda
    
    theta = np.linspace(0, 2 * np.pi, 200)
    unit_circle = np.array([np.cos(theta), np.sin(theta)]).T
    ellipse_coords_principal = unit_circle * semi_axes.reshape(1, -1)
    ellipse_points = ellipse_coords_principal @ eigenvecs.T
    ax.fill(ellipse_points[:, 0], ellipse_points[:, 1], 
            color='lightgreen', alpha=0.2, label='Ellipse Constraint Region')
    ax.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'g-', linewidth=2)
    
    # Plot convex hull
    if hull is not None:
        hull_vertices = remaining_points[hull.vertices]
        # Sort vertices to form a polygon for 2D
        center = np.mean(hull_vertices, axis=0)
        angles = np.arctan2(hull_vertices[:, 1] - center[1], hull_vertices[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        hull_vertices_sorted = hull_vertices[sorted_indices]
        
        # Close the polygon
        hull_vertices_closed = np.vstack([hull_vertices_sorted, hull_vertices_sorted[0]])
        ax.fill(hull_vertices_closed[:, 0], hull_vertices_closed[:, 1], 
                color='red', alpha=0.3, label='Convex Feasible Workspace')
        ax.plot(hull_vertices_closed[:, 0], hull_vertices_closed[:, 1], 
                'r-', linewidth=3, label='Convex Hull Boundary')
        
        # Plot hull vertices
        # ax.scatter(hull_vertices[:, 0], hull_vertices[:, 1], 
        #           c='red', s=60, marker='D', edgecolors='darkred', linewidth=1, 
        #           label=f'Hull Vertices ({len(hull_vertices)})')
    
    # Plot all valid points
    ax.scatter(remaining_points[:, 0], remaining_points[:, 1], 
              c='blue', s=20, alpha=0.6, label=f'Valid Points')
    
    # Calculate axis limits to show both cube and ellipse completely
    ellipse_range = np.max(np.abs(ellipse_points))
    cube_range = I_abs_max
    max_range = max(ellipse_range, cube_range) * 1.2
    
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    
    # Add axis lines through origin
    ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.3)
    
    # Remove tick values
    ax.set_xticks([])
    ax.set_yticks([])
    
    # ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.show()

def Get_Ellipsoid_CFW(Q, r, I_abs_max, m, plot_verbose=True, regularization_factor=1e-10):
    """
    Calculates the convex feasible workspace for combined ellipsoidal and cubic constraints.
    The constraints are i^T*Q*i <= r and -I_abs_max <= i_k <= I_abs_max.

    Args:
        Q (np.array): The matrix of the quadratic form (should be positive definite for ellipsoid).
        r (float): The scalar value for the quadratic constraint.
        I_abs_max (float): Absolute maximum allowable current for any coil.
        m (int): Number of points to generate on the ellipsoid surface.
        plot_verbose (bool): If True, generates the three requested plots for 2D case.
        regularization_factor (float): Factor to regularize ill-conditioned matrices.

    Returns:
        tuple:
            - hull (scipy.spatial.ConvexHull or None): The convex hull.
            - H_representation (dict or None): H-representation (Ax <= b).
            - remaining_points (np.array): Feasible points.
            - deleted_points (np.array): Points originally outside bounds.
    """
    n = Q.shape[0]
    I_max = I_abs_max

    # Debug: Print matrix properties
    print(f"Q matrix shape: {Q.shape}")
    print(f"Current space dimension: {n}D")
    print(f"Q matrix:\n{Q}")
    
    # Check if we can plot (only for 2D case)
    can_plot = (n == 2) and plot_verbose
    if plot_verbose and n != 2:
        print(f"Warning: Plotting is only supported for 2D problems. Current dimension: {n}D")
    
    # Check matrix conditioning
    eigenvalues = np.linalg.eigvals(Q)
    condition_number = np.linalg.cond(Q)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Condition number: {condition_number}")
    print(f"Minimum eigenvalue: {np.min(eigenvalues)}")
    print(f"Maximum eigenvalue: {np.max(eigenvalues)}")
    print(f"r value: {r}")
    
    # Check if matrix is positive definite
    min_eigenvalue = np.min(eigenvalues)
    is_positive_definite = min_eigenvalue > 1e-12  # Use a reasonable threshold
    
    print(f"Is Q positive definite? {is_positive_definite}")
    
    if not is_positive_definite:
        print(f"WARNING: Matrix Q is not positive definite!")
        print(f"The constraint i^T*Q*i <= r does NOT define an ellipsoid.")
        print(f"This constraint may define:")
        if min_eigenvalue < -1e-12:
            print("  - A hyperbolic region (indefinite quadratic form)")
        else:
            print("  - A degenerate ellipsoid (singular quadratic form)")
        
        print(f"Options:")
        print(f"1. Add regularization to make Q positive definite")
        print(f"2. Use only the cubic constraints (hypercube)")
        print(f"3. Reformulate the problem")
        
        # Option 1: Add regularization
        regularization_needed = max(regularization_factor, abs(min_eigenvalue) + 1e-8)
        Q_regularized = Q + regularization_needed * np.eye(n)
        eigenvalues_reg = np.linalg.eigvals(Q_regularized)
        print(f"Adding regularization of {regularization_needed}")
        print(f"Regularized eigenvalues: {eigenvalues_reg}")
        
        # Ask user for choice (in practice, you might want to make this automatic)
        use_regularization = True  # Set to True to proceed with regularization
        
        if use_regularization:
            print("Proceeding with regularized matrix...")
            Q = Q_regularized
            eigenvalues = eigenvalues_reg
        else:
            print("Proceeding with cubic constraints only...")
            # Generate points uniformly in the hypercube instead
            remaining_points = np.random.uniform(-I_max, I_max, (m, n))
            deleted_points = np.array([])
            
            try:
                hull = ConvexHull(remaining_points)
                A_matrix = hull.equations[:, :-1]
                b_vector = -hull.equations[:, -1]
                H_representation = {'A': A_matrix, 'b': b_vector}
                
                if can_plot:
                    _plot_2d_square_only(remaining_points, I_max)
                
                return hull, H_representation, remaining_points, deleted_points
            except Exception as e:
                print(f"Error with cubic-only approach: {e}")
                return None, None, remaining_points, deleted_points

    # Check if r is reasonable compared to eigenvalues
    max_eigenvalue = np.max(eigenvalues)
    min_eigenvalue = np.min(eigenvalues)
    print(f"r / min_eigenvalue = {r / min_eigenvalue}")
    print(f"r / max_eigenvalue = {r / max_eigenvalue}")
    
    # Check if the ellipsoid is reasonable in size
    if r / min_eigenvalue > (I_max**2):
        print(f"WARNING: The ellipsoid may be much larger than the cubic constraints!")
        print(f"Ellipsoid 'radius' in weakest direction: {np.sqrt(r / min_eigenvalue):.3f}")
        print(f"Cubic constraint limit: {I_max}")

    # --- 1. Generate m points on the ellipsoid i^T*Q*i = r ---
    try:
        eigenvals, eigenvecs = np.linalg.eigh(Q)
        # Ensure all eigenvalues are positive
        eigenvals = np.maximum(eigenvals, 1e-12)
        
        # For transformation: Q = U * Lambda * U^T, so ellipsoid transform is U * sqrt(Lambda)
        sqrt_lambda = np.sqrt(eigenvals)
        
        print(f"Eigenvalue decomposition successful")
        print(f"Semi-axes lengths: {np.sqrt(r) / sqrt_lambda}")
        
    except Exception as e:
        print(f"Eigenvalue decomposition failed: {e}")
        return None, None, np.array([]), np.array([])

    # Generate evenly distributed points on unit sphere (2D or 3D)
    if m == 0:
        points_on_unit_sphere = np.empty((0, n))
    elif n == 2:
        # Generate evenly distributed points on unit circle
        points_on_unit_sphere = np.zeros((m, 2))
        theta = np.linspace(0, 2 * np.pi, m, endpoint=False)
        points_on_unit_sphere[:, 0] = np.cos(theta)
        points_on_unit_sphere[:, 1] = np.sin(theta)
    elif n == 3:
        points_on_unit_sphere = np.zeros((m, 3))
        indices = np.arange(0, m, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / m)
        theta = np.pi * (1 + np.sqrt(5)) * indices
        points_on_unit_sphere[:, 0] = np.cos(theta) * np.sin(phi)
        points_on_unit_sphere[:, 1] = np.sin(theta) * np.sin(phi)
        points_on_unit_sphere[:, 2] = np.cos(phi)
    else:
        # Fallback for higher dimensions
        random_points = np.random.normal(size=(m, n))
        norm = np.linalg.norm(random_points, axis=1, keepdims=True)
        points_on_unit_sphere = np.zeros_like(random_points)
        non_zero_norm_mask = (norm > 1e-15).flatten()
        points_on_unit_sphere[non_zero_norm_mask] = random_points[non_zero_norm_mask] / norm[non_zero_norm_mask]
        
        # Handle zero norm case
        zero_norm_mask = ~non_zero_norm_mask
        if np.any(zero_norm_mask):
            canonical_vector = np.zeros(n)
            if n > 0: 
                canonical_vector[0] = 1.0
            points_on_unit_sphere[zero_norm_mask] = canonical_vector

    # Transform unit sphere points to ellipsoid surface
    # Method: Transform to principal axis coordinates, scale, then transform back
    
    # Step 1: Transform unit sphere points to ellipsoid coordinates (principal axes)
    # In the principal axis system, the ellipsoid equation is sum((x_i/a_i)^2) = 1
    # where a_i = sqrt(r) / sqrt(lambda_i) are the semi-axis lengths
    semi_axes = np.sqrt(r) / sqrt_lambda
    
    # Step 2: Scale unit sphere points by semi-axes lengths
    ellipsoid_coords_principal = points_on_unit_sphere * semi_axes.reshape(1, -1)
    
    # Step 3: Transform back to original coordinate system
    points_on_ellipsoid = ellipsoid_coords_principal @ eigenvecs.T
    
    print(f"Generated {points_on_ellipsoid.shape[0]} evenly distributed points on ellipsoid")
    print(f"Point range: min = {np.min(points_on_ellipsoid, axis=0)}, max = {np.max(points_on_ellipsoid, axis=0)}")
    
    # Verify that points are on the ellipsoid surface (debugging)
    if m > 0:
        # Check that i^T * Q * i ≈ r for all points
        quadratic_values = np.sum((points_on_ellipsoid @ eigenvecs @ np.diag(sqrt_lambda))**2, axis=1)
        max_deviation = np.max(np.abs(quadratic_values - r))
        print(f"Maximum deviation from ellipsoid surface: {max_deviation}")
        if max_deviation > 1e-10:
            print("Warning: Points may not be exactly on ellipsoid surface")

    # --- Generate Figure 1: Original points classification ---
    if can_plot:
        print("Generating Figure 1: Original points classification")
        plot_figure_1_original_points(points_on_ellipsoid, Q, r, I_abs_max)

    # --- 2. Project points outside the hypercube inward ---
    original_points = points_on_ellipsoid.copy()
    max_abs_coord = np.max(np.abs(original_points), axis=1)
    outside_mask = max_abs_coord > I_max
    
    print(f"Points outside bounds: {np.sum(outside_mask)} / {len(outside_mask)}")
    print(f"Percentage outside bounds: {100*np.sum(outside_mask)/len(outside_mask):.1f}%")

    points_inside_bounds = original_points.copy()
    if np.any(outside_mask):
        outside_points = original_points[outside_mask]
        max_abs_for_scaling = max_abs_coord[outside_mask].reshape(-1, 1)
        scaling_factors = I_max / max_abs_for_scaling
        points_inside_bounds[outside_mask] = outside_points * scaling_factors

    deleted_points = original_points[outside_mask]
    remaining_points = points_inside_bounds
    
    print(f"Remaining points after projection: {remaining_points.shape[0]}")

    # --- Generate Figure 2: Projection trajectories ---
    if can_plot:
        print("Generating Figure 2: Projection trajectories")
        plot_figure_2_projection_trajectories(points_on_ellipsoid, points_inside_bounds, Q, r, I_abs_max)

    # --- 3. Compute Convex Hull ---
    if remaining_points.shape[0] < n + 1:
        print(f"Warning: Not enough points ({remaining_points.shape[0]}) to form a convex hull in {n}D.")
        return None, None, remaining_points, deleted_points

    try:
        hull = ConvexHull(remaining_points)
        print(f"ConvexHull computed successfully with {len(hull.vertices)} vertices")
    except Exception as e:
        print(f"Error computing ConvexHull: {e}")
        return None, None, remaining_points, deleted_points

    A_matrix = hull.equations[:, :-1]
    b_vector = -hull.equations[:, -1]
    H_representation = {'A': A_matrix, 'b': b_vector}

    # --- Generate Figure 3: Convex hull comparison ---
    if can_plot:
        print("Generating Figure 3: Convex hull comparison")
        plot_figure_3_convex_hull_comparison(hull, remaining_points, Q, r, I_abs_max)

    return hull, H_representation, remaining_points, deleted_points

def _plot_2d_ellipsoid_hull(hull, remaining_points, deleted_points, I_abs_max, Q, r):
    """Plot for 2D current space with ellipse and square constraints"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('2D Feasible Current Space (Ellipse & Square)')
    ax.set_xlabel('I_1')
    ax.set_ylabel('I_2')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Plot convex hull of the feasible region
    if hull is not None:
        # Get hull vertices in order for 2D plotting
        hull_vertices = remaining_points[hull.vertices]
        # Sort vertices to form a polygon
        center = np.mean(hull_vertices, axis=0)
        angles = np.arctan2(hull_vertices[:, 1] - center[1], hull_vertices[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        hull_vertices_sorted = hull_vertices[sorted_indices]
        
        # Close the polygon
        hull_vertices_closed = np.vstack([hull_vertices_sorted, hull_vertices_sorted[0]])
        ax.fill(hull_vertices_closed[:, 0], hull_vertices_closed[:, 1], 
                color='cyan', alpha=0.3, edgecolor='b', linewidth=2, label='Feasible Region (Hull)')

    # Plot the square constraint boundaries
    I_min, I_max = -I_abs_max, I_abs_max
    square_corners = np.array([[I_min, I_min], [I_max, I_min], [I_max, I_max], [I_min, I_max], [I_min, I_min]])
    # ax.plot(square_corners[:, 0], square_corners[:, 1], 'k-', linewidth=2, alpha=0.7, label='Square Constraint')
    ax.plot(square_corners[:, 0], square_corners[:, 1], 'k-', linewidth=2, alpha=0.7)
    # ax.scatter([I_min, I_max, I_max, I_min], [I_min, I_min], 
    #            c='k', marker='x', s=50, label='Square Corners')

    # Plot the original ellipse using the same transformation as point generation
    eigenvals, eigenvecs = np.linalg.eigh(Q)
    eigenvals = np.maximum(eigenvals, 1e-12)
    sqrt_lambda = np.sqrt(eigenvals)
    semi_axes = np.sqrt(r) / sqrt_lambda
    
    # Generate ellipse points
    theta = np.linspace(0, 2 * np.pi, 200)
    unit_circle = np.array([np.cos(theta), np.sin(theta)]).T
    ellipse_coords_principal = unit_circle * semi_axes.reshape(1, -1)
    ellipse_points = ellipse_coords_principal @ eigenvecs.T
    
    ax.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'g-', linewidth=2, alpha=0.7, label='Ellipse Constraint')

    # Plot remaining and deleted points for debugging
    if len(remaining_points) > 0:
        # ax.scatter(remaining_points[:, 0], remaining_points[:, 1], 
        #           c='blue', s=15, alpha=0.6, label='Feasible Points')
        ax.scatter(remaining_points[:, 0], remaining_points[:, 1], 
                  c='blue', s=15, alpha=0.6)
    
    if len(deleted_points) > 0:
        ax.scatter(deleted_points[:, 0], deleted_points[:, 1], 
                  c='red', s=15, alpha=0.6, label='Projected Points')

    # Set axis limits
    all_points = np.vstack([remaining_points, ellipse_points, square_corners[:-1]])
    max_range = np.max(np.abs(all_points)) * 1.2
    max_lim = min(max_range, I_abs_max * 1.1)
    
    ax.set_xlim([-max_lim, max_lim])
    ax.set_ylim([-max_lim, max_lim])
    # ax.legend()
    plt.show()

def _plot_2d_square_only(points, I_abs_max):
    """Plot for 2D square constraints only (when Q is not positive definite)"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('2D Feasible Current Space (Square Constraints Only)')
    ax.set_xlabel('I_1')
    ax.set_ylabel('I_2')
    ax.grid(True)
    ax.set_aspect('equal')

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=15, alpha=0.6, label='Points')

    # Plot the square constraint
    # I_min, I_max = -I_abs_max, I_abs_max
    # square_corners = np.array([[I_min, I_min], [I_max, I_min], [I_max, I_max], [I_min, I_max], [I_min, I_min]])
    # ax.fill(square_corners[:, 0], square_corners[:, 1], 
    #         color='lightgray', alpha=0.3, edgecolor='k', linewidth=2, label='Square Constraint')
    # ax.scatter([I_min, I_max, I_max, I_min], [I_min, I_min, I_max, I_max], 
    #            c='k', marker='x', s=50, label='Square Corners')

    ax.set_xlim([-I_abs_max*1.1, I_abs_max*1.1])
    ax.set_ylim([-I_abs_max*1.1, I_abs_max*1.1])
    # ax.legend()
    plt.show()

# Test example - Add this at the end of the file
if __name__ == "__main__":
    # Example usage for 2D case to test plotting
    print("Testing 2D ellipsoid CFW with plotting...")
    
    # Define a simple 2D ellipsoid
    Q = np.array([[0.3, 0.5], 
                  [0.5, 2.0]])  # 2x2 positive definite matrix
    r = 1.0                    # Ellipsoid constraint value
    I_abs_max = 1.5           # Cube constraint
    m = 50                    # Number of points
    
    # Call the function with plotting enabled
    hull, H_rep, remaining_points, deleted_points = Get_Ellipsoid_CFW(
        Q=Q, 
        r=r, 
        I_abs_max=I_abs_max, 
        m=m, 
        plot_verbose=True
    )
    
    print(f"\nResults:")
    print(f"Hull vertices: {len(hull.vertices) if hull else 0}")
    print(f"Remaining points: {len(remaining_points)}")
    print(f"Deleted points: {len(deleted_points)}")












