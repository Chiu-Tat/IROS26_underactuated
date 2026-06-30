# There would be several steps to get the feasible poses.

# =================== One target one time ========================
# 1. Calibrate the OAW (omni-actuation workspace) for each target;
# 2. For each possible pose:
#       Check whether each target is within their OARs.
#          -- If yes:
#               Check whether each target is within other targets' Influence Region:
#               -- If not:
#                     Feasible pose found.
#               -- If yes:
#                     Infeasible pose, continue.
#          -- If not:
#               Infeasible pose, continue.

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import math
from lib import Extract_Map_I2B, Map_I2B, Plot_Hull_H_2D, Transform_and_Extract_Facets, HyperPlaneShiftingMethod

def Get_Ellipsoid_CFW(Q, r, I_abs_max, m, plot_verbose=True, regularization_factor=1e-10):
    """
    Calculates the convex feasible workspace for combined ellipsoidal and cubic constraints.
    The constraints are i^T*Q*i <= r and -I_abs_max <= i_k <= I_abs_max.

    Args:
        Q (np.array): The matrix of the quadratic form (should be positive definite for ellipsoid).
        r (float): The scalar value for the quadratic constraint.
        I_abs_max (float): Absolute maximum allowable current for any coil.
        m (int): Number of points to generate on the ellipsoid surface.
        plot_verbose (bool): If True, generates a plot (2D or 3D based on matrix size).
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

    # Check matrix conditioning
    eigenvalues = np.linalg.eigvals(Q)
    
    # Check if matrix is positive definite
    min_eigenvalue = np.min(eigenvalues)
    is_positive_definite = min_eigenvalue > 1e-12  # Use a reasonable threshold
    
    if not is_positive_definite:
        # Option 1: Add regularization
        regularization_needed = max(regularization_factor, abs(min_eigenvalue) + 1e-8)
        Q_regularized = Q + regularization_needed * np.eye(n)
        eigenvalues_reg = np.linalg.eigvals(Q_regularized)
        
        # Ask user for choice (in practice, you might want to make this automatic)
        use_regularization = True  # Set to True to proceed with regularization
        
        if use_regularization:
            Q = Q_regularized
            eigenvalues = eigenvalues_reg
        else:
            # Generate points uniformly in the hypercube instead
            remaining_points = np.random.uniform(-I_max, I_max, (m, n))
            deleted_points = np.array([])
            
            try:
                hull = ConvexHull(remaining_points)
                A_matrix = hull.equations[:, :-1]
                b_vector = -hull.equations[:, -1]
                H_representation = {'A': A_matrix, 'b': b_vector}
                
                return hull, H_representation, remaining_points, deleted_points
            except Exception as e:
                return None, None, remaining_points, deleted_points

    # --- 1. Generate m points on the ellipsoid i^T*Q*i = r ---
    try:
        eigenvals, eigenvecs = np.linalg.eigh(Q)
        # Ensure all eigenvalues are positive
        eigenvals = np.maximum(eigenvals, 1e-12)
        
        # For transformation: Q = U * Lambda * U^T, so ellipsoid transform is U * sqrt(Lambda)
        sqrt_lambda = np.sqrt(eigenvals)
        
    except Exception as e:
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

    # --- 2. Project points outside the hypercube inward ---
    original_points = points_on_ellipsoid.copy()
    max_abs_coord = np.max(np.abs(original_points), axis=1)
    outside_mask = max_abs_coord > I_max

    points_inside_bounds = original_points.copy()
    if np.any(outside_mask):
        outside_points = original_points[outside_mask]
        max_abs_for_scaling = max_abs_coord[outside_mask].reshape(-1, 1)
        scaling_factors = I_max / max_abs_for_scaling
        points_inside_bounds[outside_mask] = outside_points * scaling_factors

    deleted_points = original_points[outside_mask]
    remaining_points = points_inside_bounds

    # --- 3. Compute Convex Hull ---
    if remaining_points.shape[0] < n + 1:
        return None, None, remaining_points, deleted_points

    try:
        hull = ConvexHull(remaining_points)
    except Exception as e:
        return None, None, remaining_points, deleted_points

    A_matrix = hull.equations[:, :-1]
    b_vector = -hull.equations[:, -1]
    H_representation = {'A': A_matrix, 'b': b_vector}

    return hull, H_representation, remaining_points, deleted_points

B_threshold = np.array([0.057, 0.057]) # T, threshold of magnetic field for actuation
B1 = B_threshold[0]
B2 = B_threshold[1]

num_targets = B_threshold.shape[0]
I_min = -17
I_max = 17

def get_convex_hull(B):
    # Create a grid of points
    x = np.linspace(-0.06, 0.06, 50)
    y = np.linspace(-0.06, 0.06, 50)
    X, Y = np.meshgrid(x, y)
    R = 0.06
    circle_mask = (X**2 + Y**2) <= R **2
    # Initialize P with NaN values
    # P = np.full_like(X, np.nan)
    
    # Store valid points for convex hull
    valid_points = []

    # Evaluate the magnetic field at each point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if circle_mask[i, j]:
                target_points = [
                    {'X': X[i, j], 'Y': Y[i, j], 'Z': 0, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},
                ]
                A = Extract_Map_I2B(target_points) @ Map_I2B(target_points)
                G, K = HyperPlaneShiftingMethod(A, I_min, I_max)
                d = K
                # P[i, j] = 1 if (d >= B).all() else -1
                
                # Store valid points for convex hull
                if (d >= B).all():
                    valid_points.append([X[i, j], Y[i, j]])
    
    # Compute convex hull if we have enough valid points
    if len(valid_points) >= 3:
        valid_points = np.array(valid_points)
        hull = ConvexHull(valid_points)
        
        # Extract H-representation parameters (N, d)
        N = hull.equations[:, :-1]  # Normal vectors
        d = -hull.equations[:, -1].reshape(-1, 1)  # Distance parameters (note the negative sign)
        
        return N, d
    else:
        # Return empty arrays if no valid convex hull can be formed
        return np.array([]), np.array([])

def in_OAR(position, N, d):
    return np.all(N @ position.T <= d)

def get_influence_region(B, G, k):
    # Create a grid of points
    x = np.linspace(-0.06, 0.06, 50)
    y = np.linspace(-0.06, 0.0, 50)
    X, Y = np.meshgrid(x, y)
    R = 0.06
    circle_mask = (X**2 + Y**2) <= R **2
    # Initialize P with NaN values
    # P = np.full_like(X, np.nan)
    
    # Store valid points for convex hull
    valid_points = []

    # Evaluate the magnetic field at each point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if circle_mask[i, j]:
                target_points = [
                    {'X': X[i, j], 'Y': Y[i, j], 'Z': 0, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},
                ]
                A = Extract_Map_I2B(target_points) @ Map_I2B(target_points)
                N, d = Transform_and_Extract_Facets(A, G, k)
                # P[i, j] = 1 if (d >= B).all() else -1
                
                # Store valid points for convex hull
                if (d >= B).all():
                    valid_points.append([X[i, j], Y[i, j]])
    
    # Compute convex hull if we have enough valid points
    if len(valid_points) >= 3:
        valid_points = np.array(valid_points)
        
        # Check if points are collinear or have insufficient variation
        if valid_points.shape[0] >= 3:
            # Check for collinearity by computing the range of coordinates
            x_range = np.max(valid_points[:, 0]) - np.min(valid_points[:, 0])
            y_range = np.max(valid_points[:, 1]) - np.min(valid_points[:, 1])
            
            # If either dimension has very small variation, points are essentially collinear
            if x_range < 1e-10 or y_range < 1e-10:
                print(f"Warning: Points are collinear (x_range={x_range:.2e}, y_range={y_range:.2e})")
                return np.array([]), np.array([])
        
        try:
            hull = ConvexHull(valid_points)
            
            # Extract H-representation parameters (N, d)
            N = hull.equations[:, :-1]  # Normal vectors
            d = -hull.equations[:, -1].reshape(-1, 1)  # Distance parameters (note the negative sign)
            
            return N, d
        except Exception as e:
            print(f"Warning: ConvexHull computation failed: {e}")
            return np.array([]), np.array([])
    else:
        # Return empty arrays if no valid convex hull can be formed
        print(f"Warning: Insufficient valid points for convex hull ({len(valid_points)} points)")
        return np.array([]), np.array([])

# Get the position and orientation of each target, if 3 in total
def get_target_pose(x1r,y1r,angle):
    x1 = -0.02 # position x of the first target
    y1 = -0.05 # position y of the first target
    x1 = x1 + x1r
    y1 = y1 + y1r

    angle2_1 = 0 * math.pi # relative orientation of the second target with respect to the first one
    angle3_1 = 0 * math.pi # relative orientation of the third target with respect to the first one

    length_12 = 0.04 # distance between target 1 and target 2
    length_13 = 0.04 # distance between target 1 and target 3

    Position_2 = np.array([[x1 + length_12 * math.cos(angle2_1 + angle), y1 + length_12 * math.sin(angle2_1 + angle)]])
    Position_3 = np.array([[x1 + length_13 * math.cos(angle3_1 + angle), y1 + length_13 * math.sin(angle3_1 + angle)]])

    return Position_2

x1_discretized = np.linspace(-0.03, 0.03, 50)
y1_discretized = np.linspace(-0.03, 0.03, 50)
angle = 0 * math.pi # relative orientation of the total magnets

X, Y = np.meshgrid(x1_discretized, y1_discretized)

# Use P to store the feasibility of each pose
P = np.full_like(X, np.nan)

N1, d1 = get_convex_hull(B1)
N2, d2 = get_convex_hull(B2)
# N3, d3 = get_convex_hull(B3)

for i in range(len(x1_discretized)):
    for j in range(len(y1_discretized)):
        position_1 = np.array([[X[i, j]-0.02, Y[i, j]-0.05]])
        position_2 = get_target_pose(X[i, j], Y[i, j], angle)
        print('position1:')
        print(position_1)
        print('position2:')
        print(position_2)
        # print('position3:')
        # print(position_3)
        print(f"Evaluating pose ({X[i, j]:.3f}, {Y[i, j]:.3f})...")
        if in_OAR(position_1, N1, d1) and in_OAR(position_2, N2, d2):
            print(f"Pose ({X[i, j]:.3f}, {Y[i, j]:.3f}) - All targets in their OARs.")
            position123 = [position_1, position_2]
            feasible_pose = True
            for idx, point in enumerate(position123):
                target_points = [
                    {'X': point[0,0], 'Y': point[0,1], 'Z': 0, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},  
                ]
                A = Extract_Map_I2B(target_points) @ Map_I2B(target_points)
                I_abs_max = 17
                m = 1000
                Q_matrix = A.T @ A
                r_scalar = B_threshold[idx]**2

                hull, H_representation, remaining_points, deleted_points = Get_Ellipsoid_CFW(Q_matrix, r_scalar, I_abs_max, m, plot_verbose=True)
                if hull is None:
                    print(f"Warning: Could not compute ellipsoid CFW for target {idx+1}")
                    feasible_pose = False
                    break
                    
                G = hull.equations[:, :-1]
                k = -hull.equations[:, -1]
                # Check if other targets are within this target's influence region
                influence_violated = False
                for other_idx, other_point in enumerate(position123):
                    if other_idx != idx:  # Skip the current target
                        N_inf, d_inf = get_influence_region(B_threshold[other_idx], G, k)
                        if len(N_inf) > 0 and in_OAR(other_point, N_inf, d_inf):
                            influence_violated = True
                            print(f"Pose ({X[i, j]:.3f}, {Y[i, j]:.3f}) - Target {idx+1} influence region violated by target {other_idx+1}.")
                            break

                if influence_violated:
                    feasible_pose = False
                    break
                else:
                    print(f"Pose ({X[i, j]:.3f}, {Y[i, j]:.3f}) - Target {idx+1} influence region OK.")
            
            P[i, j] = 1 if feasible_pose else -1
        else:
            P[i, j] = -1  # Not all targets in their OARs
            print(f"Pose ({X[i, j]:.3f}, {Y[i, j]:.3f}) - Some targets not in their OARs.")

# Find indices for feasible and infeasible poses
feasible_mask = (P == 1)
infeasible_mask = (P == -1)

# Plot the feasible poses
plt.figure(figsize=(6, 6))

# Scatter plot for feasible poses
if np.any(feasible_mask):
    plt.scatter(X[feasible_mask], Y[feasible_mask], c='green', s=50, alpha=1, label='Feasible poses')

# Scatter plot for infeasible poses
if np.any(infeasible_mask):
    plt.scatter(X[infeasible_mask], Y[infeasible_mask], c='salmon', s=50, alpha=1, label='Infeasible poses')

# # Create contour plot for feasible/infeasible poses
# contour_levels = [-1.5, -0.5, 0.5, 1.5]
# colors = ['salmon', 'green']
# plt.contourf(X, Y, P, levels=contour_levels, colors=colors, alpha=0.7)

# # Add contour lines for clarity
# plt.contour(X, Y, P, levels=contour_levels, colors=['darkred', 'darkgreen'], linewidths=1)

# # Create custom legend
# legend_elements = [
#     mpatches.Patch(color='salmon', label='Infeasible poses'),
#     mpatches.Patch(color='green', label='Feasible poses')
# ]
# plt.legend(handles=legend_elements)

plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Feasible Poses for EM Coils')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Number of feasible poses found: {np.sum(P == 1)}")
print(f"Total poses evaluated: {np.sum(~np.isnan(P))}")
