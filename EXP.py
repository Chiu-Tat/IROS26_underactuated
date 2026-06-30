import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sympy as sp
import time
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from lib import Extract_Map_I2B, Map_I2B, Plot_Hull_H_2D, Transform_and_Extract_Facets 

# current restrictions
current_limit = 17
# Rotating magnetic field parameters
field_amplitude = 0.057 # Tesla (1 mT)
angular_velocity = 2 * np.pi /10  # rad/s (1 Hz rotation)
simulation_duration = 10.0  # seconds

# Configuration parameters
control_frequency = 20  # Hz (20 Hz = 50ms intervals)
control_period = 1.0 / control_frequency  # seconds

# List of fitted parameters for each coil
params_list = [
    # np.array([-13.04945069, -4.41557229, 6.47376799, 0.12129096, 0.00466922, -0.0174842]),  # coil 1
    np.array([-5.10083416, 13.54294901, 7.85474539, 0.05834654, -0.11165548, -0.01850546]),  # coil 2
    np.array([4.05088788, 14.23365818, 6.44760956, -0.05903076, -0.11020417, -0.01488244]),  # coil 3
    # np.array([13.89011305, -0.06092074, 4.77365608, -0.12306086, -0.00085745, -0.01378161]), # coil 4
    # np.array([11.44363813, -9.40543896, 4.46367162, -0.06806179, 0.1024875, -0.01397152]),  # coil 5
    # np.array([-9.00577939, -12.78905365, 5.98650851, 0.06473315, 0.10618968, -0.0151172]),  # coil 6
    np.array([0.92820081, 8.54965337, 8.72298349, -0.00381254, -0.08845466, -0.08874662]),   # coil 7
    # np.array([8.7302819, -4.90773115, 7.00109937, -0.07977306, 0.04481733, -0.08536032]),   # coil 8
    # np.array([-7.68962762, -6.83258326, 8.12112247, 0.07498008, 0.04542436, -0.08696975]),   # coil 9
    # np.array([2.35614001, -1.11370036, 14.00304846, -0.00722183, 0.00029277, -0.12482979])   # coil 10
]

# Define the symbols
m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z = sp.symbols('m0 m1 m2 r0_0 r0_1 r0_2 X Y Z')

# Constants
mu0 = 4 * sp.pi * 1e-7

# Calculate displacement vector
dx = X - r0_0
dy = Y - r0_1
dz = Z - r0_2

# Calculate distance to the coordinate point
r = sp.sqrt(dx**2 + dy**2 + dz**2) + 1e-9  # Add a small constant to avoid division by zero

# Calculate dot product of displacement vector and magnetic dipole moment
dot_product = m0 * dx + m1 * dy + m2 * dz

# Calculate magnetic field components
model_Bx = (mu0 / (4 * sp.pi)) * (3 * dx * dot_product / r**5 - m0 / r**3)
model_By = (mu0 / (4 * sp.pi)) * (3 * dy * dot_product / r**5 - m1 / r**3)

# Convert the symbolic functions to numerical functions
dipole_model_Bx = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx, 'numpy')
dipole_model_By = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_By, 'numpy')

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

    # Debug: Print matrix properties
    print(f"Q matrix shape: {Q.shape}")
    print(f"Current space dimension: {n}D")
    print(f"Q matrix:\n{Q}")
    
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

    return hull, H_representation, remaining_points, deleted_points

def get_convex_constraint():
    target_points = [
        # {'X': -0.02, 'Y': -0.05, 'Z': 0, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},
        {'X': 0.0, 'Y': -0.03, 'Z': 0, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},       
        # {'X': 0.02, 'Y': -0.05, 'Z': 0, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None}
    ]
    A = Extract_Map_I2B(target_points) @ Map_I2B(target_points)
    # print("A matrix:", A)

    I_abs_max = 17
    m = 6000
    Q_matrix = A.T @ A
    # print("Q matrix:", Q_matrix)
    r_scalar = 0.057**2

    # r = 3
    # Power = 675 *2
    # Q_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * r
    # r_scalar = Power

    hull, H_representation, remaining_points, deleted_points = Get_Ellipsoid_CFW(Q_matrix, r_scalar, I_abs_max, m, plot_verbose=True)
    G = hull.equations[:, :-1]
    k = -hull.equations[:, -1]
    # print("H representation:", H_representation)
    # print("Minimal b:", min(k))
    # N, d = Transform_and_Extract_Facets(A, G, k)

    return G, k

def get_target_points_at_time(time_value):
    """Get target points with rotating magnetic field values at a specific time"""
    # Calculate rotating magnetic field components
    bx = field_amplitude * np.cos(angular_velocity * time_value)
    by = field_amplitude * np.sin(angular_velocity * time_value)
    
    # Single target point (as specified)
    target_points = [
        {'X': 0.0, 'Y': -0.03, 'Z': 0.00, 'Bx': bx, 'By': by},
    ]
    
    return target_points

# Pre-compute magnetic field matrices for all coils and target points (VECTORIZED)
def precompute_field_matrices(target_points):
    """Pre-compute magnetic field matrices for all coil-point combinations"""
    n_coils = len(params_list)
    n_points = len(target_points)
    
    # Extract target coordinates and fields
    target_coords = np.array([[p['X'], p['Y'], p['Z']] for p in target_points])
    target_fields = np.array([[p['Bx'], p['By']] for p in target_points])
    
    # Initialize field matrices
    Bx_matrix = np.zeros((n_points, n_coils))
    By_matrix = np.zeros((n_points, n_coils))
    
    # Vectorized computation for all coil-point combinations
    for i, params in enumerate(params_list):
        m0, m1, m2, r0_0, r0_1, r0_2 = params
        
        # Vectorized calculation for all points at once
        X_vec = target_coords[:, 0]
        Y_vec = target_coords[:, 1] 
        Z_vec = target_coords[:, 2]
        
        Bx_matrix[:, i] = dipole_model_Bx(m0, m1, m2, r0_0, r0_1, r0_2, X_vec, Y_vec, Z_vec)
        By_matrix[:, i] = dipole_model_By(m0, m1, m2, r0_0, r0_1, r0_2, X_vec, Y_vec, Z_vec)
    
    # Stack matrices vertically to create constraint matrix A
    A_matrix = np.vstack([Bx_matrix, By_matrix])
    target_vector = np.hstack([target_fields[:, 0], target_fields[:, 1]])
    
    return A_matrix, target_vector

# Main execution: Time-series optimization
def run_time_series_optimization():
    """Run optimization for each time step at specified control frequency"""
    
    # Calculate convex constraints once at the beginning
    print("Computing convex constraints...")
    N, d = get_convex_constraint()
    print(f"Convex constraint matrix N shape: {N.shape}")
    print(f"Convex constraint vector d shape: {d.shape}")
    
    # Calculate time points for optimization based on simulation duration
    time_points = np.arange(0, simulation_duration + control_period, control_period)
    
    print(f"Running optimization at {control_frequency} Hz for {simulation_duration:.1f} seconds")
    print(f"Rotating field: amplitude={field_amplitude*1000:.1f} mT, frequency={angular_velocity/(2*np.pi):.2f} Hz")
    print(f"Total optimization points: {len(time_points)}")
    
    # Storage for results
    results_data = {
        'time': [],
        'currents': [],
        'optimization_time': [],
        'success': [],
        'objective_value': [],
        'max_field_error': [],
        'rms_field_error': [],
        'max_current': [],
        'convex_constraint_violations': []
    }
    
    # Initial guess for currents (use previous solution as warm start)
    currents_guess = np.ones(3) * 0.1
    
    total_start_time = time.time()
    
    for i, t in enumerate(time_points):
        if i % 10 == 0:  # Progress update every 10 iterations
            print(f"Processing time {t:.2f}s ({i+1}/{len(time_points)})")
        
        # Get target points for this time (now uses rotating field)
        target_points = get_target_points_at_time(t)
        
        # Pre-compute matrices for this time step
        A_matrix, target_vector = precompute_field_matrices(target_points)
        
        # Define constraints for scipy.optimize.minimize
        constraints = [
            # Equality constraint: A @ currents = target_vector
            {'type': 'eq', 'fun': lambda x: A_matrix @ x - target_vector},
            # Inequality constraint: -current_limit <= currents <= current_limit
            {'type': 'ineq', 'fun': lambda x: current_limit - np.abs(x)},
            # Convex constraint: N @ currents <= d
            {'type': 'ineq', 'fun': lambda x: d - N @ x}
        ]
        
        # Optimization for this time step
        opt_start_time = time.time()
        
        result = minimize(
            lambda x: np.sum(x**2),  # Minimize sum of squares of currents
            currents_guess,
            method='SLSQP',
            constraints=constraints,
            options={'ftol': 1e-12, 'disp': False, 'maxiter': 1000}
        )
        
        opt_end_time = time.time()
        
        # Verify constraints
        predicted_fields = A_matrix @ result.x
        field_errors = predicted_fields - target_vector
        max_current = np.max(np.abs(result.x))
        
        # Check convex constraint violations
        convex_violations = np.maximum(0, N @ result.x - d)
        max_convex_violation = np.max(convex_violations)
        
        # Store results
        results_data['time'].append(t)
        results_data['currents'].append(result.x.copy())
        results_data['optimization_time'].append(opt_end_time - opt_start_time)
        results_data['success'].append(result.success)
        results_data['objective_value'].append(result.fun)
        results_data['max_field_error'].append(np.max(np.abs(field_errors)))
        results_data['rms_field_error'].append(np.sqrt(np.mean(field_errors**2)))
        results_data['max_current'].append(max_current)
        results_data['convex_constraint_violations'].append(max_convex_violation)
        
        # Use current solution as warm start for next iteration
        if result.success:
            currents_guess = result.x
    
    total_end_time = time.time()
    
    print(f"\n=== TIME SERIES OPTIMIZATION COMPLETED ===")
    print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    print(f"Average optimization time per step: {np.mean(results_data['optimization_time']):.4f} seconds")
    print(f"Success rate: {np.mean(results_data['success']):.1%}")
    print(f"Average max field error: {np.mean(results_data['max_field_error']):.2e}")
    print(f"Average RMS field error: {np.mean(results_data['rms_field_error']):.2e}")
    print(f"Max current encountered: {np.max(results_data['max_current']):.2f} A")
    print(f"Current limit violations: {np.sum(np.array(results_data['max_current']) > current_limit)}")
    print(f"Max convex constraint violation: {np.max(results_data['convex_constraint_violations']):.2e}")
    print(f"Convex constraint violations: {np.sum(np.array(results_data['convex_constraint_violations']) > 1e-6)}")
    
    return results_data

def unconstrained_objective(currents, A_matrix, target_vector, penalty_weight=1e15):
    """
    Unconstrained objective function using penalty method
    Combines original objective with penalty terms for constraints
    """
    # Original objective: minimize sum of squares
    original_obj = np.sum(currents**2)
    
    # Magnetic field constraint penalty (equality constraints)
    predicted_fields = A_matrix @ currents
    field_error = np.sum((predicted_fields - target_vector)**2)
    
    # Current limit penalty (inequality constraints)
    # Penalty for exceeding current limits
    current_violation = np.maximum(0, np.abs(currents) - current_limit)
    current_penalty = np.sum(current_violation**2)
    
    return original_obj + penalty_weight * (field_error + current_penalty)

def unconstrained_gradient(currents, A_matrix, target_vector, penalty_weight=1e15):
    """
    Analytical gradient for faster convergence
    """
    # Gradient of original objective
    grad_original = 2 * currents
    
    # Gradient of field constraint penalty
    predicted_fields = A_matrix @ currents
    field_residual = predicted_fields - target_vector
    grad_field = 2 * penalty_weight * (A_matrix.T @ field_residual)
    
    # Gradient of current limit penalty
    current_violation = np.maximum(0, np.abs(currents) - current_limit)
    grad_current = 2 * penalty_weight * current_violation * np.sign(currents)
    
    return grad_original + grad_field + grad_current

def plot_results(results_data):
    """Plot optimization results over time"""
    times = np.array(results_data['time'])
    currents_array = np.array(results_data['currents'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot currents over time
    axes[0, 0].plot(times, currents_array * 1000)  # Convert to mA
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Current (mA)')
    axes[0, 0].set_title('Coil Currents Over Time')
    axes[0, 0].grid(True)
    axes[0, 0].legend(['Coil 1', 'Coil 2', 'Coil 3'], loc='upper right')
    
    # Plot target magnetic field
    target_bx = field_amplitude * np.cos(angular_velocity * times) * 1000  # Convert to mT
    target_by = field_amplitude * np.sin(angular_velocity * times) * 1000  # Convert to mT
    axes[0, 1].plot(times, target_bx, 'r-', label='Target Bx')
    axes[0, 1].plot(times, target_by, 'b-', label='Target By')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Magnetic Field (mT)')
    axes[0, 1].set_title('Target Rotating Magnetic Field')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # Plot field errors and convex constraint violations
    axes[0, 2].plot(times, results_data['max_field_error'], 'r-', label='Max Field Error')
    axes[0, 2].plot(times, results_data['rms_field_error'], 'b-', label='RMS Field Error')
    axes2 = axes[0, 2].twinx()
    axes2.plot(times, results_data['convex_constraint_violations'], 'g-', label='Convex Constraint Violation')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Field Error (T)', color='black')
    axes2.set_ylabel('Convex Constraint Violation', color='green')
    axes[0, 2].set_title('Errors and Constraint Violations')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True)
    axes[0, 2].legend(loc='upper left')
    axes2.legend(loc='upper right')
    
    # Plot optimization times
    axes[1, 0].plot(times, results_data['optimization_time'], 'g-')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Optimization Time (s)')
    axes[1, 0].set_title('Optimization Time per Step')
    axes[1, 0].grid(True)
    
    # Plot max current vs limit
    axes[1, 1].plot(times, np.array(results_data['max_current']), 'b-', label='Max Current')
    axes[1, 1].axhline(y=current_limit, color='r', linestyle='--', label=f'Current Limit ({current_limit}A)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Current (A)')
    axes[1, 1].set_title('Maximum Current vs Limit')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    # Calculate actual magnetic field from currents
    actual_bx_list = []
    actual_by_list = []
    
    for i, t in enumerate(times):
        target_points = get_target_points_at_time(t)
        A_matrix, _ = precompute_field_matrices(target_points)
        predicted_fields = A_matrix @ currents_array[i]
        
        # Extract Bx and By (first half is Bx, second half is By for all points)
        n_points = len(target_points)
        actual_bx = predicted_fields[:n_points] * 1000  # Convert to mT
        actual_by = predicted_fields[n_points:] * 1000  # Convert to mT
        
        actual_bx_list.append(actual_bx[0])  # Take first (and only) point
        actual_by_list.append(actual_by[0])
    
    actual_bx_array = np.array(actual_bx_list)
    actual_by_array = np.array(actual_by_list)
    
    # Plot magnetic field trajectory (Bx vs By)
    axes[1, 2].plot(target_bx, target_by, 'k-', linewidth=2, label='Target Trajectory')
    axes[1, 2].plot(actual_bx_array, actual_by_array, 'r--', linewidth=2, label='Actual Trajectory')
    axes[1, 2].set_xlabel('Bx (mT)')
    axes[1, 2].set_ylabel('By (mT)')
    axes[1, 2].set_title('Magnetic Field Trajectory')
    axes[1, 2].grid(True)
    axes[1, 2].axis('equal')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print(f"=== ROTATING MAGNETIC FIELD CONTROL ===")
    print(f"Field amplitude: {field_amplitude*1000:.1f} mT")
    print(f"Rotation frequency: {angular_velocity/(2*np.pi):.2f} Hz")
    print(f"Control point: (0.0, -0.05, 0.0) m")
    print(f"Simulation duration: {simulation_duration:.1f} s")
    print("=" * 45)
    
    # Run the time-series optimization
    results = run_time_series_optimization()
    
    # Plot results
    plot_results(results)
    
    # Save only currents and time
    times = np.array(results['time'])
    currents_array = np.array(results['currents'])  # Shape: (n_times, 10)
    
    # Create DataFrame with time and current columns (only coils 2, 3, 7 have currents)
    currents_df = pd.DataFrame()
    currents_df['Time(s)'] = times
    
    # Set all coil currents to 0 first
    for i in range(10):
        currents_df[f'Current_Coil_{i+1}(A)'] = 0.0
    
    # Only assign actual currents to coils 2, 3, 7 (indices 1, 2, 6 in params_list)
    active_coils = [2, 3, 7]  # Coil numbers
    for i, coil_num in enumerate(active_coils):
        currents_df[f'Current_Coil_{coil_num}(A)'] = currents_array[:, i]
    
    # Save to CSV
    currents_df.to_csv(f'Paper_ICRA2026/coil_currents_{field_amplitude*1000:.0f}mT.csv', index=False)
    print(f"Currents saved to coil_currents_{field_amplitude*1000:.0f}mT.csv")
