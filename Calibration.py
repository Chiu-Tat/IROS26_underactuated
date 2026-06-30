import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sympy as sp
import time
import matplotlib.pyplot as plt

# current restrictions
current_limit = 15

# Rotating magnetic field parameters
field_amplitude = 0.057 # Tesla (1 mT)
angular_velocity = 2 * np.pi /10  # rad/s (1 Hz rotation)
simulation_duration = 30.0  # seconds

# Configuration parameters
control_frequency = 20  # Hz (20 Hz = 50ms intervals)
control_period = 1.0 / control_frequency  # seconds

# List of fitted parameters for each coil
params_list = [
    np.array([-13.04945069, -4.41557229, 6.47376799, 0.12129096, 0.00466922, -0.0174842]),  # coil 1
    np.array([-5.10083416, 13.54294901, 7.85474539, 0.05834654, -0.11165548, -0.01850546]),  # coil 2
    np.array([4.05088788, 14.23365818, 6.44760956, -0.05903076, -0.11020417, -0.01488244]),  # coil 3
    np.array([13.89011305, -0.06092074, 4.77365608, -0.12306086, -0.00085745, -0.01378161]), # coil 4
    np.array([11.44363813, -9.40543896, 4.46367162, -0.06806179, 0.1024875, -0.01397152]),  # coil 5
    np.array([-9.00577939, -12.78905365, 5.98650851, 0.06473315, 0.10618968, -0.0151172]),  # coil 6
    np.array([0.92820081, 8.54965337, 8.72298349, -0.00381254, -0.08845466, -0.08874662]),   # coil 7
    np.array([8.7302819, -4.90773115, 7.00109937, -0.07977306, 0.04481733, -0.08536032]),   # coil 8
    np.array([-7.68962762, -6.83258326, 8.12112247, 0.07498008, 0.04542436, -0.08696975]),   # coil 9
    np.array([2.35614001, -1.11370036, 14.00304846, -0.00722183, 0.00029277, -0.12482979])   # coil 10
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
        'max_current': []
    }
    
    # Initial guess for currents (use previous solution as warm start)
    currents_guess = np.ones(10) * 0.1
    
    total_start_time = time.time()
    
    for i, t in enumerate(time_points):
        if i % 10 == 0:  # Progress update every 10 iterations
            print(f"Processing time {t:.2f}s ({i+1}/{len(time_points)})")
        
        # Get target points for this time (now uses rotating field)
        target_points = get_target_points_at_time(t)
        
        # Pre-compute matrices for this time step
        A_matrix, target_vector = precompute_field_matrices(target_points)
        
        # Optimization for this time step
        opt_start_time = time.time()
        
        result = minimize(
            lambda x: unconstrained_objective(x, A_matrix, target_vector),
            currents_guess,
            method='L-BFGS-B',
            jac=lambda x: unconstrained_gradient(x, A_matrix, target_vector),
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000}
        )
        
        opt_end_time = time.time()
        
        # Verify constraints
        predicted_fields = A_matrix @ result.x
        field_errors = predicted_fields - target_vector
        max_current = np.max(np.abs(result.x))
        
        # Store results
        results_data['time'].append(t)
        results_data['currents'].append(result.x.copy())
        results_data['optimization_time'].append(opt_end_time - opt_start_time)
        results_data['success'].append(result.success)
        results_data['objective_value'].append(result.fun)
        results_data['max_field_error'].append(np.max(np.abs(field_errors)))
        results_data['rms_field_error'].append(np.sqrt(np.mean(field_errors**2)))
        results_data['max_current'].append(max_current)
        
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
    axes[0, 0].legend([f'Coil {i+1}' for i in range(10)], bbox_to_anchor=(1.05, 1), loc='upper left')
    
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
    
    # Plot field errors
    axes[0, 2].plot(times, results_data['max_field_error'], 'r-', label='Max Error')
    axes[0, 2].plot(times, results_data['rms_field_error'], 'b-', label='RMS Error')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Field Error (T)')
    axes[0, 2].set_title('Magnetic Field Errors')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True)
    axes[0, 2].legend()
    
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
    
    # Plot magnetic field trajectory (Bx vs By)
    axes[1, 2].plot(target_bx, target_by, 'k-', linewidth=2, label='Target Trajectory')
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
    
    # Create DataFrame with time and 10 current columns
    currents_df = pd.DataFrame()
    currents_df['Time(s)'] = times
    for i in range(10):
        currents_df[f'Current_Coil_{i+1}(A)'] = currents_array[:, i]
    
    # Save to CSV
    currents_df.to_csv(f'Paper_ICRA2026/coil_currents_{field_amplitude*1000:.0f}mT.csv', index=False)
    print(f"Currents saved to coil_currents_rotating_field.csv")
