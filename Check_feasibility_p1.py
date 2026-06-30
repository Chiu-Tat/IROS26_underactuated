import numpy as np
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from Kinematics import robot_arm_kinematics
from Check_feasibility_one_point import if_in_OAW, if_in_IR

def check_single_point(args):
    """Worker function for parallel feasibility check."""
    alpha, beta, I_min, I_max, r1, r2, r3 = args
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(alpha, beta)
    # Mask: only consider points where tip (x3, y3) is inside the sphere x^2+y^2<=0.045^2
    if x3**2 + y3**2 > 0.045**2:
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

if __name__ == '__main__':
    xx = np.linspace(-90, 90, 20)
    yy = np.linspace(-90, 90, 20)

    # Create a meshgrid from x and y
    X, Y = np.meshgrid(xx, yy)

    I_min = -15
    I_max = 15

    # Define the amplitude of the magnetic field
    r1 = 0.02  # Units of Tesla
    r2 = 0.03
    r3 = 0.02

    ratio = 0.92

    # Build argument list for all grid points
    args_list = [
        (X[i, j], Y[i, j], I_min, I_max, r1, r2 * ratio, r3 * ratio)
        for i in range(len(xx))
        for j in range(len(yy))
    ]

    # Parallel evaluation using all available CPU cores
    # num_workers = cpu_count()
    num_workers = 10  # You can adjust this based on your system
    print(f"Running parallel feasibility check with {num_workers} workers on {len(args_list)} points...")
    with Pool(processes=num_workers) as pool:
        results = pool.map(check_single_point, args_list)

    # Reshape results back to meshgrid shape
    P = np.array(results).reshape(X.shape)

    plt.figure(figsize=(7, 6))
    # Create a 2D plot
    # plt.scatter(X[P == 1], Y[P == 1], c='green', s=5, alpha=1, label='Feasible')
    # plt.scatter(X[P == -1], Y[P == -1], c='salmon', s=5, alpha=1, label='Infeasible')
    plt.contourf(X, Y, P, levels=[0, np.inf], colors=['lightgreen'], alpha=0.9)
    plt.contourf(X, Y, P, levels=[-10, -9], colors=['lightcoral'], alpha=0.9)
    plt.contourf(X, Y, P, levels=[-1, 0], colors=['lightgray'], alpha=0.9)


    # Set the aspect ratio of the plot to be equal
    plt.gca().set_aspect('equal')

    # Add labels and title
    plt.xlabel(r'$\alpha$ (degrees)', fontsize=18)
    plt.ylabel(r'$\beta$ (degrees)', fontsize=18)

    ticksx = [-90, -45, 0, 45, 90]
    labelsx = ['-90', '-45', '0', '45', '90']
    plt.xticks(ticksx, labelsx, fontsize=18)
    ticksy = [-90, -45, 0, 45, 90]
    labelsy = ['-90', '-45', '0', '45', '90']
    plt.yticks(ticksy, labelsy, fontsize=18)

    # Set the x and y axis tick labels
    # plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'])
    # plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'])

    # Add a legend
    # plt.legend()    

    # Show the plot
    plt.grid()
    plt.gca().set_facecolor('lightgray')
    plt.show()