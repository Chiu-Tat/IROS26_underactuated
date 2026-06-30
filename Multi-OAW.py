import numpy as np
from lib import Get_MMFW
import matplotlib.pyplot as plt
from Kinematics import robot_arm_kinematics

xx = np.linspace(-90, 90, 20)
yy = np.linspace(-90, 90, 20)

# Create a meshgrid from x and y
X, Y = np.meshgrid(xx, yy)

I_min = -15
I_max = 15

# Define the amplitude of the magnetic field
r1 = 0.015  # Units of Tesla
r2 = 0.015
r3 = 0.015

# Evaluate the function at each point in the meshgrid
P = np.zeros_like(X)
for i in range(len(xx)):
    for j in range(len(yy)):
        x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(X[i, j], Y[i, j])
        target_points = [
            {'X': x1, 'Y': y1, 'Z': z1, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},
            {'X': x2, 'Y': y2, 'Z': z2, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},
            {'X': x3, 'Y': y3, 'Z': z3, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},
        ]
        N, d = Get_MMFW(target_points, I_min, I_max)
        r_values = []
        # r_result = np.zeros(len(xx))
        for k in range(N.shape[0]):  # Changed j to k to avoid variable conflict
            denom = r1* np.sqrt(N[k, 0]**2 + N[k, 1]**2) + r2 * np.sqrt(N[k, 2]**2 + N[k, 3]**2) + r3 * np.sqrt(N[k, 4]**2 + N[k, 5]**2)
            r_val = d[k] - denom
            r_values.append(r_val)
        rr = min(r_values) if r_values else np.nan

        P[i, j] = 1 if rr >= 0 else -1


# Create a 2D plot
plt.scatter(X[P == 1], Y[P == 1], c='green', s=5, alpha=1, label='In workspace')
plt.scatter(X[P == -1], Y[P == -1], c='salmon', s=5, alpha=1, label='Out workspace')

# Set the aspect ratio of the plot to be equal
plt.gca().set_aspect('equal')

# Add labels and title
plt.xlabel('x')
plt.ylabel('Z')

# Set the x and y axis tick labels
# plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'])
# plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'])

# Add a legend
plt.legend()

# Show the plot
plt.grid()
plt.gca().set_facecolor('lightgray')
plt.show()

# 123