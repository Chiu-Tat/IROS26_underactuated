import numpy as np
from lib import *
import matplotlib.pyplot as plt

xx = np.linspace(-0.08, 0.08, 50)
zz = np.linspace(-0.08, 0.08, 50)

# Create a meshgrid from x and y
X, Z = np.meshgrid(zz,xx)

I_min = -15
I_max = 15

# Define the amplitude of the magnetic field
B_amp = 0.03 # Units of Tesla

# Evaluate the function at each point in the meshgrid
P = np.zeros_like(X)
for i in range(len(xx)):
    for j in range(len(zz)):
        target_points = [
            {'X': X[i, j], 'Y': Z[i, j], 'Z': -0.04, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},
        ]
        G, K = Get_MMFW(target_points, I_min, I_max)
        d = K
        P[i, j] = 1 if (d >= B_amp).all() else -1


# Create a 2D plot
plt.scatter(X[P == 1], Z[P == 1], c='green', s=5, alpha=1, label='In workspace')
plt.scatter(X[P == -1], Z[P == -1], c='salmon', s=5, alpha=1, label='Out workspace')

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