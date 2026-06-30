import numpy as np
import math
from Kinematics import robot_arm_kinematics
from lib import *
from CCMFW import Get_Ellipsoid_CFW

def if_in_OAW(position, r, i_min, i_max):
    point_x = position[0]
    point_y = position[1]
    point_z = position[2]

    target_points = [
            {'X': point_x, 'Y': point_y, 'Z': point_z, 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},
        ]
    G, K = Get_MMFW(target_points, i_min, i_max)
    if (K>= r).all():
        return True
    else:
        return False

# first point is the actuation point
def if_in_IR(point1,point2,point3,i_max,r1,r2,r3):
    target_points = [
            {'X': point1[0], 'Y': point1[1], 'Z': point1[2], 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},
        ]
    A = Extract_Map_I2B(target_points) @ Map_I2B(target_points)
    I_abs_max = i_max
    m = 1000 # Number of random points to sample
    Q_matrix = A.T @ A
    r_scalar = r1**2
    hull, H_representation, remaining_points, deleted_points = Get_Ellipsoid_CFW(Q_matrix, r_scalar, I_abs_max, m, plot_verbose=False)
    G = hull.equations[:, :-1]
    k = -hull.equations[:, -1]
    N, d = Transform_and_Extract_Facets(A, G, k)
    # print("Minimal d1:", min(d))
    # Plot_Hull_H_2D(N, d)
    #=========Point 2============
    target_point2 = [
            {'X': point2[0], 'Y': point2[1], 'Z': point2[2], 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},
        ]
    A2 = Extract_Map_I2B(target_point2) @ Map_I2B(target_point2)
    N2, d2 = Transform_and_Extract_Facets(A2, G, k)
    # print("Minimal d2:", min(d2))
    # Plot_Hull_H_2D(N2, d2)
    #=========Point 3============
    target_point3 = [
            {'X': point3[0], 'Y': point3[1], 'Z': point3[2], 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None},
        ]
    A3 = Extract_Map_I2B(target_point3) @ Map_I2B(target_point3)
    N3, d3 = Transform_and_Extract_Facets(A3, G, k)
    # print("Minimal d3:", min(d3))
    # Plot_Hull_H_2D(N3, d3)
    
    # Return True if both not in IR of point 1
    return (min(d2)< r2) and (min(d3)< r3)

if __name__ == "__main__":
    alpha = 0 # Example angle for the first joint
    beta = 0 # Example angle for the second joint
    r1 = 0.03 
    i_min = -15
    i_max = 15

    x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(alpha, beta)
    position_1 = np.array([x1, y1, z1])
    position_2 = np.array([x2, y2, z2])
    position_3 = np.array([x3, y3, z3])
    print("Position 1:", position_1)
    print("Position 2:", position_2)
    print("Position 3:", position_3)
    a = if_in_OAW(position_1,r1,i_min,i_max)
    print(a)

    b = if_in_IR(position_1,position_2,position_3,i_max,r1,r1,r1)
    print(b)



