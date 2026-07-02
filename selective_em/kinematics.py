"""Forward kinematics of the 3-DOF magnetically-actuated arm.

Returns the 3D positions of the three magnet motors (joint 1, joint 2, and the
gripper/end-effector) for given joint angles.
"""

import math


def robot_arm_kinematics(alpha, beta):
    """Positions of the three magnets for joint angles ``alpha``, ``beta`` (degrees).

    Returns ``(x1, y1, z1, x2, y2, z2, x3, y3, z3)`` in metres, where subscript 1
    is the base magnet, 2 the second joint, and 3 the end-effector/gripper.
    """
    x1 = 0
    y1 = 0.0345
    z1 = 0.04669 - 0.085
    z2 = z1 - 0.01947
    z3 = z1

    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)

    # Link lengths.
    L1 = 0.02
    L2 = 0.02

    # Initial pose points along -y; alpha rotates from that pose.
    theta1 = alpha_rad - math.pi / 2
    x2 = x1 + L1 * math.cos(theta1)
    y2 = y1 + L1 * math.sin(theta1)

    # beta is the relative rotation of link 2 w.r.t. link 1.
    theta2 = theta1 + beta_rad
    x3 = x2 + L2 * math.cos(theta2)
    y3 = y2 + L2 * math.sin(theta2)

    return x1, y1, z1, x2, y2, z2, x3, y3, z3


if __name__ == "__main__":
    for a, b in [(0, 0)]:
        x1, y1, z1, x2, y2, z2, x3, y3, z3 = robot_arm_kinematics(a, b)
        print(f"Joint 1: ({x1:.4f}, {y1:.4f}, {z1:.4f})")
        print(f"Joint 2: ({x2:.4f}, {y2:.4f}, {z2:.4f})")
        print(f"End Effector: ({x3:.4f}, {y3:.4f}, {z3:.4f})")
