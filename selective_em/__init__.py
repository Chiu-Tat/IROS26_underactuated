"""Selective electromagnetic control of multiple magnetic motors.

Core library for the IROS 2026 paper *"Selective Electromagnetic Control of
Multiple Millimeter-sized Magnetic Motors with Limited Number of Coils"*.

Module map (see each module's docstring for details):

    coils          -- calibrated coil parameters + named subsets (single source of truth)
    field_model    -- dipole field model, actuation matrix A = map_i2b
    workspace      -- MFW / CFW geometry (Algorithms 1 & 2), MSD, hyperplane shifting
    kinematics     -- robot forward kinematics
    feasibility    -- OAR / Influence Region / selectivity theorem
    control        -- per-step current solving for a rotating field
    planner        -- dual-layer axial A* (Algorithm 3)
    visualization  -- polytope / CFW plotting
"""

from . import coils
from .coils import (
    ALL_COILS, FULL_ARRAY, WORKSPACE_COILS, EXPERIMENT_COILS, DEFAULT_COILS, MU0,
)
from .field_model import (
    calculate_b_and_derivatives, map_i2b, extract_map_i2b,
    map_i2b_2d, extract_map_i2b_2d,
)
from .workspace import (
    hyperplane_shifting_method, get_mfw, get_cfw_polytope,
    transform_and_extract_facets, minimal_supporting_distance,
    remove_duplicate_rows,
)
from .kinematics import robot_arm_kinematics
from .feasibility import (
    in_oar, is_selectively_actuable, point_in_polytope,
)
from .planner import a_star, manhattan, extract_keypoints

__all__ = [
    "coils", "ALL_COILS", "FULL_ARRAY", "WORKSPACE_COILS", "EXPERIMENT_COILS",
    "DEFAULT_COILS", "MU0",
    "calculate_b_and_derivatives", "map_i2b", "extract_map_i2b",
    "map_i2b_2d", "extract_map_i2b_2d",
    "hyperplane_shifting_method", "get_mfw", "get_cfw_polytope",
    "transform_and_extract_facets", "minimal_supporting_distance",
    "remove_duplicate_rows",
    "robot_arm_kinematics",
    "in_oar", "is_selectively_actuable", "point_in_polytope",
    "a_star", "manhattan", "extract_keypoints",
]
