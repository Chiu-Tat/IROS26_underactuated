"""Calibrated coil parameters and physical constants (single source of truth).

Each coil is modelled as a magnetic dipole described by 6 numbers::

    [m0, m1, m2, r0_0, r0_1, r0_2]
     \\_________/  \\____________/
     dipole moment    position (m)

``ALL_COILS`` holds the full 10-coil electromagnetic navigation system that was
calibrated for the paper. Individual experiments historically enabled only a
subset of the coils by commenting lines out; those subsets are named here so
each script can select the exact configuration it uses without duplicating the
parameter table.

    * ``FULL_ARRAY``       -- all 10 coils (calibration, control-sequence generation)
    * ``WORKSPACE_COILS``  -- coils 7-10, used for the workspace/feasibility analysis
    * ``EXPERIMENT_COILS`` -- coils 2, 3, 7, the 3-coil setup driving 3 motors

``DEFAULT_COILS`` is what the workspace/feasibility helpers fall back to when a
caller does not pass an explicit ``coils`` array; it matches the historical
default of the old ``lib.py`` (coils 7-10).
"""

import numpy as np

# Vacuum permeability (T*m/A).
MU0 = 4 * np.pi * 1e-7

# Full calibrated 10-coil array. Row i == coil (i + 1).
ALL_COILS = np.array([
    [-13.04945069, -4.41557229,  6.47376799,  0.12129096,  0.00466922, -0.0174842],   # coil 1
    [-5.10083416,  13.54294901,  7.85474539,  0.05834654, -0.11165548, -0.01850546],  # coil 2
    [ 4.05088788,  14.23365818,  6.44760956, -0.05903076, -0.11020417, -0.01488244],  # coil 3
    [13.89011305,  -0.06092074,  4.77365608, -0.12306086, -0.00085745, -0.01378161],  # coil 4
    [11.44363813,  -9.40543896,  4.46367162, -0.06806179,  0.1024875,  -0.01397152],  # coil 5
    [-9.00577939, -12.78905365,  5.98650851,  0.06473315,  0.10618968, -0.0151172],   # coil 6
    [ 0.92820081,   8.54965337,  8.72298349, -0.00381254, -0.08845466, -0.08874662],  # coil 7
    [ 8.7302819,   -4.90773115,  7.00109937, -0.07977306,  0.04481733, -0.08536032],  # coil 8
    [-7.68962762,  -6.83258326,  8.12112247,  0.07498008,  0.04542436, -0.08696975],  # coil 9
    [ 2.35614001,  -1.11370036, 14.00304846, -0.00722183,  0.00029277, -0.12482979],  # coil 10
])

# Named configurations (1-based coil numbers documented; 0-based indices used).
FULL_ARRAY = ALL_COILS
WORKSPACE_COILS = ALL_COILS[[6, 7, 8, 9]]      # coils 7, 8, 9, 10
EXPERIMENT_COILS = ALL_COILS[[1, 2, 6]]        # coils 2, 3, 7

# Fallback for helpers that accept an optional ``coils`` argument.
DEFAULT_COILS = WORKSPACE_COILS
