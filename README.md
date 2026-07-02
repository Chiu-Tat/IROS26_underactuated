# Selective Electromagnetic Control of Multiple Magnetic Motors

Code for the IROS 2026 paper **"Selective Electromagnetic Control of Multiple
Millimeter-sized Magnetic Motors with Limited Number of Coils"** (Da Zhao,
Jiewen Tan, Hongzhe Sun, Shing Shin Cheng, CUHK).

The paper shows how to **selectively drive one of several identical mm-scale
magnetic motors at a time using far fewer coils than full actuation requires**
(3 motors with 3 coils, versus ≥6). Each motor's reachable magnetic field is
shaped so that only the intended motor receives enough torque to rotate, and a
motor-by-motor actuation sequence is planned to reach the target pose.

## Installation

```bash
pip install -r requirements.txt
```

`pypoman` depends on `pycddlib`, which needs the **cddlib** C library and GMP.
On macOS: `brew install cddlib gmp` (set `CFLAGS`/`LDFLAGS` to the brew prefix if
the build can't find the headers); on Debian/Ubuntu: `apt install libcdd-dev libgmp-dev`.

## Layout

```
selective_em/   importable core library (the algorithms)
scripts/        runnable experiments that produce the paper's artifacts/figures
data/           generated CSV outputs (field targets, planned path, coil currents)
figures/        generated PNG figures
archive/        pre-refactor exploration scripts (unmaintained; see archive/README.md)
```

### Core library (`selective_em/`)

| Module | Contents |
|---|---|
| `coils` | Calibrated 10-coil parameters (single source of truth) + named subsets |
| `field_model` | Dipole field model; actuation matrix `A = map_i2b` |
| `workspace` | MFW/CFW geometry, MSD, hyperplane shifting (Algorithms 1 & 2) |
| `kinematics` | `robot_arm_kinematics` — the 3-DOF arm forward kinematics |
| `feasibility` | OAR / Influence Region / selectivity theorem |
| `control` | Per-step current solving for a rotating field |
| `planner` | Dual-layer axial A* (Algorithm 3) |
| `visualization` | Polytope / CFW plotting |

## Code ↔ paper terminology

| Paper term | Code |
|---|---|
| Actuation matrix `A(p)` | `field_model.map_i2b` (+ `extract_map_i2b`) |
| Magnetic-Feasible Workspace (MFW) | `workspace.get_mfw` |
| Minimal Supporting Distance (MSD) | `workspace.minimal_supporting_distance` |
| Current-Feasible Workspace (CFW), **Algorithm 1** | `workspace.get_cfw_polytope` |
| MFW from a general CFW, **Algorithm 2** | `workspace.transform_and_extract_facets` |
| Omni-Actuation Region (OAR) | `feasibility.in_oar` |
| Influence Region (IR) + selectivity theorem | `feasibility.is_selectively_actuable` |
| Dual-layer axial A*, **Algorithm 3** | `planner.a_star` |
| Per-step selective controller | `scripts/run_field_optimization.py` |

## Coil configurations

`selective_em.coils` holds all 10 calibrated coils and the named subsets each
experiment uses (this replaces the old habit of commenting lines in/out):

- `WORKSPACE_COILS` (coils 7–10) — workspace/feasibility analysis (the default).
- `EXPERIMENT_COILS` (coils 2, 3, 7) — the 3-coil / 3-motor demonstration.
- `FULL_ARRAY` (all 10) — calibration and control-sequence generation.

Helpers accept an explicit `coils=` argument, so one import serves any subset.

## Running the experiments

Each script opens a matplotlib figure (blocking) and/or writes to `data/` or
`figures/`. Run from the repo root:

```bash
python scripts/plot_oar_map.py --mode single    # single-target OAR map
python scripts/plot_oar_map.py --mode multi      # 3-motor OAR over joint angles
python scripts/plot_cfw_mfw.py                   # selectivity figure (paper Fig. 10)
python scripts/plot_influence_region.py          # spatial influence/actuation map
python scripts/compute_feasibility_maps.py --motor 1   # per-motor feasibility map (1|2|3)
python scripts/plan_path.py                      # feasibility maps + A* -> data/path_result.csv
python scripts/plan_path.py --demo               # fast synthetic-obstacle A* demo
python scripts/visualize_planner_steps.py        # -> figures/planner_process.png
python scripts/generate_control_sequence.py      # -> data/field_targets.csv + 3D preview
python scripts/calibrate_coils.py                # unconstrained current tracking (10 coils)
python scripts/run_field_optimization.py         # selective SLSQP controller (3 coils)
```

The feasibility/planning scripts are CPU-heavy (parallelized with
`multiprocessing`).
