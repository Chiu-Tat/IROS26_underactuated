# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Code for the IROS 2026 paper *"Selective Electromagnetic Control of Multiple Millimeter-sized Magnetic Motors with Limited Number of Coils"* (Da Zhao et al., CUHK). It shows how to **selectively drive one of several identical mm-scale magnetic motors at a time using fewer coils than full actuation needs** (3 motors, 3 coils). The demonstrator is a 3-DOF arm — two revolute joints plus a gripper, each joint a permanent magnet + 1:50 gearbox with a z-parallel rotation axis — actuated by rotating magnetic fields.

The scientific core is **workspace analysis via polytope/zonotope geometry**: given box-bounded (and optionally ellipsoid-bounded) coil currents, what fields are reachable at each motor, and which configurations let one motor rotate without driving the others. See `PAPER_SUMMARY.md` in the sibling paper repo (`../Da_ZHAO_s_IROS2026/`) for the full contribution.

## Layout

The code is a package plus runnable scripts (it was reorganized from a flat pile of scripts):

- `selective_em/` — importable core library (the algorithms).
- `scripts/` — runnable experiments that produce the paper's artifacts/figures.
- `data/`, `figures/` — generated outputs (CSV, PNG).
- `archive/` — pre-refactor exploration scripts, **unmaintained**, import the removed flat `lib` module; don't build on them (see `archive/README.md`).

## Running

```bash
pip install -r requirements.txt          # needs cddlib+gmp for pypoman (brew install cddlib gmp)
python scripts/<name>.py                  # run from the repo root
```

There is no test suite. Scripts open a **blocking** matplotlib window (`plt.show()`) and/or write to `data/`/`figures/`; they are interactive/visual, not batch. The feasibility/planning scripts use `multiprocessing` and are CPU-heavy. See `README.md` for the full script list and what each produces.

Dependencies: `numpy`, `scipy`, `sympy`, `matplotlib`, `pandas`, `pypoman`, `scikit-learn`.

## Core library (`selective_em/`)

Bottom-up dependency order — `coils` → `field_model` → `workspace` → `feasibility`/`control`; `kinematics` and `planner` are independent; `visualization` is imported lazily.

- **`coils.py`** — single source of truth for the calibrated 10-coil dipole array (`ALL_COILS`, rows of `[m0,m1,m2, r0_0,r0_1,r0_2]` = dipole moment + position) plus named subsets and `MU0`.
- **`field_model.py`** — symbolic dipole B-field + spatial derivatives (built once via sympy `lambdify` at import). `map_i2b(target_points, coils)` is the actuation matrix `A` (8 field/gradient rows per target × one column per coil); `extract_map_i2b` selects the constrained rows. The universal `target_points` dict is `{'X','Y','Z', 'Bx','By','Bz','Bx_dx',...}` where field keys are `True` (constrain) or `None` (ignore).
- **`workspace.py`** — `get_mfw` (Magnetic-Feasible Workspace), `hyperplane_shifting_method` (zonotope H-rep for box currents), `minimal_supporting_distance` (MSD), `get_cfw_polytope` (Algorithm 1, ellipsoid∩box CFW via surface sampling + projection + hull), `transform_and_extract_facets` (Algorithm 2).
- **`kinematics.py`** — `robot_arm_kinematics(alpha, beta)` → 3 magnet positions (angles in **degrees**).
- **`feasibility.py`** — `in_oar`, `is_selectively_actuable` (the selectivity theorem: in own OAR **and** no other motor in its influence region), `point_in_polytope`.
- **`control.py`** — per-step current solving for a rotating field: `precompute_field_matrices`, penalty-method `unconstrained_objective`/`unconstrained_gradient`, `plot_optimization_results`.
- **`planner.py`** — dual-layer axial A* (Algorithm 3) with direction-dependent feasibility maps + `extract_keypoints`.

## Key conventions & gotchas

- **The physics model lives in exactly one place** (`field_model.py`); every function takes an explicit `coils` array — no module-global coil list. This deliberately replaced the old duplication (the model was copy-pasted into 4 files, each enabling a different coil subset by commenting lines).
- **Coil subsets are named, not commented.** `WORKSPACE_COILS` (7–10, the default), `EXPERIMENT_COILS` (2,3,7 — the 3-coil demo), `FULL_ARRAY` (all 10). Pass the intended subset explicitly; the same target yields different results per subset.
- **Naming matches the paper**: MFW / MSD / CFW / OAR / IR ↔ `get_mfw` / `minimal_supporting_distance` / `get_cfw_polytope` / `in_oar` / influence-region checks. Algorithms 1/2/3 map to `get_cfw_polytope` / `transform_and_extract_facets` / `a_star`. There's a code↔paper table in `README.md`.
- **Units:** positions/lengths in metres, fields in Tesla, currents in amperes, kinematics angles in **degrees**. Current limits and field radii are tuned per experiment (e.g. 15 A vs 17 A; radii 0.02/0.03/0.02 T) and set in each script, not globally.
- **`get_cfw_polytope` is stochastic when the current-space dimension n > 3** (random sphere sampling); it is deterministic for n = 2 (circle) and n = 3 (Fibonacci sphere). So feasibility maps / paths that go through the CFW for >3 coils vary run-to-run. The deterministic core (`map_i2b`, `get_mfw`, kinematics, control sequence) is exactly reproducible.
- **Behavior parity:** the refactor was verified behavior-preserving — `map_i2b`/`get_mfw`/`get_cfw_polytope` are bit-identical to the pre-refactor code, and a regenerated `data/field_targets.csv` matches the committed baseline to ~1e-15.
