# archive/

Pre-refactor exploration scripts kept for reference only. They are **not
maintained** and predate the `selective_em` package — they import the old flat
`lib` module (removed) and may not run as-is. See the git history for the full
original layout.

- `CFW.py` — early power-constrained CFW (`Get_CFW`) using DBSCAN/KMeans point
  clustering; superseded by the ellipsoid-CFW `get_cfw_polytope`
  (`selective_em.workspace`).
- `Test.py` — scratch 2D test harness for the ellipsoid-CFW routine.
- `Get_feasible_pose.py` — incomplete 2-target static pose-search exploration.
  Its one reusable helper (point-in-polytope membership) now lives in
  `selective_em.feasibility.point_in_polytope`.
