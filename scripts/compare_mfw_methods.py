"""Visualize the effect of the new support-function MFW/MSD method.

Produces a single figure with two rows:

  * Top row -- for each of the three motors (motor 1 actuated), the field-space
    MFW: the Monte-Carlo ground-truth field cloud, the new method's MFW polygon
    and its two-sided MSD bracket, the old Algorithm 1-2 approximation, and the
    motor's minimum-actuation circle. Shows which motors actually enclose their
    circle (i.e. would rotate).
  * Bottom row -- MSD comparison across methods: Monte-Carlo truth vs the new
    deterministic bracket vs three stochastic runs of the old method, against
    each motor's actuation threshold.

Saved headless to ``figures/mfw_method_comparison.png``. Run from the repo root:

    python scripts/compare_mfw_methods.py
"""

import sys
import pathlib

import numpy as np
np.seterr(all="ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from selective_em import (map_i2b, extract_map_i2b, robot_arm_kinematics,
                          WORKSPACE_COILS, mfw_support, get_cfw_polytope)

ALPHA, BETA, I_MAX = 50, -30, 15
RADII = (0.02, 0.03, 0.02)          # minimum-actuation-circle radius per motor
COILS = WORKSPACE_COILS


def target(pos):
    x, y, z = pos
    return [{"X": x, "Y": y, "Z": z, "Bx": True, "By": True, "Bz": None,
             "Bx_dx": None, "Bx_dy": None, "Bx_dz": None,
             "By_dy": None, "By_dz": None}]


def A_of(pos):
    tp = target(pos)
    return extract_map_i2b(tp) @ map_i2b(tp, COILS)


def mc_cloud(A_target, A_act, r, n=2_000_000, seed=0):
    """Ground-truth MFW as a field cloud (rejection-sample the box ∩ slab)."""
    rng = np.random.default_rng(seed)
    I = rng.uniform(-I_MAX, I_MAX, size=(n, A_act.shape[1]))
    return I[np.linalg.norm(I @ A_act.T, axis=1) <= r] @ A_target.T


def old_polygon_and_msd(A_target, A_act, r_field, seed):
    """Old Algorithm 1+2 result (scipy-only reproduction): polygon + MSD."""
    np.random.seed(seed)
    Q = A_act.T @ A_act
    hull, _, _, _ = get_cfw_polytope(Q, r_field ** 2, I_MAX, 2000)
    verts = hull.points[hull.vertices]
    W = verts @ A_target.T
    h2 = ConvexHull(W)
    poly = W[h2.vertices]
    msd = float(np.min(-h2.equations[:, -1]))
    return poly, msd


def main():
    xs = robot_arm_kinematics(ALPHA, BETA)
    motors = [tuple(xs[0:3]), tuple(xs[3:6]), tuple(xs[6:9])]
    A_act = A_of(motors[0])            # motor 1 actuated, CFW radius r1
    r = RADII[0]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1.1], hspace=0.32, wspace=0.28)

    new_lo, new_up, mc_msd, old_runs = [], [], [], []
    old_seeds = (1, 2, 3)

    for j, (pos, rc) in enumerate(zip(motors, RADII)):
        A_t = A_of(pos)
        cloud = mc_cloud(A_t, A_act, r) * 1000.0            # -> mT
        res = mfw_support(A_t, A_act, r, I_MAX, n_dirs=180)
        new_poly = res["inner_points"] * 1000.0
        polys_old = [old_polygon_and_msd(A_t, A_act, r, s) for s in old_seeds]

        new_lo.append(res["msd_lower"] * 1000)
        new_up.append(res["msd_upper"] * 1000)
        old_runs.append([m * 1000 for _, m in polys_old])
        # MC MSD (min support over directions of the cloud)
        th = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        U = np.column_stack([np.cos(th), np.sin(th)])
        mc_msd.append(float((cloud @ U.T).max(axis=0).min()))

        ax = fig.add_subplot(gs[0, j])
        ax.scatter(cloud[::40, 0], cloud[::40, 1], s=2, c="0.8",
                   label="Ground truth (MC cloud)", rasterized=True)
        old_poly = polys_old[0][0] * 1000.0
        oc = np.vstack([old_poly, old_poly[0]])
        ax.plot(oc[:, 0], oc[:, 1], "r--", lw=1.8, label="Old Alg. 1-2 (1 run)")
        nc = np.vstack([new_poly, new_poly[0]])
        ax.fill(nc[:, 0], nc[:, 1], color="tab:blue", alpha=0.18)
        ax.plot(nc[:, 0], nc[:, 1], color="tab:blue", lw=2,
                label="New MFW (support fn.)")
        circ = plt.Circle((0, 0), rc * 1000, fill=False, ec="green", lw=2.2,
                          label=f"Actuation circle r={rc*1000:.0f} mT")
        ax.add_patch(circ)
        ax.set_aspect("equal")
        lim = max(np.abs(cloud).max(), rc * 1000) * 1.15
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        if j == 0:
            title, col = "Motor 1 (actuated: MFW = disk)", "black"
        else:
            encl = res["msd"] >= rc - 1e-9
            title = f"Motor {j+1} " + ("(ENCLOSES → would rotate)" if encl
                                       else "(excluded)")
            col = "firebrick" if encl else "black"
        ax.set_title(title, fontsize=11, color=col)
        ax.set_xlabel("$B_x$ (mT)"); ax.set_ylabel("$B_y$ (mT)")
        ax.grid(True, ls=":", lw=0.5)
        if j == 0:
            ax.legend(fontsize=8, loc="upper left")

    # --- bottom: MSD comparison ---
    axb = fig.add_subplot(gs[1, :])
    x = np.arange(3)
    for j in range(3):
        # old stochastic runs
        axb.scatter([x[j]] * 3, old_runs[j], c="red", marker="x", s=70, zorder=3,
                    label="Old Alg. 1-2 (3 seeds)" if j == 0 else None)
        # new deterministic bracket
        axb.errorbar(x[j], (new_lo[j] + new_up[j]) / 2,
                     yerr=[[(new_up[j]-new_lo[j])/2], [(new_up[j]-new_lo[j])/2]],
                     fmt="o", color="tab:blue", capsize=6, ms=8, zorder=4,
                     label="New MSD bracket" if j == 0 else None)
        # MC truth
        axb.scatter(x[j], mc_msd[j], c="black", marker="_", s=800, lw=2.5,
                    zorder=2, label="Ground truth (MC)" if j == 0 else None)
        # threshold
        axb.hlines(RADII[j] * 1000, x[j] - 0.35, x[j] + 0.35, colors="green",
                   ls="--", label="Actuation threshold" if j == 0 else None)
    axb.set_xticks(x); axb.set_xticklabels([f"Motor {i+1}" for i in range(3)])
    axb.set_ylabel("MSD (mT)")
    axb.set_title("MSD: new method hugs ground truth; old method is low and "
                  "run-to-run non-deterministic", fontsize=11)
    axb.grid(True, ls=":", lw=0.5, axis="y")
    axb.legend(fontsize=9, ncol=4, loc="upper center")

    fig.suptitle(f"Effect of the new support-function MFW/MSD method  "
                 f"(config $\\alpha$={ALPHA}$^\\circ$, $\\beta$={BETA}$^\\circ$, "
                 f"motor 1 actuated)", fontsize=13)
    out = ROOT / "figures" / "mfw_method_comparison.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"  MC truth (mT):      {[f'{v:.2f}' for v in mc_msd]}")
    print(f"  New bracket (mT):   "
          f"{[f'[{lo:.2f},{up:.2f}]' for lo, up in zip(new_lo, new_up)]}")
    print(f"  Old 3 seeds (mT):   "
          f"{[[f'{v:.2f}' for v in run] for run in old_runs]}")


if __name__ == "__main__":
    main()
