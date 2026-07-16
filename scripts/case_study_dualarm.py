"""Case study II (simulation): gearbox-aware selective control of a 6-motor
dual-arm platform.

Two 3-motor micromanipulator arms (6 magnetic motors) laid out in 3D, all
rotating about z (planar Bx/By field control), driven by 8 of the 10 calibrated
coils (vs 12 needed for full 2-D actuation of 6 motors).

Selectivity is assessed *dynamically*: to actuate motor m we track a rotating
field of magnitude ``DRIVE_K * b_min`` at m with a spillover-minimizing current
    i(theta) = argmin  sum_{j!=m} ||A_j i||^2 + lambda ||i||^2   s.t.  A_m i = b_m,
which uses the coils' spare DOF to suppress the field at the other motors. We
then integrate every magnet's rotational dynamics under its induced field and
count NET winding of the input magnet; the 1:50 gearbox transmits only sustained
winding, so a magnet that merely oscillates leaves its output joint (nearly)
still. This is far less conservative than the worst-case MFW influence-region
test and reflects the mechanical low-pass filtering of the gearbox.

Outputs (headless): figures/dualarm_{layout,winding,trajectories,fields,
scalability}.png and data/dualarm_case_study.csv.

    python scripts/case_study_dualarm.py
"""
import sys
import math
import pathlib
import csv

import numpy as np
np.seterr(all="ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.integrate import solve_ivp

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from selective_em.field_model import map_i2b, extract_map_i2b
from selective_em.coils import ALL_COILS

# --- scenario + model parameters (verified operating point) ---
COILS_IDX = [1, 2, 4, 5, 6, 7, 8, 9]      # coils 2,3,5,6,7,8,9,10 (8 of 10)
RADII = (0.02, 0.03, 0.02, 0.02, 0.03, 0.02)   # b_min per motor (T)
SEP = 0.035               # arm base offset in +/- y
XB = -0.015               # arm base x-offset
YAW = 65                  # base yaw: arms extend in +x (parallel, non-overlapping)
CFG_A, CFG_B = (20, -30), (20, -30)
L1 = L2 = 0.02; DZ = 0.01947; BASE_Z = 0.04669 - 0.085
OMEGA = 2 * math.pi        # field rotation rate (rad/s)
DRIVE_K = 1.4              # actuated field = DRIVE_K * b_min
LAMBDA = 3e-6             # current-norm regularization
JR, CR = 2e-4, 6e-4       # per-|m| inertia, viscous damping
EPS = 0.05 * OMEGA        # Coulomb-friction smoothing
GEAR = 50                 # gearbox reduction
N_CYCLES = 6
ARM_COLORS = ("#1f77b4", "#d62728")
# Distinct, colorblind-friendly joint colors.  Arm 1 stays in a cool family
# and arm 2 in a warm family, while every motor remains easy to identify.
MOTOR_COLORS = ["#0072B2", "#009E73", "#7E57C2",
                "#D55E00", "#E69F00", "#CC79A7"]
PAPER_DPI = 300

# Coil profile for the layout drawing only (magnetics stay in
# selective_em.coils). Real core/winding are Ø45/Ø105 mm, 180/120 mm long; the
# layout draws a shortened, slightly slimmed solenoid (exposed core tip +
# winding barrel + open bore) so all eight coils render as complete, distinct
# bodies that fit fully inside the frame instead of being cropped at its edges.
CORE_R, WIND_R = 0.019, 0.036
CORE_TIP, WIND_LEN = 0.010, 0.040   # exposed core tip + winding barrel length
COIL_CU, COIL_CU_EDGE = "#b86b45", "#5b2c1c"
CORE_COL, CORE_EDGE = "#d8d5ca", "#4d4d4d"
# coil number -> (unit axis toward the workspace, core-top position (m));
# nominal mounting poses of the 10-coil array (body extends along -axis)
COIL_POSES = {
    1:  ((-1.0, 0.0, 0.0),                (0.085, 0.0, 0.0)),
    2:  ((-0.5, 0.866025, 0.0),           (0.0425, -0.073612, 0.0)),
    3:  ((0.5, 0.866025, 0.0),            (-0.0425, -0.073612, 0.0)),
    4:  ((1.0, 0.0, 0.0),                 (-0.085, 0.0, 0.0)),
    5:  ((0.5, -0.866025, 0.0),           (-0.0425, 0.073612, 0.0)),
    6:  ((-0.5, -0.866025, 0.0),          (0.0425, 0.073612, 0.0)),
    7:  ((0.0, 0.766045, 0.642787),       (0.0, -0.0651138, -0.0546369)),
    8:  ((0.663414, -0.383022, 0.642787), (-0.0563902, 0.0325569, -0.0546369)),
    9:  ((-0.663414, -0.383022, 0.642787), (0.0563902, 0.0325569, -0.0546369)),
    10: ((0.0, 0.0, 1.0),                 (0.0, 0.0, -0.085)),
}


PAPER_STYLE = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.5,
}


# ------------------------------------------------------------------ geometry --
def arm(base, yaw, alpha, beta):
    bx, by, bz = base
    a = math.radians(alpha); b = math.radians(beta); y = math.radians(yaw)
    t1 = a - math.pi / 2 + y
    m2 = np.array([bx + L1 * math.cos(t1), by + L1 * math.sin(t1), bz - DZ])
    t2 = t1 + b
    m3 = np.array([m2[0] + L2 * math.cos(t2), m2[1] + L2 * math.sin(t2), bz])
    return [np.array([bx, by, bz]), m2, m3]


def dual_arm(cfgA=CFG_A, cfgB=CFG_B):
    return arm((XB, SEP, BASE_Z), YAW, *cfgA) + arm((XB, -SEP, BASE_Z), YAW, *cfgB)


def A_planar(pos, coils):
    tp = [{"X": pos[0], "Y": pos[1], "Z": pos[2], "Bx": True, "By": True,
           "Bz": None, "Bx_dx": None, "Bx_dy": None, "Bx_dz": None,
           "By_dy": None, "By_dz": None}]
    return extract_map_i2b(tp) @ map_i2b(tp, coils)


# ------------------------------------------------------------------ control ---
def spillover_S(A, m, lam=LAMBDA):
    """i = S @ b_m, minimizing spillover to other motors under the tracking eq."""
    n = A[0].shape[1]
    M = lam * np.eye(n) + sum(A[j].T @ A[j] for j in range(len(A)) if j != m)
    Minv = np.linalg.inv(M); Am = A[m]
    return Minv @ Am.T @ np.linalg.inv(Am @ Minv @ Am.T)


def current(S, r_drive, theta):
    return S @ (r_drive * np.array([math.cos(theta), math.sin(theta)]))


# ------------------------------------------------------------------ dynamics --
def magnet_rhs(phi, w, B, b_min):
    mag = math.hypot(B[0], B[1]); ang = math.atan2(B[1], B[0])
    return w, (mag * math.sin(ang - phi) - CR * w - b_min * math.tanh(w / EPS)) / JR


def net_winding(A_j, S, r_drive, b_min_j):
    def rhs(t, s):
        return magnet_rhs(s[0], s[1], A_j @ current(S, r_drive, OMEGA * t), b_min_j)
    T = N_CYCLES * 2 * math.pi / OMEGA
    sol = solve_ivp(rhs, [0, T], [0, 0], max_step=T / (N_CYCLES * 200),
                    rtol=1e-6, atol=1e-8)
    i0 = np.searchsorted(sol.t, T / N_CYCLES)
    return (sol.y[0][-1] - sol.y[0][i0]) / (2 * math.pi) / (N_CYCLES - 1)


def winding_matrix(A):
    n = len(A)
    W = np.zeros((n, n)); curr = np.zeros(n)
    for m in range(n):
        S = spillover_S(A, m)
        th = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        curr[m] = max(np.max(np.abs(current(S, DRIVE_K * RADII[m], t))) for t in th)
        for j in range(n):
            W[m, j] = net_winding(A[j], S, DRIVE_K * RADII[m], RADII[j])
    return W, curr


# ------------------------------------------------------------------ figures ---
def _frame(d):
    d = np.asarray(d, float); d = d / (np.linalg.norm(d) + 1e-12)
    a = np.array([1., 0, 0]) if abs(d[0]) < 0.9 else np.array([0., 1, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    return d, u, np.cross(d, u)


def _cylinder(ax, p0, p1, r, color, n=24, alpha=1.0, cap=True):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    if np.linalg.norm(d) < 1e-12:
        return
    _, u, v = _frame(d)
    th = np.linspace(0, 2 * np.pi, n); t = np.array([0.0, 1.0])
    TH, T = np.meshgrid(th, t)
    P = lambda A, B, C: (p0[A] + T * d[A] + r * np.cos(TH) * u[A]
                         + r * np.sin(TH) * v[A])
    ax.plot_surface(P(0, 0, 0), P(1, 1, 1), P(2, 2, 2), color=color, alpha=alpha,
                    linewidth=0, antialiased=True, shade=True)
    if cap:
        rr = np.linspace(0, r, 2)
        for c in (p0, p1):
            RR, TH2 = np.meshgrid(rr, th)
            Xc = c[0] + RR * np.cos(TH2) * u[0] + RR * np.sin(TH2) * v[0]
            Yc = c[1] + RR * np.cos(TH2) * u[1] + RR * np.sin(TH2) * v[1]
            Zc = c[2] + RR * np.cos(TH2) * u[2] + RR * np.sin(TH2) * v[2]
            ax.plot_surface(Xc, Yc, Zc, color=color, alpha=alpha, linewidth=0,
                            shade=True)


def _annulus(ax, center, axis, r0, r1, color, alpha=1.0, n=48):
    c = np.asarray(center, float)
    _, u, v = _frame(axis)
    th = np.linspace(0, 2 * np.pi, n); rr = np.array([r0, r1])
    RR, TH = np.meshgrid(rr, th)
    X = c[0] + RR * np.cos(TH) * u[0] + RR * np.sin(TH) * v[0]
    Y = c[1] + RR * np.cos(TH) * u[1] + RR * np.sin(TH) * v[1]
    Z = c[2] + RR * np.cos(TH) * u[2] + RR * np.sin(TH) * v[2]
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=True)


def _ring(ax, center, axis, r, color, lw=0.5, alpha=1.0, n=72):
    c = np.asarray(center, float)
    _, u, v = _frame(axis)
    th = np.linspace(0, 2 * np.pi, n)
    ax.plot(c[0] + r * (np.cos(th) * u[0] + np.sin(th) * v[0]),
            c[1] + r * (np.cos(th) * u[1] + np.sin(th) * v[1]),
            c[2] + r * (np.cos(th) * u[2] + np.sin(th) * v[2]),
            color=color, lw=lw, alpha=alpha)


def _draw_coil(ax, num, view, alpha=0.92):
    """Complete schematic solenoid for coil `num`: exposed iron core tip, copper
    winding barrel with an open bore, and front/rear end rings."""
    axis, top = COIL_POSES[num]
    axis = np.asarray(axis, float); top = np.asarray(top, float)
    head = top - CORE_TIP * axis          # winding front face
    back = head - WIND_LEN * axis         # winding rear face
    core = [(lambda: _cylinder(ax, head, top, CORE_R, CORE_COL, alpha=alpha)),
            (lambda: _ring(ax, top, axis, CORE_R, CORE_EDGE, lw=0.6, alpha=alpha))]
    front = [(lambda: _annulus(ax, head, axis, CORE_R, WIND_R, COIL_CU, alpha=alpha)),
             (lambda: _ring(ax, head, axis, WIND_R, COIL_CU_EDGE, lw=0.7, alpha=alpha)),
             (lambda: _ring(ax, head, axis, CORE_R, COIL_CU_EDGE, lw=0.5, alpha=alpha))]
    barrel = [(lambda: _cylinder(ax, back, head, WIND_R, COIL_CU, alpha=alpha, cap=False))]
    rear = [(lambda: _annulus(ax, back, axis, CORE_R, WIND_R, COIL_CU, alpha=alpha)),
            (lambda: _ring(ax, back, axis, WIND_R, COIL_CU_EDGE, lw=0.7, alpha=alpha))]
    # painter's algorithm within the coil: far parts first
    if np.dot(axis, view) > 0:            # front face toward the camera
        parts = rear + barrel + front + core
    else:                                 # rear face toward the camera
        parts = core + front + barrel + rear
    for draw in parts:
        draw()


def _box(ax, center, sx, sy, sz, color, alpha=1.0):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cx, cy, cz = center
    x = [cx - sx / 2, cx + sx / 2]; y = [cy - sy / 2, cy + sy / 2]
    z = [cz - sz / 2, cz + sz / 2]
    v = np.array([[x[i], y[j], z[k]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    f = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([v[i] for i in f], facecolor=color,
                                         edgecolor="k", linewidths=0.4, alpha=alpha))


def _draw_arm(ax, motors, ai):
    """Draw one 3-motor arm (base, links, joint magnets, gripper). Labels are
    added by `_label_motors` so they never overlap the geometry."""
    from matplotlib.colors import to_rgb
    m1, m2, m3 = motors
    z = np.array([0, 0, 1.0]); col = ARM_COLORS[ai]
    joint_cols = MOTOR_COLORS[ai * 3:ai * 3 + 3]
    link_col = tuple(0.40 * c + 0.60 for c in to_rgb(col))  # pale arm tint
    _box(ax, (m1[0], m1[1], m1[2] - 0.010), 0.024, 0.024, 0.008, "0.35")  # base
    _cylinder(ax, (m1[0], m1[1], m1[2] - 0.006), m1, 0.006, "0.5")        # pillar
    _cylinder(ax, m1, m2, 0.0060, link_col)                                # link 1
    _cylinder(ax, m2, m3, 0.0060, link_col)                                # link 2
    for p, joint_col in zip((m1, m2), joint_cols[:2]):                    # joint motors
        _cylinder(ax, p - z * 0.006, p + z * 0.006, 0.0085, joint_col)
    _cylinder(ax, m3 - z * 0.005, m3 + z * 0.005, 0.0072,
              joint_cols[2])                                               # gripper hub
    d = m3 - m2; d = d / (np.linalg.norm(d) + 1e-9)
    perp = np.array([-d[1], d[0], 0.0])
    for s in (1, -1):                                                      # gripper prongs
        root = m3 + perp * s * 0.005
        _cylinder(ax, root, root + d * 0.013 + perp * s * 0.004, 0.0018,
                  joint_cols[2])


def _label_motors(ax, motors, fan, lift=0.030, spread=0.016, fs=13):
    """Leader-lined M-labels fanned out of the arm cluster so each sits clear of
    the geometry. `fan[k]` is the (dx, dy) offset direction for motor k."""
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    for k, p in enumerate(motors):
        col = MOTOR_COLORS[k]
        dx, dy = fan[k]
        q = np.array([p[0] + dx * spread, p[1] + dy * spread, p[2] + lift])
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=col, lw=0.9,
                alpha=0.85, zorder=60)
        ax.scatter([p[0]], [p[1]], [p[2]], s=8, color=col, zorder=61)
        t = ax.text(q[0], q[1], q[2], f"M{k+1}", fontsize=fs, weight="bold",
                    color=col, ha="center", va="center", zorder=62)
        t.set_path_effects(halo)


def _clean3d(ax):
    """Strip ticks, panes, grid and axis lines for the arm-detail panel."""
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_visible(False); a.line.set_visible(False)
    ax.grid(False)


def _panel_arms(ax, motors):
    """Panel (a): the two arms zoomed in, with per-motor leader labels."""
    ax.computed_zorder = False
    _draw_arm(ax, motors[0:3], 0)
    _draw_arm(ax, motors[3:6], 1)
    fan = {0: (0.55, 1.05), 1: (0.95, -0.65), 2: (0.15, 1.15),
           3: (-0.55, 1.05), 4: (-1.25, 0.15), 5: (-0.15, -1.20)}
    _label_motors(ax, motors, fan)
    M = np.array(motors)
    ax.set_xlim(M[:, 0].min() - 0.014, M[:, 0].max() + 0.014)
    ax.set_ylim(M[:, 1].min() - 0.022, M[:, 1].max() + 0.022)
    ax.set_zlim(M[:, 2].min() - 0.010, M[:, 2].max() + 0.036)
    ax.set_box_aspect((np.ptp(M[:, 0]) + 0.028, np.ptp(M[:, 1]) + 0.044, 0.055))
    ax.view_init(elev=42, azim=-70)
    _clean3d(ax)


def _panel_system(ax, motors, coil_nums):
    """Panel (b): the full platform — six motors ringed by the eight coils."""
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    # draw order (coils -> arms -> tags) instead of depth sort keeps the arms
    # and their tags in front of the translucent coils
    ax.computed_zorder = False
    elev, azim = 24, -58
    view = np.array([math.cos(math.radians(elev)) * math.cos(math.radians(azim)),
                     math.cos(math.radians(elev)) * math.sin(math.radians(azim)),
                     math.sin(math.radians(elev))])
    for num in sorted(coil_nums, key=lambda n: np.dot(COIL_POSES[n][1], view)):
        _draw_coil(ax, num, view)
    _draw_arm(ax, motors[0:3], 0)
    _draw_arm(ax, motors[3:6], 1)
    for ms, ai, txt in [(motors[0:3], 0, "Arm 1"), (motors[3:6], 1, "Arm 2")]:
        c = np.mean(ms, axis=0); q = c + np.array([0.0, 0.0, 0.078])
        ax.plot([c[0], q[0]], [c[1], q[1]], [c[2], q[2]], color=ARM_COLORS[ai],
                lw=1.0, alpha=0.85)
        t = ax.text(q[0], q[1], q[2], txt, fontsize=12.5, weight="bold",
                    color=ARM_COLORS[ai], ha="center", va="center")
        t.set_path_effects(halo)
    ax.set_xlabel("$x$ (m)", labelpad=8)
    ax.set_ylabel("$y$ (m)", labelpad=8)
    ax.set_zlabel("$z$ (m)", labelpad=6)
    L = 0.135
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-0.135, 0.06)
    ax.set_xticks([-0.1, 0, 0.1]); ax.set_yticks([-0.1, 0, 0.1])
    ax.set_zticks([-0.1, -0.05, 0, 0.05])
    ax.set_box_aspect((1, 1, 0.72)); ax.view_init(elev=elev, azim=azim)
    ax.tick_params(pad=1)


def fig_layout(motors, coil_nums, out):
    """Two-panel platform layout: (a) a zoomed arm detail with per-motor labels,
    (b) the full platform — six motors ringed by the eight complete coils."""
    layout_style = dict(PAPER_STYLE)
    layout_style.update({
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    with plt.rc_context(layout_style):
        fig = plt.figure(figsize=(7.6, 4.0))
        axa = fig.add_axes([-0.01, 0.04, 0.40, 0.90], projection="3d")
        axb = fig.add_axes([0.35, 0.00, 0.66, 0.96], projection="3d")
        _panel_arms(axa, motors)
        _panel_system(axb, motors, coil_nums)
        axa.text2D(0.5, 0.97, "(a) arm detail", transform=axa.transAxes,
                   ha="center", fontsize=13)
        axb.text2D(0.5, 0.96, "(b) platform: 6 motors, 8 coils",
                   transform=axb.transAxes, ha="center", fontsize=13)
        fig.savefig(out, dpi=PAPER_DPI)
        plt.close(fig)


def fig_winding(W, curr, out):
    n = len(W)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.abs(W), cmap="RdYlGn_r", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(W[i, j]) > 0.5 else "black")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"M{j+1}" for j in range(n)])
    ax.set_yticklabels([f"M{i+1}\n({curr[i]:.0f} A)" for i in range(n)])
    ax.set_xlabel("Observed motor"); ax.set_ylabel("Actuated motor (peak current)")
    ax.set_title("Input-magnet net winding (turns/cycle)\n"
                 "diagonal = 1 (target rotates); off-diagonal $\\approx$0 (others held)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("|net winding| (turns/cycle)")
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_fields(A, actuated, out):
    """Induced field trajectory at every motor while `actuated` is driven."""
    n = len(A)
    S = spillover_S(A, actuated)
    th = np.linspace(0, 2 * math.pi, 200)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for j, ax in enumerate(axes.ravel()):
        B = np.array([A[j] @ current(S, DRIVE_K * RADII[actuated], t) for t in th]) * 1000
        is_t = (j == actuated)
        col = "tab:red" if is_t else "tab:blue"
        ax.plot(B[:, 0], B[:, 1], color=col, lw=2, label="induced field")
        ax.add_patch(plt.Circle((0, 0), RADII[j] * 1000, fill=False, ec="green",
                                lw=2, label="$b_{min}$ circle"))
        ax.set_aspect("equal")
        lim = max(np.abs(B).max(), RADII[j] * 1000) * 1.25
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        tag = "ACTUATED: field circles $b_{min}$" if is_t else "held: field inside $b_{min}$"
        ax.set_title(f"Motor {j+1} — {tag}", color=(col if is_t else "black"),
                     fontsize=10)
        ax.set_xlabel("$B_x$ (mT)"); ax.set_ylabel("$B_y$ (mT)")
        ax.grid(True, ls=":", lw=0.5)
        if j == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Induced field at all 6 motors while Motor {actuated+1} is actuated",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_sequence(A, out):
    """Sequential actuation M1..M6; integrate all magnets, plot output angles."""
    n = len(A)
    Ss = [spillover_S(A, m) for m in range(n)]
    seg_cycles = 4
    Tseg = seg_cycles * 2 * math.pi / OMEGA
    t_all, out_all = [], [[] for _ in range(n)]
    phi = np.zeros(n); w = np.zeros(n); t0 = 0.0
    net_in = np.zeros(n)
    for m in range(n):     # segment: actuate motor m
        S = Ss[m]
        def rhs(t, s):
            ph, ww = s[:n], s[n:]
            d = np.zeros(2 * n)
            i = current(S, DRIVE_K * RADII[m], OMEGA * (t - t0))
            for j in range(n):
                B = A[j] @ i
                _, acc = magnet_rhs(ph[j], ww[j], B, RADII[j])
                d[j] = ww[j]; d[n + j] = acc
            return d
        sol = solve_ivp(rhs, [t0, t0 + Tseg], np.concatenate([phi, w]),
                        max_step=Tseg / (seg_cycles * 120), rtol=1e-6, atol=1e-8)
        for k in range(len(sol.t)):
            t_all.append(sol.t[k])
            for j in range(n):
                out_all[j].append((sol.y[j][k]) / GEAR * 180 / math.pi)  # deg output
        phi = sol.y[:n, -1].copy(); w = sol.y[n:, -1].copy(); t0 += Tseg
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(3.55, 2.25))
        for j in range(n):
            ax.plot(t_all, out_all[j], color=MOTOR_COLORS[j], lw=1.55,
                    label=f"M{j+1}")
        for m in range(n):
            ax.axvspan(m * Tseg, (m + 1) * Tseg, color=MOTOR_COLORS[m],
                       alpha=0.045, lw=0)
            ax.axvline(m * Tseg, color="0.82", lw=0.45, zorder=0)
            ax.text((m + 0.5) * Tseg, 30.3, f"M{m+1}", ha="center", va="top",
                    fontsize=6.7, color=MOTOR_COLORS[m])
        ax.axvline(n * Tseg, color="0.82", lw=0.45, zorder=0)
        ax.set_ylim(-1.2, 30.8)
        ax.set_yticks([0, 15, 30])
        ax.set_xlim(0, n * Tseg)
        ax.set_xticks([m * Tseg for m in range(n + 1)])
        ax.set_xlabel("time (s)", labelpad=1.5)
        ax.set_ylabel("angle (deg)", labelpad=1.5)
        ax.grid(True, ls=":", lw=0.45, color="0.82")
        ax.legend(ncol=3, loc="lower right", frameon=True, framealpha=0.9,
                  borderpad=0.2, handlelength=1.3, columnspacing=0.7)
        ax.tick_params(direction="out", length=2.5, width=0.6, pad=2)
        fig.subplots_adjust(left=0.14, right=0.995, bottom=0.20, top=0.98)
        fig.savefig(out, dpi=PAPER_DPI, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)


def _winding_selectable(motors, idx):
    """#motors selective under the winding criterion (same as the main result)."""
    A = [A_planar(p, ALL_COILS[idx]) for p in motors]
    W, curr = winding_matrix(A)
    diag = np.abs(np.diag(W)); off = np.abs(W - np.diag(np.diag(W)))
    return int(np.sum([(diag[m] > 0.6) and np.all(off[m] < 0.4) and curr[m] <= 17.5
                       for m in range(6)]))


def fig_scalability(motors, out):
    """#selectable motors vs #coils, nested coil sets, winding criterion."""
    # nested sequence: add one coil at a time toward the 8-coil operating set
    seq = [[5, 6, 8], [5, 6, 8, 9], [3, 5, 6, 8, 9], [2, 3, 5, 6, 8, 9],
           [2, 3, 5, 6, 7, 8, 9], [1, 2, 4, 5, 6, 7, 8, 9]]
    ncoils = [len(s) for s in seq]
    nsel = [_winding_selectable(motors, s) for s in seq]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([2, 12], [1, 6], "k--", lw=1.2, alpha=0.6,
            label="full actuation (2 coils / motor)")
    ax.plot(ncoils, nsel, "o-", color="tab:blue", ms=9, lw=2.5,
            label="proposed method")
    for x, y in zip(ncoils, nsel):
        ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.set_xlabel("number of coils"); ax.set_ylabel("motors selectively actuable")
    ax.set_title("Scalability: more coils $\\Rightarrow$ more selectable motors\n"
                 "6 motors with 8 coils, vs 12 for full actuation")
    ax.set_xticks(range(2, 13, 2)); ax.set_yticks(range(0, 7)); ax.set_ylim(0, 6.5)
    ax.grid(True, ls=":", lw=0.5); ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return ncoils, nsel


def main():
    coils = ALL_COILS[COILS_IDX]
    motors = dual_arm()
    A = [A_planar(p, coils) for p in motors]
    figdir = ROOT / "figures"; figdir.mkdir(exist_ok=True)
    datadir = ROOT / "data"; datadir.mkdir(exist_ok=True)

    print("computing winding matrix...")
    W, curr = winding_matrix(A)
    diag = np.abs(np.diag(W)); off = np.abs(W - np.diag(np.diag(W)))
    all_sel = bool(np.all(diag > 0.6) and np.all(off < 0.4))

    print("rendering figures...")
    fig_layout(motors, [i + 1 for i in COILS_IDX], figdir / "dualarm_layout.png")
    fig_winding(W, curr, figdir / "dualarm_winding.png")
    fig_fields(A, 0, figdir / "dualarm_fields.png")
    fig_sequence(A, figdir / "dualarm_trajectories.png")
    ncoils, nsel = fig_scalability(motors, figdir / "dualarm_scalability.png")

    with open(datadir / "dualarm_case_study.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actuated_motor"] + [f"winding_M{j+1}" for j in range(6)]
                   + ["peak_current_A"])
        for m in range(6):
            w.writerow([m + 1] + [f"{W[m, j]:.3f}" for j in range(6)]
                       + [f"{curr[m]:.1f}"])
        w.writerow([])
        w.writerow(["n_coils"] + [str(c) for c in ncoils])
        w.writerow(["n_selective"] + [str(s) for s in nsel])

    print(f"\nALL 6 SELECTIVE: {all_sel}")
    print(f"max off-target winding: {off.max():.3f} turns/cyc "
          f"({off.max()*N_CYCLES/GEAR*360:.1f} deg output creep over {N_CYCLES} cyc)")
    print(f"peak coil current: {curr.max():.1f} A (limit 17 A)")
    print(f"scalability (coils->selective): {list(zip(ncoils, nsel))}")
    print(f"figures in {figdir}, data in {datadir}")


if __name__ == "__main__":
    main()
