"""Figures from results/*.json and counterexamples/*.json.

    python -m optcurves.figures     # writes figures/consecutive.png, strong_threshold.png, counterexample_2d.png
"""
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"

# Categorical slots 1-3 of the reference palette (blue, orange, aqua), fixed order by n.
COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
MARKERS = {1: "o", 2: "s", 3: "^"}
INK, MUTED, GRID = "#3d3c37", "#8a887f", "#e4e2dc"


def h1_closed_form(kappa):
    """Exact threshold for one violation, mu-strongly convex, kappa = L/mu (proved in theorem.py), in floating point.
    The smaller root (1 + kappa) - sqrt((1 + kappa)^2 - (7 kappa + 1)/2) subtracts two nearly equal numbers (it
    gives 0.0 at kappa = 1e17), so it is computed as ((7 kappa + 1)/2) / (1 + kappa + sqrt((kappa - 1)(kappa - 1/2))),
    the square root taken factor by factor so that it does not overflow. kappa = inf (mu = 0) gives 7/4."""
    if math.isinf(kappa):
        return 1.75
    return (3.5 * kappa + 0.5) / (1 + kappa + math.sqrt(kappa - 1) * math.sqrt(kappa - 0.5))


def _axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)


def consecutive(plt):
    cons = json.loads((RESULTS / "consecutive.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for n in (1, 2, 3):
        pts = cons["curves"][str(n)]
        ax.plot([p[0] for p in pts], [max(p[1], 0.0) for p in pts], color=COLORS[n], linewidth=2,
                label="1 violation" if n == 1 else f"{n} consecutive violations")
        t = cons["thresholds"]["convex_with_minimizer"][str(n)]["lo"]
        ax.axvline(t, color=COLORS[n], linewidth=1, linestyle=":")
        ax.text(t - 0.003, 0.97, f"h$_{n}$ ≈ {t:.4f}", transform=ax.get_xaxis_transform(), rotation=90,
                va="top", ha="right", color=INK, fontsize=8)
    ax.set_xlabel("step size h (units of 1/L)", color=INK, fontsize=9)
    ax.set_ylabel("worst case of min(v$_1$, …, v$_n$),  ‖x$_0$ − x*‖ ≤ 1", color=INK, fontsize=9)
    ax.set_title("Consecutive convexity violations of the GD curve, L-smooth convex f (PEP)",
                 fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    _axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "consecutive.png")
    plt.close(fig)


def strong(plt):
    st = json.loads((RESULTS / "strong.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    finite = [r for r in st["rows"] if r["kappa"] != "inf"]
    inf_row = next(r for r in st["rows"] if r["kappa"] == "inf")
    ks = np.geomspace(1.0001, 1e4, 400)
    ax.plot(ks, [h1_closed_form(k) for k in ks], color=INK, linewidth=1, linestyle="--",
            label="h$_1$(κ), exact (theorem)")
    for n in (1, 2, 3):
        recs = [(r["kappa"], r[f"h{n}"]) for r in finite if r[f"h{n}"] and r[f"h{n}"]["hi"] is not None]
        # a bracket that points no solver could decide left wider than 1e-5 is drawn whole; narrower ones are points
        wide = lambda t: t["hi"] - t["lo"] > 1e-5
        pts = [(k, t["lo"]) for k, t in recs if not wide(t)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[n], linewidth=0, marker=MARKERS[n],
                markersize=5, label=f"h$_{n}$(κ), PEP")
        for i, (k, t) in enumerate((k, t) for k, t in recs if wide(t)):
            ax.plot([k, k], [t["lo"], t["hi"]], color=COLORS[n], linewidth=2, solid_capstyle="butt")
            ax.plot([k], [t["lo"]], color=COLORS[n], linewidth=0, marker=MARKERS[n], markersize=5,
                    markerfacecolor="white", label=None if i else f"h$_{n}$(κ), PEP bracket left open")
        ax.axhline(inf_row[f"h{n}"]["lo"], color=COLORS[n], linewidth=1, linestyle=":")
    exact = [json.loads(p.read_text()) for p in sorted((ROOT / "counterexamples").glob("*.json"))
             if not p.name.startswith("._")]
    exact = [c for c in exact if "mu" in c and c["n"] > 1]
    if exact:
        ns = ", ".join(str(n) for n in sorted({c["n"] for c in exact}))
        ax.plot([float(1 / F(c["mu"])) for c in exact], [float(F(c["h"])) for c in exact], color=INK,
                linewidth=0, marker="*", markersize=9, label=f"{ns} violations, exact counterexample")
    ax.set_xscale("log")
    ax.set_ylim(1.74, 2.005)
    ax.set_xlabel("condition number κ = L/μ   (dotted lines: μ = 0)", color=INK, fontsize=9)
    ax.set_ylabel("threshold h$_n$ (units of 1/L)", color=INK, fontsize=9)
    ax.set_title("PEP threshold for n consecutive violations (n = 1: exact), μ-strongly convex f",
                 fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "strong_threshold.png")
    plt.close(fig)


def interpolant(ce):
    """Largest 1-smooth convex interpolant of the counterexample data, evaluated by its dual formula
    f(x) = min over lam in the simplex of 1/2 ||sum lam_i (x - x_i + g_i)||^2 + sum lam_i (f_i - 1/2 ||g_i||^2)
    (the conjugate of the smallest 1-strongly convex function through the dual data, the maximum of the quadratic
    lower bounds; conjugation reverses the order). Returns a vectorized f.

    The minimum is a convex quadratic program over the simplex, solved exactly (up to rounding) face by face: on a
    face the KKT conditions are a linear system, and a face whose solution satisfies all of them (lam >= 0 and no
    gradient entry outside the face below the multiplier) gives the minimum. Each point is solved on its own, so its
    value does not depend on the other points of the call. f.kkt_residual is the largest KKT residual of the last
    call."""
    g = [np.array([float(F(a)) for a in gk]) for gk in ce["g"]]
    x = [np.array([float(a) for a in xk]) for xk in _positions(ce)]
    fv = [float(F(a)) for a in ce["f"]]
    pts_x = np.array([np.zeros(len(x[0]))] + x)
    pts_g = np.array([np.zeros(len(x[0]))] + g)
    c = np.array([0.0] + [fk - 0.5 * gk @ gk for fk, gk in zip(fv, g)])
    m = len(c)

    def f(X):
        A = X[:, None, :] - pts_x[None] + pts_g[None]          # (P, m, d): rows x - x_i + g_i
        best, resid_best = np.full(len(X), np.inf), np.full(len(X), np.inf)
        for r in range(1, m + 1):
            for face in itertools.combinations(range(m), r):
                S, out = list(face), [i for i in range(m) if i not in face]
                K = np.zeros((len(X), r + 1, r + 1))             # [A_S A_S^T, -1; 1^T, 0] [lam_S; nu] = [-c_S; 1]
                K[:, :r, :r] = np.einsum("pid,pjd->pij", A[:, S], A[:, S])
                K[:, :r, r], K[:, r, :r] = -1.0, 1.0
                rhs = np.zeros((len(X), r + 1))
                rhs[:, :r], rhs[:, r] = -c[S], 1.0
                sol = np.einsum("pij,pj->pi", np.linalg.pinv(K), rhs)
                lam = np.zeros((len(X), m))
                lam[:, S] = sol[:, :r]
                nu = sol[:, r]
                v = np.einsum("pm,pmd->pd", lam, A)
                grad = np.einsum("pmd,pd->pm", A, v) + c          # gradient of the objective in lam
                parts = [np.abs(lam.sum(axis=1) - 1), np.maximum(-lam, 0).max(axis=1),
                         np.abs(grad[:, S] - nu[:, None]).max(axis=1)]
                if out:
                    parts.append(np.maximum(nu[:, None] - grad[:, out], 0).max(axis=1))
                resid = np.max(parts, axis=0)
                val = 0.5 * np.einsum("pd,pd->p", v, v) + lam @ c
                take = (resid < 1e-9) & (val < best)
                best[take], resid_best[take] = val[take], resid[take]
        f.kkt_residual = float(resid_best.max())
        return best
    return f


def plot_grid(ce, points=140):
    """Iterates of the counterexample and the grid on which its contours are drawn."""
    xs = np.array([[float(a) for a in xk] for xk in _positions(ce)])
    lo, hi = xs.min(axis=0), xs.max(axis=0)
    pad = 0.35 * (hi - lo).max()
    gx = np.linspace(min(lo[0], 0) - pad, max(hi[0], 0) + pad, points)
    gy = np.linspace(min(lo[1], 0) - pad, max(hi[1], 0) + pad, points)
    GX, GY = np.meshgrid(gx, gy)
    return xs, GX, GY


def _positions(ce):
    h = F(ce["h"])
    xs = [[F(a) for a in ce["x0"]]]
    for gk in ce["g"][:-1]:
        xs.append([a - h * F(b) for a, b in zip(xs[-1], gk)])
    return xs


def counterexample_2d(plt, name="n2_h19_10_d2.json"):
    ce = json.loads((ROOT / "counterexamples" / name).read_text())
    f = interpolant(ce)
    xs, GX, GY = plot_grid(ce)
    fv = [float(F(a)) for a in ce["f"]]
    assert np.allclose(f(xs), fv, atol=1e-9), "interpolant does not reproduce the data"
    Z = f(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    assert f.kkt_residual < 1e-9, "contour values not solved to optimality"

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]})
    a1.contour(GX, GY, Z, levels=18, colors=MUTED, linewidths=0.6)
    a1.plot(xs[:, 0], xs[:, 1], color=COLORS[2], linewidth=2, marker="o", markersize=6)
    for k, (px, py) in enumerate(xs):
        a1.annotate(f"x$_{k}$", (px, py), xytext=(6, 4), textcoords="offset points", fontsize=9, color=INK)
    a1.plot([0], [0], marker="*", markersize=10, color=INK)
    a1.annotate("x*", (0, 0), xytext=(6, -10), textcoords="offset points", fontsize=9, color=INK)
    a1.set_aspect("equal")
    a1.set_title(f"GD with h = {ce['h']} on a 1-smooth convex f: R² → R", fontsize=9, color=INK)
    _axes(a1)
    dec = [fv[k] - fv[k + 1] for k in range(len(fv) - 1)]
    # Dots, not bars: the axis does not start at 0.
    a2.plot(range(len(dec)), dec, color=COLORS[2], linewidth=2, marker="o", markersize=8)
    for k, d in enumerate(dec):
        a2.annotate(f"{d:.4f}", (k, d), xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=8, color=INK)
    a2.set_xticks(range(len(dec)), [f"f(x$_{k}$) − f(x$_{k + 1}$)" for k in range(len(dec))], fontsize=8)
    a2.set_xlim(-0.4, len(dec) - 0.6)
    a2.set_ylim(min(dec) - 0.25 * (max(dec) - min(dec)), max(dec) + 0.3 * (max(dec) - min(dec)))
    viol = ", ".join(f"v$_{k + 1}$ = {float(F(v)):.4f}" for k, v in enumerate(ce["violations"]))
    a2.set_title(f"decrease per step grows twice in a row ({viol})", fontsize=9, color=INK)
    _axes(a2)
    fig.tight_layout()
    fig.savefig(FIGURES / "counterexample_2d.png")
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(exist_ok=True)
    consecutive(plt)
    strong(plt)
    counterexample_2d(plt)
    print("wrote figures/consecutive.png, figures/strong_threshold.png, figures/counterexample_2d.png")


if __name__ == "__main__":
    main()
