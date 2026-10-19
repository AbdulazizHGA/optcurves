"""Numerical study (PEP, PEPit + Clarabel). Writes results/*.json; `plots` turns them into figures/*.png.

    python -m optcurves.study consecutive   # h_n for n = 1..6, both convex classes; curves for n = 1..3
    python -m optcurves.study strong        # h_n(kappa) for mu-strongly convex f, n = 1..3
    python -m optcurves.study plots         # figures from the JSON files
"""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path

from . import pep

ROOT = Path(__file__).resolve().parents[2]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
H_MAX = 1.9999  # thresholds are searched in [1.5, H_MAX]; "none" means no violation found below H_MAX
ITERS = 24


def _meta():
    return {
        "pepit": importlib.metadata.version("PEPit"),
        "cvxpy": importlib.metadata.version("cvxpy"),
        "solver": pep.SOLVER,
        "fallback": pep.FALLBACK,
        "violation_tol": pep.TOL,
        "bisection": {"interval": [1.5, H_MAX], "iterations": ITERS,
                      "bracket_width": (H_MAX - 1.5) / 2 ** ITERS},
        "L": 1.0,
    }


def _threshold(n, mu=0.0, minimizer=True):
    value = lambda h: pep.max_min_violation(h, n, mu=mu, minimizer=minimizer)
    pep.INACCURATE[0] = 0
    if value(H_MAX) <= pep.TOL:
        return None
    lo, hi = pep.bisect_threshold(value, 1.5, H_MAX, iters=ITERS)
    return {"lo": lo, "hi": hi, "value_at_lo": value(lo), "value_at_hi": value(hi),
            "inaccurate_solves": pep.INACCURATE[0]}


def consecutive(n_max=6, grid=None):
    grid = grid or [round(1.70 + 0.005 * i, 4) for i in range(60)] + [1.9975, 1.999]
    out = {"meta": _meta(), "thresholds": {}, "curves": {}}
    for minimizer in (True, False):
        cls = "convex_with_minimizer" if minimizer else "convex_all"
        out["thresholds"][cls] = {}
        for n in range(1, n_max + 1):
            out["thresholds"][cls][n] = _threshold(n, minimizer=minimizer)
            print(cls, n, out["thresholds"][cls][n], flush=True)
    for n in (1, 2, 3):
        out["curves"][n] = [[h, pep.max_min_violation(h, n)] for h in grid]
    _write("consecutive.json", out)


KAPPAS = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 2, 2.25, 2.5, 3, 3.5, 4, 5, 6, 8,
          10, 15, 20, 30, 50, 100, 300, 1000, 10000, math.inf]


def strong(kappas=KAPPAS, n_max=3):
    out = {"meta": _meta(), "class": "L-smooth mu-strongly convex, ||x0 - x*||^2 <= 1, kappa = L/mu",
           "rows": []}
    for kappa in kappas:
        mu = 0.0 if math.isinf(kappa) else 1.0 / kappa
        row = {"kappa": "inf" if math.isinf(kappa) else kappa,
               **{f"h{n}": _threshold(n, mu=mu) for n in range(1, n_max + 1)}}
        print(row["kappa"], {k: (v and round(v["lo"], 6)) for k, v in row.items() if k != "kappa"}, flush=True)
        out["rows"].append(row)
    _write("strong.json", out)


def _write(name, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1) + "\n")


# Categorical slots 1-3 of the reference palette (blue, orange, aqua), fixed order by n.
COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
MARKERS = {1: "o", 2: "s", 3: "^"}


def _axes(ax):
    ax.grid(True, color="#e4e2dc", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8a887f")


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    cons = json.loads((RESULTS / "consecutive.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for n in (1, 2, 3):
        pts = cons["curves"][str(n)]
        ax.plot([p[0] for p in pts], [max(p[1], 0.0) for p in pts], color=COLORS[n], linewidth=2,
                label=f"{n} consecutive" if n > 1 else "1 violation")
        t = cons["thresholds"]["convex_with_minimizer"][str(n)]
        ax.axvline(t["lo"], color=COLORS[n], linewidth=1, linestyle=":")
        ax.annotate(f"h$_{n}$ ≈ {t['lo']:.4f}", (t["lo"], 0.0), xytext=(4, 6 + 16 * n),
                    textcoords="offset points", color="#3d3c37", fontsize=8)
    ax.set_xlabel("step size h (in units of 1/L)")
    ax.set_ylabel("max min(v$_1$, …, v$_n$)   (‖x$_0$ − x*‖ ≤ 1)")
    ax.set_title("Worst-case consecutive convexity violations, L-smooth convex f", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "consecutive.png")
    plt.close(fig)

    st = json.loads((RESULTS / "strong.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    finite = [r for r in st["rows"] if r["kappa"] != "inf"]
    inf_row = next(r for r in st["rows"] if r["kappa"] == "inf")
    for n in (1, 2, 3):
        pts = [(r["kappa"], r[f"h{n}"]["lo"]) for r in finite if r[f"h{n}"]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[n], linewidth=2,
                marker=MARKERS[n], markersize=4, label=f"h$_{n}$(κ)")
        ax.axhline(inf_row[f"h{n}"]["lo"], color=COLORS[n], linewidth=1, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("condition number κ = L/μ   (dotted: κ = ∞, i.e. μ = 0)")
    ax.set_ylabel("threshold h$_n$ (units of 1/L)")
    ax.set_title("Step size above which n consecutive violations occur, μ-strongly convex f", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "strong_threshold.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["consecutive", "strong", "plots"])
    {"consecutive": consecutive, "strong": strong, "plots": plots}[ap.parse_args().what]()
