"""Numerical study (PEP, PEPit + Clarabel). Writes results/*.json.

    python -m optcurves.study consecutive   # h_n for n = 1..6, both convex classes; curves for n = 1..3
    python -m optcurves.study strong        # h_n(kappa) for mu-strongly convex f, n = 1..3
Figures: python -m optcurves.figures
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["consecutive", "strong"])
    {"consecutive": consecutive, "strong": strong}[ap.parse_args().what]()
