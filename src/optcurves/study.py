"""Numerical study (PEP, PEPit + Clarabel). Writes results/*.json.

    python -m optcurves.study consecutive   # h_n for n = 1..6, both convex classes; curves for n = 1..3
    python -m optcurves.study strong        # h_n(kappa) for mu-strongly convex f, n = 1..3
Figures: python -m optcurves.figures
"""
import argparse
import importlib.metadata
import json
import math
import platform
from pathlib import Path

from . import pep

ROOT = Path(__file__).resolve().parents[2]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
H_MAX = 1.9999  # thresholds are searched in [1.5, H_MAX]; None means no violation at H_MAX
# Upper ends tried in turn until one is decided reliably (pep._solve); H_MAX first.
TOPS = (H_MAX, 1.9995, 1.999, 1.998, 1.995, 1.99)
ITERS = 24
WIDTH = (H_MAX - 1.5) / 2 ** ITERS  # bracket width after ITERS halvings of [1.5, H_MAX]


def _meta():
    version = importlib.metadata.version
    return {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "pepit": version("PEPit"),
        "cvxpy": version("cvxpy"),
        "clarabel": version("clarabel"),
        "scs": version("scs"),
        "solver": pep.SOLVER,
        "fallback": pep.FALLBACK,
        "unresolved": "point where neither the solver nor the fallback reaches status optimal; it decides nothing",
        "violation_tol": pep.TOL,
        "bisection": {"interval": [1.5, H_MAX], "upper_ends_tried": list(TOPS), "iterations": ITERS,
                      "bracket_width": WIDTH, "points_tried_per_step": list(pep.TRIES)},
        "L": 1.0,
    }


def _threshold(n, mu=0.0, minimizer=True):
    """Bracket of the infimum of the h in [1.5, H_MAX] with n consecutive violations, assuming that these h form an
    interval: proved for convex f (rescaling, README section 1), assumed for mu > 0. lo has no violation and hi has
    one, both from solves that reached "optimal";
    lo = 1.5 needs no solve (v_1 <= 0 for h <= 7/4 by Barzilai, Shamir and Zamani, also for mu > 0).
    None: no violation at H_MAX. hi = None: no upper end could be decided (then lo is the largest top end without a
    violation, or 1.5). resolved: the bracket reached the width WIDTH. The counts of pep.STATS and the points left
    unresolved are stored with it."""
    pep.reset_stats()
    values, unresolved_at = {}, []

    def decide(h):
        value, reliable = pep.evaluate(h, n, mu=mu, minimizer=minimizer)
        if not reliable:
            unresolved_at.append(h)
            return None
        values[h] = value
        return value > pep.TOL

    lo, hi, resolved = 1.5, None, False
    for top in TOPS:
        answer = decide(top)
        if answer is not None:
            break
    if answer is False and top == H_MAX:
        return None
    if answer:
        lo, hi, resolved = pep.bisect_threshold(decide, 1.5, top, WIDTH)
    elif answer is False:
        lo = top
    return {"lo": lo, "hi": hi, "value_at_lo": values.get(lo), "value_at_hi": values.get(hi),
            "resolved": resolved, **pep.STATS, "unresolved_at": unresolved_at}


def consecutive(n_max=6, grid=None):
    grid = grid or [round(1.70 + 0.005 * i, 4) for i in range(60)] + [1.9975, 1.999]
    out = {"meta": _meta(), "thresholds": {}, "curves": {}}
    for minimizer in (True, False):
        cls = "convex_with_minimizer" if minimizer else "convex_all"
        out["thresholds"][cls] = {}
        for n in range(1, n_max + 1):
            out["thresholds"][cls][n] = _threshold(n, minimizer=minimizer)
            print(cls, n, out["thresholds"][cls][n], flush=True)
    out["curves_unresolved"] = {}
    for n in (1, 2, 3):
        pep.reset_stats()
        out["curves"][n] = [[h, pep.max_min_violation(h, n)] for h in grid]
        out["curves_unresolved"][n] = pep.STATS["unresolved"]  # values kept but not reliable
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
