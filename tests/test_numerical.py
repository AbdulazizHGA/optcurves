"""Fast numerical checks: PEPit against known values, and against the independently built SDPs."""
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from optcurves import certificate, counterexample, figures, pep

ROOT = Path(__file__).resolve().parents[1]


def _jsons(folder):
    return [json.loads(p.read_text()) for p in (ROOT / folder).glob("*.json") if not p.name.startswith("._")]


@pytest.mark.parametrize("h, n, positive", [
    (1.5, 1, False), (1.74, 1, False), (1.76, 1, True),    # Barzilai & Shamir: threshold 7/4
    (1.85, 2, False), (1.87, 2, True), (1.89, 3, False), (1.91, 3, True),
])
def test_pep_signs(h, n, positive):
    for minimizer in (True, False):
        v = pep.max_min_violation(h, n, minimizer=minimizer)
        assert (v > 1e-4) if positive else (v <= pep.TOL)


def test_pep_one_violation_matches_known_slope():
    # Worst single violation grows linearly past 7/4 (PEP value at 1.80 is 0.0618, ||x0 - x*|| <= 1).
    assert abs(pep.max_min_violation(1.8, 1) - 0.0618) < 1e-3


@pytest.mark.parametrize("h, n", [("9/5", 2), ("19/10", 3)])
def test_independent_dual_sdp_agrees(h, n):
    """Our hand-built dual SDP (certificate.py) is feasible exactly where PEPit says 'no violation'."""
    N, pairs, Q, A, b = certificate._system(h, n)
    _, margin = certificate._numeric(N, Q, A, b, len(pairs), n)
    assert margin > 0
    assert pep.max_min_violation(float(Fraction(h)), n, minimizer=False) <= pep.TOL


def test_independent_primal_sdp_agrees():
    """Our hand-built primal SDP (counterexample.py) finds slack exactly where PEPit finds violations."""
    _, _, t = counterexample._sdp(1.9, 2)
    assert t > 0 and pep.max_min_violation(1.9, 2) > 0.02


@pytest.mark.parametrize("kappa", [2.0, 7 / 5, 10.0])
def test_h1_conjecture_spot(kappa):
    h1 = figures.h1_conjecture(kappa)
    assert pep.max_min_violation(h1 - 2e-3, 1, mu=1 / kappa) <= pep.TOL
    assert pep.max_min_violation(h1 + 2e-3, 1, mu=1 / kappa) > pep.TOL


def test_results_consistent_with_exact_brackets():
    """Every numerical threshold h_n (n = 2..6, both classes) lies inside its exact bracket."""
    cons = json.loads((ROOT / "results" / "consecutive.json").read_text())
    lo, hi = {}, {}
    for c in _jsons("certificates"):
        lo[c["n"]] = max(lo.get(c["n"], 0.0), float(Fraction(c["h"])))
    for c in _jsons("counterexamples"):
        hi[c["n"]] = min(hi.get(c["n"], math.inf), float(Fraction(c["h"])))
    for cls in ("convex_with_minimizer", "convex_all"):
        for n in range(2, 7):
            t = cons["thresholds"][cls][str(n)]
            assert lo[n] <= t["lo"] <= t["hi"] <= hi[n], (cls, n)
