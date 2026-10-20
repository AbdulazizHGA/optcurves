"""Fast numerical checks: PEPit against known values, and against the independently built SDPs."""
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from optcurves import certificate, counterexample, figures, pep, theorem

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


@pytest.mark.parametrize("h", ["1799/1000", "1801/1000"])
def test_strong_sdps_agree_with_pepit(h):
    """kappa = 7/5 (mu = 5/7), one violation, on both sides of 9/5: our dual SDP (certificate.py, no face) has a
    positive margin exactly where PEPit (SmoothStronglyConvexFunction) finds no violation, and our primal SDP
    (counterexample.py, minimizer included) finds slack exactly where PEPit finds one."""
    hf = float(Fraction(h))
    N, pairs, Q, A, b = certificate._system(h, 1, "5/7")
    _, margin = certificate._numeric(N, Q, A, b, len(pairs), 1, reduced=False)
    _, _, t = counterexample._sdp(hf, 1, 5 / 7)
    v = pep.max_min_violation(hf, 1, mu=5 / 7)
    if hf < 1.8:  # the primal optimum is then t = 0 (all data zero), up to solver noise
        assert margin > 1e-4 and t <= pep.TOL and v <= pep.TOL
    else:
        assert margin < -1e-4 and t > 1e-5 and v > 1e-5


@pytest.mark.parametrize("kappa", [2.0, 7 / 5, 10.0])
def test_h1_closed_form_against_pepit(kappa):
    h1 = figures.h1_closed_form(kappa)
    assert abs(h1 - float(theorem.h1(sp.nsimplify(kappa)))) < 1e-12
    assert pep.max_min_violation(h1 - 2e-3, 1, mu=1 / kappa) <= pep.TOL
    assert pep.max_min_violation(h1 + 2e-3, 1, mu=1 / kappa) > pep.TOL


def test_results_consistent_with_exact_brackets():
    """Every numerical threshold h_n (n = 2..6, both classes) lies inside its exact bracket."""
    cons = json.loads((ROOT / "results" / "consecutive.json").read_text())
    lo, hi = {}, {}
    for c in _jsons("certificates"):
        if "mu" not in c:
            lo[c["n"]] = max(lo.get(c["n"], 0.0), float(Fraction(c["h"])))
    for c in _jsons("counterexamples"):
        if "mu" not in c:
            hi[c["n"]] = min(hi.get(c["n"], math.inf), float(Fraction(c["h"])))
    for cls in ("convex_with_minimizer", "convex_all"):
        for n in range(2, 7):
            t = cons["thresholds"][cls][str(n)]
            assert lo[n] <= t["lo"] <= t["hi"] <= hi[n], (cls, n)


def test_strong_results_consistent_with_exact_bracket():
    """The PEP threshold h_1 at kappa = 1.4 (results/strong.json) lies in the exact bracket [9/5, 9/5 + 1e-6].
    It lies about 1e-7 above 9/5 because the bisection counts PEP values <= 1e-7 as "no violation"."""
    row = next(r for r in json.loads((ROOT / "results" / "strong.json").read_text())["rows"] if r["kappa"] == 1.4)
    lo = max(float(Fraction(c["h"])) for c in _jsons("certificates") if c.get("mu") == "5/7")
    hi = min(float(Fraction(c["h"])) for c in _jsons("counterexamples") if c.get("mu") == "5/7")
    assert lo == 1.8 and lo <= row["h1"]["lo"] <= row["h1"]["hi"] <= hi


def test_strong_pep_h1_matches_theorem():
    """Every PEP threshold h_1(kappa) in results/strong.json (30 finite kappa and mu = 0) lies within 5e-6 of the exact
    h_1(kappa) of theorem.py. Since the closed form is proved, this measures the PEP pipeline (violation tolerance
    1e-7, SCS fallback for kappa <= 3); the largest gap is 4.4e-6, at kappa = 1.15."""
    rows = json.loads((ROOT / "results" / "strong.json").read_text())["rows"]
    assert len(rows) == 31
    for r in rows:
        exact = 1.75 if r["kappa"] == "inf" else figures.h1_closed_form(r["kappa"])
        assert max(abs(r["h1"]["lo"] - exact), abs(r["h1"]["hi"] - exact)) < 5e-6, r["kappa"]


@pytest.mark.parametrize("mu, h", [(2 / 3, 1.9), (0.5, 1.95), (0.1, 1.85), (0.0, 1.8)])
def test_strong_worst_case_matches_extremal_data(mu, h):
    """[conjecture] Above h_1(kappa) the one-dimensional data of theorem.py are the worst case: the PEP maximum of v_1
    over ||x_0 - x*|| <= 1 equals their value -h^2 q / (2 D), D = 2 + h^2 (1 - mu) (for mu = 0: h^2 (4h - 7) /
    (2 (2 + h^2)), 0.0618 at h = 1.8)."""
    q = 2 * mu * h**2 - 4 * (1 + mu) * h + mu + 7
    v_data = -h**2 * q / (2 * (2 + h**2 * (1 - mu)))
    assert v_data > 0
    assert abs(pep.max_min_violation(h, 1, mu=mu) - v_data) <= 1e-4 * v_data
