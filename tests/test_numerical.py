"""Fast numerical checks: PEPit against known values, and against the independently built SDPs."""
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from optcurves import certificate, counterexample, figures, pep, theorem

ROOT = Path(__file__).resolve().parents[1]


def _jsons(folder):
    return [json.loads(p.read_text()) for p in (ROOT / folder).glob("*.json") if not p.name.startswith("._")]


@pytest.mark.parametrize("h, n, positive", [
    (1.5, 1, False), (1.74, 1, False), (1.76, 1, True),    # Barzilai, Shamir and Zamani: threshold 7/4
    (1.85, 2, False), (1.87, 2, True), (1.89, 3, False), (1.91, 3, True),
])
def test_pep_signs(h, n, positive):
    for minimizer in (True, False):
        v = pep.max_min_violation(h, n, minimizer=minimizer)
        assert (v > 1e-4) if positive else (v <= pep.TOL)


def test_pep_one_violation_matches_worst_case():
    # Worst single violation past 7/4 is h^2 (4h - 7) / (2 (2 + h^2)) for ||x0 - x*|| <= 1 (worstcase.py, mu = 0):
    # 0.0618 at h = 1.8.
    assert abs(pep.max_min_violation(1.8, 1) - 0.0618) < 1e-3


@pytest.mark.parametrize("h, n", [("9/5", 2), ("19/10", 3)])
def test_independent_dual_sdp_agrees(h, n):
    """Our hand-built dual SDP (certificate.py) has a positive margin where PEPit says 'no violation': at h = 9/5
    (n = 2) and h = 19/10 (n = 3)."""
    N, pairs, Q, A, b = certificate._system(h, n)
    _, margin = certificate._numeric(N, Q, A, b, len(pairs), n)
    assert margin > 0
    assert pep.max_min_violation(float(Fraction(h)), n, minimizer=False) <= pep.TOL


def test_independent_primal_sdp_agrees():
    """Our hand-built primal SDP (counterexample.py) finds positive slack where PEPit finds a violation: at h = 1.9,
    n = 2."""
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
    """Every PEP bracket of h_1(kappa) in results/strong.json (30 finite kappa and mu = 0) contains the exact h_1(kappa)
    of theorem.py up to 1e-6 (PEP values <= 1e-7 count as "no violation", which puts the bracket about 1e-7 above), and
    every bracket that reached full width lies within 5e-6 of it. Since the closed form is proved, this measures the
    PEP pipeline."""
    rows = json.loads((ROOT / "results" / "strong.json").read_text())["rows"]
    assert len(rows) == 31
    for r in rows:
        exact = 1.75 if r["kappa"] == "inf" else figures.h1_closed_form(r["kappa"])
        t = r["h1"]
        assert t["hi"] is not None and t["lo"] - 1e-6 <= exact <= t["hi"] + 1e-6, r["kappa"]
        if t["resolved"]:
            assert max(abs(t["lo"] - exact), abs(t["hi"] - exact)) < 5e-6, r["kappa"]


def _threshold_records():
    cons = json.loads((ROOT / "results" / "consecutive.json").read_text())
    for cls, thresholds in cons["thresholds"].items():
        for n, t in thresholds.items():
            yield f"{cls}, n = {n}", t
    for r in json.loads((ROOT / "results" / "strong.json").read_text())["rows"]:
        for n in (1, 2, 3):
            yield f"kappa = {r['kappa']}, n = {n}", r[f"h{n}"]


def test_threshold_brackets_are_consistent():
    """Both ends of every bracket were decided by solves that reached "optimal": no violation at lo, one at hi. The
    PEP value is continuous in h, so across a full-width bracket (3e-8) it crosses TOL = 1e-7 but cannot jump; a
    value at hi far above TOL (3.6e-4 at kappa = 1.05 before unresolved points were set aside) means a wrong end."""
    for name, t in _threshold_records():
        if t is None:
            continue
        assert t["value_at_lo"] is None or t["value_at_lo"] <= pep.TOL, name
        assert t["hi"] is None or t["value_at_hi"] > pep.TOL, name
        if t["resolved"]:
            assert t["value_at_hi"] < 1e-6 and t["hi"] - t["lo"] < 3e-8, name


def test_strong_results_consistent_with_strong_counterexamples():
    """Every strongly convex counterexample (n violations at step h, modulus mu) lies above the lower end of the PEP
    bracket of h_n(1/mu), where the search found no violation."""
    rows = {r["kappa"]: r for r in json.loads((ROOT / "results" / "strong.json").read_text())["rows"]}
    found = 0
    for c in _jsons("counterexamples"):
        if "mu" in c:
            t = rows[float(1 / Fraction(c["mu"]))][f"h{c['n']}"]
            assert t is not None and t["lo"] < float(Fraction(c["h"])), c["h"]
            found += 1
    assert found >= 5


@pytest.mark.parametrize("mu, h", [(2 / 3, 1.9), (0.5, 1.95), (0.1, 1.85), (0.0, 1.8)])
def test_strong_worst_case_matches_extremal_data(mu, h):
    """Above h_1(kappa) the one-dimensional data of theorem.py are the worst case (proved in worstcase.py): the PEP
    maximum of v_1 over ||x_0 - x*|| <= 1 equals their value -h^2 q / (2 D), D = 2 + h^2 (1 - mu) (for mu = 0:
    h^2 (4h - 7) / (2 (2 + h^2)), 0.0618 at h = 1.8). Since the value is proved, this checks the PEP pipeline."""
    q = 2 * mu * h**2 - 4 * (1 + mu) * h + mu + 7
    v_data = -h**2 * q / (2 * (2 + h**2 * (1 - mu)))
    assert v_data > 0
    assert abs(pep.max_min_violation(h, 1, mu=mu) - v_data) <= 1e-4 * v_data


def test_side_experiments():
    """results/side.json (python -m optcurves.side): the PEP worst case of one violation against the closed form of
    worstcase.py at 20 points, all solved to "optimal", and the max-slack search in R^1, whose best margins stay below
    the feasibility tolerance 1e-9 while the same search in R^2 has margins of order 1e-2."""
    side = json.loads((ROOT / "results" / "side.json").read_text())
    pts = side["worst_case_pep"]["points"]
    assert len(pts) == 20 and all(p["reliable"] for p in pts)
    for p in pts:
        mu, h = 1 / p["kappa"], p["h"]
        q = 2 * mu * h**2 - 4 * (1 + mu) * h + mu + 7
        assert abs(p["closed_form"] + h**2 * q / (2 * (2 + h**2 * (1 - mu)))) < 1e-15
        assert abs(p["pep"] - p["closed_form"]) <= 2e-7 * p["closed_form"]
    assert side["worst_case_pep"]["max_relative_error"] < 1.6e-7  # README: "within a relative 1.6e-7"
    searches = side["one_dimensional_search"]
    assert [(r["h"], r["n"]) for r in searches] == [("1.9", 2), ("1.99", 3)]
    for r in searches:
        assert (r["best_margin_R1"] is None or r["best_margin_R1"] < 1e-9) and r["margin_R2"] > 1e-3


def test_h1_closed_form_is_stable():
    """The floating-point h_1(kappa) agrees with the exact theorem.h1 to rounding (relative 1e-14; near kappa = 1 the
    root is ill-conditioned, dh_1/dkappa ~ -34 at 1.0001), and stays near 7/4 for huge kappa (the textbook root
    formula was off by 4e-13 at kappa = 1e4, gave 0.0 at 1e17 and overflowed at 1e200)."""
    for kappa in [1, 1.0001, 1.05, 7 / 5, 2, 10, 100, 1e4]:
        exact = float(sp.N(theorem.h1(sp.nsimplify(kappa)), 30))
        assert abs(figures.h1_closed_form(kappa) - exact) < 1e-14 * exact, kappa
    for kappa in [1e8, 1e17, 1e200, 1e300, 1e308, sys.float_info.max]:  # 7 kappa overflowed to NaN at 1e308
        assert 1.75 <= figures.h1_closed_form(kappa) < 1.75 + 1e-6  # h_1 - 7/4 ~ 1 / (32 kappa)
    assert figures.h1_closed_form(math.inf) == 1.75


def test_strong_trends_stated_in_the_readme():
    """README section 4: "Every bracket we computed lies above its convex counterpart and decreases in kappa". For
    n = 1, 2, 3 and every record (None, no violation up to H_MAX, counts as h_n >= H_MAX): its lower end is at least
    that of mu = 0, the lower ends do not increase with kappa, and the brackets admit a threshold that does not
    increase with kappa (no lower end at a larger kappa lies above the upper end at a smaller one)."""
    from optcurves import study
    rows = json.loads((ROOT / "results" / "strong.json").read_text())["rows"]
    convex = next(r for r in rows if r["kappa"] == "inf")
    finite = sorted((r for r in rows if r["kappa"] != "inf"), key=lambda r: r["kappa"])
    for n in (1, 2, 3):
        lo = [study.H_MAX if r[f"h{n}"] is None else r[f"h{n}"]["lo"] for r in finite]
        hi = [math.inf if r[f"h{n}"] is None or r[f"h{n}"]["hi"] is None else r[f"h{n}"]["hi"] for r in finite]
        assert all(v >= convex[f"h{n}"]["lo"] for v in lo), n
        assert all(a >= b for a, b in zip(lo, lo[1:])), n
        assert all(max(lo[i + 1:], default=-math.inf) <= hi[i] for i in range(len(finite))), n
