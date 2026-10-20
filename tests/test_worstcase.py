"""The exact worst case of one violation above h_1(kappa) (worstcase.py): the symbolic proof, an independent exact
check of the certificate identity, exact instances on both sides of the threshold, and the stored counterexamples."""
import json
import random
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from optcurves import certificate, counterexample, theorem, worstcase

ROOT = Path(__file__).resolve().parents[1]
F = lambda r: Fraction(str(r))


def q(hv, muv):
    return 2 * muv * hv**2 - 4 * (1 + muv) * hv + muv + 7


def c(hv, muv):
    """The theorem's bound h^2 (-q) / (2 D) (the worst case when q < 0)."""
    return hv**2 * (-q(hv, muv)) / (2 * (2 + hv**2 * (1 - muv)))


def threshold_point(t):
    """(h, mu) with q(h, mu) = 0 and h = t: mu = (4t - 7) / (2t^2 - 4t + 1), in (0, 1) for 7/4 < t < 2."""
    t = Fraction(t)
    return t, (4 * t - 7) / (2 * t**2 - 4 * t + 1)


def valid(cert):
    """Multipliers >= 0 and S positive semidefinite, by an exact LDL^T (zero pivots only with vanishing columns)."""
    res = certificate._ldl(cert["S"], psd=True)
    return (all(v >= 0 for v in cert["lam"].values()) and res is not None and all(d >= 0 for d in res[1])
            and res[0] * sp.diag(*res[1]) * res[0].T == cert["S"])


@pytest.fixture(scope="module")
def checks():
    return worstcase.verify()


def test_every_step_of_the_proof(checks):
    failed = [text for text, ok in checks if not ok]
    assert not failed and len(checks) >= 15, failed


def test_proof_check_is_not_vacuous(monkeypatch):
    """A wrong Gram matrix, multiplier or bound is caught."""
    monkeypatch.setattr(worstcase, "W_MATRIX", worstcase.W_MATRIX + sp.diag(0, 0, 0, sp.Rational(1, 10**6)))
    bad = {text for text, ok in worstcase.verify() if not ok}
    assert any("equals W" in t for t in bad) and any("= 0 for symbolic" in t for t in bad)
    monkeypatch.undo()
    lam = dict(worstcase.LAM)
    lam[0, 1] += sp.Rational(1, 10**6)
    monkeypatch.setattr(worstcase, "LAM", lam)
    assert any("coefficients of v_1" in t for t, ok in worstcase.verify() if not ok)
    monkeypatch.undo()
    monkeypatch.setattr(worstcase, "C", worstcase.C * (1 + sp.Rational(1, 10**6)))
    bad = {text for text, ok in worstcase.verify() if not ok}
    assert any("= 0 for symbolic" in t for t in bad) and any("v_1 = c" in t for t in bad)


def test_identity_at_random_rational_points():
    """Independent of the symbolic code: at random rational (h, mu) and random rational x*, x_0, g_0, g_1, g_2 in Q^3
    and f_*, f_0, f_1, f_2 (no interpolation needed, it is an identity),
    v_1 - c ||x_0 - x*||^2 + sum lam Q + sum S_ab <u_a, u_b> = 0 in Fractions."""
    rng = random.Random(1)
    rnd = lambda: Fraction(rng.randint(-99, 99), rng.randint(1, 30))
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    sub = lambda a, b: [x - y for x, y in zip(a, b)]

    def interp(P, R, m):
        (xp, gp, fp), (xr, gr, fr) = P, R
        dx, dg = sub(xp, xr), sub(gp, gr)
        return fp - fr - dot(gr, dx) - (dot(dg, dg) + m * dot(dx, dx) - 2 * m * dot(dg, dx)) / (2 * (1 - m))

    for _ in range(25):
        hv, muv = Fraction(rng.randint(1, 1999), 1000), Fraction(rng.randint(0, 999), 1000)
        cert = worstcase.certificate_at(str(hv), str(muv))
        xs, x0 = [rnd() for _ in range(3)], [rnd() for _ in range(3)]
        g = [[rnd() for _ in range(3)] for _ in range(3)]
        f = [rnd() for _ in range(4)]
        x1 = [a - hv * b for a, b in zip(x0, g[0])]
        x2 = [a - hv * b for a, b in zip(x1, g[1])]
        pts = {"s": (xs, [0] * 3, f[0]), 0: (x0, g[0], f[1]), 1: (x1, g[1], f[2]), 2: (x2, g[2], f[3])}
        u = [sub(x0, xs)] + g
        total = (2 * f[2] - f[1] - f[3] - F(cert["c"]) * dot(u[0], u[0])
                 + sum(F(v) * interp(pts[p], pts[r], muv) for (p, r), v in cert["lam"].items())
                 + sum(F(cert["S"][a, b]) * dot(u[a], u[b]) for a in range(4) for b in range(4)))
        assert total == 0


def test_certificate_valid_exactly_above_threshold():
    """At random rational (h, mu), 1 < h < 2, 0 <= mu < 1: the certificate is valid iff q < 0 (h > h_1(kappa)), and
    then S has rank 3 (kernel spanned by the data u = (1, 1, g_1, x_2)) and c equals the data's v_1 exactly."""
    rng = random.Random(0)
    sides = {True: 0, False: 0}
    for _ in range(150):
        muv = Fraction(rng.randint(0, 999), 1000)
        hv = Fraction(rng.randint(1001, 1999), 1000) if rng.random() < 0.3 else Fraction(rng.randint(1700, 1999), 1000)
        above = q(hv, muv) < 0
        sides[above] += 1
        cert = worstcase.certificate_at(str(hv), str(muv))
        assert valid(cert) == above and F(cert["c"]) == c(hv, muv)
        if above:
            data = theorem.data_at(str(hv), str(muv))
            ok, slack, viol = counterexample.check_exact(data)
            ud = sp.Matrix([1] + [sp.Rational(gk[0]) for gk in data["g"]])
            assert ok and slack >= 0 and viol == [c(hv, muv)] and data["x0"] == ["1"]
            assert cert["S"].rank() == 3 and cert["S"] * ud == sp.zeros(4, 1)
    assert min(sides.values()) >= 30


@pytest.mark.parametrize("t", ["1751/1000", "9/5", "15/8", "19/10", "1999/1000"])
def test_threshold_points_continue_the_h1_certificate(t):
    """On q = 0 the certificate is that of theorem.py (a): c = 0, no minimizer terms, the same lam_01, lam_20, lam_21,
    and S = 0 (+) the Gram matrix that certificate.py builds from theorem.py's multipliers."""
    hv, muv = threshold_point(t)
    cert = worstcase.certificate_at(str(hv), str(muv))
    old = theorem.certificate_at(str(hv), str(muv))
    assert cert["c"] == 0 and cert["lam"]["s", 0] == cert["lam"][2, "s"] == 0
    assert {f"{p},{r}": str(v) for (p, r), v in cert["lam"].items() if "s" not in (p, r)} == old["lam"]
    assert cert["S"][0, :] == sp.zeros(1, 4) and cert["S"][1:, 1:] == certificate.gram_S(old)
    assert valid(cert)


@pytest.mark.parametrize("hv", ["1751/1000", "9/5", "19/10", "1999/1000"])
def test_mu_zero_convex_worst_case(hv):
    """mu = 0 (1-smooth convex f with a minimizer), 7/4 < h < 2: the certificate is valid and
    c = h^2 (4h - 7) / (2 (2 + h^2)), the v_1 of theorem.py's data (0.0618 at h = 1.8 is the PEP value of
    test_numerical.py)."""
    hf = Fraction(hv)
    cert = worstcase.certificate_at(hv, "0")
    assert valid(cert) and F(cert["c"]) == hf**2 * (4 * hf - 7) / (2 * (2 + hf**2))
    assert counterexample.check_exact(theorem.data_at(hv, "0"))[2] == [F(cert["c"])]


def test_stored_counterexamples_obey_the_bound():
    """Every stored counterexample (all include x* = 0) is data of a 1-smooth mu-strongly convex function, so each of
    its violations obeys v_k <= c ||x_{k-1} - x*||^2 with c = c(h, mu) > 0 (exact check, Fractions)."""
    files = sorted(p for p in (ROOT / "counterexamples").glob("*.json") if not p.name.startswith("._"))
    assert len(files) >= 11
    for path in files:
        ce = json.loads(path.read_text())
        ok, _, viol = counterexample.check_exact(ce)
        hv, muv = Fraction(ce["h"]), Fraction(ce.get("mu", "0"))
        xs = counterexample.trajectory(hv, [Fraction(a) for a in ce["x0"]], [[Fraction(a) for a in g] for g in ce["g"]])
        assert ok and q(hv, muv) < 0 and len(viol) == ce["n"]
        for k, v in enumerate(viol, start=1):
            assert 0 < v <= c(hv, muv) * sum(a * a for a in xs[k - 1]), (path.name, k)
