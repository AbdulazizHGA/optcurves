"""The exact one-violation threshold h_1(kappa) for strongly convex f (theorem.py): the symbolic proof, its link to
the Gram-matrix and interpolation code, and exact instances on both sides of the threshold."""
import json
import random
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from optcurves import certificate, counterexample, theorem

ROOT = Path(__file__).resolve().parents[1]


def q(hv, muv):
    return 2 * muv * hv**2 - 4 * (1 + muv) * hv + muv + 7


def threshold_point(t):
    """(h, mu) with q(h, mu) = 0 and h = t: mu = (4t - 7) / (2t^2 - 4t + 1), in (0, 1) for 7/4 < t < 2."""
    t = Fraction(t)
    return t, (4 * t - 7) / (2 * t**2 - 4 * t + 1)


@pytest.fixture(scope="module")
def checks():
    return theorem.verify()


def test_every_step_of_the_proof(checks):
    failed = [text for text, ok in checks if not ok]
    assert not failed and len(checks) >= 20, failed


def test_proof_check_is_not_vacuous(monkeypatch):
    """A wrong Gram matrix or a wrong multiplier is caught."""
    monkeypatch.setattr(theorem, "W_MATRIX", theorem.W_MATRIX + sp.diag(0, 0, sp.Rational(1, 10**6)))
    bad = {text for text, ok in theorem.verify() if not ok}
    assert any("equals W" in t for t in bad) and any("= 0 for symbolic" in t for t in bad)
    monkeypatch.undo()
    lam = dict(theorem.LAM)
    lam[0, 1] += sp.Rational(1, 10**6)
    monkeypatch.setattr(theorem, "LAM", lam)
    assert any("coefficients of v_1" in t for t, ok in theorem.verify() if not ok)


@pytest.mark.parametrize("hv, muv", [("1/3", "1/2"), ("7/4", "5/7"), ("9/5", "5/7"), ("19/10", "9/10"),
                                     ("1999/1000", "1/1000")])
def test_symbolic_gram_matches_certificate_code(hv, muv):
    """W / (2 (1 - mu)(2 - h mu)) is the S that certificate.py builds from the same multipliers."""
    sub = {theorem.h: sp.Rational(hv), theorem.mu: sp.Rational(muv)}
    S = (theorem.W_MATRIX / (2 * (1 - theorem.mu) * (2 - theorem.h * theorem.mu))).subs(sub)
    assert S == certificate.gram_S(theorem.certificate_at(hv, muv))


def test_both_sides_at_random_rational_points():
    """Exact instances with the repo's own checkers: the certificate of (a) is valid iff q >= 0 (h <= h_1), and the
    data of (b) satisfy all 12 interpolation inequalities (checked for h >= 1, where they are claimed), with
    v_1 = -h^2 q / (2D) > 0 iff q < 0."""
    rng = random.Random(0)
    sides = {True: 0, False: 0}
    for _ in range(150):
        muv = Fraction(rng.randint(1, 999), 1000)
        hv = Fraction(rng.randint(1, 1999), 1000) if rng.random() < 0.3 else Fraction(rng.randint(1700, 1999), 1000)
        below = q(hv, muv) >= 0
        sides[below] += 1
        assert certificate.verify_certificate(theorem.certificate_at(str(hv), str(muv))) == below
        if hv >= 1:
            ok, slack, viol = counterexample.check_exact(theorem.data_at(str(hv), str(muv)))
            assert slack >= 0 and viol == [-hv**2 * q(hv, muv) / (2 * (2 + hv**2 * (1 - muv)))]
            assert ok == (not below)
    assert min(sides.values()) >= 30


@pytest.mark.parametrize("t", ["1751/1000", "9/5", "15/8", "19/10", "1999/1000"])
def test_threshold_points_are_tight(t):
    """On the curve q = 0 (rational points h = t): h_1(1/mu) = t exactly, S has rank 2 with the data's gradients as
    null vector, the data have v_1 = 0, and 10^-6 above t they have v_1 > 0 while the certificate fails."""
    hv, muv = threshold_point(t)
    assert 0 < muv < 1 and theorem.h1(sp.Rational(1) / sp.Rational(muv)) == sp.Rational(hv)
    cert = theorem.certificate_at(str(hv), str(muv))
    data = theorem.data_at(str(hv), str(muv))
    cert["kernel"] = [gk[0] for gk in data["g"]]
    assert certificate.verify_certificate(cert) and len(cert["sos"]) == 2
    assert certificate.gram_S(cert).rank() == 2
    assert counterexample.check_exact(data)[1:] == (0, [0])
    up = hv + Fraction(1, 10**6)
    assert counterexample.check_exact(theorem.data_at(str(up), str(muv)))[0]
    assert not certificate.verify_certificate(theorem.certificate_at(str(up), str(muv)))


def test_stored_kappa_7_5_certificate_is_an_instance():
    """The closed-form tight certificate at kappa = 7/5, h = 9/5 (certificates/n1_h9_5_mu5_7.json) and its equality
    data are the theorem's certificate and data at (h, mu) = (9/5, 5/7)."""
    stored = json.loads((ROOT / "certificates" / "n1_h9_5_mu5_7.json").read_text())
    assert stored["lam"] == theorem.certificate_at("9/5", "5/7")["lam"]
    data = theorem.data_at("9/5", "5/7")
    assert stored["attained_by"] == {k: data[k] for k in ("x0", "g", "f")}
    assert theorem.h1(sp.Rational(7, 5)) == sp.Rational(9, 5)


@pytest.mark.parametrize("hv", ["1/2", "3/2", "17/10", "1749/1000"])
def test_mu_zero_certificate_recovers_convex_threshold(hv):
    """mu = 0, h < 7/4: lam = (3/2, 1/2, 1/2) is a valid convex certificate (facial reduction path of
    certificate.py), as in Barzilai, Shamir and Zamani."""
    cert = theorem.certificate_at(hv, "0")
    assert "mu" not in cert and cert["lam"] == {"0,1": "3/2", "2,0": "1/2", "2,1": "1/2"}
    assert certificate.verify_certificate(cert)


@pytest.mark.parametrize("hv", ["7/4", "1751/1000", "9/5", "19/10"])
def test_mu_zero_data_recover_convex_threshold(hv):
    ok, slack, viol = counterexample.check_exact(theorem.data_at(hv, "0"))
    hf = Fraction(hv)
    assert slack >= 0 and viol == [-hf**2 * (7 - 4 * hf) / (2 * (2 + hf**2))] and ok == (hf > Fraction(7, 4))


def test_two_parameter_family_degenerates_on_q():
    """With lam_10 = lam_12 = 0 the f-equations leave lam_01 = a, lam_02 = b, lam_20 = a + b - 1, lam_21 = 2 - a, and
    8 (1 - mu)^2 det S = (a, b, 1) M (a, b, 1)^T with det M = h^6 mu (2 - h mu)^2 q^2 (how q was found); the theorem's
    certificate is the member b = 0, a = lam_01, where det S = h^2 mu q / (8 (1 - mu)^2)."""
    h, mu = theorem.h, theorem.mu
    a, b, c = sp.symbols("a b c")
    lam = {(0, 1): a, (0, 2): b, (2, 0): a + b - 1, (2, 1): 2 - a}
    for k, coeff in ((0, -1), (1, 2), (2, -1)):
        assert sp.expand(coeff + sum(v * ((p == k) - (r == k)) for (p, r), v in lam.items())) == 0
    S = theorem.gram_from_multipliers(lam)
    P = sp.expand(sp.cancel(8 * (1 - mu) ** 2 * S.det()))
    homog = sp.expand(P.subs({a: a / c, b: b / c}) * c**2)
    M = sp.Matrix(3, 3, lambda i, j: sp.diff(homog, (a, b, c)[i], (a, b, c)[j]) / 2)
    assert sp.expand(M.det() - h**6 * mu * (2 - h * mu) ** 2 * theorem.q**2) == 0
    member = S.det().subs({a: theorem.LAM[0, 1], b: 0})
    assert sp.cancel(member - h**2 * mu * theorem.q / (8 * (1 - mu) ** 2)) == 0
