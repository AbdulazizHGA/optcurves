"""Exact checks (SymPy / Fractions, no floating point) of every stored certificate and counterexample."""
import json
import random
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from optcurves import certificate, counterexample, lean_gen

ROOT = Path(__file__).resolve().parents[1]
# Skip "._*" AppleDouble files (macOS writes them on exFAT/FAT volumes).
CERTS = sorted(p for p in (ROOT / "certificates").glob("*.json") if not p.name.startswith("._"))
CES = sorted(p for p in (ROOT / "counterexamples").glob("*.json") if not p.name.startswith("._"))
# Strongly convex files carry "mu": 5/7 (kappa = 7/5, where the threshold h_1 of theorem.py is exactly 9/5), and one
# counterexample with 20/21 (kappa = 1.05, three consecutive violations).
STRONG_CERTS = [p for p in CERTS if "mu" in json.loads(p.read_text())]
STRONG_CES = [p for p in CES if "mu" in json.loads(p.read_text())]
KAPPA_7_5_CES = [p for p in STRONG_CES if json.loads(p.read_text())["mu"] == "5/7"]
CONVEX_CERTS = [p for p in CERTS if p not in STRONG_CERTS]
TIGHT = ROOT / "certificates" / "n1_h9_5_mu5_7.json"


def _rationals(obj):
    """Every number in a certificate (apart from numerical metadata) must be an exact rational string."""
    skip = {"margin_numeric", "denominator_bound", "n", "N", "dim", "numeric_slack"}
    if isinstance(obj, dict):
        return all(_rationals(v) for k, v in obj.items() if k not in skip)
    if isinstance(obj, list):
        return all(_rationals(v) for v in obj)
    if isinstance(obj, str):
        try:
            sp.Rational(obj)
            return "." not in obj and "e" not in obj.lower()
        except (TypeError, ValueError, SyntaxError):
            return True  # descriptive text
    return not isinstance(obj, float)


def test_files_present():
    assert len(CERTS) >= 15 and len(CES) >= 11
    assert len(STRONG_CERTS) >= 5 and len(KAPPA_7_5_CES) >= 4 and len(STRONG_CES) >= 5


@pytest.mark.parametrize("path", CERTS, ids=lambda p: p.stem)
def test_certificate_exact(path):
    cert = json.loads(path.read_text())
    assert _rationals(cert)
    assert certificate.verify(path)


@pytest.mark.parametrize("path", CONVEX_CERTS[:2] + [STRONG_CERTS[0], TIGHT], ids=lambda p: p.stem)
def test_certificate_tampered_fails(path):
    cert = json.loads(path.read_text())
    cert.pop("sos")
    key = next(iter(cert["lam"]))
    cert["lam"][key] = str(sp.Rational(cert["lam"][key]) + sp.Rational(1, 10**6))
    assert not certificate.verify_certificate(cert)


@pytest.mark.parametrize("path", CES, ids=lambda p: p.stem)
def test_counterexample_exact(path):
    ce = json.loads(path.read_text())
    assert _rationals({k: ce[k] for k in ("h", "mu", "x0", "g", "f") if k in ce})
    ok, slack, viol = counterexample.check_exact(ce)
    assert ok and slack >= 0 and len(viol) == ce["n"] and all(v > 0 for v in viol)
    assert str(slack) == ce["min_interpolation_slack"] and [str(v) for v in viol] == ce["violations"]


def test_counterexample_tampered_fails():
    ce = json.loads((ROOT / "counterexamples" / "n2_h19_10_d2.json").read_text())
    ce["f"][2] = str(Fraction(ce["f"][2]) - Fraction(1, 10))  # breaks v_2 > 0 or interpolation
    assert not counterexample.check_exact(ce)[0]


def test_exact_brackets():
    """For every n, an exact certificate at a and an exact counterexample at b > a, b - a <= 1e-4 (convex f)."""
    certs = {}
    for p in CERTS:
        c = json.loads(p.read_text())
        if "mu" not in c:
            certs.setdefault(c["n"], []).append(Fraction(c["h"]))
    ces = {}
    for p in CES:
        c = json.loads(p.read_text())
        if "mu" not in c:
            ces.setdefault(c["n"], []).append(Fraction(c["h"]))
    for n in range(2, 7):
        a, b = max(certs[n]), min(ces[n])
        assert a < b and b - a <= Fraction(1, 10**4), (n, a, b)


def test_lean_file_up_to_date():
    assert (ROOT / "lean" / "Optcurves" / "Certificates.lean").read_text() == lean_gen.generate()


# --- mu = 5/7 (kappa = 7/5): one violation, threshold h_1(7/5) = 9/5 (proved for all kappa in theorem.py) ---

def test_strong_exact_bracket():
    """A certificate at exactly h_1(7/5) and counterexamples above it, the closest within 1e-6."""
    kappa = sp.Rational(7, 5)
    closed_form = (1 + kappa) - sp.sqrt((1 + kappa) ** 2 - (7 * kappa + 1) / 2)
    assert closed_form == sp.Rational(9, 5)
    certs = [json.loads(p.read_text()) for p in STRONG_CERTS]
    ces = [json.loads(p.read_text()) for p in KAPPA_7_5_CES]
    assert all(c["mu"] == "5/7" and c["n"] == 1 for c in certs + ces)
    a, b = max(Fraction(c["h"]) for c in certs), min(Fraction(c["h"]) for c in ces)
    assert a == Fraction(9, 5) and a < b <= a + Fraction(1, 10**6)


def test_tight_certificate_structure():
    """At h = 9/5, S = -sum lam Q has rank 2 and null vector (16, -11, 7) (so the margin is exactly 0), and the
    bound v_1 <= 0 is attained by one-dimensional data with those gradients."""
    cert = json.loads(TIGHT.read_text())
    S = certificate.gram_S(cert)
    assert S.rank() == 2 and S * sp.Matrix([16, -11, 7]) == sp.zeros(3, 1)
    assert len(cert["sos"]) == 2 and len(cert["lam"]) == 3
    assert certificate.verify_attained(cert)
    eq = {"h": cert["h"], "mu": cert["mu"], **cert["attained_by"]}
    assert counterexample.check_exact(eq)[1:] == (0, [0])  # some inequality tight, v_1 = 0
    eq["f"][1] = str(Fraction(eq["f"][1]) + Fraction(1, 10**9))
    assert counterexample.check_exact(eq)[1] < 0  # raising f_1 (v_1 > 0) breaks interpolation


def test_tight_face_is_forced():
    """With lam_10 = lam_12 = 0 the f-equations leave lam_01 = a, lam_02 = b, lam_20 = a + b - 1, lam_21 = 2 - a,
    and det S = -81 (5a + 2b - 8)^2 / 800 at h = 9/5: S >= 0 only on the line 5a + 2b = 8. The stored certificate
    is its point b = 0."""
    a, b = sp.symbols("a b")
    lam = {(0, 1): a, (0, 2): b, (2, 0): a + b - 1, (2, 1): 2 - a}
    for k in range(3):
        assert sp.expand(certificate.violation(1).get(k, 0) + sum(v * ((p == k) - (q == k))
                                                                    for (p, q), v in lam.items())) == 0
    _, pos, grad, _ = certificate.setup("9/5", 1)
    S = sp.zeros(3, 3)
    for (p, q), v in lam.items():
        S -= v * certificate.interp_quadratic(pos, grad, p, q, "5/7")
    assert sp.expand(S.det() + sp.Rational(81, 800) * (5 * a + 2 * b - 8) ** 2) == 0
    stored = json.loads(TIGHT.read_text())["lam"]
    assert stored == {f"{p},{q}": str(v.subs({a: sp.Rational(8, 5), b: 0}))
                      for (p, q), v in lam.items() if v.subs({a: sp.Rational(8, 5), b: 0}) != 0}


def test_tight_certificate_off_the_line_fails():
    """Moving lam along the f-equations off 5a + 2b = 8 keeps every linear check but makes det S < 0."""
    cert = json.loads(TIGHT.read_text())
    for k in ("sos", "kernel", "attained_by"):
        cert.pop(k)
    eps = sp.Rational(1, 10**6)
    for key, s in (("0,1", 1), ("2,0", 1), ("2,1", -1)):
        cert["lam"][key] = str(sp.Rational(cert["lam"][key]) + s * eps)
    assert certificate.gram_S(cert).det() < 0
    assert not certificate.verify_certificate(cert)


def test_strong_certificates_need_mu():
    """The same multipliers are no certificate for convex f (one violation is possible there for h > 7/4):
    with mu = 0, S = -sum lam Q is not even positive semidefinite."""
    for p in STRONG_CERTS:
        cert = json.loads(p.read_text())
        cert.pop("sos")
        cert["mu"] = "0"
        assert not certificate.verify_certificate(cert)
        res = certificate._ldl(certificate.gram_S(cert), psd=True)
        assert res is None or min(res[1]) < 0


@pytest.mark.parametrize("path", STRONG_CERTS, ids=lambda p: p.stem)
def test_strong_identity_on_random_data(path):
    """Independent of the Gram-matrix code: for random rational x_0, g_a in Q^3 and values f_a (no interpolation
    assumed), sum w_k v_k + sum lam_pq Q_pq + sum_k d_k ||sum_a coeffs_a g_a||^2 = 0 exactly. The left side is
    linear in the Gram entries of (x_0, g_0, ..., g_N) and in f, and the samples span all those coordinates, so
    this proves the identity."""
    cert = json.loads(path.read_text())
    h, mu, N = Fraction(cert["h"]), Fraction(cert["mu"]), cert["N"]
    rng = random.Random(0)
    rnd = lambda: Fraction(rng.randint(-60, 60), rng.randint(1, 30))
    dot = lambda u, v: sum(s * t for s, t in zip(u, v))
    sub = lambda u, v: [s - t for s, t in zip(u, v)]
    samples = []
    for _ in range(20):
        x0 = [rnd() for _ in range(3)]
        g = [[rnd() for _ in range(3)] for _ in range(N + 1)]
        f = [rnd() for _ in range(N + 1)]
        x = counterexample.trajectory(h, x0, g)

        def Q(p, q):
            dx, dg = sub(x[p], x[q]), sub(g[p], g[q])
            return (f[p] - f[q] - dot(g[q], dx)
                    - (dot(dg, dg) + mu * dot(dx, dx) - 2 * mu * dot(dg, dx)) / (2 * (1 - mu)))

        total = sum(Fraction(w) * (2 * f[k] - f[k - 1] - f[k + 1]) for k, w in enumerate(cert["w"], 1))
        total += sum(Fraction(v) * Q(*map(int, key.split(","))) for key, v in cert["lam"].items())
        for term in cert["sos"]:
            vec = [sum(Fraction(c) * g[a][i] for a, c in enumerate(term["coeffs"])) for i in range(3)]
            total += Fraction(term["d"]) * dot(vec, vec)
        assert total == 0
        vecs = [x0] + g
        samples.append([dot(u, v) for i, u in enumerate(vecs) for v in vecs[i:]] + f)
    assert sp.Matrix(samples).rank() == len(samples[0])  # (N + 2)(N + 3)/2 Gram entries + N + 1 values


@pytest.mark.parametrize("path", KAPPA_7_5_CES, ids=lambda p: p.stem)
def test_strong_counterexample_one_dimensional(path):
    """The kappa = 7/5 examples are functions of one variable; being 5/7-strongly convex data, they also pass the
    convex (mu = 0) check."""
    ce = json.loads(path.read_text())
    assert ce["dim"] == 1 and len(ce["x0"]) == 1 and all(len(gk) == 1 for gk in ce["g"])
    assert counterexample.check_exact({**ce, "mu": "0"})[0]


def test_strong_counterexample_tampered_fails():
    ce = json.loads((ROOT / "counterexamples" / "n1_h1801_1000_mu5_7_d1.json").read_text())
    ce["f"][1] = str(Fraction(ce["f"][1]) - Fraction(1, 10**4))  # v_1 ~ 6e-5 drops below 0
    assert not counterexample.check_exact(ce)[0]
