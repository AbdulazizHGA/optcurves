"""Exact checks (SymPy / Fractions, no floating point) of every stored certificate and counterexample."""
import json
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from optcurves import certificate, counterexample, lean_gen

ROOT = Path(__file__).resolve().parents[1]
# Skip "._*" AppleDouble files (macOS writes them on exFAT/FAT volumes).
CERTS = sorted(p for p in (ROOT / "certificates").glob("*.json") if not p.name.startswith("._"))
CES = sorted(p for p in (ROOT / "counterexamples").glob("*.json") if not p.name.startswith("._"))


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
    assert len(CERTS) >= 10 and len(CES) >= 7


@pytest.mark.parametrize("path", CERTS, ids=lambda p: p.stem)
def test_certificate_exact(path):
    cert = json.loads(path.read_text())
    assert _rationals(cert)
    assert certificate.verify(path)


@pytest.mark.parametrize("path", CERTS[:2], ids=lambda p: p.stem)
def test_certificate_tampered_fails(path):
    cert = json.loads(path.read_text())
    cert.pop("sos")
    key = next(iter(cert["lam"]))
    cert["lam"][key] = str(sp.Rational(cert["lam"][key]) + sp.Rational(1, 10**6))
    assert not certificate.verify_certificate(cert)


@pytest.mark.parametrize("path", CES, ids=lambda p: p.stem)
def test_counterexample_exact(path):
    ce = json.loads(path.read_text())
    assert _rationals({k: ce[k] for k in ("h", "x0", "g", "f")})
    ok, slack, viol = counterexample.check_exact(ce)
    assert ok and slack >= 0 and len(viol) == ce["n"] and all(v > 0 for v in viol)
    assert str(slack) == ce["min_interpolation_slack"] and [str(v) for v in viol] == ce["violations"]


def test_counterexample_tampered_fails():
    ce = json.loads((ROOT / "counterexamples" / "n2_h19_10_d2.json").read_text())
    ce["f"][2] = str(Fraction(ce["f"][2]) - Fraction(1, 10))  # breaks v_2 > 0 or interpolation
    assert not counterexample.check_exact(ce)[0]


def test_exact_brackets():
    """For every n, an exact certificate at a and an exact counterexample at b > a, b - a <= 1e-4."""
    certs = {}
    for p in CERTS:
        c = json.loads(p.read_text())
        certs.setdefault(c["n"], []).append(Fraction(c["h"]))
    ces = {}
    for p in CES:
        c = json.loads(p.read_text())
        ces.setdefault(c["n"], []).append(Fraction(c["h"]))
    for n in range(2, 7):
        a, b = max(certs[n]), min(ces[n])
        assert a < b and b - a <= Fraction(1, 10**4), (n, a, b)


def test_lean_file_up_to_date():
    assert (ROOT / "lean" / "Optcurves" / "Certificates.lean").read_text() == lean_gen.generate()
