"""Exact rational certificates that gradient descent cannot break curve convexity n times in a row.

Setting (L = 1, every 1-smooth mu-strongly convex f, 0 <= mu < 1, no minimizer assumed): x_{k+1} = x_k - h g_k,
k = 0..n, with g_k = grad f(x_k) and f_k = f(x_k). Curve violations v_k = 2 f_k - f_{k-1} - f_{k+1}, k = 1..n.

Claim: w_1 v_1 + ... + w_n v_n <= 0 for explicit rational weights w_k >= 0 with sum 1, hence
not (v_1 > 0 and ... and v_n > 0). (By minimax this weighted form is exactly the dual of the PEP
max_f min_k v_k, so nothing is lost against a case split on the smallest v_k.) Certificate: the identity

    sum_k w_k v_k + sum_{p != q} lam_pq Q_pq + sum_{a,b} S_ab <g_a, g_b> = 0,

where Q_pq >= 0 are the interpolation inequalities of 1-smooth mu-strongly convex functions (Taylor, Hendrickx &
Glineur, Math. Program. 2017, Thm. 4; mu = 0 gives the convex ones)

    Q_pq = f_p - f_q - <g_q, x_p - x_q>
           - (||g_p - g_q||^2 + mu ||x_p - x_q||^2 - 2 mu <g_p - g_q, x_p - x_q>) / (2 (1 - mu)),

lam >= 0, and S is positive semidefinite. Q_pq involves positions only through x_p - x_q = -h (g_q + ... + g_{p-1})
(p > q), so x_0 drops out by translation invariance and no minimizer is used.

mu = 0. Linear functions make every term zero with all g_a equal, which forces S 1 = 0 (facial reduction). We
impose it and certify instead that M = S[:n+1, :n+1] (the form in the coordinates c_a = g_a - g_{n+1}) is positive
definite, by an exact LDL^T:

    sum S_ab <g_a, g_b> = sum_k d_k || sum_a Lo_ka (g_a - g_{n+1}) ||^2,   d_k > 0.

mu > 0. Linear functions are not in the class and nothing forces a face: the quadratics (mu/2)||x||^2 and
(1/2)||x||^2 make every Q_pq vanish, but their v_k are < 0. Below the threshold S can be positive definite, and the
LDL^T is of S itself, in the coordinates g_0..g_{n+1}. At the threshold S is singular: the LDL^T may then have zero
pivots, allowed only when the rest of their column vanishes, so that S = Lo diag(d) Lo^T with d >= 0 still proves
S >= 0 (only the squares with d_k > 0 are stored). TIGHT gives such a certificate in closed form.

The multipliers come from a numerical SDP (cvxpy + Clarabel), are rounded to rationals, and the linear
equalities are then repaired exactly; every check below is in exact rational arithmetic (SymPy).
"""
import itertools
import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import sympy as sp

from .counterexample import check_exact

ROOT = Path(__file__).resolve().parents[2]


def setup(h, n):
    """Points 0..N (N = n+1), positions in the basis g_0..g_N with x_0 = 0 (translation invariance)."""
    N = n + 1
    h = sp.Rational(h)
    pos, cur = [], [sp.Integer(0)] * (N + 1)
    for k in range(N + 1):
        pos.append(cur[:])
        cur = [c - (h if a == k else 0) for a, c in enumerate(cur)]
    grad = [[sp.Integer(int(a == k)) for a in range(N + 1)] for k in range(N + 1)]
    pairs = list(itertools.permutations(range(N + 1), 2))
    return N, pos, grad, pairs


def _sym(u, v):
    M = sp.Matrix(len(u), len(v), lambda r, c: u[r] * v[c])
    return (M + M.T) / 2


def interp_quadratic(pos, grad, p, q, mu=0):
    """Matrix of the quadratic part of Q_pq:
    -<g_q, x_p - x_q> - (||g_p - g_q||^2 + mu ||x_p - x_q||^2 - 2 mu <g_p - g_q, x_p - x_q>) / (2 (1 - mu))."""
    dx = [a - b for a, b in zip(pos[p], pos[q])]
    dg = [a - b for a, b in zip(grad[p], grad[q])]
    mu = sp.Rational(mu)
    return -_sym(grad[q], dx) - (_sym(dg, dg) + mu * _sym(dx, dx) - 2 * mu * _sym(dg, dx)) / (2 * (1 - mu))


def violation(k):
    """f-coefficients of v_k = 2 f_k - f_{k-1} - f_{k+1}."""
    return {k: 2, k - 1: -1, k + 1: -1}


def _ldl(M, psd=False):
    """Exact LDL^T (unit lower-triangular Lo, diagonal d) of a symmetric rational matrix, or None.
    psd=True also allows a zero pivot whose column below it vanishes in the Schur complement."""
    n = M.rows
    Lo, d = sp.eye(n), [sp.Integer(0)] * n
    for j in range(n):
        d[j] = M[j, j] - sum(Lo[j, k] ** 2 * d[k] for k in range(j))
        if d[j] == 0 and not psd:
            return None
        for i in range(j + 1, n):
            r = M[i, j] - sum(Lo[i, k] * Lo[j, k] * d[k] for k in range(j))
            if d[j] != 0:
                Lo[i, j] = r / d[j]
            elif r != 0:
                return None
    return Lo, d


def _system(h, n, mu=0):
    """Exact linear system A z = b for z = (lam_pq for all pairs, w_1..w_n), and the Q matrices.
    For mu = 0 it includes the face S 1 = 0."""
    N, pos, grad, pairs = setup(h, n)
    Q = [interp_quadratic(pos, grad, p, q, mu) for p, q in pairs]
    ones = sp.ones(N + 1, 1)
    rows, rhs = [], []
    for k in range(N + 1):  # f_k coefficient of the identity vanishes
        rows.append([sp.Integer((p == k) - (q == k)) for p, q in pairs]
                    + [sp.Integer(violation(j).get(k, 0)) for j in range(1, n + 1)])
        rhs.append(0)
    rows.append([sp.Integer(0)] * len(pairs) + [sp.Integer(1)] * n)  # weights sum to 1
    rhs.append(1)
    if sp.Rational(mu) == 0:
        SQ1 = [-(Qm * ones) for Qm in Q]  # S 1 = 0 with S = -sum lam Q
        for r in range(N + 1):
            rows.append([SQ1[m][r] for m in range(len(pairs))] + [sp.Integer(0)] * n)
            rhs.append(0)
    return N, pairs, Q, sp.Matrix(rows), sp.Matrix(rhs)


def _numeric(N, Q, A, b, n_lam, n_w, cap=50.0, reduced=True):
    """Maximize the smallest eigenvalue of S[:N, :N] (reduced, mu = 0) or of S (mu > 0)."""
    import cvxpy as cp
    z = cp.Variable(n_lam + n_w, nonneg=True)
    t = cp.Variable()
    Qf = [np.array(Qm.tolist(), dtype=float) for Qm in Q]
    S = -sum(z[m] * Qf[m] for m in range(n_lam))
    M = S[:N, :N] if reduced else S
    cons = [M - t * np.eye(M.shape[0]) >> 0, cp.sum(z) <= cap,
            np.array(A.tolist(), dtype=float) @ z == np.array(b.tolist(), dtype=float).ravel()]
    cp.Problem(cp.Maximize(t), cons).solve(solver="CLARABEL")
    return (None, None) if t.value is None else (z.value, float(t.value))


def _cert(h, n, mu, w, lam):
    """Certificate record (without the sum of squares, which verify_certificate adds)."""
    mu = sp.Rational(mu)
    cert = {"h": str(sp.Rational(h)), "n": n, "N": n + 1}
    if mu != 0:
        cert["mu"] = str(mu)
    cert["claim"] = " + ".join(f"w_{k} v_{k}" for k in range(1, n + 1)) + " <= 0"
    cert["class"] = ("1-smooth convex functions on any real inner-product space (no minimizer assumed)" if mu == 0
                     else f"1-smooth {mu}-strongly convex functions (kappa = {1 / mu}) on any real inner-product space")
    cert["w"] = [str(v) for v in w]
    cert["lam"] = {f"{p},{q}": str(v) for (p, q), v in lam.items() if v != 0}
    return cert


def find_certificate(h, n, mu=0, denominators=(100, 1000, 10**4, 10**5, 10**6, 10**7)):
    """Exact certificate for sum_k w_k v_k <= 0 at step h over 1-smooth mu-strongly convex f, or None if
    rounding fails (or the numerical margin is not positive, as at a threshold)."""
    mu = sp.Rational(mu)
    N, pairs, Q, A, b = _system(h, n, mu)
    zf, margin = _numeric(N, Q, A, b, len(pairs), n, reduced=mu == 0)
    if zf is None or margin < 1e-10:
        return None
    nvar = len(zf)
    for D in denominators:
        zr = [sp.Rational(F(float(x)).limit_denominator(D)) if x > 1e-9 else sp.Integer(0) for x in zf]
        order = sorted(range(nvar), key=lambda m: -zf[m])  # large multipliers as pivots
        _, piv = A[:, order].rref()
        pivots = [order[c] for c in piv]
        rest = [m for m in range(nvar) if m not in pivots]
        rhs = b - A[:, rest] * sp.Matrix([zr[m] for m in rest]) if rest else b
        try:
            sol, params = A[:, pivots].gauss_jordan_solve(rhs)
        except ValueError:
            continue
        if params.shape[0]:
            continue
        if not all(v.is_Rational for v in sol):
            continue
        z = list(zr)
        for k, m in enumerate(pivots):
            z[m] = sol[k]
        cert = _cert(h, n, mu, z[len(pairs):], dict(zip(pairs, z)))
        if verify_certificate(cert):
            cert["margin_numeric"] = margin
            cert["denominator_bound"] = D
            return cert
    return None


def gram_S(cert):
    """S = -sum lam_pq Q_pq, exactly."""
    N, pos, grad, _ = setup(cert["h"], cert["n"])
    S = sp.zeros(N + 1, N + 1)
    for key, val in cert["lam"].items():
        p, q = map(int, key.split(","))
        S -= sp.Rational(val) * interp_quadratic(pos, grad, p, q, cert.get("mu", 0))
    return S


def verify_certificate(cert):
    """Exact check; also stores the LDL^T sum of squares in cert["sos"]. Returns bool.
    An optional cert["kernel"] must be a nonzero vector u with S u = 0 (a tight certificate)."""
    n, N = cert["n"], cert["N"]
    mu = sp.Rational(cert.get("mu", 0))
    lam = {tuple(map(int, k.split(","))): sp.Rational(v) for k, v in cert["lam"].items()}
    w = [sp.Rational(v) for v in cert["w"]]
    if not 0 <= mu < 1 or any(v < 0 for v in [*lam.values(), *w]) or sum(w) != 1:
        return False
    for k in range(N + 1):  # function values cancel
        c = sum(w[j - 1] * violation(j).get(k, 0) for j in range(1, n + 1))
        c += sum(v * ((p == k) - (q == k)) for (p, q), v in lam.items())
        if c != 0:
            return False
    S = gram_S(cert)
    if "kernel" in cert:
        u = sp.Matrix([sp.Rational(c) for c in cert["kernel"]])
        if u.rows != N + 1 or u.is_zero_matrix or not (S * u).is_zero_matrix:
            return False
    if mu > 0:
        # No face: S = Lo diag(d) Lo^T in the coordinates g_0..g_N, d >= 0.
        res = _ldl(S, psd=True)
        if res is None or any(dk < 0 for dk in res[1]):
            return False
        Lo, d = res
        if Lo * sp.diag(*d) * Lo.T != S:
            return False
        cert["sos"] = [{"d": str(d[k]), "coeffs": [str(Lo[a, k]) for a in range(N + 1)]}
                       for k in range(N + 1) if d[k] != 0]
        return True
    if S * sp.ones(N + 1, 1) != sp.zeros(N + 1, 1):
        return False
    res = _ldl(S[:N, :N])
    if res is None or any(dk <= 0 for dk in res[1]):
        return False
    Lo, d = res
    # Sum of squares in c_a = g_a - g_N: S = E Lo diag(d) Lo^T E^T. Check that it reconstructs S.
    E = sp.zeros(N + 1, N)
    for a in range(N):
        E[a, a], E[N, a] = 1, -1
    if E * Lo * sp.diag(*d) * Lo.T * E.T != S:
        return False
    cert["sos"] = [{"d": str(d[k]), "coeffs": [str(Lo[a, k]) for a in range(N)]} for k in range(N)]
    return True


def verify_attained(cert):
    """Tightness of a certificate: cert["attained_by"] is data (minimizer x* = 0, g* = 0, f* = 0 included) that
    satisfies every interpolation inequality of the class exactly, has sum_k w_k v_k = 0, and has nonzero
    gradients g_a = kernel_a e for one vector e (the null vector of S)."""
    data = {"h": cert["h"], "mu": cert.get("mu", "0"), **cert["attained_by"]}
    _, slack, viol = check_exact(data)
    g = [[F(c) for c in gk] for gk in data["g"]]
    u = [F(c) for c in cert["kernel"]]
    j = next(a for a, c in enumerate(u) if c != 0)
    e = [c / u[j] for c in g[j]]
    return (slack >= 0 and len(viol) == cert["n"] and sum(F(wk) * vk for wk, vk in zip(cert["w"], viol)) == 0
            and any(e) and all(gk == [ua * c for c in e] for gk, ua in zip(g, u)))


def verify(cert_file):
    cert = json.loads(Path(cert_file).read_text())
    stored = cert.pop("sos")
    if not (verify_certificate(cert) and cert["sos"] == stored):
        return False
    return "attained_by" not in cert or verify_attained(cert)


def filename(h, n, mu="0"):
    """File name from the target strings (not from the reduced fraction in the file)."""
    return f"n{n}_h{h.replace('/', '_')}" + ("" if mu == "0" else f"_mu{mu.replace('/', '_')}") + ".json"


# (h, n): the first per n is the one just below the numerical threshold h_n (tightest exact bound).
TARGETS = [("9317/5000", 2), ("3/2", 2), ("9/5", 2), ("37/20", 2), ("93/50", 2),
           ("19/10", 3), ("47/25", 3), ("38391/20000", 4), ("38643/20000", 5), ("19411/10000", 6)]
# (h, n, mu): one violation for mu = 5/7 (kappa = 7/5), 9/5 - 10^-k below h_1(7/5) = 9/5 (theorem.py).
STRONG_TARGETS = [("1799/1000", 1, "5/7"), ("17999/10000", 1, "5/7"), ("179999/100000", 1, "5/7"),
                  ("1799999/1000000", 1, "5/7")]
# Closed-form certificate at h = 9/5 itself, where every certificate is tight (margin 0). With lam_10 = lam_12 = 0
# the f-equations leave lam_01 = a, lam_02 = b, lam_20 = a + b - 1, lam_21 = 2 - a, and then
# det S = -81 (5a + 2b - 8)^2 / 800, so S >= 0 forces 5a + 2b = 8 (checked in the tests); b = 0 uses the fewest
# inequalities. S has rank 2 and null vector (16, -11, 7): the gradients of the one-dimensional data in
# "attained_by", on which v_1 = 0. So v_1 <= 0 is sharp at h = 9/5.
TIGHT = [{"h": "9/5", "n": 1, "mu": "5/7", "w": ["1"], "lam": {(0, 1): "8/5", (2, 0): "3/5", (2, 1): "2/5"},
          "kernel": ["16", "-11", "7"],
          "attained_by": {"x0": ["1"], "g": [["1"], ["-11/16"], ["7/16"]], "f": ["1/2", "305/1024", "49/512"]}}]


def tight_certificate(spec):
    cert = _cert(spec["h"], spec["n"], spec["mu"], spec["w"], spec["lam"])
    cert["kernel"], cert["attained_by"] = spec["kernel"], spec["attained_by"]
    return cert if verify_certificate(cert) and verify_attained(cert) else None


def main():
    """Writes every certificate that is found. Exits with an error if any target fails, so that a stale file left
    from an earlier run is never mistaken for a regenerated one (reproduce.sh stops there)."""
    out = ROOT / "certificates"
    out.mkdir(exist_ok=True)
    failed = []
    for h, n, mu in [(h, n, "0") for h, n in TARGETS] + STRONG_TARGETS:
        cert = find_certificate(h, n, mu)
        status = (f"ok, w = {cert['w']}, denominators <= {cert['denominator_bound']}, "
                  f"numeric margin {cert['margin_numeric']:.2e}") if cert else "FAILED"
        label = "" if mu == "0" else f", mu = {mu}"
        print(f"h = {h} ({float(sp.Rational(h)):.4f}), n = {n}{label}: {status}", flush=True)
        if cert:
            (out / filename(h, n, mu)).write_text(json.dumps(cert, indent=1) + "\n")
        else:
            failed.append(filename(h, n, mu))
    for spec in TIGHT:
        cert = tight_certificate(spec)
        print(f"h = {spec['h']}, n = {spec['n']}, mu = {spec['mu']}: "
              + (f"tight, lam = {cert['lam']}, kernel {cert['kernel']}" if cert else "FAILED"), flush=True)
        if cert:
            (out / filename(spec["h"], spec["n"], spec["mu"])).write_text(json.dumps(cert, indent=1) + "\n")
        else:
            failed.append(filename(spec["h"], spec["n"], spec["mu"]))
    if failed:
        raise SystemExit(f"{len(failed)} certificate(s) FAILED, not regenerated: {', '.join(failed)}")


if __name__ == "__main__":
    main()
