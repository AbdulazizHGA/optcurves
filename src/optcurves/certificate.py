"""Exact rational certificates that gradient descent cannot break curve convexity n times in a row.

Setting (L = 1, every 1-smooth convex f, no minimizer assumed): x_{k+1} = x_k - h g_k, k = 0..n,
with g_k = grad f(x_k) and f_k = f(x_k). Curve violations v_k = 2 f_k - f_{k-1} - f_{k+1}, k = 1..n.

Claim: w_1 v_1 + ... + w_n v_n <= 0 for explicit rational weights w_k >= 0 with sum 1, hence
not (v_1 > 0 and ... and v_n > 0). (By minimax this weighted form is exactly the dual of the PEP
max_f min_k v_k, so nothing is lost against a case split on the smallest v_k.) Certificate: the identity

    sum_k w_k v_k + sum_{p != q} lam_pq Q_pq + sum_{a,b} S_ab <g_a, g_b> = 0,

where Q_pq = f_p - f_q - <g_q, x_p - x_q> - 1/2 ||g_p - g_q||^2 >= 0 are the interpolation inequalities of
1-smooth convex functions, lam >= 0, and S is positive semidefinite. Linear functions make every term
zero with all g_a equal, which forces S 1 = 0 (facial reduction). We impose it and certify instead that
M = S[:n+1, :n+1] (the form in the coordinates c_a = g_a - g_{n+1}) is positive definite, by an exact LDL^T:

    sum S_ab <g_a, g_b> = sum_k d_k || sum_a Lo_ka (g_a - g_{n+1}) ||^2,   d_k > 0.

The multipliers come from a numerical SDP (cvxpy + Clarabel), are rounded to rationals, and the linear
equalities are then repaired exactly; every check below is in exact rational arithmetic (SymPy).
"""
import itertools
import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import sympy as sp

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


def interp_quadratic(pos, grad, p, q):
    """Matrix of the quadratic part of Q_pq: -<g_q, x_p - x_q> - 1/2 ||g_p - g_q||^2."""
    dx = [a - b for a, b in zip(pos[p], pos[q])]
    dg = [a - b for a, b in zip(grad[p], grad[q])]
    return -_sym(grad[q], dx) - _sym(dg, dg) / 2


def violation(k):
    """f-coefficients of v_k = 2 f_k - f_{k-1} - f_{k+1}."""
    return {k: 2, k - 1: -1, k + 1: -1}


def _ldl(M):
    """Exact LDL^T (unit lower-triangular Lo, diagonal d) of a symmetric rational matrix, or None."""
    n = M.rows
    Lo, d = sp.eye(n), [sp.Integer(0)] * n
    for j in range(n):
        d[j] = M[j, j] - sum(Lo[j, k] ** 2 * d[k] for k in range(j))
        if d[j] == 0:
            return None
        for i in range(j + 1, n):
            Lo[i, j] = (M[i, j] - sum(Lo[i, k] * Lo[j, k] * d[k] for k in range(j))) / d[j]
    return Lo, d


def _system(h, n):
    """Exact linear system A z = b for z = (lam_pq for all pairs, w_1..w_n), and the Q matrices."""
    N, pos, grad, pairs = setup(h, n)
    Q = [interp_quadratic(pos, grad, p, q) for p, q in pairs]
    ones = sp.ones(N + 1, 1)
    rows, rhs = [], []
    for k in range(N + 1):  # f_k coefficient of the identity vanishes
        rows.append([sp.Integer((p == k) - (q == k)) for p, q in pairs]
                    + [sp.Integer(violation(j).get(k, 0)) for j in range(1, n + 1)])
        rhs.append(0)
    rows.append([sp.Integer(0)] * len(pairs) + [sp.Integer(1)] * n)  # weights sum to 1
    rhs.append(1)
    SQ1 = [-(Qm * ones) for Qm in Q]  # S 1 = 0 with S = -sum lam Q
    for r in range(N + 1):
        rows.append([SQ1[m][r] for m in range(len(pairs))] + [sp.Integer(0)] * n)
        rhs.append(0)
    return N, pairs, Q, sp.Matrix(rows), sp.Matrix(rhs)


def _numeric(N, Q, A, b, n_lam, n_w, cap=50.0):
    import cvxpy as cp
    z = cp.Variable(n_lam + n_w, nonneg=True)
    t = cp.Variable()
    Qf = [np.array(Qm.tolist(), dtype=float) for Qm in Q]
    S = -sum(z[m] * Qf[m] for m in range(n_lam))
    M = S[:N, :N]
    cons = [M - t * np.eye(N) >> 0, cp.sum(z) <= cap,
            np.array(A.tolist(), dtype=float) @ z == np.array(b.tolist(), dtype=float).ravel()]
    cp.Problem(cp.Maximize(t), cons).solve(solver="CLARABEL")
    return (None, None) if t.value is None else (z.value, float(t.value))


def find_certificate(h, n, denominators=(100, 1000, 10**4, 10**5, 10**6, 10**7)):
    """Exact certificate for sum_k w_k v_k <= 0 at step h, or None if rounding fails."""
    N, pairs, Q, A, b = _system(h, n)
    zf, margin = _numeric(N, Q, A, b, len(pairs), n)
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
        cert = {"h": str(sp.Rational(h)), "n": n, "N": N,
                "claim": " + ".join(f"w_{k} v_{k}" for k in range(1, n + 1)) + " <= 0",
                "class": "1-smooth convex functions on any real inner-product space (no minimizer assumed)",
                "w": [str(z[len(pairs) + k]) for k in range(n)],
                "lam": {f"{p},{q}": str(z[m]) for m, (p, q) in enumerate(pairs) if z[m] != 0}}
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
        S -= sp.Rational(val) * interp_quadratic(pos, grad, p, q)
    return S


def verify_certificate(cert):
    """Exact check; also stores the LDL^T sum of squares in cert["sos"]. Returns bool."""
    n, N = cert["n"], cert["N"]
    lam = {tuple(map(int, k.split(","))): sp.Rational(v) for k, v in cert["lam"].items()}
    w = [sp.Rational(v) for v in cert["w"]]
    if any(v < 0 for v in [*lam.values(), *w]) or sum(w) != 1:
        return False
    for k in range(N + 1):  # function values cancel
        c = sum(w[j - 1] * violation(j).get(k, 0) for j in range(1, n + 1))
        c += sum(v * ((p == k) - (q == k)) for (p, q), v in lam.items())
        if c != 0:
            return False
    S = gram_S(cert)
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


def verify(cert_file):
    cert = json.loads(Path(cert_file).read_text())
    stored = cert.pop("sos")
    return verify_certificate(cert) and cert["sos"] == stored


# (h, n): the first per n is the one just below the numerical threshold h_n (tightest exact bound).
TARGETS = [("9317/5000", 2), ("3/2", 2), ("9/5", 2), ("37/20", 2), ("93/50", 2),
           ("19/10", 3), ("47/25", 3), ("38391/20000", 4), ("38643/20000", 5), ("19411/10000", 6)]


def main():
    out = ROOT / "certificates"
    out.mkdir(exist_ok=True)
    for h, n in TARGETS:
        cert = find_certificate(h, n)
        status = (f"ok, w = {cert['w']}, denominators <= {cert['denominator_bound']}, "
                  f"numeric margin {cert['margin_numeric']:.2e}") if cert else "FAILED"
        print(f"h = {h} ({float(sp.Rational(h)):.4f}), n = {n}: {status}", flush=True)
        if cert:
            (out / f"n{n}_h{h.replace('/', '_')}.json").write_text(json.dumps(cert, indent=1) + "\n")


if __name__ == "__main__":
    main()
