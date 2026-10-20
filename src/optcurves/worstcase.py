"""The exact worst case of one curve violation above the threshold h_1(kappa), for strongly convex and convex f.

    python -m optcurves.worstcase     # checks every step of the proof below, prints it, exits 1 on a failure

Notation of theorem.py: L = 1, x_{k+1} = x_k - h g_k, v_1 = 2 f_1 - f_0 - f_2, the interpolation inequalities
Q_pq >= 0, and

    q = 2 mu h^2 - 4 (1 + mu) h + mu + 7 = (1 - mu)(7 - 4 h) + 2 mu (2 - h)^2,     D = 2 + h^2 (1 - mu).

Now 0 <= mu < 1 and f has a minimizer x*; it is the point "s" with g_* = 0, f_* = f(x*). For mu > 0 a minimizer
exists on R^d or any Hilbert space, but not on every inner-product space. On the finite sequences inside l^2, with
S the shift, f(x) = <A x, x> / 2 - x_1 and A = (4 I - S - S^T) / 6 is 1-smooth and 1/3-strongly convex, and its
gradient A x - e_1 is again a finite sequence, so gradient descent stays in the space. Yet f has no minimizer
there: the only candidate, the solution x*_k = 6 (2 - sqrt 3)^k of A x = e_1 in l^2, is not a finite sequence.
By the second form of q, q < 0 forces h > 7/4; for 0 < h < 2, q < 0 means h_1(kappa) < h < 2 (theorem.py), and
7/4 < h < 2 when mu = 0.

Theorem. Let 0 <= mu < 1 and 0 < h < 2 with q < 0. Then for every 1-smooth mu-strongly convex f (convex for
mu = 0) with a minimizer x*, on any real inner-product space, and every x_0,

    v_1 <= c ||x_0 - x*||^2,     c = h^2 (-q) / (2 D) > 0,

with equality for the one-dimensional data of theorem.py (b), which have ||x_0 - x*|| = 1. Together with theorem.py
(a) (v_1 <= 0 for h <= h_1(kappa); for mu = 0 this is the threshold 7/4 of Barzilai, Shamir and Zamani) and x_0 = x*
(v_1 = 0), the worst case of one violation is, for all 0 < h < 2,

    max of v_1 over f and x_0 with ||x_0 - x*|| <= 1  =  max(0, -h^2 q / (2 D))

(for a general L, multiply by L). For h > h_1(kappa) the maximum is taken at radius 1, since f -> r^2 f(. / r)
scales v_1 by r^2. For h <= h_1(kappa) it is 0, taken at x_0 = x*; for mu > 0 and h < h_1(kappa) only there: the
Gram part of the certificate of theorem.py (a) is then positive definite, so v_1 < 0 unless g_0 = g_1 = g_2 = 0,
that is, unless x_0 = x*.

Proof. With y = x_0 - x* and u = (y, g_0, g_1, g_2), the identity (x_1 = x_0 - h g_0, x_2 = x_1 - h g_1)

    v_1 - c ||y||^2 + 2c (Q_*0 + Q_2*) + 2 (h - 1) Q_01 + lam_20 Q_20 + 2 (2 - h) Q_21 + sum_ab S_ab <u_a, u_b> = 0,
    lam_20 = 2 (h - 1) E / D,   E = h^3 mu - 2 h^2 mu - h^2 + h + 3 = (1 - mu)(1 + (2 - h)(1 + h))
                                                                       + mu ((h - 1)^3 + 2 (2 - h)),

holds with S = W / (2 (1 - mu) D) and the polynomial matrix W below (W_MATRIX, indices 0, 1, 2, 3 for
y, g_0, g_1, g_2). Only the sum Q_*0 + Q_2* involves f_*, and there it cancels. For 1 < h < 2 and q < 0 all five
multipliers are positive. The y-row of W is h^2 (-q) (1 + mu, -(1 + h mu), -h mu, -mu), and W is positive
semidefinite: its LDL^T in the pivot order g_2, g_1, g_0, y,

    sum_ab W_ab <u_a, u_b> = d_1 ||g_2 + (W_23 g_1 + W_13 g_0 + W_03 y) / W_33||^2 + d_2 ||g_1 + l_1 g_0 + l_0 y||^2
                             + d_3 ||g_0 - y||^2,

has the pivots
    d_1 = W_33 = D                                                                  > 0,
    d_2 = D (mu D - 2 q)                                                            > 0,
    d_3 = h^2 (-q) P / (mu D - 2 q),   P = -2 q + mu T                              > 0,
    T = (1 - mu)(h^2 + 8 (h - 3/2)) + 2 mu ((h - 1)^4 + 4 (h - 1)(2 - h) + 1) + 2 h^3 (2 - h) mu (1 - mu)   >= 0,
and a fourth pivot d_4 = 0, because W u = 0 for the data u = (1, 1, g_1, x_2) of theorem.py (b) (so det W = 0; l_0
and l_1 are rational functions of h and mu, computed by `ldl`). The signs need only q < 0, mu >= 0 and
3/2 <= h <= 2. So v_1 - c ||y||^2 = -(multiplier terms) - (squares) <= 0. On the data of theorem.py (b) the five Q
above are tight, W u = 0 and v_1 = c, so the bound is attained.

On the threshold q = 0 this certificate is the one of theorem.py (a): 2c = 0, the y-row of W vanishes, and the
other multipliers and the g-block of S differ from theorem.py's by multiples of q (for instance
2 (h - 1) - lam_01 = -q / (2 - h mu) and 2 (2 - h) - lam_21 = q / (2 - h mu)). With mu = 0:
c = h^2 (4 h - 7) / (2 (2 + h^2)), lam_20 = 2 (h - 1)(3 + h - h^2) / (2 + h^2) and d_3 = h^2 (4 h - 7).

`verify()` checks every identity above in exact polynomial arithmetic in the symbols h, mu (no sampling, no floating
point). The certificate identity is checked with symbolic vectors x*, x_0, g_0, g_1, g_2 in R^2 and symbolic
f_*, f_0, f_1, f_2; as in theorem.py it is linear in the inner products of y, g_0, g_1, g_2, so it holds in every
real inner-product space. The signs are then read off the displayed forms.
"""
import sys

import sympy as sp

from .theorem import D, LAM as LAM_H1, W_MATRIX as W_H1, _divisible_by_q, _zero, extremal_data, h, interp, mu, q

C = h**2 * (-q) / (2 * D)
E = h**3 * mu - 2 * h**2 * mu - h**2 + h + 3
LAM = {("s", 0): 2 * C, (2, "s"): 2 * C, (0, 1): 2 * (h - 1), (2, 0): 2 * (h - 1) * E / D, (2, 1): 2 * (2 - h)}
T = ((1 - mu) * (h**2 + 8 * (h - sp.Rational(3, 2))) + 2 * mu * ((h - 1) ** 4 + 4 * (h - 1) * (2 - h) + 1)
     + 2 * h**3 * (2 - h) * mu * (1 - mu))
P = -2 * q + mu * T
# W = 2 (1 - mu) D S in the basis y = x_0 - x*, g_0, g_1, g_2.
W_MATRIX = sp.Matrix([
    [h**2 * (1 + mu) * (-q), -h**2 * (1 + h * mu) * (-q), -h**3 * mu * (-q), -h**2 * mu * (-q)],
    [-h**2 * (1 + h * mu) * (-q),
     -4 * h**5 * mu**2 + 9 * h**4 * mu**2 + 3 * h**4 * mu + 4 * h**4 - 4 * h**3 * mu**2 - 4 * h**3 - 13 * h**2 * mu
     - 13 * h**2 + 8 * h * mu + 20 * h - 10,
     -2 * h**5 * mu**2 + 3 * h**4 * mu**2 + h**4 * mu + 4 * h**4 + 4 * h**3 * mu - 8 * h**3 - 8 * h**2 * mu + 2 * h**2
     - 2 * h + 4,
     -2 * h**4 * mu**2 + 3 * h**3 * mu**2 + 3 * h**3 * mu + 2 * h**3 - 4 * h**2 - 6 * h * mu - 4 * h + 6],
    [-h**3 * mu * (-q),
     -2 * h**5 * mu**2 + 3 * h**4 * mu**2 + h**4 * mu + 4 * h**4 + 4 * h**3 * mu - 8 * h**3 - 8 * h**2 * mu + 2 * h**2
     - 2 * h + 4,
     D * (h**2 * mu + 4 * h**2 - 8 * h + 2), D * (h * mu + 2 * h - 4)],
    [-h**2 * mu * (-q),
     -2 * h**4 * mu**2 + 3 * h**3 * mu**2 + 3 * h**3 * mu + 2 * h**3 - 4 * h**2 - 6 * h * mu - 4 * h + 6,
     D * (h * mu + 2 * h - 4), D]])


def _points(y, g, f):
    """Points {s, 0, 1, 2} as (x, g, f) with x* = 0 and x_0 = y (translation invariance)."""
    zero = y * 0
    return {"s": (zero, zero, f[0]), 0: (y, g[0], f[1]), 1: (y - h * g[0], g[1], f[2]),
            2: (y - h * g[0] - h * g[1], g[2], f[3])}


def gram_from_multipliers(lam, c, m=mu):
    """S with u^T S u = c ||y||^2 - sum lam_pq (quadratic part of Q_pq), in the basis y, g_0, g_1, g_2, built from
    scalar symbols: the coefficient matrix of a quadratic form."""
    u = sp.symbols("u0:4")
    pts = _points(u[0], u[1:], [0] * 4)
    form = c * u[0] ** 2 - sum(v * interp(pts[p], pts[r], lambda a, b: a * b, m) for (p, r), v in lam.items())
    return sp.Matrix(4, 4, lambda i, j: sp.diff(form, u[i], u[j]) / 2).applyfunc(sp.cancel)


def ldl(W, order=(3, 2, 1, 0)):
    """LDL^T of W in the given pivot order (default g_2, g_1, g_0, y): pivots d and rows l_j (coefficients of
    y, g_0, g_1, g_2) with u^T W u = sum_j d_j (l_j . u)^2 when the pivots are nonzero or their columns vanish."""
    n = len(order)
    Wp = W.extract(list(order), list(order))
    Lo, d = sp.zeros(n, n), []
    for j in range(n):
        Lo[j, j] = 1
        d.append(sp.cancel(Wp[j, j] - sum(Lo[j, k] ** 2 * d[k] for k in range(j))))
        for i in range(j + 1, n):
            r = sp.cancel(Wp[i, j] - sum(Lo[i, k] * Lo[j, k] * d[k] for k in range(j)))
            Lo[i, j] = sp.cancel(r / d[j]) if d[j] != 0 else r
    rows = []
    for j in range(n):
        row = [sp.Integer(0)] * n
        for i in range(n):
            row[order[i]] = Lo[i, j]
        rows.append(row)
    return d, rows


def verify():
    """Every step of the proof as (description, bool), in exact arithmetic."""
    checks = []
    add = lambda text, ok: checks.append((text, bool(ok)))

    add("q = (1 - mu)(7 - 4h) + 2 mu (2 - h)^2: q < 0 with 0 <= mu < 1 forces h > 7/4",
        sp.expand(q - (1 - mu) * (7 - 4 * h) - 2 * mu * (2 - h) ** 2) == 0)

    # The certificate.
    viol = {"s": 0, 0: -1, 1: 2, 2: -1}
    add("the f_*, f_0, f_1, f_2 coefficients of v_1 + sum lam Q vanish (f_* only in Q_*0 + Q_2*)",
        all(_zero(viol[k] + sum(c * ((p == k) - (r == k)) for (p, r), c in LAM.items())) for k in viol)
        and LAM["s", 0] == LAM[2, "s"])
    add("multipliers: 2c = h^2 (-q) / D, 2 (h - 1), 2 (h - 1) E / D, 2 (2 - h); "
        "E = (1 - mu)(1 + (2 - h)(1 + h)) + mu ((h - 1)^3 + 2 (2 - h))",
        _zero(LAM["s", 0] - h**2 * (-q) / D)
        and sp.expand(E - (1 - mu) * (1 + (2 - h) * (1 + h)) - mu * ((h - 1) ** 3 + 2 * (2 - h))) == 0)
    S = gram_from_multipliers(LAM, C)
    add("S = c E_yy - sum lam Q_quad equals W / (2 (1 - mu) D)",
        all(_zero(e) for e in S - W_MATRIX / (2 * (1 - mu) * D)))
    # The identity itself, with vectors in R^2 (x* and x_0 both symbolic, nothing but dot products).
    xs = sp.Matrix(sp.symbols("xs_:2"))
    x0 = sp.Matrix(sp.symbols("x0_:2"))
    gs = [sp.Matrix(sp.symbols(f"g{a}_:2")) for a in range(3)]
    fs = sp.symbols("fs f0:3")
    pts = {"s": (xs, sp.zeros(2, 1), fs[0]), 0: (x0, gs[0], fs[1]), 1: (x0 - h * gs[0], gs[1], fs[2]),
           2: (x0 - h * gs[0] - h * gs[1], gs[2], fs[3])}
    dot = lambda a, b: (a.T * b)[0, 0]
    u = [x0 - xs] + gs
    total = (2 * fs[2] - fs[1] - fs[3] - C * dot(u[0], u[0])
             + sum(c * interp(pts[p], pts[r], dot) for (p, r), c in LAM.items())
             + sum(W_MATRIX[a, b] * dot(u[a], u[b]) for a in range(4) for b in range(4)) / (2 * (1 - mu) * D))
    add("v_1 - c ||x_0 - x*||^2 + sum lam Q + sum S_ab <u_a, u_b> = 0 for symbolic x*, x_0, g_a in R^2 and f_*, f_a",
        _zero(total))
    add("the y-row of W is h^2 (-q) (1 + mu, -(1 + h mu), -h mu, -mu)",
        all(sp.expand(W_MATRIX[0, j] - h**2 * (-q) * w) == 0
            for j, w in enumerate([1 + mu, -(1 + h * mu), -h * mu, -mu])))

    # W is positive semidefinite.
    data = extremal_data()
    ud = sp.Matrix([data[0][0], data[0][1], data[1][1], data[2][1]])
    add("W u = 0 for the data u = (1, 1, g_1, x_2) of theorem.py (b)", all(_zero(e) for e in W_MATRIX * ud))
    d, rows = ldl(W_MATRIX)
    us = sp.symbols("u0:4")
    sq = lambda row: sum(c * x for c, x in zip(row, us)) ** 2
    form = sum(W_MATRIX[a, b] * us[a] * us[b] for a in range(4) for b in range(4))
    add("LDL^T in the order g_2, g_1, g_0, y: d_4 = 0 and d_1 sq_1 + d_2 sq_2 + d_3 sq_3 reproduces W",
        _zero(d[3]) and _zero(sum(d[j] * sq(rows[j]) for j in range(3)) - form))
    add("the third square is ||g_0 - y||^2 and the second one vanishes on u",
        [sp.cancel(e) for e in rows[2]] == [-1, 1, 0, 0]
        and _zero(sum(c * x for c, x in zip(rows[1], ud))))
    add("d_1 = W_33 = D", _zero(d[0] - D))
    add("d_2 = D (mu D - 2 q)", _zero(d[1] - D * (mu * D - 2 * q)))
    add("d_3 = h^2 (-q) P / (mu D - 2 q) with P = -2 q + mu T (T as displayed)",
        _zero(d[2] - h**2 * (-q) * P / (mu * D - 2 * q)))
    add("det of the g-block of W = h^2 D^2 (-q) P, det W = 0",
        sp.expand(W_MATRIX[1:, 1:].det() - h**2 * D**2 * (-q) * P) == 0 and sp.expand(W_MATRIX.det()) == 0)

    # Equality.
    one = lambda a, b: a * b
    Qd = {k: sp.cancel(interp(data[k[0]], data[k[1]], one)) for k in LAM}
    v1 = 2 * data[1][2] - data[0][2] - data[2][2]
    add("on the data of theorem.py (b): ||x_0 - x*|| = 1, Q_*0 = Q_2* = Q_01 = Q_20 = Q_21 = 0 and v_1 = c",
        data[0][0] == 1 and data["s"] == (0, 0, 0) and all(e == 0 for e in Qd.values()) and _zero(v1 - C))

    # The threshold and mu = 0.
    add("q = 0: 2c = 0, and lam_01, lam_20, lam_21 differ from theorem.py's by multiples of q",
        _divisible_by_q(LAM["s", 0]) and all(_divisible_by_q(LAM[k] - LAM_H1[k]) for k in LAM_H1)
        and _zero(2 * (h - 1) - LAM_H1[0, 1] + q / (2 - h * mu))
        and _zero(2 * (2 - h) - LAM_H1[2, 1] - q / (2 - h * mu)))
    add("q = 0: the y-row of S vanishes and its g-block is the S of theorem.py",
        all(_divisible_by_q(e) for e in W_MATRIX[0, :])
        and all(_divisible_by_q(e) for e in S[1:, 1:] - W_H1 / (2 * (1 - mu) * (2 - h * mu))))
    add("mu = 0: c = h^2 (4h - 7) / (2 (2 + h^2)), lam_20 = 2 (h - 1)(3 + h - h^2) / (2 + h^2), d_3 = h^2 (4h - 7)",
        _zero(C.subs(mu, 0) - h**2 * (4 * h - 7) / (2 * (2 + h**2)))
        and _zero(LAM[2, 0].subs(mu, 0) - 2 * (h - 1) * (3 + h - h**2) / (2 + h**2))
        and _zero(d[2].subs(mu, 0) - h**2 * (4 * h - 7)))
    return checks


def certificate_at(hv, muv):
    """The certificate at rational (h, mu): {"c", "lam", "S"} with Fractions-compatible SymPy rationals."""
    sub = {h: sp.Rational(hv), mu: sp.Rational(muv)}
    return {"h": sp.Rational(hv), "mu": sp.Rational(muv), "c": sp.Rational(C.subs(sub)),
            "lam": {k: sp.Rational(v.subs(sub)) for k, v in LAM.items()},
            "S": (W_MATRIX / (2 * (1 - mu) * D)).subs(sub)}


def main():
    checks = verify()
    for text, ok in checks:
        print(("ok      " if ok else "FAILED  ") + text, flush=True)
    failed = sum(not ok for _, ok in checks)
    print(f"{len(checks) - failed} of {len(checks)} steps verified")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
