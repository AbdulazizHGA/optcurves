"""The exact threshold for one curve violation of gradient descent on smooth strongly convex functions.

    python -m optcurves.theorem     # checks every step of the proof below, prints it, exits 1 on a failure

Setting as in certificate.py: L = 1, x_{k+1} = x_k - h g_k with g_k = grad f(x_k), f_k = f(x_k), and the curve
violation v_1 = 2 f_1 - f_0 - f_2. Let 0 < mu < 1, kappa = 1/mu, 0 < h < 2 and

    q(h, mu) = 2 mu h^2 - 4 (1 + mu) h + mu + 7 = mu (2 h^2 - 4 (1 + kappa) h + 7 kappa + 1),
    h_1(kappa) = (1 + kappa) - sqrt((1 + kappa)^2 - (7 kappa + 1) / 2)      (the smaller root of q).

q(0) = 7 + mu > 0 > mu - 1 = q(2) and q is convex in h, so for 0 < h < 2: q >= 0 iff h <= h_1(kappa).

Theorem.
 (a) If h <= h_1(kappa), then v_1 <= 0 for every 1-smooth mu-strongly convex f on any real inner-product space and
     every x_0 (no minimizer needed). Applied at every step, the whole curve k -> f(x_k) is convex.
 (b) If h_1(kappa) < h < 2, then v_1 > 0 for some 1-smooth mu-strongly convex f: R -> R with a minimizer.
 (c) At h = h_1(kappa) the data of (b) have v_1 = 0, so (a) is attained.
 (d) h_1 decreases strictly in kappa, from 2 (kappa -> 1) to 7/4 (kappa -> infinity).

Proof of (a). With the interpolation inequalities Q_pq >= 0 of certificate.py, the identity

    v_1 + lam_01 Q_01 + lam_20 Q_20 + lam_21 Q_21 + sum_ab S_ab <g_a, g_b> = 0,
    lam_01 = (3 + mu - 2 h mu) / (2 - h mu),  lam_20 = (1 + mu - h mu) / (2 - h mu),  lam_21 = (1 - mu) / (2 - h mu),

holds with S = W / (2 (1 - mu) (2 - h mu)) and the polynomial matrix W below (W_MATRIX). The numerators of the
multipliers are 3 (1 - mu) + 2 mu (2 - h), (1 - mu) + mu (2 - h) and 1 - mu, all positive. W is positive
semidefinite: its LDL^T in the pivot order g_1, g_0, g_2,

    sum_ab W_ab <g_a, g_b> = d_1 ||g_1 + (W_01 g_0 + W_12 g_2) / W_11||^2 + d_2 ||g_0 + l g_2||^2 + d_3 ||g_2||^2,

has the pivots
    d_1 = W_11 = (4 - 2 h) + h^2 mu (2 - h mu)                                      > 0,
    d_2 = m / W_11,   m = W_00 W_11 - W_01^2 = q (mu (h^2 mu - 2 h + 1)^2 + 1 - mu) + mu (1 - mu)   > 0 if q >= 0,
    d_3 = det W / m,  det W = h^2 mu (1 - mu) (2 - h mu)^3 q                          >= 0 if q >= 0,
and l = (W_02 W_11 - W_01 W_12) / m. So v_1 = -(lam Q terms) - (squares) <= 0.

Proof of (b), (c). One-dimensional data with minimizer x* = 0 (g* = f* = 0) and D = 2 + h^2 (1 - mu):

    x_0 = 1,      g_0 = 1,                                    f_0 = 1/2,
    x_1 = 1 - h,  g_1 = -(h - 1) (2 + h (1 - mu)) / D,         f_1 = x_1^2 / 2 - (g_1 - x_1)^2 / (2 (1 - mu)),
    x_2 = x_1 - h g_1 = 2 (h - 1)^2 / D,  g_2 = x_2,           f_2 = x_2^2 / 2.

Of the 12 interpolation inequalities, the six among x*, x_0, x_2 (data of x^2 / 2) and the three Q_*1, Q_01, Q_21
hold with equality; the other three are

    Q_1* = h (h - 1)^3 (1 - mu) (2 + h - h^2 mu) / D^2,     2 + h - h^2 mu = (1 - mu) h^2 + (2 - h)(1 + h),
    Q_10 = h^2 (h - 1)^2 (1 - mu) (1 + 2 h - h^2 mu) / D^2,  1 + 2 h - h^2 mu = (1 - mu) h^2 + 2 - (h - 1)^2,
    Q_12 = h^2 (h - 1)^3 (1 - mu) (3 - h mu) / D^2,

all >= 0 for 1 <= h < 2 (and h_1(kappa) > 7/4 since q(7/4) = mu/8 > 0). Finally

    v_1 = -h^2 q / (2 D),

which is > 0 iff q < 0 iff h > h_1(kappa), and 0 at h = h_1(kappa); there the gradient vector (1, g_1, g_2) spans the
kernel of W. By the interpolation theorem of Taylor, Hendrickx and Glineur (Math. Program. 2017, Thm. 4) the data
come from a 1-smooth mu-strongly convex f: R -> R, and gradient descent from x_0 visits x_1, x_2.

Proof of (d). With R = (1 + kappa)^2 - (7 kappa + 1)/2 = (kappa - 1)(kappa - 1/2):
dh_1/dkappa = 1 - (kappa - 3/4) / sqrt(R) < 0 because (kappa - 3/4)^2 - R = 1/16 > 0, and
h_1 = ((7 kappa + 1)/2) / (1 + kappa + sqrt(R)) -> 7/4.

`verify()` checks every identity above in exact polynomial arithmetic in the symbols h, mu, kappa (no sampling, no
floating point). The identity of (a) is checked with symbolic vectors in R^2 (no Gram-matrix code); it is linear in
the inner products of x_0, g_0, g_1, g_2, and Gram matrices of vectors in R^2 span all symmetric matrices, so it holds
in every real inner-product space. The signs are then read off the displayed forms. With mu = 0 the same formulas give
lam = (3/2, 1/2, 1/2), W 1 = 0, sum W_ab <g_a, g_b> = 1/2 ||g_0 + g_1 - 2 g_2||^2 + (7 - 4h)/2 ||g_0 - g_1||^2 and
v_1 = -h^2 (7 - 4 h) / (2 (2 + h^2)): they recover the threshold 7/4 of Barzilai, Shamir and Zamani for convex f.
"""
import itertools
import sys

import sympy as sp

h, mu, kappa = sp.symbols("h mu kappa", positive=True)
q = 2 * mu * h**2 - 4 * (1 + mu) * h + mu + 7
D = 2 + h**2 * (1 - mu)
LAM = {(0, 1): (3 + mu - 2 * h * mu) / (2 - h * mu),
       (2, 0): (1 + mu - h * mu) / (2 - h * mu),
       (2, 1): (1 - mu) / (2 - h * mu)}
# W = 2 (1 - mu) (2 - h mu) S in the basis g_0, g_1, g_2.
W_MATRIX = sp.Matrix([
    [4 - 2 * h + 2 * mu - 11 * h * mu + 6 * h**2 * mu - 2 * h * mu**2 + 6 * h**2 * mu**2 - 3 * h**3 * mu**2,
     -3 - mu + 2 * h * (1 + mu) + h**2 * mu**2 * (1 - h),
     -(1 - h * mu) * (1 + mu - h * mu)],
    [-3 - mu + 2 * h * (1 + mu) + h**2 * mu**2 * (1 - h),
     4 - 2 * h + h**2 * mu * (2 - h * mu),
     mu - (1 - h * mu) ** 2],
    [-(1 - h * mu) * (1 + mu - h * mu), mu - (1 - h * mu) ** 2, 2 - h * mu]])


def h1(k=kappa):
    """The threshold h_1(kappa) (a SymPy expression; exact for rational kappa)."""
    return (1 + k) - sp.sqrt((1 + k) ** 2 - (7 * k + 1) / sp.Integer(2))


def interp(P, R, dot, m=mu):
    """Q_PR = f_P - f_R - <g_R, x_P - x_R> - (||g_P - g_R||^2 + m ||x_P - x_R||^2 - 2 m <g_P - g_R, x_P - x_R>)
    / (2 (1 - m)) for points P = (x, g, f), with the inner product `dot`."""
    (xp, gp, fp), (xr, gr, fr) = P, R
    dx, dg = xp - xr, gp - gr
    return fp - fr - dot(gr, dx) - (dot(dg, dg) + m * dot(dx, dx) - 2 * m * dot(dg, dx)) / (2 * (1 - m))


def gram_from_multipliers(lam, m=mu):
    """S = -sum lam_pq (quadratic part of Q_pq) in the basis g_0, g_1, g_2 (x_0 = 0 by translation invariance),
    built from scalar symbols: the coefficient matrix of a quadratic form in g_0, g_1, g_2."""
    g = sp.symbols("g0:3")
    x = [sp.Integer(0), -h * g[0], -h * g[0] - h * g[1]]
    form = -sum(c * interp((x[p], g[p], 0), (x[r], g[r], 0), lambda u, v: u * v, m) for (p, r), c in lam.items())
    return sp.Matrix(3, 3, lambda i, j: sp.diff(form, g[i], g[j]) / 2).applyfunc(sp.cancel)


def extremal_data(m=mu):
    """The one-dimensional data of (b): {point: (x, g, f)} for the minimizer 's' and x_0, x_1, x_2."""
    Dm = 2 + h**2 * (1 - m)
    x1 = 1 - h
    g1 = -(h - 1) * (2 + h * (1 - m)) / Dm
    x2 = x1 - h * g1
    return {"s": (0, 0, 0), 0: (1, 1, sp.Rational(1, 2)),
            1: (x1, g1, x1**2 / 2 - (g1 - x1) ** 2 / (2 * (1 - m))), 2: (x2, x2, x2**2 / 2)}


def ldl_102(W):
    """Pivots and multipliers of the LDL^T of W in the pivot order g_1, g_0, g_2."""
    m = W[0, 0] * W[1, 1] - W[0, 1] ** 2
    return {"d": [W[1, 1], m / W[1, 1], W.det() / m],
            "first": [W[0, 1] / W[1, 1], 1, W[1, 2] / W[1, 1]],  # coefficients of g_0, g_1, g_2
            "second": [1, 0, (W[0, 2] * W[1, 1] - W[0, 1] * W[1, 2]) / m],
            "third": [0, 0, 1]}


def _zero(expr):
    return sp.cancel(sp.together(expr)) == 0


def _divisible_by_q(expr):
    num = sp.numer(sp.together(expr))
    return sp.Poly(sp.expand(num), h, mu).rem(sp.Poly(q, h, mu)).is_zero


def verify():
    """Every step of the proof as (description, bool), in exact arithmetic."""
    checks = []
    add = lambda text, ok: checks.append((text, bool(ok)))

    # The threshold.
    R = (1 + kappa) ** 2 - (7 * kappa + 1) / sp.Integer(2)
    add("q = mu (2 h^2 - 4 (1 + kappa) h + 7 kappa + 1) with kappa = 1/mu",
        _zero(q - mu * (2 * h**2 - 4 * (1 + 1 / mu) * h + 7 / mu + 1)))
    add("q(h_1(kappa), 1/kappa) = 0", sp.expand(q.subs({h: h1(), mu: 1 / kappa}) * kappa) == 0)
    add("radicand R = (kappa - 1)(kappa - 1/2)", sp.expand(R - (kappa - 1) * (kappa - sp.Rational(1, 2))) == 0)
    add("q(0) = 7 + mu, q(2) = mu - 1, q(7/4) = mu/8: one root of q in (0, 2), in (7/4, 2)",
        sp.expand(q.subs(h, 0) - 7 - mu) == 0 and sp.expand(q.subs(h, 2) - mu + 1) == 0
        and sp.expand(q.subs(h, sp.Rational(7, 4)) - mu / 8) == 0 and sp.diff(q, h, 2) == 4 * mu)
    add("(d) dh_1/dkappa = 1 - (kappa - 3/4)/sqrt(R), (kappa - 3/4)^2 - R = 1/16, (1 + kappa)^2 - R = (7 kappa + 1)/2",
        sp.simplify(sp.diff(h1(), kappa) - (1 - (kappa - sp.Rational(3, 4)) / sp.sqrt(R))) == 0
        and sp.expand((kappa - sp.Rational(3, 4)) ** 2 - R) == sp.Rational(1, 16)
        and sp.expand((1 + kappa) ** 2 - R - (7 * kappa + 1) / sp.Integer(2)) == 0)

    # (a) The certificate.
    viol = {0: -1, 1: 2, 2: -1}
    add("(a) the f_0, f_1, f_2 coefficients of v_1 + sum lam Q vanish",
        all(_zero(viol[k] + sum(c * ((p == k) - (r == k)) for (p, r), c in LAM.items())) for k in range(3)))
    add("(a) multipliers: numerators 3(1 - mu) + 2 mu (2 - h), (1 - mu) + mu (2 - h), 1 - mu over 2 - h mu",
        _zero(LAM[0, 1] - (3 * (1 - mu) + 2 * mu * (2 - h)) / (2 - h * mu))
        and _zero(LAM[2, 0] - ((1 - mu) + mu * (2 - h)) / (2 - h * mu)) and _zero(LAM[2, 1] - (1 - mu) / (2 - h * mu)))
    S = gram_from_multipliers(LAM)
    add("(a) S = -sum lam Q_quad equals W / (2 (1 - mu)(2 - h mu))",
        all(_zero(e) for e in S - W_MATRIX / (2 * (1 - mu) * (2 - h * mu))))
    # The identity itself, with vectors in R^2 (x_0 included; nothing but dot products of symbolic vectors).
    xs = sp.Matrix(sp.symbols("x0_:2"))
    gs = [sp.Matrix(sp.symbols(f"g{a}_:2")) for a in range(3)]
    fs = sp.symbols("f0:3")
    pts = []
    for a in range(3):
        pts.append((xs, gs[a], fs[a]))
        xs = xs - h * gs[a]
    dot = lambda u, v: (u.T * v)[0, 0]
    total = (2 * fs[1] - fs[0] - fs[2] + sum(c * interp(pts[p], pts[r], dot) for (p, r), c in LAM.items())
             + sum(W_MATRIX[a, b] * dot(gs[a], gs[b]) for a in range(3) for b in range(3))
             / (2 * (1 - mu) * (2 - h * mu)))
    add("(a) v_1 + sum lam Q + sum S_ab <g_a, g_b> = 0 for symbolic x_0, g_a in R^2 and f_a", _zero(total))
    L = ldl_102(W_MATRIX)
    g = sp.symbols("g0:3")
    sq = lambda c: sum(ci * gi for ci, gi in zip(c, g)) ** 2
    form = sum(W_MATRIX[a, b] * g[a] * g[b] for a in range(3) for b in range(3))
    add("(a) LDL^T in the order g_1, g_0, g_2 reproduces W",
        _zero(L["d"][0] * sq(L["first"]) + L["d"][1] * sq(L["second"]) + L["d"][2] * sq(L["third"]) - form))
    m = W_MATRIX[0, 0] * W_MATRIX[1, 1] - W_MATRIX[0, 1] ** 2
    add("(a) d_1 = W_11 = (4 - 2h) + h^2 mu (2 - h mu)", sp.expand(W_MATRIX[1, 1] - (4 - 2 * h)
                                                               - h**2 * mu * (2 - h * mu)) == 0)
    add("(a) m = W_00 W_11 - W_01^2 = q (mu (h^2 mu - 2h + 1)^2 + 1 - mu) + mu (1 - mu)",
        sp.expand(m - q * (mu * (h**2 * mu - 2 * h + 1) ** 2 + 1 - mu) - mu * (1 - mu)) == 0)
    add("(a) det W = h^2 mu (1 - mu) (2 - h mu)^3 q",
        sp.expand(W_MATRIX.det() - h**2 * mu * (1 - mu) * (2 - h * mu) ** 3 * q) == 0)

    # (b), (c) The one-dimensional data.
    data = extremal_data()
    one = lambda u, v: u * v
    Q = {(P, R2): sp.cancel(interp(data[P], data[R2], one)) for P, R2 in itertools.permutations(data, 2)}
    slack = {(1, "s"): h * (h - 1) ** 3 * (1 - mu) * (2 + h - h**2 * mu) / D**2,
             (1, 0): h**2 * (h - 1) ** 2 * (1 - mu) * (1 + 2 * h - h**2 * mu) / D**2,
             (1, 2): h**2 * (h - 1) ** 3 * (1 - mu) * (3 - h * mu) / D**2}
    add("(b) x_2 = x_1 - h g_1 = 2 (h - 1)^2 / D", _zero(data[2][0] - (data[1][0] - h * data[1][1]))
        and _zero(data[2][0] - 2 * (h - 1) ** 2 / D))
    add("(b) the 9 inequalities among x*, x_0, x_2 and into x_1 hold with equality",
        all(Q[k] == 0 for k in Q if k not in slack) and len(Q) == 12)
    add("(b) Q_1*, Q_10, Q_12 equal the displayed factored forms", all(_zero(Q[k] - v) for k, v in slack.items()))
    add("(b) 2 + h - h^2 mu = (1 - mu) h^2 + (2 - h)(1 + h), 1 + 2h - h^2 mu = (1 - mu) h^2 + 2 - (h - 1)^2",
        sp.expand(2 + h - h**2 * mu - (1 - mu) * h**2 - (2 - h) * (1 + h)) == 0
        and sp.expand(1 + 2 * h - h**2 * mu - (1 - mu) * h**2 - 2 + (h - 1) ** 2) == 0)
    v1 = 2 * data[1][2] - data[0][2] - data[2][2]
    add("(b) v_1 = -h^2 q / (2 D)", _zero(v1 + h**2 * q / (2 * D)))
    kernel = W_MATRIX * sp.Matrix([data[0][1], data[1][1], data[2][1]])
    add("(c) W (g_0, g_1, g_2) is a multiple of q: at q = 0 the gradients span the kernel of W",
        all(_divisible_by_q(e) for e in kernel))

    # mu = 0: the convex case of Barzilai, Shamir and Zamani.
    W0 = W_MATRIX.subs(mu, 0)
    add("mu = 0: lam = (3/2, 1/2, 1/2), W 1 = 0, form = 1/2 (g0 + g1 - 2 g2)^2 + (7 - 4h)/2 (g0 - g1)^2",
        [LAM[k].subs(mu, 0) for k in ((0, 1), (2, 0), (2, 1))] == [sp.Rational(3, 2), sp.Rational(1, 2),
                                                                   sp.Rational(1, 2)]
        and W0 * sp.ones(3, 1) == sp.zeros(3, 1)
        and sp.expand(form.subs(mu, 0) - (g[0] + g[1] - 2 * g[2]) ** 2 / 2
                      - (7 - 4 * h) * (g[0] - g[1]) ** 2 / 2) == 0)
    add("mu = 0: v_1 of the data = -h^2 (7 - 4h) / (2 (2 + h^2))",
        _zero(v1.subs(mu, 0) + h**2 * (7 - 4 * h) / (2 * (2 + h**2))))
    return checks


def certificate_at(hv, muv):
    """The certificate of (a) at rational (h, mu), in the format of certificate.py (verify_certificate adds sos)."""
    from . import certificate
    sub = {h: sp.Rational(hv), mu: sp.Rational(muv)}
    return certificate._cert(str(sp.Rational(hv)), 1, str(sp.Rational(muv)), [sp.Integer(1)],
                             {k: sp.Rational(v.subs(sub)) for k, v in LAM.items()})


def data_at(hv, muv):
    """The data of (b) at rational (h, mu), in the format of counterexample.check_exact."""
    sub = {h: sp.Rational(hv), mu: sp.Rational(muv)}
    pt = {k: [sp.Rational(sp.sympify(c).subs(sub)) for c in v] for k, v in extremal_data().items()}
    return {"h": str(sp.Rational(hv)), "mu": str(sp.Rational(muv)), "x0": [str(pt[0][0])],
            "g": [[str(pt[k][1])] for k in range(3)], "f": [str(pt[k][2]) for k in range(3)]}


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
