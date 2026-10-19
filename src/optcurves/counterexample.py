"""Exact counterexamples: rational data on which gradient descent breaks curve convexity n times in a row.

A counterexample is rational data  x* = 0, g* = 0, f* = 0  and  (x_k, g_k, f_k), k = 0..n+1, in Q^d with
x_{k+1} = x_k - h g_k, such that
  (i)  every interpolation inequality of 1-smooth convex functions holds between every two points
       (including x*):  f_p - f_q - <g_q, x_p - x_q> - 1/2 ||g_p - g_q||^2 >= 0, and
  (ii) v_k = 2 f_k - f_{k-1} - f_{k+1} > 0 for k = 1..n.
By the interpolation theorem of Taylor, Hendrickx & Glineur (Math. Program. 2017, Thm. 4), (i) is
equivalent to the existence of a 1-smooth convex f: R^d -> R with f(x_k) = f_k and grad f(x_k) = g_k
(and minimizer x* = 0, since g* = 0). Gradient descent from x_0 then visits exactly x_0, ..., x_{n+1}.
Both (i) and (ii) are checked in exact rational arithmetic; the theorem is the only external input.

The data come from a numerical problem that maximizes the smallest slack t in (i) and (ii) (an SDP over
the Gram matrix, or a small nonlinear program in fixed dimension d), then are rounded to rationals.
"""
import itertools
import json
from fractions import Fraction as F
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def trajectory(h, x0, grads):
    xs = [list(x0)]
    for g in grads[:-1]:
        xs.append([a - h * b for a, b in zip(xs[-1], g)])
    return xs


def check_exact(ce):
    """Exact check of (i) and (ii). Returns (ok, min interpolation slack, violations) as Fractions."""
    h = F(ce["h"])
    x0 = [F(a) for a in ce["x0"]]
    g = [[F(a) for a in gk] for gk in ce["g"]]
    f = [F(a) for a in ce["f"]]
    d = len(x0)
    pts = {"s": ([F(0)] * d, [F(0)] * d, F(0))}
    for k, xk in enumerate(trajectory(h, x0, g)):
        pts[k] = (xk, g[k], f[k])
    dot = lambda u, v: sum(a * b for a, b in zip(u, v))
    slack = min(
        fp - fq - dot(gq, [a - b for a, b in zip(xp, xq)]) - dot([a - b for a, b in zip(gp, gq)],
                                                                 [a - b for a, b in zip(gp, gq)]) / 2
        for (xp, gp, fp), (xq, gq, fq) in itertools.permutations(pts.values(), 2))
    viol = [2 * f[k] - f[k - 1] - f[k + 1] for k in range(1, len(f) - 1)]
    return slack >= 0 and all(v > 0 for v in viol), slack, viol


def _sdp(h, n):
    """Max-slack data via the Gram matrix of (x_0, g_0..g_N); returns (X, f, t), X with rows = coordinates."""
    import cvxpy as cp
    N = n + 1
    P = N + 2
    G = cp.Variable((P, P), PSD=True)
    fv = cp.Variable(N + 1)
    t = cp.Variable()
    e = np.eye(P)
    pos = {"s": np.zeros(P)}
    grad = {"s": np.zeros(P)}
    cur = e[0].copy()
    for k in range(N + 1):
        pos[k], grad[k] = cur.copy(), e[1 + k]
        cur = cur - h * e[1 + k]
    fval = lambda p: 0 if p == "s" else fv[p]
    ip = lambda u, v: cp.sum(cp.multiply(G, np.outer(u, v)))
    cons = [ip(pos[0], pos[0]) <= 1]
    for p, q in itertools.permutations(pos, 2):
        dg = grad[p] - grad[q]
        cons.append(fval(p) - fval(q) - ip(grad[q], pos[p] - pos[q]) - 0.5 * ip(dg, dg) >= t)
    cons += [2 * fv[k] - fv[k - 1] - fv[k + 1] >= t for k in range(1, n + 1)]
    cp.Problem(cp.Maximize(t), cons).solve(solver="CLARABEL")
    lam, V = np.linalg.eigh(G.value)
    keep = lam > 1e-9 * lam.max()
    X = (V[:, keep] * np.sqrt(lam[keep])).T
    return X, fv.value, float(t.value)


def _nlp(h, n, d, trials=200, seed=0):
    """Max-slack data in fixed dimension d (local search with random restarts)."""
    from scipy.optimize import minimize
    N = n + 1
    P = N + 2
    pairs = list(itertools.permutations(["s"] + list(range(N + 1)), 2))
    rng = np.random.default_rng(seed)

    def cons(z):
        X = z[:d * P].reshape(d, P)
        f = {"s": 0.0, **{k: z[d * P + k] for k in range(N + 1)}}
        x, g, cur = {"s": np.zeros(d)}, {"s": np.zeros(d)}, X[:, 0].copy()
        for k in range(N + 1):
            x[k], g[k] = cur.copy(), X[:, 1 + k]
            cur = cur - h * g[k]
        c = [f[p] - f[q] - g[q] @ (x[p] - x[q]) - 0.5 * np.sum((g[p] - g[q]) ** 2) - z[-1] for p, q in pairs]
        c += [2 * f[k] - f[k - 1] - f[k + 1] - z[-1] for k in range(1, n + 1)]
        return np.array(c + [1 - X[:, 0] @ X[:, 0]])

    best = None
    for _ in range(trials):
        z0 = np.concatenate([rng.normal(size=d * P), rng.normal(size=N + 1), [0.0]])
        r = minimize(lambda z: -z[-1], z0, constraints=[{"type": "ineq", "fun": cons}], method="SLSQP",
                     options={"maxiter": 2000, "ftol": 1e-12})
        if r.success and cons(r.x).min() > -1e-9 and (best is None or r.x[-1] > best[-1]):
            best = r.x
    if best is None:
        return None
    return best[:d * P].reshape(d, P), best[d * P:d * P + N + 1], float(best[-1])


def find(h, n, dim=None, denominators=(10, 100, 1000, 10**4, 10**5, 10**6, 10**7, 10**8)):
    """Exact counterexample at step h (a rational string) with n consecutive violations, or None.
    dim=None: SDP (any dimension); dim=d: search in R^d."""
    hq = F(h)
    found = _sdp(float(hq), n) if dim is None else _nlp(float(hq), n, dim)
    if found is None:
        return None
    X, fv, t = found
    if t <= 0:
        return None
    for D in denominators:
        r = lambda a: F(float(a)).limit_denominator(D)
        ce = {"h": str(hq), "n": n,
              "x0": [str(r(a)) for a in X[:, 0]],
              "g": [[str(r(a)) for a in X[:, 1 + k]] for k in range(n + 2)],
              "f": [str(r(a)) for a in fv]}
        ok, slack, viol = check_exact(ce)
        if ok:
            ce.update(dim=len(ce["x0"]), numeric_slack=t, denominator_bound=D,
                      min_interpolation_slack=str(slack), violations=[str(v) for v in viol])
            return ce
    return None


# (h, n, dim): dim=None is the SDP (tight, just above the numerical h_n); dim=2 gives plottable examples.
TARGETS = [("3727/2000", 2, None), ("19/10", 2, 2), ("19001/10000", 3, None), ("199/100", 3, 2),
           ("38393/20000", 4, None), ("38645/20000", 5, None), ("9706/5000", 6, None)]


def main():
    out = ROOT / "counterexamples"
    out.mkdir(exist_ok=True)
    for h, n, dim in TARGETS:
        ce = find(h, n, dim)
        print(f"h = {h} ({float(F(h)):.4f}), n = {n}, dim = {dim or 'any'}: "
              + (f"ok in R^{ce['dim']}, denominators <= {ce['denominator_bound']}, "
                 f"violations {[f'{float(F(v)):.2e}' for v in ce['violations']]}" if ce else "FAILED"),
              flush=True)
        if ce:
            (out / f"n{n}_h{h.replace('/', '_')}_d{ce['dim']}.json").write_text(json.dumps(ce, indent=1) + "\n")


if __name__ == "__main__":
    main()
