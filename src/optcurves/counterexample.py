"""Exact counterexamples: rational data on which gradient descent breaks curve convexity n times in a row.

A counterexample is rational data  x* = 0, g* = 0, f* = 0  and  (x_k, g_k, f_k), k = 0..n+1, in Q^d with
x_{k+1} = x_k - h g_k, and a modulus 0 <= mu < 1 ("mu" in the file, 0 if absent), such that
  (i)  every interpolation inequality of 1-smooth mu-strongly convex functions holds between every two points
       (including x*):  f_p - f_q - <g_q, x_p - x_q>
                        - (||g_p - g_q||^2 + mu ||x_p - x_q||^2 - 2 mu <g_p - g_q, x_p - x_q>) / (2 (1 - mu)) >= 0
       (for mu = 0 the last term is 1/2 ||g_p - g_q||^2, the 1-smooth convex case), and
  (ii) v_k = 2 f_k - f_{k-1} - f_{k+1} > 0 for k = 1..n.
By the interpolation theorem of Taylor, Hendrickx & Glineur (Math. Program. 2017, Thm. 4), (i) is
equivalent to the existence of a 1-smooth mu-strongly convex f: R^d -> R with f(x_k) = f_k and grad f(x_k) = g_k
(and minimizer x* = 0, since g* = 0). Gradient descent from x_0 then visits exactly x_0, ..., x_{n+1}.
Both (i) and (ii) are checked in exact rational arithmetic; the theorem is the only external input.

The data come from a numerical problem that maximizes the smallest slack t in (i) and (ii) (an SDP over
the Gram matrix, or a small nonlinear program in fixed dimension d), then are rounded to rationals.
"""
import itertools
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from . import sdp

ROOT = Path(__file__).resolve().parents[2]


def trajectory(h, x0, grads):
    xs = [list(x0)]
    for g in grads[:-1]:
        xs.append([a - h * b for a, b in zip(xs[-1], g)])
    return xs


def _require_types(ce):
    """Exact field types, else ValueError: "n" and "dim" (when present) are int, not bool (a subclass of int, so
    {"n": true} passed as n = 1) and not float; "x0", "f" and the gradients in "g" are lists; every rational ("h",
    "mu", the coordinates and values) is a string such as "19/10"."""
    for key in ("h", "x0", "g", "f"):
        if key not in ce:
            raise ValueError(f"malformed counterexample: no field {key!r}")
    for key in ("n", "dim"):
        if key in ce and type(ce[key]) is not int:
            raise ValueError(f"malformed counterexample: {key} is a {type(ce[key]).__name__}, not an int")
    lists = [ce["x0"], ce["f"], ce["g"], *ce["g"]] if isinstance(ce["g"], list) else [ce["g"]]
    if not all(isinstance(v, list) for v in lists):
        raise ValueError("malformed counterexample: x0, f, g and each gradient must be lists")
    rationals = [ce["h"], ce.get("mu", "0"), *ce["x0"], *ce["f"], *(a for gk in ce["g"] for a in gk)]
    if not all(isinstance(a, str) for a in rationals):
        raise ValueError('malformed counterexample: every rational must be a string such as "19/10"')


def check_exact(ce):
    """Exact check of (i) and (ii). Returns (ok, min interpolation slack, violations) as Fractions.
    A record whose sizes disagree (gradients of another dimension than x_0, len(g) != len(f), "n" or "dim" not
    matching the data) is not a counterexample of any size and raises ValueError, as does mu outside [0, 1) or a field
    of the wrong type (see _require_types)."""
    _require_types(ce)
    h = F(ce["h"])
    mu = F(ce.get("mu", "0"))
    if not 0 <= mu < 1:
        raise ValueError(f"mu = {mu} is not in [0, 1)")
    x0 = [F(a) for a in ce["x0"]]
    g = [[F(a) for a in gk] for gk in ce["g"]]
    f = [F(a) for a in ce["f"]]
    d = len(x0)
    n = len(f) - 2
    if not (h > 0 and d >= 1 and n >= 1 and len(g) == len(f) and all(len(gk) == d for gk in g)
            and ce.get("n", n) == n and ce.get("dim", d) == d):
        raise ValueError(f"malformed counterexample: dim {d}, {len(g)} gradients of dimensions "
                         f"{sorted({len(gk) for gk in g})}, {len(f)} values, n = {ce.get('n')}, h = {h}")
    pts = {"s": ([F(0)] * d, [F(0)] * d, F(0))}
    for k, xk in enumerate(trajectory(h, x0, g)):
        pts[k] = (xk, g[k], f[k])
    dot = lambda u, v: sum(a * b for a, b in zip(u, v))

    def interp(p, q):
        (xp, gp, fp), (xq, gq, fq) = p, q
        dx = [a - b for a, b in zip(xp, xq)]
        dg = [a - b for a, b in zip(gp, gq)]
        return fp - fq - dot(gq, dx) - (dot(dg, dg) + mu * dot(dx, dx) - 2 * mu * dot(dg, dx)) / (2 * (1 - mu))

    slack = min(interp(p, q) for p, q in itertools.permutations(pts.values(), 2))
    viol = [2 * f[k] - f[k - 1] - f[k + 1] for k in range(1, len(f) - 1)]
    return slack >= 0 and all(v > 0 for v in viol), slack, viol


def _sdp(h, n, mu=0.0, rank=None, which=0, label="counterexample"):
    """Max-slack data via the Gram matrix of (x_0, g_0..g_N), solved with sdp.SOLVERS[which]; returns (X, f, t),
    X with rows = coordinates, or None when the solver gives no usable candidate (see sdp.py). rank=r keeps only the
    r largest eigenvalues of the Gram matrix (for an optimum of numerical rank r). _sdp.last is the (solver, status)
    of the call."""
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
        dg, dx = grad[p] - grad[q], pos[p] - pos[q]
        if mu == 0:
            curv = 0.5 * ip(dg, dg)
        else:
            curv = (ip(dg, dg) + mu * ip(dx, dx) - 2 * mu * ip(dg, dx)) / (2 * (1 - mu))
        cons.append(fval(p) - fval(q) - ip(grad[q], pos[p] - pos[q]) - curv >= t)
    cons += [2 * fv[k] - fv[k - 1] - fv[k + 1] >= t for k in range(1, n + 1)]
    _sdp.last = sdp.solve(cp.Problem(cp.Maximize(t), cons), which, label)
    if _sdp.last[1] not in sdp.USABLE or G.value is None or fv.value is None or t.value is None:
        return None
    lam, V = np.linalg.eigh(G.value)
    keep = lam > 1e-9 * lam.max() if rank is None else np.arange(P) >= P - rank
    X = (V[:, keep] * np.sqrt(lam[keep])).T
    return X, fv.value, float(t.value)


def _nlp(h, n, d, trials=200, seed=0, mu=0.0):
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
        c = []
        for p, q in pairs:
            dg, dx = g[p] - g[q], x[p] - x[q]
            curv = 0.5 * np.sum(dg ** 2) if mu == 0 else (dg @ dg + mu * dx @ dx - 2 * mu * dg @ dx) / (2 * (1 - mu))
            c.append(f[p] - f[q] - g[q] @ dx - curv - z[-1])
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


def find(h, n, dim=None, mu="0", rank=None,
         denominators=(10, 100, 1000, 10**4, 10**5, 10**6, 10**7, 10**8)):
    """Exact counterexample at step h (a rational string) with n consecutive violations over 1-smooth
    mu-strongly convex f, or None. dim=None: SDP (any dimension; rank=r truncates its optimum to R^r);
    dim=d: search in R^d."""
    hq, muq = F(h), F(mu)
    if dim is not None:
        return _round_counterexample(hq, muq, n, _nlp(float(hq), n, dim, mu=float(muq)), denominators)
    label = f"counterexample h = {h}, n = {n}" + (f", mu = {mu}" if muq else "")
    for which in range(len(sdp.SOLVERS)):  # Clarabel; SCS only if Clarabel gives no exact counterexample
        found = _sdp(float(hq), n, float(muq), rank, which=which, label=label)
        if found is None:
            continue  # sdp.solve has reported the status
        ce = _round_counterexample(hq, muq, n, found, denominators)
        if ce:
            ce["sdp_solver"], ce["sdp_status"] = _sdp.last
            return ce
        if which + 1 < len(sdp.SOLVERS):
            reason = ("its slack is not positive" if found[2] <= 0 else "its candidate fails the exact check")
            print(f"  SDP {label}: {_sdp.last[0]} {reason}; trying {sdp.SOLVERS[which + 1][0]}",
                  file=sys.stderr, flush=True)
    return None


def _round_counterexample(hq, muq, n, found, denominators):
    """Round the numerical data found = (X, f, t) to rationals and check them exactly (check_exact). The first
    denominator bound that gives a valid counterexample wins; None if none does (or found is None, or t <= 0)."""
    if found is None:
        return None
    X, fv, t = found
    if t <= 0:
        return None
    for D in denominators:
        r = lambda a: F(float(a)).limit_denominator(D)
        ce = {"h": str(hq), "n": n}
        if muq:
            ce["mu"] = str(muq)
        ce.update(x0=[str(r(a)) for a in X[:, 0]],
                  g=[[str(r(a)) for a in X[:, 1 + k]] for k in range(n + 2)],
                  f=[str(r(a)) for a in fv])
        ok, slack, viol = check_exact(ce)
        if ok:
            ce.update(dim=len(ce["x0"]), numeric_slack=t, denominator_bound=D,
                      min_interpolation_slack=str(slack), violations=[str(v) for v in viol])
            return ce
    return None


# (h, n, dim): dim=None is the SDP (tight, just above the numerical h_n); dim=2 gives plottable examples.
TARGETS = [("3727/2000", 2, None), ("19/10", 2, 2), ("19001/10000", 3, None), ("199/100", 3, 2),
           ("38393/20000", 4, None), ("38645/20000", 5, None), ("9706/5000", 6, None)]
# (h, n, mu, rank): one violation for mu = 5/7 (kappa = 7/5), 9/5 + 10^-k above h_1(7/5) = 9/5 (theorem.py).
# The SDP optimum is numerically of rank 1 (second eigenvalue ~1e-9 against ~2.7), so it is truncated to R^1.
# Last: three consecutive violations for mu = 20/21 (kappa = 1.05), where the PEP search cannot decide every point
# near 2 (both solvers inaccurate); the SDP optimum (rank 4) is kept whole.
STRONG_TARGETS = [("1801/1000", 1, "5/7", 1), ("18001/10000", 1, "5/7", 1), ("180001/100000", 1, "5/7", 1),
                  ("1800001/1000000", 1, "5/7", 1), ("1999/1000", 3, "20/21", None)]


def main():
    """Writes every counterexample that is found. Exits with an error if any target fails, so that a stale file left
    from an earlier run is never mistaken for a regenerated one (reproduce.sh stops there)."""
    out = ROOT / "counterexamples"
    out.mkdir(exist_ok=True)
    jobs = [dict(h=h, n=n, dim=dim) for h, n, dim in TARGETS]
    jobs += [dict(h=h, n=n, mu=mu, rank=rank) for h, n, mu, rank in STRONG_TARGETS]
    failed = []
    for job in jobs:
        h, n, mu = job["h"], job["n"], job.get("mu", "0")
        ce = find(**job)
        label = (f"dim = {job['dim'] or 'any'}" if mu == "0"
                 else f"mu = {mu}, SDP" + (f" truncated to rank {job['rank']}" if job["rank"] else ""))
        print(f"h = {h} ({float(F(h)):.4f}), n = {n}, {label}: "
              + (f"ok in R^{ce['dim']}, denominators <= {ce['denominator_bound']}, "
                 f"violations {[f'{float(F(v)):.2e}' for v in ce['violations']]}" if ce else "FAILED"),
              flush=True)
        if ce:
            tag = "" if mu == "0" else f"_mu{mu.replace('/', '_')}"
            (out / f"n{n}_h{h.replace('/', '_')}{tag}_d{ce['dim']}.json").write_text(json.dumps(ce, indent=1) + "\n")
        else:
            failed.append(f"h = {h}, n = {n}, {label}")
    if failed:
        raise SystemExit(f"{len(failed)} counterexample(s) FAILED, not regenerated: {'; '.join(failed)}")


if __name__ == "__main__":
    main()
