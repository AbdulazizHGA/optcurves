"""Performance-estimation (PEP) worst cases for convexity of the gradient-descent optimization curve.

Gradient descent with constant step h/L:  x_{k+1} = x_k - (h/L) grad f(x_k).
Curve violation at step k:                v_k = 2 f(x_k) - f(x_{k-1}) - f(x_{k+1}).
The curve k -> f(x_k) is convex at step k iff v_k <= 0.

`max_min_violation(h, n)` returns the PEP value  max_f max_{x_0} min(v_1, ..., v_n)  over the window
x_0, ..., x_{n+1}. The starting point x_0 is free, so the window covers every n consecutive steps of
every trajectory. The value is > 0 (up to solver tolerance) iff n consecutive violations are possible.

Function classes (L = 1 without loss of generality: the problem is invariant under rescaling f):
  * mu = 0, minimizer=True:  L-smooth convex f with a minimizer x*, normalised by ||x_0 - x*||^2 <= 1.
  * mu = 0, minimizer=False: all L-smooth convex f, normalised by ||grad f(x_0)||^2 <= 1.
  * mu > 0:                  L-smooth mu-strongly convex f (always has a minimizer), ||x_0 - x*||^2 <= 1.
Both normalisations only fix the scale (v_k is 2-homogeneous), so the sign of the value is unaffected.
"""
import warnings

from PEPit import PEP
from PEPit.functions import SmoothConvexFunction, SmoothStronglyConvexFunction

warnings.filterwarnings("ignore")

SOLVER = "CLARABEL"
# Clarabel occasionally stops with "optimal_inaccurate" (mostly for mu > 0 or values at the noise floor).
# Those points are re-solved with SCS at eps = 1e-9. If SCS is also "optimal_inaccurate", its value is used
# and counted in INACCURATE (reported next to every threshold); any other status is an error.
FALLBACK = dict(solver="SCS", eps=1e-9, max_iters=200000)
# Bisection tolerance: "no violation" means PEP value <= TOL. Solver noise is ~1e-9 (Clarabel) / ~1e-12 (SCS).
TOL = 1e-7
INACCURATE = [0]


def _solve(pep):
    value = pep.solve(verbose=0, wrapper="cvxpy", solver=SOLVER)
    if pep.wrapper.prob.status == "optimal":
        return value
    value = pep.solve(verbose=0, wrapper="cvxpy", **FALLBACK)
    status = pep.wrapper.prob.status
    if status == "optimal_inaccurate":
        INACCURATE[0] += 1
    elif status != "optimal":
        raise RuntimeError(f"PEP not solved to optimality (Clarabel and SCS): {status}")
    return value


def _build(h, n, mu=0.0, minimizer=True, L=1.0):
    pep = PEP()
    if mu > 0:
        f = pep.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    else:
        f = pep.declare_function(SmoothConvexFunction, L=L)
    x0 = pep.set_initial_point()
    if minimizer or mu > 0:
        xs = f.stationary_point()
        pep.set_initial_condition((x0 - xs) ** 2 <= 1)
    else:
        pep.set_initial_condition(f.gradient(x0) ** 2 <= 1)
    x = [x0]
    for _ in range(n + 1):
        x.append(x[-1] - h / L * f.gradient(x[-1]))
    fv = [f(xi) for xi in x]
    viol = [2 * fv[k] - fv[k - 1] - fv[k + 1] for k in range(1, n + 1)]
    return pep, f, x, fv, viol


def max_min_violation(h, n=1, mu=0.0, minimizer=True, L=1.0):
    """max over the class of min(v_1, ..., v_n); PEPit maximizes the minimum of all set metrics."""
    pep, _, _, _, viol = _build(h, n, mu, minimizer, L)
    for v in viol:
        pep.set_performance_metric(v)
    return _solve(pep)


def bisect_threshold(value, lo, hi, tol=TOL, iters=24):
    """Largest h in [lo, hi] with value(h) <= tol, assuming value(lo) <= tol < value(hi).

    Returns (lo, hi): the final bracket. Bracket width is (hi - lo) / 2**iters."""
    assert value(lo) <= tol < value(hi), "bracket does not straddle the threshold"
    for _ in range(iters):
        mid = (lo + hi) / 2
        if value(mid) <= tol:
            lo = mid
        else:
            hi = mid
    return lo, hi

