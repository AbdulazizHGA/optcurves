"""Performance-estimation (PEP) worst cases for convexity of the gradient-descent optimization curve.

Gradient descent with constant step h/L:  x_{k+1} = x_k - (h/L) grad f(x_k).
Curve violation at step k:                v_k = 2 f(x_k) - f(x_{k-1}) - f(x_{k+1}).
The curve k -> f(x_k) is convex at step k iff v_k <= 0.

`evaluate(h, n)` returns the PEP value  max_f max_{x_0} min(v_1, ..., v_n)  over the window x_0, ..., x_{n+1},
and whether a solver reached status "optimal" for it. The starting point x_0 is free, so the window covers every
n consecutive steps of every trajectory. The value is > 0 (up to solver tolerance) iff n consecutive violations are
possible.

Function classes (L = 1 without loss of generality: the problem is invariant under rescaling f):
  * mu = 0, minimizer=True:  L-smooth convex f with a minimizer x*, normalized by ||x_0 - x*||^2 <= 1.
  * mu = 0, minimizer=False: all L-smooth convex f, normalized by ||grad f(x_0)||^2 <= 1.
  * mu > 0:                  L-smooth mu-strongly convex f, ||x_0 - x*||^2 <= 1 (a minimizer exists here because
                             PEPit works in R^d; on an incomplete inner-product space it need not).
Both normalizations only fix the scale (v_k is 2-homogeneous), so the sign of the value is unaffected.
"""
import warnings

from cvxpy.error import SolverError
from PEPit import PEP
from PEPit.functions import SmoothConvexFunction, SmoothStronglyConvexFunction

SOLVER = "CLARABEL"
# Clarabel occasionally stops with "optimal_inaccurate" (mostly for mu > 0 near kappa = 1). Such a point is re-solved
# with SCS at eps = 1e-9. If SCS does not reach "optimal" either, the point is UNRESOLVED: its value is kept for the
# record but never decides whether there is a violation. (Accepting the inaccurate SCS value instead made the search
# miss three consecutive violations at kappa = 1.05, where SCS returned 0.0 and Clarabel 3.5e-4.)
FALLBACK = dict(solver="SCS", eps=1e-9, max_iters=200000)
# "No violation" means PEP value <= TOL. Solver noise is ~1e-9 (Clarabel) / ~1e-12 (SCS).
TOL = 1e-7
# What each CVXPY status means here. A PEP of this repository is always feasible (every iterate at the minimizer with
# zero gradients, e.g. for f = mu ||x - x*||^2 / 2) and bounded (the normalization), so a certified "infeasible" or
# "unbounded" means the model itself is wrong: an error.
# The other non-optimal statuses are numerical trouble: the next solver is tried, and if none reaches "optimal" the
# point is unresolved. Any status not listed is an error too, so that a new one cannot pass unnoticed.
DECIDES = ("optimal",)
UNRESOLVED = ("optimal_inaccurate", "user_limit", "infeasible_inaccurate", "unbounded_inaccurate", "solver error")
MODEL_FAILURE = ("infeasible", "unbounded", "infeasible_or_unbounded")
# Counts since the last reset_stats(): PEP solves, fallbacks to SCS, unresolved points.
STATS = {"solves": 0, "fallbacks": 0, "unresolved": 0}
# The solver statuses of the last evaluate(), e.g. "CLARABEL optimal_inaccurate, SCS optimal_inaccurate".
LAST = {"status": ""}
# Points of a bracket [lo, hi] tried in turn by one bisection step: lo + t (hi - lo).
TRIES = (0.5, 0.25, 0.75)


def reset_stats():
    for key in STATS:
        STATS[key] = 0


def _quiet_solve(pep, **options):
    """pep.solve, silencing only CVXPY's "Solution may be inaccurate" warning (the status "optimal_inaccurate", which
    _solve handles); any other warning is shown."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Solution may be inaccurate", category=UserWarning)
        return pep.solve(verbose=0, wrapper="cvxpy", **options)


def _attempt(pep, options):
    """(value, status) of one solver; the status is "solver error" when the solver raises."""
    try:
        value = _quiet_solve(pep, **options)
        return value, pep.wrapper.prob.status
    except SolverError:
        return None, "solver error"


def _solve(pep):
    """(value, reliable): Clarabel, then the SCS fallback if Clarabel does not reach "optimal". reliable is False
    when neither does (the point is unresolved; the value is SCS's, or None). Statuses are classified above; a model
    failure or an unknown status raises RuntimeError. LAST["status"] records the statuses of this call."""
    STATS["solves"] += 1
    trail = []
    for k, options in enumerate(({"solver": SOLVER}, FALLBACK)):
        STATS["fallbacks"] += k
        value, status = _attempt(pep, options)
        trail.append(f"{options['solver']} {status}")
        LAST["status"] = ", ".join(trail)
        if status in MODEL_FAILURE:
            raise RuntimeError(f"PEP model failure ({LAST['status']}): this PEP is feasible and bounded by "
                               "construction, so the model is wrong")
        if status not in DECIDES + UNRESOLVED:
            raise RuntimeError(f"unclassified solver status ({LAST['status']}); add it to pep.py")
        if status in DECIDES:
            return value, True
    STATS["unresolved"] += 1
    return value, False


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


def evaluate(h, n=1, mu=0.0, minimizer=True, L=1.0):
    """(value, reliable): max over the class of min(v_1, ..., v_n) (PEPit maximizes the minimum of all set
    metrics), and whether it was solved to status "optimal"."""
    pep, _, _, _, viol = _build(h, n, mu, minimizer, L)
    for v in viol:
        pep.set_performance_metric(v)
    return _solve(pep)


def max_min_violation(h, n=1, mu=0.0, minimizer=True, L=1.0):
    """The PEP value of evaluate(), which must be reliable: raises RuntimeError when no solver reached status
    "optimal", so that an undecided value cannot be used by mistake. Code that handles unresolved points
    (study.py, side.py) calls evaluate() instead."""
    value, reliable = evaluate(h, n, mu, minimizer, L)
    if not reliable:
        raise RuntimeError(f"PEP value at h = {h}, n = {n}, mu = {mu} is unresolved ({LAST['status']})")
    return value


def bisect_threshold(decide, lo, hi, width, tries=TRIES, max_steps=60):
    """Bracket [lo, hi] of the threshold, from lo without a violation and hi with one.

    decide(h) is True (violation), False (none) or None (unresolved). A step tests the midpoint (lo + hi) / 2 and, if
    it is unresolved, lo + t (hi - lo) for the other t in tries; the first answer replaces one end, unresolved points
    never do. Without unresolved points this is plain bisection. Stops when hi - lo <= width or when every point of a
    step is unresolved. Returns (lo, hi, resolved) with resolved = (hi - lo <= width)."""
    done = lambda: hi - lo <= width * (1 + 1e-6)  # the slack absorbs rounding: n halvings of hi - lo, not n + 1
    for _ in range(max_steps):
        if done():
            break
        for t in tries:
            h = (lo + hi) / 2 if t == 0.5 else lo + t * (hi - lo)
            answer = decide(h)
            if answer is not None:
                break
        else:
            break
        if answer:
            hi = h
        else:
            lo = h
    return lo, hi, done()
