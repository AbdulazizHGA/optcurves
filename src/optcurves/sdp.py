"""Solving the hand-built SDPs of certificate.py and counterexample.py, with a status check.

Their numerical solutions are only candidates: every certificate and counterexample is rounded to rationals and
checked exactly afterwards, and only that check decides. The status still matters for clear failures:

  * "optimal": the candidate is used.
  * "optimal_inaccurate": the candidate is used too, with a message on stderr, and the status is stored in the
    record. One published counterexample needs this (three violations at kappa = 1.05, h = 1.999): Clarabel stops
    "optimal_inaccurate" there, its candidate passes the exact check, and SCS (also "optimal_inaccurate") gives a
    candidate that fails it. Rejecting such candidates would lose a valid result.
  * anything else (infeasible, unbounded, solver error, no values): no candidate, with a message on stderr.

The callers try Clarabel first and SCS only when Clarabel gives no candidate or its candidate fails the exact check.
"""
import sys
import warnings

SOLVERS = (("CLARABEL", {}), ("SCS", dict(eps=1e-9, max_iters=200000)))
USABLE = ("optimal", "optimal_inaccurate")


def solve(problem, which, label):
    """Solve a cvxpy problem with SOLVERS[which]; returns (solver name, status). The status is "solver error"
    when the solver raises. Only CVXPY's "Solution may be inaccurate" warning is silenced (the status
    "optimal_inaccurate", reported below instead); any other warning is shown."""
    from cvxpy.error import SolverError
    name, options = SOLVERS[which]
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Solution may be inaccurate", category=UserWarning)
            problem.solve(solver=name, **options)
        status = problem.status
    except SolverError:
        status = "solver error"
    if status != "optimal":
        note = ("candidate kept, the exact check decides" if status in USABLE
                else "no candidate" + (", trying " + SOLVERS[which + 1][0] if which + 1 < len(SOLVERS) else ""))
        print(f"  SDP {label}: {name} status {status!r}; {note}", file=sys.stderr, flush=True)
    return name, status
