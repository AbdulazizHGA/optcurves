"""The contour values of figures/counterexample_2d.png: exact evaluation of the largest smooth interpolant."""
import json
from fractions import Fraction

import cvxpy as cp
import numpy as np

from optcurves import figures

CE = json.loads((figures.ROOT / "counterexamples" / "n2_h19_10_d2.json").read_text())


def _grid():
    f = figures.interpolant(CE)
    xs, GX, GY = figures.plot_grid(CE)
    P = np.column_stack([GX.ravel(), GY.ravel()])
    return f, xs, P, f(P)


def test_contour_values_are_optimal_on_the_plotting_grid():
    """Every one of the 140 x 140 grid points of the figure is solved to its KKT conditions, the data values are
    reproduced, and a point gives the same value alone as inside the grid (the earlier iterative evaluation was off
    by up to 9.5e-4 inside the grid, at (-0.0206, 0.5011), while exact on its own)."""
    f, xs, P, Z = _grid()
    assert len(P) == 140 * 140 and f.kkt_residual < 1e-10
    assert np.abs(f(xs) - [float(Fraction(a)) for a in CE["f"]]).max() < 1e-12
    worst_before = int(np.argmin(np.abs(P - [-0.0206301823, 0.5011047865]).sum(axis=1)))
    for i in [worst_before, *np.random.default_rng(0).choice(len(P), 40, replace=False)]:
        assert abs(f(P[i:i + 1])[0] - Z[i]) < 1e-14


def test_contour_values_against_an_independent_qp_solver():
    """The same minimum over the simplex solved by cvxpy (Clarabel, tolerance ~1e-8) at 30 grid points."""
    _, xs, P, Z = _grid()
    g = [np.array([float(Fraction(a)) for a in gk]) for gk in CE["g"]]
    fv = [float(Fraction(a)) for a in CE["f"]]
    pts_x, pts_g = np.vstack([np.zeros(2), xs]), np.vstack([np.zeros(2), g])
    c = np.array([0.0] + [fk - gk @ gk / 2 for fk, gk in zip(fv, g)])
    for i in np.random.default_rng(1).choice(len(P), 30, replace=False):
        A = P[i] - pts_x + pts_g
        lam = cp.Variable(len(c), nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A.T @ lam) / 2 + c @ lam), [cp.sum(lam) == 1])
        prob.solve(solver="CLARABEL")
        assert abs(prob.value - Z[i]) < 1e-7
