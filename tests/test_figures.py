"""Figure data: the contour values of figures/counterexample_2d.png (the largest smooth interpolant, evaluated
numerically by face enumeration with a KKT check) and the reliability of the curve points of figures/consecutive.png."""
import json
import math
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


def test_curve_points_are_reliable():
    """Every published curve point [h, PEP value, reliable] of results/consecutive.json was solved to status
    "optimal", so the figure has no gaps."""
    cons = json.loads((figures.RESULTS / "consecutive.json").read_text())
    for n, pts in cons["curves"].items():
        assert len(pts) == 62 and all(len(p) == 3 and p[2] is True for p in pts), n
        assert cons["curves_unresolved"][n] == 0, n


def test_unreliable_curve_point_is_a_gap():
    xs, ys = figures.curve_xy([[1.8, 0.06, True], [1.85, 0.12, False], [1.9, -1e-9, True]])
    assert xs == [1.8, 1.85, 1.9] and ys[0] == 0.06 and math.isnan(ys[1]) and ys[2] == 0.0


# --- threshold figures: the marker follows the record's "resolved" field, not the bracket's width ---

def _t(lo, hi, resolved):
    return {"lo": lo, "hi": hi, "resolved": resolved}


def test_threshold_kind():
    assert figures.threshold_kind(None) is None                        # no violation up to H_MAX
    assert figures.threshold_kind(_t(1.5, None, False)) is None        # nothing decided
    assert figures.threshold_kind(_t(1.99, None, False)) == "lower_only"
    assert figures.threshold_kind(_t(1.8, 1.8 + 3e-8, True)) == "resolved"
    assert figures.threshold_kind(_t(1.8, 1.8 + 7e-8, False)) == "open"  # narrow, but stopped early: still open


def test_published_threshold_markers():
    """The brackets drawn open in figures/strong_threshold.png are exactly those the README calls open: h_1 at
    kappa = 1.5 and 2, h_3 at kappa = 1.05 and 3 (bracket), h_2 at kappa = 1.05 (lower end only). Those of h_1 at
    kappa = 1.5 and 2 and h_3 at kappa = 3 are narrower than 1e-6 and were drawn as resolved points before."""
    rows = json.loads((figures.RESULTS / "strong.json").read_text())["rows"]
    kinds = {(r["kappa"], n): figures.threshold_kind(r[f"h{n}"]) for r in rows for n in (1, 2, 3)}
    assert {key for key, kind in kinds.items() if kind not in ("resolved", None)} == {
        (1.5, 1), (2, 1), (1.05, 3), (3, 3), (1.05, 2)}
    assert kinds[(1.05, 2)] == "lower_only"
    for r in rows:
        for n in (1, 2, 3):
            t = r[f"h{n}"]
            assert t is None or (kinds[(r["kappa"], n)] == "resolved") == t["resolved"]


def test_consecutive_figure_skips_a_missing_threshold(monkeypatch, tmp_path):
    """study._threshold returns None when there is no violation up to H_MAX; the figure must not index it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cons = json.loads((figures.RESULTS / "consecutive.json").read_text())
    cons["thresholds"]["convex_with_minimizer"]["3"] = None
    (tmp_path / "consecutive.json").write_text(json.dumps(cons))
    monkeypatch.setattr(figures, "RESULTS", tmp_path)
    monkeypatch.setattr(figures, "FIGURES", tmp_path)
    figures.consecutive(plt)
    assert (tmp_path / "consecutive.png").exists()
