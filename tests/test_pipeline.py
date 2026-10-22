"""Failure handling of the pipeline: a target that cannot be regenerated must stop reproduce.sh, not leave the old
file looking regenerated; a PEP point that no solver can decide must not decide a threshold."""
import warnings

import cvxpy as cp
import pytest

from optcurves import certificate, counterexample, pep, sdp, study


def test_certificate_generator_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setattr(certificate, "ROOT", tmp_path)
    monkeypatch.setattr(certificate, "find_certificate", lambda *args, **kwargs: None)
    monkeypatch.setattr(certificate, "tight_certificate", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit) as exit_info:
        certificate.main()
    assert exit_info.value.code  # a message, so the process exits with status 1
    assert not any((tmp_path / "certificates").iterdir())


def test_counterexample_generator_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setattr(counterexample, "ROOT", tmp_path)
    monkeypatch.setattr(counterexample, "find", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit) as exit_info:
        counterexample.main()
    assert exit_info.value.code
    assert not any((tmp_path / "counterexamples").iterdir())


def _plain_bisection(decide, lo, hi, iters):
    for _ in range(iters):
        mid = (lo + hi) / 2
        if decide(mid):
            hi = mid
        else:
            lo = mid
    return lo, hi


def test_bisection_without_unresolved_points_is_plain_bisection():
    for threshold in (1.75, 1.8634462, 1.94, 1.9998):
        decide = lambda h: h > threshold
        lo, hi, resolved = pep.bisect_threshold(decide, 1.5, study.H_MAX, study.WIDTH)
        assert resolved and (lo, hi) == _plain_bisection(decide, 1.5, study.H_MAX, study.ITERS)


def test_bisection_steps_around_an_unresolved_point():
    decide = lambda h: None if abs(h - 1.75) < 1e-3 else h > 1.9  # the first midpoint, 1.74995, is unresolved
    lo, hi, resolved = pep.bisect_threshold(decide, 1.5, study.H_MAX, study.WIDTH)
    assert resolved and lo <= 1.9 < hi


def test_bisection_stops_where_nothing_can_be_decided():
    decide = lambda h: None if 1.98 < h < 1.995 else h > 1.99
    lo, hi, resolved = pep.bisect_threshold(decide, 1.5, study.H_MAX, study.WIDTH)
    assert not resolved and lo <= 1.98 and hi >= 1.995


def _fake_evaluate(threshold, unresolved=lambda h: False):
    """Stand-in for pep.evaluate: PEP value max(h - threshold, 0), unresolved where unresolved(h)."""
    def evaluate(h, n=1, mu=0.0, minimizer=True, L=1.0):
        pep.STATS["solves"] += 1
        if unresolved(h):
            pep.STATS["unresolved"] += 1
            return 0.0, False  # what SCS returned at kappa = 1.05, n = 3, h = 1.9999
        return max(h - threshold, 0.0), True
    return evaluate


def test_threshold_skips_an_undecidable_upper_end(monkeypatch):
    """kappa = 1.05, n = 3 in miniature: H_MAX and 1.9995 cannot be decided; this used to give "no violation"."""
    monkeypatch.setattr(pep, "evaluate", _fake_evaluate(1.99, lambda h: h > 1.9992))
    t = study._threshold(3, mu=20 / 21)
    assert t["resolved"] and t["lo"] <= 1.99 + pep.TOL < t["hi"]
    assert t["unresolved_at"] == [study.H_MAX, 1.9995] and t["unresolved"] == 2


def test_threshold_is_none_only_when_h_max_is_decided(monkeypatch):
    monkeypatch.setattr(pep, "evaluate", _fake_evaluate(5.0))
    assert study._threshold(3) is None
    monkeypatch.setattr(pep, "evaluate", _fake_evaluate(5.0, lambda h: True))
    t = study._threshold(3)
    assert t["hi"] is None and t["lo"] == 1.5 and not t["resolved"] and t["unresolved_at"] == list(study.TOPS)


def test_sdp_falls_back_to_scs_when_clarabel_gives_no_candidate(monkeypatch):
    real = sdp.solve
    monkeypatch.setattr(sdp, "solve", lambda problem, which, label:
                        ("CLARABEL", "solver error") if which == 0 else real(problem, which, label))
    cert = certificate.find_certificate("9/5", 2)
    assert cert and (cert["sdp_solver"], cert["sdp_status"]) == ("SCS", "optimal")
    ce = counterexample.find(h="3727/2000", n=2)
    assert ce and (ce["sdp_solver"], ce["sdp_status"]) == ("SCS", "optimal")


def test_sdp_without_a_usable_status_gives_no_record(monkeypatch):
    monkeypatch.setattr(sdp, "solve", lambda problem, which, label: (sdp.SOLVERS[which][0], "infeasible"))
    assert certificate.find_certificate("9/5", 2) is None
    assert counterexample.find(h="3727/2000", n=2) is None


def test_sdp_status_of_an_infeasible_problem():
    x = cp.Variable()
    name, status = sdp.solve(cp.Problem(cp.Minimize(x), [x >= 1, x <= 0]), 0, "test")
    assert name == "CLARABEL" and status not in sdp.USABLE


# --- solver statuses (pep.py): decides / unresolved / model failure ---

class _FakePep:
    """Stand-in for a PEPit problem whose solves return given CVXPY statuses in turn."""
    def __init__(self, statuses):
        self.statuses, self.calls = list(statuses), []

    def solve(self, verbose, wrapper, **options):
        status = self.statuses.pop(0)
        self.calls.append(options["solver"])
        if status == "raise":
            raise cp.error.SolverError("solver crashed")
        self.wrapper = type("W", (), {"prob": type("P", (), {"status": status})()})()
        return 0.25


@pytest.mark.parametrize("statuses, reliable, calls", [
    (["optimal"], True, ["CLARABEL"]),
    (["optimal_inaccurate", "optimal"], True, ["CLARABEL", "SCS"]),
    (["raise", "optimal"], True, ["CLARABEL", "SCS"]),
    (["optimal_inaccurate", "optimal_inaccurate"], False, ["CLARABEL", "SCS"]),
    (["user_limit", "infeasible_inaccurate"], False, ["CLARABEL", "SCS"]),
    (["unbounded_inaccurate", "raise"], False, ["CLARABEL", "SCS"]),
])
def test_status_decides_or_leaves_unresolved(statuses, reliable, calls):
    pep.reset_stats()
    fake = _FakePep(statuses)
    value, ok = pep._solve(fake)
    assert ok is reliable and fake.calls == calls
    assert pep.STATS == {"solves": 1, "fallbacks": len(calls) - 1, "unresolved": int(not reliable)}
    assert pep.LAST["status"].startswith("CLARABEL ") and pep.LAST["status"].count(",") == len(calls) - 1


@pytest.mark.parametrize("statuses", [["infeasible"], ["unbounded"], ["optimal_inaccurate", "infeasible"],
                                      ["optimal_inaccurate", "infeasible_or_unbounded"]])
def test_certified_infeasible_or_unbounded_is_a_model_failure(statuses):
    with pytest.raises(RuntimeError, match="model failure"):
        pep._solve(_FakePep(statuses))


def test_unknown_status_is_an_error():
    with pytest.raises(RuntimeError, match="unclassified"):
        pep._solve(_FakePep(["optimal_inaccurate", "some new status"]))


def test_max_min_violation_refuses_an_unresolved_value(monkeypatch):
    monkeypatch.setattr(pep, "_solve", lambda problem: (0.0, False))
    assert pep.evaluate(1.9, 2) == (0.0, False)  # evaluate reports it ...
    with pytest.raises(RuntimeError, match="unresolved"):
        pep.max_min_violation(1.9, 2)  # ... the shortcut refuses it
    monkeypatch.setattr(pep, "_solve", lambda problem: (0.25, True))
    assert pep.max_min_violation(1.9, 2) == 0.25


def test_threshold_stores_the_statuses_of_unresolved_points(monkeypatch):
    fake = _fake_evaluate(1.99, lambda h: h > 1.9992)

    def evaluate(*args, **kwargs):
        pep.LAST["status"] = "CLARABEL optimal_inaccurate, SCS optimal_inaccurate"
        return fake(*args, **kwargs)
    monkeypatch.setattr(pep, "evaluate", evaluate)
    t = study._threshold(3, mu=20 / 21)
    assert t["unresolved_at"] == [study.H_MAX, 1.9995]
    assert t["unresolved_status"] == ["CLARABEL optimal_inaccurate, SCS optimal_inaccurate"] * 2


# --- warnings: only CVXPY's "Solution may be inaccurate" is silenced ---

def _warn_both():
    warnings.warn("Solution may be inaccurate. Try another solver.", UserWarning)
    warnings.warn("some other warning", DeprecationWarning)


def test_pep_solve_shows_other_warnings():
    class Warns(_FakePep):
        def solve(self, verbose, wrapper, **options):
            _warn_both()
            return super().solve(verbose, wrapper, **options)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pep._quiet_solve(Warns(["optimal"]), solver="CLARABEL")
    assert [str(w.message) for w in caught] == ["some other warning"]


def test_sdp_solve_shows_other_warnings():
    class Problem:
        status = "optimal"

        def solve(self, solver, **options):
            _warn_both()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert sdp.solve(Problem(), 0, "test") == ("CLARABEL", "optimal")
    assert [str(w.message) for w in caught] == ["some other warning"]
