"""Failure handling of the pipeline: a target that cannot be regenerated must stop reproduce.sh, not leave the old
file looking regenerated; a PEP point that no solver can decide must not decide a threshold."""
import pytest

from optcurves import certificate, counterexample, pep, study


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
