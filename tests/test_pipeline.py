"""Failure handling of the pipeline: a target that cannot be regenerated must stop reproduce.sh, not leave the old
file looking regenerated."""
import pytest

from optcurves import certificate, counterexample


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
