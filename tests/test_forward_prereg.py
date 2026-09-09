"""Pre-registration record tests (audit item 1.4).

The point of pre-registration is that a rule cannot be edited after its results
are known. The tests therefore focus on the append-only and tamper-evident
properties, not on the shape of any particular rule.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.forward.prereg import (
    PREREG_FIELDS,
    PreregError,
    list_preregistrations,
    load_preregistration,
    new_record,
    record_path,
    record_sha256,
    trial_count,
    validate_record,
    verify_preregistration,
    write_preregistration,
)


def _record(**over) -> dict:
    kwargs = dict(
        rule_id="d_forward_test",
        scope={"account": "D_5W", "window": ["2026-09-10", "2027-03-09"]},
        decision={"hard": ["tracking_error_daily_pp<0.2"], "verdict": "all hard gates"},
        stopping={"window_days": 120, "kill": "any hard failure 3 days"},
        trials={"family": "d_forward", "prior_trials": 2, "this_trial": 3},
        frozen_at="2026-09-08T18:00:00",
        code_commit="a" * 40,
    )
    kwargs.update(over)
    return new_record(**kwargs)


def test_record_has_the_six_template_fields():
    rec = _record()
    for f in PREREG_FIELDS:
        assert f in rec and rec[f]
    assert rec["record_sha256"] == record_sha256(rec)


def test_roundtrip_and_verify(tmp_path):
    rec = _record()
    path = write_preregistration(rec, dir=tmp_path)
    assert path.name == "prereg_d_forward_test_v1.json"
    loaded = load_preregistration(path)
    assert loaded["rule_id"] == "d_forward_test"
    assert verify_preregistration(path)["record_sha256"] == rec["record_sha256"]


def test_editing_a_frozen_record_is_detected(tmp_path):
    path = write_preregistration(_record(), dir=tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["decision"]["hard"] = ["anything I like after seeing the results"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PreregError) as exc:
        verify_preregistration(path)
    assert "record_sha256 mismatch" in str(exc.value)


def test_append_only_refuses_overwrite(tmp_path):
    path = write_preregistration(_record(), dir=tmp_path)
    assert path.exists()
    changed = _record(decision={"hard": ["different"], "verdict": "x"})
    with pytest.raises(PreregError) as exc:
        write_preregistration(changed, dir=tmp_path)
    assert "append-only" in str(exc.value)
    # a new version is the sanctioned way to change a rule
    v2 = _record(version=2, supersedes="d_forward_test_v1",
                 decision={"hard": ["new"], "verdict": "x"})
    p2 = write_preregistration(v2, dir=tmp_path)
    assert p2.name == "prereg_d_forward_test_v2.json"
    assert load_preregistration(p2)["supersedes"] == "d_forward_test_v1"


def test_identical_content_may_be_rewritten(tmp_path):
    rec = _record()
    path = write_preregistration(rec, dir=tmp_path)
    assert write_preregistration(rec, dir=tmp_path) == path  # idempotent repair


def test_missing_field_rejected():
    rec = _record()
    del rec["stopping"]
    rec["record_sha256"] = record_sha256(rec)
    with pytest.raises(PreregError) as exc:
        validate_record(rec)
    assert "stopping" in str(exc.value)


def test_empty_scope_rejected():
    with pytest.raises(PreregError):
        new_record(rule_id="x", scope={}, decision={"a": 1}, stopping={"b": 2},
                   trials={"c": 3}, code_commit="a" * 40)


def test_future_freeze_rejected(tmp_path):
    future = (pd.Timestamp.now() + pd.Timedelta(days=30)).isoformat()
    path = write_preregistration(_record(frozen_at=future), dir=tmp_path)
    with pytest.raises(PreregError) as exc:
        verify_preregistration(path)
    assert "future" in str(exc.value)


def test_require_commit_detects_code_drift(tmp_path):
    path = write_preregistration(_record(code_commit="b" * 40), dir=tmp_path)
    assert verify_preregistration(path)["code_commit"] == "b" * 40
    with pytest.raises(PreregError):
        verify_preregistration(path, require_commit=True, repo_root=".")


def test_list_and_trial_count(tmp_path):
    write_preregistration(_record(), dir=tmp_path)
    write_preregistration(_record(rule_id="other", trials={"family": "other", "this_trial": 1}),
                          dir=tmp_path)
    recs = list_preregistrations(tmp_path)
    assert len(recs) == 2
    assert trial_count(tmp_path, "d_forward") == 1
    assert trial_count(tmp_path, "other") == 1
    assert trial_count(tmp_path, "absent") == 0


def test_record_path_sanitizes_rule_id():
    assert record_path("a/b c", 3).name == "prereg_a_b_c_v3.json"
