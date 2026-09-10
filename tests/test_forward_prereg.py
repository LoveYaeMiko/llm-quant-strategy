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
    policy_fingerprint,
    prereg_gate,
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
        policy_sha256="p" * 64,
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


def test_write_refuses_a_record_with_no_policy_binding(tmp_path):
    """A freeze that cannot bind is not a freeze — refuse it at write time.

    ``verify_preregistration`` only checks the six template fields and the
    self-hash, so a record written without ``policy_sha256`` verifies happily and
    then fails :func:`prereg_gate` forever. Hit on 2026-09-10 by a re-signature
    script that bypassed ``scripts/prereg.py new`` (which stamps the fingerprint).
    """
    unbound = _record(policy_sha256=None)
    assert unbound["policy_sha256"] is None          # new_record does not invent one
    with pytest.raises(PreregError) as exc:
        write_preregistration(unbound, dir=tmp_path)
    assert "no policy binding" in str(exc.value)
    # the CLI's `config_sha256` spelling (records frozen before the rename) is
    # accepted as a binding too
    legacy = _record(policy_sha256=None)
    legacy["config_sha256"] = "c" * 64
    legacy["record_sha256"] = record_sha256(legacy)
    assert write_preregistration(legacy, dir=tmp_path).is_file()


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


# --------------------------------------------------------------------------- #
# the binding gate (audit H-1, 2026-09-10): a freeze that is not enforced is a
# decoration — the whole point is that editing a threshold after seeing the
# results must FAIL until a new version is frozen.
# --------------------------------------------------------------------------- #
class _Cfg:
    """Minimal config stub for :func:`policy_fingerprint`."""

    def __init__(self, payload=None):
        self._payload = payload if payload is not None else {
            "forward": {"risk_gate": {"hard": {"violations_max": 0}}},
            "deployment": {"mode": "observe"},
            "shadow": {"accounts": [{"name": "D_5W", "pb_stop_lo": 0.035}]},
        }

    def get(self, path, default=None):
        node = self._payload
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


WINDOW = {"start": "2026-09-10", "end": "2027-03-09"}   # matches _record()'s scope.window


def test_policy_fingerprint_tracks_the_policy_surface():
    base = _Cfg()
    a = policy_fingerprint(base)
    changed = dict(base._payload)
    changed["forward"] = {"risk_gate": {"hard": {"violations_max": 1}}}
    assert policy_fingerprint(_Cfg(changed)) != a, "a threshold change must move the hash"
    unrelated = dict(base._payload, llm={"api_key": "sk-rotated"})
    assert policy_fingerprint(_Cfg(unrelated)) == a, "unrelated keys must not invalidate a freeze"


def test_prereg_gate_passes_a_properly_bound_record():
    rec = _record(policy_sha256="p" * 64)
    out = prereg_gate(record=rec, window=WINDOW, data_as_of="2026-09-10",
                      policy_sha256="p" * 64, code_commit="a" * 40)
    assert out["ok"], out
    assert out["frozen_before_window"] and out["window_match"]
    assert out["policy_sha256_match"] and out["code_commit_match"]


def test_prereg_gate_fails_without_a_record():
    out = prereg_gate(record=None, window=WINDOW, data_as_of="2026-09-10",
                      policy_sha256="p" * 64, code_commit="a" * 40)
    assert not out["ok"] and "no verified pre-registration" in out["issues"][0]


def test_prereg_gate_fails_on_policy_drift():
    rec = _record(policy_sha256="p" * 64)
    out = prereg_gate(record=rec, window=WINDOW, data_as_of="2026-09-10",
                      policy_sha256="q" * 64, code_commit="a" * 40)
    assert not out["ok"] and not out["policy_sha256_match"]
    assert any("fingerprint" in i for i in out["issues"])


def test_prereg_gate_fails_on_a_hand_picked_window():
    rec = _record(policy_sha256="p" * 64)
    out = prereg_gate(record=rec, window={"start": "2026-10-01", "end": "2027-03-09"},
                      data_as_of="2026-10-01", policy_sha256="p" * 64, code_commit="a" * 40)
    assert not out["ok"] and not out["window_match"]
    assert any("sub-range" in i for i in out["issues"])


def test_prereg_gate_fails_when_frozen_inside_the_window():
    rec = _record(frozen_at="2026-09-15T10:00:00", policy_sha256="p" * 64)
    out = prereg_gate(record=rec, window=WINDOW, data_as_of="2026-09-15",
                      policy_sha256="p" * 64, code_commit="a" * 40)
    assert not out["ok"] and not out["frozen_before_window"]
    assert any("not before the window start" in i for i in out["issues"])


def test_prereg_gate_code_drift_needs_an_explicit_waiver():
    rec = _record(policy_sha256="p" * 64)
    strict = prereg_gate(record=rec, window=WINDOW, data_as_of="2026-09-10",
                         policy_sha256="p" * 64, code_commit="b" * 40)
    assert not strict["ok"] and not strict["code_commit_match"]
    waived = prereg_gate(record=rec, window=WINDOW, data_as_of="2026-09-10",
                         policy_sha256="p" * 64, code_commit="b" * 40, allow_code_drift=True)
    assert waived["ok"] and waived["waived"] and "waived" in waived["waiver_reason"]


def test_prereg_gate_fails_on_a_dirty_working_tree():
    """Records bound the OLD way (commit only) must still catch a dirty tree."""
    rec = _record(policy_sha256="p" * 64)
    dirty = prereg_gate(record=rec, window=WINDOW, data_as_of="2026-09-10",
                        policy_sha256="p" * 64, code_commit="a" * 40, code_dirty=True)
    assert not dirty["ok"] and dirty["code_commit_match"] is True
    assert any("DIRTY" in i for i in dirty["issues"])
    ok = prereg_gate(record=rec, window=WINDOW, data_as_of="2026-09-10",
                     policy_sha256="p" * 64, code_commit="a" * 40, code_dirty=False)
    assert ok["ok"]


def test_prereg_gate_binds_to_the_code_fingerprint_not_the_commit():
    """A docs-only commit must not invalidate a freeze; a code edit must.

    Binding to HEAD made every commit — even a README — force a re-freeze, which
    trains everyone to re-freeze reflexively and defeats the lock (found
    2026-09-10 when an evidence commit tripped the gate it had just passed).
    """
    rec = _record(policy_sha256="p" * 64, code_fingerprint="c" * 64)
    # same code content, different commit: still bound
    same = prereg_gate(record=rec, window=WINDOW, data_as_of="2026-09-10",
                       policy_sha256="p" * 64, code_commit="b" * 40,
                       code_fingerprint="c" * 64, code_dirty=False)
    assert same["ok"], same["issues"]
    # code content changed (committed or not): NOT bound
    moved = prereg_gate(record=rec, window=WINDOW, data_as_of="2026-09-10",
                        policy_sha256="p" * 64, code_commit="b" * 40,
                        code_fingerprint="d" * 64)
    assert not moved["ok"] and not moved["code_commit_match"]
    assert any("behaviour-deciding code changed" in i for i in moved["issues"])


def test_code_fingerprint_tracks_content_not_commit(tmp_path):
    from src.provenance import code_fingerprint

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "s.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "README.md").write_text("hello\n", encoding="utf-8")
    first = code_fingerprint(tmp_path)
    # a docs-only edit does not move it
    (tmp_path / "docs" / "README.md").write_text("hello again\n", encoding="utf-8")
    assert code_fingerprint(tmp_path) == first
    # a code edit does
    (tmp_path / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert code_fingerprint(tmp_path) != first
    # a non-python artifact under a code prefix does not
    (tmp_path / "src" / "notes.txt").write_text("nope\n", encoding="utf-8")
    assert code_fingerprint(tmp_path) != first  # (already moved by a.py)
    before = code_fingerprint(tmp_path)
    (tmp_path / "src" / "notes.txt").write_text("changed\n", encoding="utf-8")
    assert code_fingerprint(tmp_path) == before
