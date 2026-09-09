"""Provenance contract tests (audit item P-6).

The contract is only useful if it is hard to satisfy accidentally: a missing
field, a wrong hash and an "unknown" commit must all fail, and re-stamping an
unchanged payload must be idempotent.
"""

from __future__ import annotations

import json

import pytest

from src.provenance import (
    PROVENANCE_REQUIRED,
    ProvenanceError,
    canonical_json,
    check_artifact_file,
    check_artifacts,
    check_provenance,
    require_provenance,
    sha256_of,
    stamp_artifact,
)

GOOD = dict(window={"start": "2025-09-01", "end": "2025-12-31"},
            convention="auction-close fills, adjusted-close basis, T+1",
            data_as_of="2025-12-31")


def _payload() -> dict:
    return {"label": "demo", "metrics": {"sharpe": 1.66}, "checks": {"a": True}}


def test_stamp_then_check_passes():
    art = stamp_artifact(_payload(), code_commit="a" * 40, **GOOD)
    report = check_provenance(art)
    assert report["ok"], report
    assert report["missing"] == [] and report["problems"] == []
    assert report["values"]["code_commit"] == "a" * 40
    assert report["values"]["window"] == GOOD["window"]


def test_stamp_is_idempotent_and_hash_covers_payload_only():
    a = stamp_artifact(_payload(), code_commit="a" * 40, **GOOD)
    b = stamp_artifact(_payload(), code_commit="b" * 40, **GOOD)
    # different provenance metadata, identical payload hash
    assert a["provenance"]["artifact_sha256"] == b["provenance"]["artifact_sha256"]
    # re-stamping the already-stamped artifact reproduces the same block hash
    again = stamp_artifact(a, code_commit="a" * 40, **GOOD)
    assert again["provenance"]["artifact_sha256"] == a["provenance"]["artifact_sha256"]


def test_tampered_payload_is_detected():
    art = stamp_artifact(_payload(), code_commit="a" * 40, **GOOD)
    art["metrics"]["sharpe"] = 9.99
    report = check_provenance(art)
    assert not report["ok"]
    assert any("mismatch" in p for p in report["problems"]), report


def test_missing_field_fails():
    art = stamp_artifact(_payload(), code_commit="a" * 40, **GOOD)
    del art["provenance"]["data_as_of"]
    report = check_provenance(art)
    assert not report["ok"] and report["missing"] == ["data_as_of"]


def test_unknown_commit_fails():
    art = stamp_artifact(_payload(), code_commit="unknown", **GOOD)
    report = check_provenance(art)
    assert not report["ok"]
    assert any("code_commit" in p for p in report["problems"])


def test_bad_window_and_date_are_refused_at_stamp_time():
    with pytest.raises(ProvenanceError):
        stamp_artifact(_payload(), window={"start": "2025-09-01"},
                       convention="x", data_as_of="2025-12-31", code_commit="a" * 40)
    with pytest.raises(ProvenanceError):
        stamp_artifact(_payload(), window=GOOD["window"],
                       convention="x", data_as_of="not-a-date", code_commit="a" * 40)
    with pytest.raises(ProvenanceError):
        stamp_artifact(_payload(), window=GOOD["window"],
                       convention="  ", data_as_of="2025-12-31", code_commit="a" * 40)


def test_legacy_top_level_block_is_found_but_not_citable():
    """Legacy artifacts carry the fields at top level; the checker must find them
    (so the error message is useful) yet still fail on the missing hash."""
    legacy = dict(_payload(), window=GOOD["window"], convention=GOOD["convention"])
    report = check_provenance(legacy)
    assert report["where"] == "top"
    assert not report["ok"]
    assert set(report["missing"]) == {"data_as_of", "artifact_sha256", "code_commit"}


def test_require_provenance_raises_with_detail():
    with pytest.raises(ProvenanceError) as exc:
        require_provenance({"label": "x"}, label="demo.json")
    assert "demo.json" in str(exc.value) and "window" in str(exc.value)


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"a": 1, "b": [2, {"c": 3}]}) == canonical_json({"b": [2, {"c": 3}], "a": 1})
    assert sha256_of({"a": 1, "b": 2}) == sha256_of({"b": 2, "a": 1})


def test_check_artifact_file_and_sweep(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(stamp_artifact(_payload(), code_commit="a" * 40, **GOOD)), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"label": "legacy"}), encoding="utf-8")
    assert check_artifact_file(good)["ok"]
    assert not check_artifact_file(bad)["ok"]
    sweep = check_artifacts([good, bad])
    assert sweep["n_checked"] == 2 and sweep["n_failed"] == 1
    assert sweep["failures"][0]["path"].endswith("bad.json")
    # a non-JSON file is a failure, not a crash
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    rep = check_artifact_file(broken)
    assert not rep["ok"] and "unreadable" in rep["problems"][0]


def test_required_fields_are_the_documented_five():
    assert PROVENANCE_REQUIRED == ("window", "convention", "data_as_of",
                                   "artifact_sha256", "code_commit")
