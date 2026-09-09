"""Offline tests for the versioned price-adjustment anchor (defect C3).

The PIT price series is backward-adjusted against an implicit, unversioned
anchor (the newest bar of the ingest batch, ``adjust_factor == 1.0`` there).
These tests pin the contract that makes that anchor explicit and
drift-detectable — capture, canonical hashing, write/refuse/round-trip and
drift comparison — without any database and without mutating a stored price.

Every fixture is hand-built. The "re-ingest re-bases history" case is simulated
by scaling a captured factor map (a new corporate-action event multiplies every
earlier bar's factor by ``1/ex_factor``), never by touching the real store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.adjust_anchor import (
    ANCHOR_POLICY,
    HISTORY_DIR_ENV,
    RECORD_HASH_KEY,
    anchor_from_frame,
    build_anchor,
    canonical_json,
    capture_anchor,
    compare_anchors,
    factors_sha256,
    format_drift_report,
    load_anchor,
    record_sha256,
    write_anchor,
)

CAPTURED_AT = "2026-09-09T18:00:00+08:00"


def _anchor(per_symbol: dict[str, float], data_as_of: str = "2026-09-09", captured_at: str = CAPTURED_AT) -> dict:
    """A minimal anchor fingerprint with a coherent ``factors_sha256``."""
    rows = [(sym, factor, pd.Timestamp(data_as_of), 10) for sym, factor in per_symbol.items()]
    return build_anchor(rows, captured_at=captured_at)


def _frame(spec: dict[str, list[float]], dates=("2026-09-07", "2026-09-08", "2026-09-09")) -> pd.DataFrame:
    """Price-record frame: one row per (symbol, date) with the last-bar factor."""
    rows = []
    for sym, factors in spec.items():
        for date, factor in zip(dates, factors):
            rows.append(
                {
                    "symbol": sym,
                    "valid_from": pd.Timestamp(date),
                    "adjust_factor": factor,
                    "raw_close": 10.0,
                    "close": round(10.0 * factor, 4),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compare_anchors
# ---------------------------------------------------------------------------


def test_compare_identical_is_stable():
    anchor = _anchor({"000001.SZ": 1.0, "600519.SH": 0.5})
    report = compare_anchors(anchor, anchor)
    assert report["drifted"] is False
    assert report["data_as_of_moved"] is False
    assert report["data_as_of_prev"] == report["data_as_of_curr"] == "2026-09-09"
    assert report["n_common"] == 2
    assert report["n_new_symbols"] == report["n_lost_symbols"] == 0
    assert report["n_drifted"] == 0
    assert report["drifted_symbols"] == []
    assert report["max_rel_change"] == 0.0
    assert report["hash_changed"] is False
    assert report["factors_sha256_prev"] == report["factors_sha256_curr"]


def test_compare_detects_factor_drift():
    prev = _anchor({"000001.SZ": 1.0, "600519.SH": 0.5})
    curr = _anchor({"000001.SZ": 1.0, "600519.SH": 0.49})
    report = compare_anchors(prev, curr)
    assert report["drifted"] is True
    assert report["n_drifted"] == 1
    assert report["drifted_symbols"][0]["symbol"] == "600519.SH"
    assert report["drifted_symbols"][0]["prev"] == 0.5
    assert report["drifted_symbols"][0]["curr"] == 0.49
    assert report["drifted_symbols"][0]["rel_change"] == pytest.approx(-0.02)
    assert report["max_rel_change"] == pytest.approx(0.02)
    assert report["hash_changed"] is True


def test_compare_detects_lost_symbols():
    prev = _anchor({"000001.SZ": 1.0, "600519.SH": 0.5, "300750.SZ": 0.9})
    curr = _anchor({"000001.SZ": 1.0})
    report = compare_anchors(prev, curr)
    assert report["drifted"] is True
    assert report["n_lost_symbols"] == 2
    assert report["lost_symbols"] == ["300750.SZ", "600519.SH"]
    assert report["n_common"] == 1
    assert report["n_drifted"] == 0


def test_compare_new_symbols_alone_is_not_drift():
    prev = _anchor({"000001.SZ": 1.0})
    curr = _anchor({"000001.SZ": 1.0, "600519.SH": 0.5})
    report = compare_anchors(prev, curr)
    assert report["drifted"] is False
    assert report["n_new_symbols"] == 1
    assert report["new_symbols"] == ["600519.SH"]


def test_compare_moved_data_as_of_alone_is_not_drift():
    prev = _anchor({"000001.SZ": 1.0}, data_as_of="2026-09-08")
    curr = _anchor({"000001.SZ": 1.0}, data_as_of="2026-09-09")
    report = compare_anchors(prev, curr)
    assert report["data_as_of_moved"] is True
    assert report["data_as_of_prev"] == "2026-09-08"
    assert report["data_as_of_curr"] == "2026-09-09"
    assert report["drifted"] is False


def test_compare_respects_tolerance():
    prev = _anchor({"600519.SH": 1.0})
    curr = _anchor({"600519.SH": 1.0 + 1e-12})
    assert compare_anchors(prev, curr, tol=1e-9)["drifted"] is False
    assert compare_anchors(prev, curr, tol=1e-15)["drifted"] is True


def test_compare_report_is_json_serializable_and_sorted():
    prev = _anchor({"A": 1.0, "B": 1.0, "C": 1.0})
    curr = _anchor({"A": 1.0, "B": 0.5, "C": 0.75})
    report = compare_anchors(prev, curr)
    json.dumps(report)  # must not raise
    assert [d["symbol"] for d in report["drifted_symbols"]] == ["B", "C"]
    assert "DRIFTED" in format_drift_report(report)


def test_reingest_rebase_is_detected_offline():
    """A new corporate-action event multiplies every earlier bar's factor.

    This is the C3 failure mode: same symbols, same window, same parameters —
    only the (implicit) anchor moved.
    """
    frame = _frame({"600519.SH": [1.0, 1.0, 1.0], "000001.SZ": [1.0, 1.0, 1.0]})
    baseline = anchor_from_frame(frame, captured_at=CAPTURED_AT)

    rebased = frame.copy()
    rebased["adjust_factor"] = rebased["adjust_factor"] * 0.98  # one new event
    current = anchor_from_frame(rebased, captured_at=CAPTURED_AT)

    report = compare_anchors(baseline, current)
    assert baseline["data_as_of"] == current["data_as_of"] == "2026-09-09"
    assert report["data_as_of_moved"] is False
    assert report["drifted"] is True
    assert report["n_drifted"] == 2
    assert report["max_rel_change"] == pytest.approx(0.02)
    assert all(d["rel_change"] == pytest.approx(-0.02) for d in report["drifted_symbols"])


# ---------------------------------------------------------------------------
# capture (offline, from a frame)
# ---------------------------------------------------------------------------


def test_anchor_from_frame_reports_last_bar_factor_and_counts():
    frame = _frame({"600519.SH": [0.9, 0.8, 0.5], "000001.SZ": [1.0, 1.0, 1.0]})
    anchor = anchor_from_frame(frame, captured_at=CAPTURED_AT)
    assert anchor["anchor_policy"] == ANCHOR_POLICY
    assert anchor["captured_at"] == CAPTURED_AT
    assert anchor["data_as_of"] == "2026-09-09"
    assert anchor["n_symbols"] == 2
    assert anchor["n_rows"] == 6
    # the factor of the LAST bar, not the first
    assert anchor["per_symbol_factor"] == {"000001.SZ": 1.0, "600519.SH": 0.5}
    assert anchor["factor_stats"] == {
        "n_symbols_factor_ne_1": 1,
        "n_symbols_factor_lt_1": 1,
        "n_symbols_factor_gt_1": 0,
        "min": 0.5,
        "max": 1.0,
    }
    assert anchor["top_factors"][0] == ["600519.SH", 0.5]
    assert anchor["factors_sha256"] == factors_sha256({"000001.SZ": 1.0, "600519.SH": 0.5})


def test_anchor_from_frame_is_order_and_subset_invariant():
    frame = _frame({"600519.SH": [1.0, 0.8, 0.5], "000001.SZ": [1.0, 1.0, 1.0]})
    shuffled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    a = anchor_from_frame(frame, captured_at=CAPTURED_AT)
    b = anchor_from_frame(shuffled, captured_at=CAPTURED_AT)
    assert a == b

    subset = anchor_from_frame(frame, symbols=["600519.SH"], captured_at=CAPTURED_AT)
    assert subset["n_symbols"] == 1 and subset["n_rows"] == 3
    assert subset["per_symbol_factor"] == {"600519.SH": 0.5}


def test_anchor_from_frame_treats_missing_or_nan_factor_as_one():
    frame = _frame({"600519.SH": [1.0, 1.0, 1.0]})
    frame.loc[frame.index[-1], "adjust_factor"] = float("nan")
    assert anchor_from_frame(frame, captured_at=CAPTURED_AT)["per_symbol_factor"] == {"600519.SH": 1.0}

    without = frame.drop(columns=["adjust_factor"])
    assert anchor_from_frame(without, captured_at=CAPTURED_AT)["per_symbol_factor"] == {"600519.SH": 1.0}


def test_build_anchor_handles_empty_store():
    anchor = build_anchor([], captured_at=CAPTURED_AT)
    assert anchor["n_symbols"] == 0 and anchor["n_rows"] == 0
    assert anchor["data_as_of"] is None
    assert anchor["top_factors"] == [] and anchor["per_symbol_factor"] == {}
    assert anchor["factor_stats"]["min"] is None
    assert compare_anchors(anchor, anchor)["drifted"] is False


def test_capture_anchor_rejects_empty_url():
    with pytest.raises(ValueError):
        capture_anchor("")


# ---------------------------------------------------------------------------
# canonical JSON / hashing
# ---------------------------------------------------------------------------


def test_canonical_json_is_sorted_compact_and_unicode_literal():
    text = canonical_json({"b": 1, "a": "中文", "c": [2, 1]})
    assert text == '{"a":"中文","b":1,"c":[2,1]}'
    assert " " not in text


def test_factors_sha256_is_key_order_invariant_and_stable():
    a = factors_sha256({"600519.SH": 0.5, "000001.SZ": 1.0})
    b = factors_sha256({"000001.SZ": 1.0, "600519.SH": 0.5})
    assert a == b
    assert a == factors_sha256({"000001.SZ": 1.0, "600519.SH": 0.5})  # same input → same hash
    assert a != factors_sha256({"000001.SZ": 1.0, "600519.SH": 0.5000001})
    assert len(a) == 64


def test_record_sha256_ignores_key_order_and_the_hash_field():
    payload = _anchor({"600519.SH": 0.5})
    reordered = {k: payload[k] for k in sorted(payload, reverse=True)}
    assert record_sha256(payload) == record_sha256(reordered)
    assert record_sha256({**payload, RECORD_HASH_KEY: "bogus"}) == record_sha256(payload)


# ---------------------------------------------------------------------------
# write / load
# ---------------------------------------------------------------------------


def test_write_anchor_refuses_overwrite_and_force_overrides(tmp_path: Path):
    path = tmp_path / "anchor.json"
    first = _anchor({"600519.SH": 0.5})
    write_anchor(path, first)
    with pytest.raises(FileExistsError):
        write_anchor(path, _anchor({"600519.SH": 0.4}))
    assert load_anchor(path)["per_symbol_factor"] == {"600519.SH": 0.5}

    write_anchor(path, _anchor({"600519.SH": 0.4}), force=True)
    assert load_anchor(path)["per_symbol_factor"] == {"600519.SH": 0.4}


def test_write_anchor_round_trip_and_integrity(tmp_path: Path):
    path = tmp_path / "anchor.json"
    anchor = _anchor({"000001.SZ": 1.0, "600519.SH": 0.456273903049096})
    write_anchor(path, anchor)

    raw = path.read_text(encoding="utf-8")
    assert "\n" not in raw and ": " not in raw  # canonical: no whitespace
    stored = json.loads(raw)
    assert stored[RECORD_HASH_KEY] == record_sha256(anchor)

    loaded = load_anchor(path)
    assert {k: v for k, v in loaded.items() if k != RECORD_HASH_KEY} == anchor
    assert loaded[RECORD_HASH_KEY] == record_sha256(loaded)

    # a hand-edited payload must not load silently
    tampered = json.loads(raw)
    tampered["per_symbol_factor"]["600519.SH"] = 0.1
    path.write_text(canonical_json(tampered), encoding="utf-8")
    with pytest.raises(ValueError):
        load_anchor(path)
    assert load_anchor(path, verify=False)["per_symbol_factor"]["600519.SH"] == 0.1


def test_write_anchor_is_byte_deterministic_regardless_of_key_order(tmp_path: Path):
    a = _anchor({"000001.SZ": 1.0, "600519.SH": 0.5})
    b = {k: a[k] for k in reversed(list(a))}
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    write_anchor(p1, a)
    write_anchor(p2, b)
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")
    assert record_sha256(a) == record_sha256(b)


def test_write_anchor_appends_to_configured_history(tmp_path: Path):
    path = tmp_path / "adjust_anchor.json"
    history = tmp_path / "history"
    write_anchor(path, _anchor({"600519.SH": 0.5}), history_dir=history)
    write_anchor(path, _anchor({"600519.SH": 0.5}), force=True, history_dir=history)
    copies = sorted(history.glob("*.json"))
    assert len(copies) == 2  # same captured_at → suffixed, never clobbered
    assert all(load_anchor(c)["per_symbol_factor"] == {"600519.SH": 0.5} for c in copies)


def test_write_anchor_history_dir_from_env(tmp_path: Path, monkeypatch):
    history = tmp_path / "env_history"
    monkeypatch.setenv(HISTORY_DIR_ENV, str(history))
    write_anchor(tmp_path / "anchor.json", _anchor({"600519.SH": 0.5}))
    assert len(list(history.glob("*.json"))) == 1


def test_write_anchor_without_history_configuration_writes_only_the_target(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(HISTORY_DIR_ENV, raising=False)
    path = tmp_path / "anchor.json"
    write_anchor(path, _anchor({"600519.SH": 0.5}))
    assert path.is_file()
    assert not (tmp_path / "history").exists()
