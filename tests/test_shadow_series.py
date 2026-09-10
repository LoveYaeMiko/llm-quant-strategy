"""Shadow-series annotation tests (follow-up ①, 2026-09-10).

The point of the annotation is that a quoted number belongs to a SEGMENT: within a
segment the code, the parameters, the execution regime and the cross-section width
are constant. These tests pin the segmentation logic and the documented era table
offline (no git, no market, no DB).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import shadow_series as ss  # noqa: E402


def _row(day: str, **conv) -> dict:
    base = {"date": day, "code_commit": "abc123", "panel_symbols": 301,
            "stop": "flat 2.5%", "tr_basis": "fixed", "cash_guard": "guarded",
            "params": {"stop_lo": 0.025, "stop_hi": 0.025, "atr_mult": 1.5,
                       "stop_trigger": "close", "stop_open_minutes": 30,
                       "tail_vol_max": 0.5, "full_invest": True, "k": 6,
                       "universe": "hs300_500"},
            "live_regime": False}
    base.update(conv)
    return base


def test_panel_width_eras():
    """The cross-section width is a documented external fact, not a derivation."""
    assert ss.panel_width("2026-01-05") == 301
    assert ss.panel_width("2026-09-07") == 301
    assert ss.panel_width("2026-09-08") == 800
    assert ss.panel_width("2026-12-31") == 800
    assert ss.panel_width("2025-10-15") is None      # outside the documented table


def test_documented_era_is_the_latest_at_or_before_the_day():
    assert ss.documented_era("2026-01-05")["from"] == "2026-01-05"
    assert ss.documented_era("2026-09-03")["from"] == "2026-01-05"
    assert ss.documented_era("2026-09-04")["from"] == "2026-09-04"
    assert ss.documented_era("2026-09-08")["from"] == "2026-09-08"
    assert ss.documented_era("2026-09-09")["from"] == "2026-09-09"
    # each era must cite where its claim comes from
    for era in ss.DOCUMENTED_ERAS:
        assert era["source"] and "docs/" in era["source"] or "configs/" in era["source"]


def test_segments_split_on_every_convention_change():
    rows = [
        _row("2026-01-05"), _row("2026-01-06"), _row("2026-01-07"),
        _row("2026-01-08", live_regime=True),                       # execution regime
        _row("2026-01-09", live_regime=True, panel_symbols=800),    # cross-section width
        _row("2026-01-12", live_regime=True, panel_symbols=800,
             params={**_row("x")["params"], "stop_lo": 0.035, "stop_hi": 0.035},
             stop="flat 3.5%"),                                     # stop width
    ]
    segs = ss.segment_days(rows)
    assert len(segs) == 4
    assert [len(s["days"]) for s in segs] == [3, 1, 1, 1]
    assert segs[0]["days"] == ["2026-01-05", "2026-01-06", "2026-01-07"]
    assert segs[-1]["convention"]["stop"] == "flat 3.5%"


def test_segments_are_contiguous_and_cover_every_day():
    rows = [_row(f"2026-01-{d:02d}", live_regime=(d > 5)) for d in range(5, 10)]
    segs = ss.segment_days(rows)
    flat = [d for s in segs for d in s["days"]]
    assert flat == [r["date"] for r in rows]
    for seg in segs:
        assert seg["convention"]["date"] == seg["days"][0]


def test_segment_metrics_use_the_segment_slice_only():
    idx = pd.date_range("2026-01-05", periods=5, freq="B")
    eq = pd.Series([100.0, 110.0, 90.0, 99.0, 108.9], index=idx)
    fills = pd.DataFrame({"date": [str(d.date()) for d in idx[:3]]})
    days = [str(d.date()) for d in idx[:3]]
    m = ss._segment_metrics(eq, days, fills)
    assert m["n_days"] == 3
    assert m["return"] == pytest.approx(90.0 / 100.0 - 1.0, abs=1e-4)
    assert m["max_drawdown"] == pytest.approx(90.0 / 110.0 - 1.0, abs=1e-4)
    assert m["n_fills"] == 3
    assert m["first_equity"] == 100.0 and m["last_equity"] == 90.0


def test_segment_metrics_tolerate_days_outside_the_curve():
    idx = pd.date_range("2026-01-05", periods=2, freq="B")
    eq = pd.Series([100.0, 101.0], index=idx)
    m = ss._segment_metrics(eq, ["2026-01-05", "2026-06-01"], pd.DataFrame())
    assert m["n_days"] == 1 and m["return"] == 0.0
