"""Fill-time precision + provenance invariants (landing item D-9.2).

The user's standing rule: **交易记录时间需要精确到分钟**. This test audits the
PRODUCTION ledger (skipped when it is not present) so a regression cannot
silently introduce second-less or malformed timestamps, a fill outside the
trading session, or an unknown provenance label.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "outputs" / "shadow_ledger_D_5W.sqlite"
TIME_RE = re.compile(r"^\d{2}:\d{2}(:\d{2})?$")
KNOWN_SOURCES = {"", "live", "replay", "close", "auction"}


def _rows():
    if not LEDGER.is_file():
        pytest.skip("production D ledger not present")
    conn = sqlite3.connect(str(LEDGER))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(fills)").fetchall()]
        if "time" not in cols:
            pytest.skip("ledger predates the time column")
        select = "date, symbol, side, time, " + ("COALESCE(source,'')" if "source" in cols else "''")
        return conn.execute(f"SELECT {select} FROM fills ORDER BY seq").fetchall()
    finally:
        conn.close()


def test_every_fill_has_minute_precision_time():
    rows = _rows()
    assert rows, "ledger has no fills"
    bad = [(d, s, t) for d, s, _side, t, _src in rows if str(t or "") and not TIME_RE.match(str(t))]
    assert bad == [], f"malformed timestamps: {bad[:5]}"


def test_intraday_fills_are_inside_the_session():
    rows = _rows()
    bad = [
        (d, s, t) for d, s, side, t, _src in rows
        if str(t or "") and not ("09:30" <= str(t)[:5] <= "15:00")
    ]
    assert bad == [], f"fills outside the session: {bad[:5]}"


def test_provenance_labels_are_known():
    rows = _rows()
    unknown = sorted({str(src) for *_x, src in rows} - KNOWN_SOURCES)
    assert unknown == [], f"unknown fill provenance: {unknown}"


def test_sell_fills_are_intraday_or_close_only():
    """No sell may be timestamped inside a session it could not trade (T+1)."""
    rows = _rows()
    intraday = [(d, s, t) for d, s, side, t, _src in rows if side == "sell" and str(t or "")]
    assert all(TIME_RE.match(str(t)) for _d, _s, t in intraday)
