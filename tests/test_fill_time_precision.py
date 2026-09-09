"""Fill-time precision + provenance invariants (landing item D-9.2).

The user's standing rule: **交易记录时间需要精确到分钟**. This audits the
PRODUCTION ledger (skipped when it is not present) so a regression cannot
silently introduce second-less or malformed timestamps, a fill outside the
trading session, or an unknown provenance label.

Every test first asserts the ledger actually contains the rows it audits — the
earlier version wrapped each check in ``if str(t or "")``, so on a ledger whose
fills were almost all close fills (197 of 252 have no ``time``) it passed
without checking anything.
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
CLOSE_TIMES = {"", "15:00"}


def _rows():
    if not LEDGER.is_file():
        pytest.skip("production D ledger not present")
    conn = sqlite3.connect(str(LEDGER))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(fills)").fetchall()]
        if "time" not in cols:
            pytest.skip("ledger predates the time column")
        select = "date, symbol, side, time, " + ("COALESCE(source,'')" if "source" in cols else "''")
        return conn.execute(f"SELECT {select} FROM fills ORDER BY seq").fetchall(), cols
    finally:
        conn.close()


def test_ledger_has_both_intraday_and_close_fills():
    rows, _cols = _rows()
    assert rows, "ledger has no fills"
    intraday = [r for r in rows if str(r[3] or "")]
    close = [r for r in rows if not str(r[3] or "")]
    # the audit only means something when both kinds exist
    assert intraday, "no intraday fills to audit"
    assert close, "no close fills to audit"


def test_every_non_empty_time_has_minute_precision():
    rows, _cols = _rows()
    bad = [(d, s, t) for d, s, _side, t, _src in rows if str(t or "") and not TIME_RE.match(str(t))]
    assert bad == [], f"malformed timestamps: {bad[:5]}"


def test_close_fills_use_the_auction_stamp():
    rows, _cols = _rows()
    bad = [(d, s, t) for d, s, _side, t, _src in rows if str(t or "") in CLOSE_TIMES and t not in CLOSE_TIMES]
    assert bad == []
    # a close fill is exactly "" or 15:00
    odd = [(d, s, t) for d, s, _side, t, _src in rows
           if not str(t or "") and t not in CLOSE_TIMES]
    assert odd == []


def test_intraday_fills_are_inside_the_session():
    rows, _cols = _rows()
    bad = [
        (d, s, t) for d, s, side, t, _src in rows
        if str(t or "") and not ("09:30" <= str(t)[:5] <= "15:00")
    ]
    assert bad == [], f"fills outside the session: {bad[:5]}"


def test_provenance_column_exists_and_labels_are_known():
    rows, cols = _rows()
    assert "source" in cols, "ledger has no source column — the migration never ran"
    unknown = sorted({str(src) for *_x, src in rows} - KNOWN_SOURCES)
    assert unknown == [], f"unknown fill provenance: {unknown}"


def test_sell_fills_are_intraday_or_close_only():
    """No sell may be timestamped inside a session it could not trade (T+1)."""
    rows, _cols = _rows()
    sells = [(d, s, t) for d, s, side, t, _src in rows if side == "sell"]
    assert sells, "no sell fills to audit"
    assert all(TIME_RE.match(str(t)) or str(t) in CLOSE_TIMES for _d, _s, t in sells)
