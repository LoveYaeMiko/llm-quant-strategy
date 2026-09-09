"""Ledger forking tests (forward candidate seeding + replay).

A shadow copy must start from the same account state the live book was in on the
eve of its window; starting flat silently turns the inherited sell list into a
short position.
"""

from __future__ import annotations

import pandas as pd

from src.online.order_executor import Fill
from src.paper.ledger import PaperLedger, clone_ledger_before


def _seed(path, days=("2026-09-04", "2026-09-07", "2026-09-08", "2026-09-09")) -> None:
    led = PaperLedger(str(path))
    for i, d in enumerate(days):
        led.record_day(d, 50_000.0 - i * 100, 50_000.0 - i * 90,
                       {"600000.SH": 1000.0}, [], 10_000.0)
    led.append_fill(Fill(date=days[-1], symbol="600000.SH", side="sell", shares=-1000.0,
                         price=10.0, commission=5.0, notional=10_000.0, source="auction"))
    led.close()


def test_clone_keeps_only_rows_before_the_cutoff(tmp_path):
    src = tmp_path / "prod.sqlite"
    dst = tmp_path / "cand.sqlite"
    _seed(src)
    out = clone_ledger_before(src, dst, "2026-09-09")
    assert out["days"] == 3 and out["last_date"] == "2026-09-08"
    # one positions row per recorded day (the table is date × symbol)
    assert out["positions"] == 3 and out["fills"] == 0
    clone = PaperLedger(str(dst))
    try:
        last, cash, positions = clone.latest_state()
        assert last == "2026-09-08"
        assert cash == 49_800.0
        assert positions == {"600000.SH": 1000.0}
    finally:
        clone.close()


def test_clone_does_not_touch_the_source(tmp_path):
    src = tmp_path / "prod.sqlite"
    _seed(src)
    before = src.read_bytes()
    clone_ledger_before(src, tmp_path / "cand.sqlite", "2026-09-09")
    assert src.read_bytes() == before
    led = PaperLedger(str(src))
    try:
        assert led.last_date() == "2026-09-09" and len(led.fills()) == 1
    finally:
        led.close()


def test_clone_overwrites_an_existing_destination(tmp_path):
    src = tmp_path / "prod.sqlite"
    dst = tmp_path / "cand.sqlite"
    _seed(src)
    _seed(dst, days=("2026-09-01",))
    out = clone_ledger_before(src, dst, "2026-09-08")
    assert out["days"] == 2 and out["last_date"] == "2026-09-07"


def test_clone_of_an_empty_ledger_is_empty(tmp_path):
    src = tmp_path / "empty.sqlite"
    PaperLedger(str(src)).close()
    out = clone_ledger_before(src, tmp_path / "cand.sqlite", pd.Timestamp("2026-09-09"))
    assert out["days"] == 0 and out["last_date"] is None and out["positions"] == 0
