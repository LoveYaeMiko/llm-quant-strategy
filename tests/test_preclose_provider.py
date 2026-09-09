"""Close-execution provider tests (audit item 2 wiring).

The 15:00 auction layer must execute EXACTLY the 14:50 order list, and a forward
candidate must inherit the production list (``pb_preclose_account``) so the paired
comparison isolates the one rule under test.
"""

from __future__ import annotations

import json

import pandas as pd

from src import cli


def _write_orders(root, name: str, date: str, orders: list[dict]) -> None:
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / f"preclose_orders_{name}.json").write_text(
        json.dumps({"date": date, "orders": orders}), encoding="utf-8"
    )


def test_before_the_live_date_the_runner_computes_its_own_targets(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    prov = cli._make_preclose_provider(
        {"name": "D_5W", "pb_live_intraday_from": "2026-09-04"}
    )
    assert prov("2026-09-03") == "__normal__"


def test_on_a_live_date_without_orders_no_close_trades(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    prov = cli._make_preclose_provider(
        {"name": "D_5W", "pb_live_intraday_from": "2026-09-04"}
    )
    assert prov("2026-09-09") is None


def test_orders_are_returned_only_for_their_own_date(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    _write_orders(tmp_path, "D_5W", "2026-09-09",
                  [{"symbol": "601872.SH", "side": "sell", "shares": -1300}])
    prov = cli._make_preclose_provider(
        {"name": "D_5W", "pb_live_intraday_from": "2026-09-04"}
    )
    assert prov("2026-09-09") == [{"symbol": "601872.SH", "side": "sell", "shares": -1300}]
    # a stale file (another day's list) must NOT be executed
    assert prov("2026-09-10") is None


def test_candidate_inherits_the_production_order_list(monkeypatch, tmp_path):
    """The candidate reads D_5W's list, not its own (which never exists)."""
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    _write_orders(tmp_path, "D_5W", "2026-09-09",
                  [{"symbol": "600000.SH", "side": "buy", "shares": 500}])
    _write_orders(tmp_path, "D_5W_FWD_ATR", "2026-09-09",
                  [{"symbol": "999999.SZ", "side": "buy", "shares": 100}])
    prov = cli._make_preclose_provider({
        "name": "D_5W_FWD_ATR", "pb_preclose_account": "D_5W",
        "pb_live_intraday_from": "2026-09-04",
    })
    orders = prov("2026-09-09")
    assert orders == [{"symbol": "600000.SH", "side": "buy", "shares": 500}]


def test_corrupt_order_file_is_treated_as_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs" / "preclose_orders_D_5W.json").write_text("{not json", encoding="utf-8")
    prov = cli._make_preclose_provider(
        {"name": "D_5W", "pb_live_intraday_from": "2026-09-04"}
    )
    assert prov("2026-09-09") is None


def test_no_live_date_always_uses_the_normal_path(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    _write_orders(tmp_path, "D_5W", "2026-09-09", [{"symbol": "600000.SH"}])
    prov = cli._make_preclose_provider({"name": "D_5W"})
    assert prov(pd.Timestamp("2026-09-09")) == "__normal__"
