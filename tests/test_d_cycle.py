"""D model-cycle loop — embargo windows + forward promotion gate logic.

The heavy refit/challenger paths need the PIT store and are exercised by the
production smoke; these tests lock the PURE decision logic (cutoff embargo,
margin gate, legality veto, artifact isolation/promotion moves) on synthetic
data.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.config import Config
from src.data.synthetic import make_synthetic_market
from src.d_cycle import _deploy_from, _last_month_cutoff, decide_promotion
from src.online.order_executor import Fill
from src.paper.ledger import PaperLedger

CFG = Config({"d_model_cycle": {
    "enabled": True,
    "challenger_ledger": "outputs/shadow_ledger_D_5W_CH.sqlite",
    "margin_pp": 0.3,
    "min_fills": 5,
    "window_days": 30,
    # Promotion is human-in-the-loop by default (2026-09-09); this fixture
    # exercises the automatic path explicitly.
    "auto_promote": True,
}})


def _market(days=90, seed=3):
    return make_synthetic_market(symbols=8, days=90, seed=seed)


def _write_ledger(db_path, market, n_days, daily_g=0.0):
    """Deterministic equity curve: eq *= (1+daily_g) per day; 5+ legal buy fills."""
    led = PaperLedger(str(db_path))
    dates = sorted(market.price_panel.index)[:n_days]
    a01 = market.price_panel["A01"]
    eq = 50_000.0
    for i, d in enumerate(dates):
        eq = eq * (1.0 + daily_g)
        fills = []
        if i > 0 and i % 4 == 0:
            px = round(float(a01.loc[d]), 2)
            fills.append(Fill(str(d.date()), "A01", "buy", 100.0, px, 5.0, 100.0 * px))
        led.record_day(d, cash=10_000.0, equity=eq, positions={"A01": 100.0},
                       fills=fills, gross_exposure=0.0)
    led.close()


@pytest.fixture
def cycle_paths(tmp_path, monkeypatch):
    import src.d_cycle as dcy

    monkeypatch.setattr(dcy, "ROOT", tmp_path)
    monkeypatch.setattr(dcy, "CHALLENGER_DIR", tmp_path / "models_challenger")
    monkeypatch.setattr(dcy, "STATE_PATH", tmp_path / "d_model_cycle.json")
    (tmp_path / "outputs").mkdir(exist_ok=True)
    dcy.CHALLENGER_DIR.mkdir(exist_ok=True)
    (dcy.CHALLENGER_DIR / "ml_ch.txt").write_text("fake-model", encoding="utf-8")
    (dcy.CHALLENGER_DIR / "ml_ch.json").write_text("{}", encoding="utf-8")
    (dcy.CHALLENGER_DIR / "active.json").write_text(
        json.dumps({"model": str(dcy.CHALLENGER_DIR / "ml_ch.txt"),
                    "meta": str(dcy.CHALLENGER_DIR / "ml_ch.json")}),
        encoding="utf-8",
    )
    return dcy


def _patch_artifact_dir(tmp_path, monkeypatch):
    import src.paper.ml_book as mlb

    dest = tmp_path / "models"
    dest.mkdir(exist_ok=True)
    monkeypatch.setattr(mlb, "_ARTIFACT_DIR", dest)
    return dest


def test_cutoff_embargo_and_deploy_boundary():
    # fake market over Jun-Sep 2026 so the month boundary exists
    idx = pd.bdate_range("2026-06-01", "2026-09-10")
    panel = pd.DataFrame({"A01": 10.0 + 0.01 * pd.Series(range(len(idx)), dtype=float).to_numpy()},
                         index=idx)
    market = type("M", (), {"price_panel": panel})()
    today = pd.Timestamp("2026-09-07")  # first Sunday of September 2026
    cutoff = pd.Timestamp(_last_month_cutoff(market, today=today))
    deploy = pd.Timestamp(_deploy_from(market, pd.Timestamp("2026-08-31")))
    aug = idx[idx <= pd.Timestamp("2026-08-31")]
    assert cutoff == aug[-1 - 10]  # 10 trading days before month end
    assert deploy == idx[idx > pd.Timestamp("2026-08-31")][0]  # first Sep bar
    assert deploy > cutoff + pd.Timedelta(days=10)  # label embargo fully expired


def test_decide_promotes_winner(cycle_paths, tmp_path, monkeypatch):
    import src.d_cycle as dcy

    market = _market()
    dest = _patch_artifact_dir(tmp_path, monkeypatch)
    dates = sorted(market.price_panel.index)
    _write_ledger(tmp_path / "outputs" / "shadow_ledger_D_5W.sqlite", market, 60, daily_g=0.0001)
    _write_ledger(tmp_path / "outputs" / "shadow_ledger_D_5W_CH.sqlite", market, 60, daily_g=0.0006)

    res = decide_promotion(CFG, market=market, today=dates[59] + pd.Timedelta(days=1))
    assert res["ok"] is True
    d = res["decision"]
    assert d["promote"] is True, d["reasons"]
    assert d["promoted_artifact"].endswith("ml_ch.json")
    assert (dest / "ml_ch.json").is_file()
    assert (dest / "ml_ch.txt").is_file()
    state = json.loads((tmp_path / "d_model_cycle.json").read_text(encoding="utf-8"))
    assert state["history"][-1]["promote"] is True


def test_decide_drops_loser(cycle_paths, tmp_path, monkeypatch):
    import src.d_cycle as dcy

    market = _market()
    dest = _patch_artifact_dir(tmp_path, monkeypatch)
    dates = sorted(market.price_panel.index)
    _write_ledger(tmp_path / "outputs" / "shadow_ledger_D_5W.sqlite", market, 60, daily_g=0.0006)
    _write_ledger(tmp_path / "outputs" / "shadow_ledger_D_5W_CH.sqlite", market, 60, daily_g=0.0001)

    res = decide_promotion(CFG, market=market, today=dates[59] + pd.Timedelta(days=1))
    d = res["decision"]
    assert d["promote"] is False
    assert d["challenger_dropped"] is True
    assert not list(dcy.CHALLENGER_DIR.glob("ml_*"))
    assert not (dcy.CHALLENGER_DIR / "active.json").exists()
    assert not list(dest.glob("ml_*"))


def test_decide_legality_veto(cycle_paths, tmp_path, monkeypatch):
    import src.d_cycle as dcy

    market = _market()
    dest = _patch_artifact_dir(tmp_path, monkeypatch)
    dates = sorted(market.price_panel.index)
    _write_ledger(tmp_path / "outputs" / "shadow_ledger_D_5W.sqlite", market, 60, daily_g=0.0001)
    _write_ledger(tmp_path / "outputs" / "shadow_ledger_D_5W_CH.sqlite", market, 60, daily_g=0.0006)

    # inject a same-day flip (T+1 violation) into the challenger ledger
    led = PaperLedger(str(tmp_path / "outputs" / "shadow_ledger_D_5W_CH.sqlite"))
    d0 = str(dates[10].date())
    led.append_fill(Fill(d0, "A01", "buy", 100.0, 10.0, 5.0, 1000.0))
    led.append_fill(Fill(d0, "A01", "sell", -100.0, 10.5, 5.0, 1050.0))
    led.close()

    res = decide_promotion(CFG, market=market, today=dates[59] + pd.Timedelta(days=1))
    d = res["decision"]
    assert d["promote"] is False
    assert d["violations"]["t_plus_1"] >= 1
    assert any("合法性" in r for r in d["reasons"])
    assert not list(dest.glob("ml_*"))


def test_decide_short_window_reports_error(cycle_paths, tmp_path, monkeypatch):
    market = _market()
    dest = _patch_artifact_dir(tmp_path, monkeypatch)
    dates = sorted(market.price_panel.index)
    _write_ledger(tmp_path / "outputs" / "shadow_ledger_D_5W.sqlite", market, 60, daily_g=0.0006)
    _write_ledger(tmp_path / "outputs" / "shadow_ledger_D_5W_CH.sqlite", market, 10, daily_g=0.0006)

    res = decide_promotion(CFG, market=market, today=dates[59] + pd.Timedelta(days=1))
    assert res["ok"] is False
    assert "window too short" in res["error"]
