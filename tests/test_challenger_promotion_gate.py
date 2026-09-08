"""Challenger promotion gate tests (landing item: human-in-the-loop promotion).

``decide_promotion`` used to swap the production model automatically whenever a
30-day, ≥5-fill challenger window beat the incumbent by the configured margin.
A window that short cannot distinguish a real edge (Sharpe SE ≈ √(252/N); see
docs/D_TRACK_EVIDENCE.md §五), so the gate now reports "qualifies" and KEEPS the
challenger artifacts, but only promotes when ``d_model_cycle.auto_promote`` is
explicitly true.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import d_cycle


class _Cfg:
    def __init__(self, section: dict) -> None:
        self._s = section

    def section(self, name: str):
        return self._s if name == "shadow" else {}

    def get(self, key, default=None):
        if key == "d_model_cycle":
            return self._s
        return default


def _ledger(path: Path, values: list[float], dates: list[str]) -> None:
    import sqlite3

    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS daily_state (date TEXT PRIMARY KEY, cash REAL, equity REAL, "
        "gross_exposure REAL DEFAULT 0, n_positions INTEGER DEFAULT 0, n_fills INTEGER DEFAULT 0, "
        "commission REAL DEFAULT 0, notional REAL DEFAULT 0);"
        "CREATE TABLE IF NOT EXISTS fills (seq INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, symbol TEXT, "
        "side TEXT, shares REAL, price REAL, commission REAL, notional REAL, time TEXT DEFAULT '', source TEXT DEFAULT '');"
    )
    for d, v in zip(dates, values):
        conn.execute("INSERT OR REPLACE INTO daily_state (date, cash, equity) VALUES (?, ?, ?)", (d, 0.0, v))
    # ≥ min_fills challenger fills so the "enough" gate can pass
    for i in range(6):
        conn.execute(
            "INSERT INTO fills (date, symbol, side, shares, price, commission, notional) "
            "VALUES (?, '600000.SH', 'buy', 100, 10.0, 5.0, 1000.0)",
            (dates[i],),
        )
    conn.commit()
    conn.close()


@pytest.fixture()
def cycle_dirs(tmp_path, monkeypatch):
    challenger = tmp_path / "challenger"
    challenger.mkdir()
    monkeypatch.setattr(d_cycle, "CHALLENGER_DIR", challenger)
    monkeypatch.setattr(d_cycle, "ROOT", tmp_path)
    (tmp_path / "outputs").mkdir(exist_ok=True)
    return tmp_path, challenger


def _prepare(cycle_dirs, monkeypatch, *, auto: bool, challenger_beats: bool = True):
    root, challenger = cycle_dirs
    dates = list(pd.bdate_range("2026-08-01", periods=25).strftime("%Y-%m-%d"))
    ch_path = challenger / "shadow_ledger_D_5W_CH.sqlite"
    _ledger(ch_path, [100.0 + (i if challenger_beats else -i) for i in range(25)], dates)
    main_path = root / "outputs" / "shadow_ledger_D_5W.sqlite"
    _ledger(main_path, [100.0 for _ in range(25)], dates)
    monkeypatch.setattr(d_cycle, "_challenger_ledger_path", lambda cfg: ch_path)
    monkeypatch.setattr(d_cycle, "legality_audit", lambda *a, **k: {}, raising=False)
    # audit_tracks is imported inside the function; patch it at the source module
    import scripts.audit_tracks as at

    monkeypatch.setattr(at, "legality_audit", lambda *a, **k: {})
    monkeypatch.setattr(d_cycle, "_update_state", lambda decision: None)
    monkeypatch.setattr(d_cycle, "_market_data", lambda *a, **k: object(), raising=False)
    cfg = _Cfg({"margin_pp": 0.3, "min_fills": 5, "window_days": 60, "auto_promote": auto})
    return cfg


def test_qualifying_challenger_is_not_auto_promoted(cycle_dirs, monkeypatch):
    cfg = _prepare(cycle_dirs, monkeypatch, auto=False)
    out = d_cycle.decide_promotion(cfg, market=object(), today=pd.Timestamp("2026-09-08"))
    assert out["ok"] is True
    decision = out["decision"]
    assert decision["qualifies"] is True
    assert decision["promote"] is False
    assert decision["pending_human_approval"] is True
    assert decision.get("challenger_kept") is True
    assert "auto_promote=false" in decision["reasons"][0]


def test_auto_promote_true_swaps_the_model(cycle_dirs, monkeypatch):
    cfg = _prepare(cycle_dirs, monkeypatch, auto=True)
    root, challenger = cycle_dirs
    # NEVER let the promotion copy into the real outputs/models: redirect the
    # artifact dir into the tmp tree (a previous version of this test polluted
    # the production model dir with a fixture artifact and broke
    # _resolve_artifact's newest-wins lookup).
    import src.paper.ml_book as ml_book

    fake_artifacts = root / "models"
    fake_artifacts.mkdir(exist_ok=True)
    monkeypatch.setattr(ml_book, "_ARTIFACT_DIR", fake_artifacts)
    (challenger / "active.json").write_text(
        json.dumps({"meta": str(challenger / "ml_x.json"), "model": str(challenger / "ml_x.txt")}),
        encoding="utf-8",
    )
    (challenger / "ml_x.json").write_text("{}", encoding="utf-8")
    (challenger / "ml_x.txt").write_text("model", encoding="utf-8")
    out = d_cycle.decide_promotion(cfg, market=object(), today=pd.Timestamp("2026-09-08"))
    assert out["decision"]["qualifies"] is True
    assert out["decision"]["auto_promote"] is True
    assert (fake_artifacts / "ml_x.json").is_file()
    assert (fake_artifacts / "ml_x.txt").is_file()


def test_losing_challenger_is_dropped(cycle_dirs, monkeypatch):
    cfg = _prepare(cycle_dirs, monkeypatch, auto=False, challenger_beats=False)
    out = d_cycle.decide_promotion(cfg, market=object(), today=pd.Timestamp("2026-09-08"))
    assert out["decision"]["qualifies"] is False
    assert out["decision"]["promote"] is False
    assert out["decision"].get("challenger_dropped") is True
