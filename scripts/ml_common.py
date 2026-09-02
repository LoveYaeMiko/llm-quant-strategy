"""Shared helpers for the margin-augmented ML training scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.data.margin import RECORD_TYPE, margin_ml_features  # noqa: E402
from src.data.point_in_time_loader import from_url  # noqa: E402


def load_margin_extras(cfg) -> dict[str, pd.Series]:
    """Margin crowding factors (raw + reversed) from the PIT store.

    PIT-safe: factors are computed on the ``valid_from`` grid (a balance dated
    d is visible from d+1), trailing 20-day growth — no look-ahead.
    """
    url = cfg.get("data.pit_database_url")
    if not url:
        raise RuntimeError("PIT_DATABASE_URL not set")
    store = from_url(url)
    snap = store.snapshot(RECORD_TYPE)
    if snap.empty:
        raise RuntimeError("no margin records — run scripts/margin_ingest.py first")
    long = pd.DataFrame(
        {
            "date": pd.to_datetime(snap["valid_from"]),
            "symbol": snap["symbol"],
            **{
                c: snap[c]
                for c in ("fin_balance", "fin_buy", "sl_balance", "sl_volume", "sl_sell_volume")
                if c in snap.columns
            },
        }
    )
    extras = margin_ml_features(long)
    print(f"margin extras: {len(extras)} series, "
          f"{long['date'].min().date()}..{long['date'].max().date()}", flush=True)
    return extras


def load_zoo_formulas() -> list[str]:
    """All translated zoo formulas (alpha101 + gtja191 + alpha158), deduped."""
    translated = json.loads(
        (ROOT / "paper" / "factor_zoo" / "translated.json").read_text(encoding="utf-8")
    )
    zoo = [
        r["fqa"]
        for fam in ("alpha101", "gtja191", "alpha158")
        for r in translated[fam].values()
        if r.get("status") == "ok"
    ]
    return list(dict.fromkeys(zoo))


__all__ = ["load_margin_extras", "load_zoo_formulas"]
