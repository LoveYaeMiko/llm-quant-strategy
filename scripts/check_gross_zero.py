"""Inspect the A-shadow market panel around the gross=0 stretch."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sqlite3

import pandas as pd


def main() -> int:
    from src.cli import _build_market_for_paper
    from src.config import load_config
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    symbols = resolve_shadow_universe(cfg, "hs300")
    market = _build_market_for_paper(cfg, symbols, "2026-01-01", None, seed=1)
    panel = market.price_panel
    print("panel index unique:", panel.index.is_unique, "| type:", type(panel.index))
    dup = panel.index[panel.index.duplicated(keep=False)].unique()
    print("duplicated dates:", list(dup)[:10])

    conn = sqlite3.connect("outputs/shadow_ledger_A_200W.sqlite")
    cur = conn.cursor()
    held = [r[0] for r in cur.execute(
        "SELECT symbol FROM positions WHERE date = '2026-06-29'"
    ).fetchall()]
    conn.close()
    print("held on 06-29:", held[:5], "...", len(held), "names")

    d = pd.Timestamp("2026-06-29")
    row = panel.loc[d]
    print("loc[d] type:", type(row))
    if isinstance(row, pd.DataFrame):
        print("DataFrame shape:", row.shape, "columns:", list(row.columns)[:5])
        return 0
    print("row index sample:", list(row.index[:5]))
    print("held prices sample:")
    for s in held[:8]:
        print(f"  {s}: {row.get(s, 'MISSING')}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
