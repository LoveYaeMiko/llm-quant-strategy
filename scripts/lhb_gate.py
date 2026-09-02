"""Dragon-tiger factor gate — discovery/validation split, PIT-safe.

Factors are built from the store's ``valid_from`` grid (a listing dated d is
visible from d+1). Daily panel = listing facts reindexed onto the market
calendar with 0 fill for non-listed days, so rolling sums are time-consistent.

Factors:
* ``net_buy`` — 1-day 龙虎榜净买额;
* ``net_buy_5d`` — trailing 5-day cumulative net buy (smart-money persistence);
* ``net_ratio`` — net buy / market turnover (relative footprint).

Split: discovery 2024-2025, validation 2026 (OOS).

Usage:  python scripts/lhb_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.backtest.metrics import daily_ic  # noqa: E402
from src.cli import _market_data  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.lhb import NET_BUY, NET_RATIO, RECORD_TYPE  # noqa: E402
from src.data.point_in_time_loader import from_url  # noqa: E402


def main() -> int:
    cfg = load_config()
    url = cfg.get("data.pit_database_url")
    if not url:
        print("PIT_DATABASE_URL not set — aborting", file=sys.stderr)
        return 2

    store = from_url(url)
    snap = store.snapshot(RECORD_TYPE)
    if snap.empty:
        print("no lhb records in the PIT store — run scripts/lhb_ingest.py first")
        return 1
    print(f"lhb records: {len(snap):,}  {snap['valid_from'].min().date()} .. {snap['valid_from'].max().date()}")

    market = _market_data(cfg, seed=1)
    fwd = market.forward_returns
    calendar = market.price_panel.index

    # daily panel: (date=valid_from, symbol) reindexed to the market grid, 0-filled
    snap = snap.copy()
    snap["date"] = pd.to_datetime(snap["valid_from"])
    snap[NET_BUY] = pd.to_numeric(snap[NET_BUY], errors="coerce").fillna(0.0)
    snap[NET_RATIO] = pd.to_numeric(snap[NET_RATIO], errors="coerce").fillna(0.0)
    wide_nb = snap.pivot_table(index="date", columns="symbol", values=NET_BUY, aggfunc="sum")
    wide_nr = snap.pivot_table(index="date", columns="symbol", values=NET_RATIO, aggfunc="sum")
    symbols = list(market.price_panel.columns)
    wide_nb = wide_nb.reindex(index=calendar, columns=symbols).fillna(0.0)
    wide_nr = wide_nr.reindex(index=calendar, columns=symbols).fillna(0.0)

    factors = pd.DataFrame(
        {
            "net_buy": wide_nb.stack(dropna=False).rename("net_buy"),
            "net_buy_5d": wide_nb.rolling(5, min_periods=1).sum().stack(dropna=False).rename("net_buy_5d"),
            "net_ratio": wide_nr.stack(dropna=False).rename("net_ratio"),
        },
        index=wide_nb.stack(dropna=False).index,
    )
    factors.index.names = ["date", "symbol"]

    spans = [
        ("discovery 2024-2025", "2024-01-01", "2025-12-31"),
        ("validation 2026 (OOS)", "2026-01-01", "2026-12-31"),
    ]
    for label, s, e in spans:
        fac = factors[
            (factors.index.get_level_values(0) >= pd.Timestamp(s))
            & (factors.index.get_level_values(0) <= pd.Timestamp(e))
        ]
        rows = []
        for col in fac.columns:
            sig = fac[col]
            sig = sig[sig != 0.0].dropna() if col != "net_buy_5d" else sig.dropna()
            ic = daily_ic(sig, fwd, method="spearman").dropna()
            if len(ic) < 10:
                rows.append((col, len(ic), None, None))
                continue
            rows.append((col, len(ic), float(ic.mean()),
                         float(ic.mean() / ic.std() * 252 ** 0.5) if ic.std() > 0 else None))
        out = pd.DataFrame(rows, columns=["factor", "n_days", "rank_ic", "icir"])
        print(f"\n== lhb factor gate — {label} ==")
        print(out.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
