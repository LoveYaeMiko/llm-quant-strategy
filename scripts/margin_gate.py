"""Margin-factor gate — rank_ic / ICIR of leverage-structure factors.

PIT discipline: factors are computed on the store's ``valid_from`` grid (a
balance dated d is visible from d+1 — encoded at ingestion), so the factor at
date d only ever uses margin facts visible at d; the forward return is the
[d, d+1) return of the research universe. Reports per-factor rank_ic / ICIR
over the covered window plus the same metrics on the research test window
(2022-2025) where overlap exists.

Usage:  python scripts/margin_gate.py
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
from src.data.margin import RECORD_TYPE, margin_factors  # noqa: E402
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
        print("no margin records in the PIT store — run scripts/margin_ingest.py first")
        return 1
    print(f"margin records: {len(snap):,}  {snap['valid_from'].min().date()} .. "
          f"{snap['valid_from'].max().date()}")

    market = _market_data(cfg, seed=1)
    fwd = market.forward_returns

    long = pd.DataFrame(
        {
            "date": pd.to_datetime(snap["valid_from"]),
            "symbol": snap["symbol"],
            **{c: snap[c] for c in ("fin_balance", "fin_buy", "sl_balance", "sl_volume", "sl_sell_volume") if c in snap.columns},
        }
    )
    factors = margin_factors(long).dropna(how="all")

    spans = [
        ("discovery 2010-2023", "2010-01-01", "2023-12-31"),
        ("validation 2024-2026 (OOS)", "2024-01-01", "2026-12-31"),
    ]
    for label, s, e in spans:
        fac = factors[
            (factors.index.get_level_values(0) >= pd.Timestamp(s))
            & (factors.index.get_level_values(0) <= pd.Timestamp(e))
        ]
        rows = []
        for col in fac.columns:
            sig = fac[col].dropna()
            ic = daily_ic(sig, fwd, method="spearman").dropna()
            if len(ic) < 10:
                rows.append((col, len(ic), None, None, "insufficient"))
                continue
            rows.append(
                (
                    col,
                    len(ic),
                    float(ic.mean()),
                    float(ic.mean() / ic.std() * 252 ** 0.5) if ic.std() > 0 else None,
                    "ok",
                )
            )
        out = pd.DataFrame(rows, columns=["factor", "n_days", "rank_ic", "icir", "note"])
        print(f"\n== margin factor gate — {label} ==")
        print(out.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        print("   (reversed = short-leverage-crowding: negate the sign)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
