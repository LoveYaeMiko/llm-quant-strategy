"""Corporate-action (复权因子) event features from AlphaFeed ex_factors.

The adjusted-close contract (ADR-0002) already uses ex_factors inside the price
pipeline; this module extracts the EVENT STREAM itself (ex-dividend / split /
bonus) and turns it into PIT-safe features:

* ``ca_days_since`` — calendar days since the symbol's last corporate action
  (capped at 365) — recent events mark ex-date re-pricing/liquidity effects;
* ``ca_last_mag`` — ln(ex_factor) of the last event (magnitude of the action);
* ``ca_cum_1y`` — ln of the cumulative factor over the trailing year.

Validation: rank-IC of each feature vs the 10d forward return on the
walk-forward test window, plus Spearman correlation with the deployed ML score
— a feature only gets wired into training when it adds signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

_EVENTS_CACHE = ROOT / "data" / "corp_actions" / "events.parquet"


def fetch_events(cfg) -> pd.DataFrame:
    """Pull ex_factors for the research universe (200/call, 120/min — fast)."""
    from src.data.ingestion.alphafeed_adapter import AlphaFeedAdapter
    from src.paper.shadow import resolve_shadow_universe

    symbols = resolve_shadow_universe(cfg, "hs300_500")
    adapter = AlphaFeedAdapter(api_key=str(cfg.get("data.alphafeed.api_key", "")))
    frames = []
    for i in range(0, len(symbols), 200):
        chunk = symbols[i : i + 200]
        batch = adapter.fetch_ex_factors(chunk)
        for sym, df in batch.items():
            if df is None or df.empty:
                continue
            d = df.copy()
            d["symbol"] = sym
            if "timestamp" in d.columns and pd.api.types.is_numeric_dtype(d["timestamp"]):
                d["date"] = (
                    pd.to_datetime(d["timestamp"], unit="ms", utc=True)
                    .dt.tz_convert("Asia/Shanghai")
                    .dt.tz_localize(None)
                    .dt.normalize()
                )
            elif "trade_date" in d.columns:
                d["date"] = pd.to_datetime(d["trade_date"]).dt.normalize()
            else:
                continue
            frames.append(d[["symbol", "date", "ex_factor"]])
    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    _EVENTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_EVENTS_CACHE)
    return out


def _grid_long(dates: pd.DatetimeIndex, syms: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        index=pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    ).reset_index()


def event_features(cfg, close_wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """PIT-safe event features on the market's daily date grid (wide frames)."""
    if not _EVENTS_CACHE.is_file():
        fetch_events(cfg)
    events = pd.read_parquet(_EVENTS_CACHE)
    events["date"] = pd.to_datetime(events["date"])
    events["mag"] = np.log(events["ex_factor"].astype(float))
    ev = events.sort_values("date").rename(columns={"date": "ev_date"})

    dates = pd.DatetimeIndex(close_wide.index)
    syms = list(close_wide.columns)

    # days since last event + last magnitude (backward merge_asof per symbol)
    grid = _grid_long(dates, syms)
    merged = pd.merge_asof(
        grid, ev,
        left_on="date", right_on="ev_date", by="symbol",
        direction="backward", allow_exact_matches=True,
    )
    merged["days_since"] = (merged["date"] - merged["ev_date"]).dt.days
    days_since = merged.pivot(index="date", columns="symbol", values="days_since").clip(upper=365)
    last_mag = merged.pivot(index="date", columns="symbol", values="mag")

    # trailing-year cumulative magnitude (rolling sum on events, then ffill)
    ev2 = events.set_index("date").sort_index()
    cum = ev2.groupby("symbol")["mag"].rolling("365D").sum().reset_index(level=0)
    cum.columns = ["symbol", "cum_1y"]
    cum_wide = cum.reset_index().pivot(index="date", columns="symbol", values="cum_1y")
    cum_wide = cum_wide.reindex(index=dates).ffill()

    for k in ("ca_days_since", "ca_last_mag", "ca_cum_1y"):
        f = {"ca_days_since": days_since, "ca_last_mag": last_mag, "ca_cum_1y": cum_wide}[k]
        f = f.reindex(index=dates, columns=syms).ffill()
    return {
        "ca_days_since": days_since.reindex(index=dates, columns=syms).ffill(),
        "ca_last_mag": last_mag.reindex(index=dates, columns=syms).ffill(),
        "ca_cum_1y": cum_wide.reindex(index=dates, columns=syms).ffill(),
    }


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    events = fetch_events(cfg)
    print(f"events: {len(events)} rows, {events['symbol'].nunique()} symbols, "
          f"range {events['date'].min().date()} .. {events['date'].max().date()}")
    print(events.tail(3).to_string())
