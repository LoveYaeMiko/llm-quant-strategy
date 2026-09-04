"""Intraday feature pack from AlphaFeed minute klines (ADR-0002 contract).

Daily aggregates from 1-minute bars, cached per symbol under
``data/intraday/`` and rolled up into one wide-frame parquet:

* ``vwap_gap`` — close vs true minute-weighted VWAP (the AVWAP pullback anchor);
* ``rv`` — daily realized volatility (sqrt of summed squared 1m log returns);
* ``tail_vol`` — last-30-minute volume share (尾盘资金异动);
* ``gap`` — open vs previous close.

The AlphaFeed minute endpoint serves the trailing ~1 year, so the pack covers
the 2026 shadow window and rolls forward daily. Fetches are rate-limited via
:data:`RateLimiters.alphafeed_minute_batch` (60/min).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from .ingestion.alphafeed_adapter import AlphaFeedAdapter
from .schema.rate_limiter import RateLimiters

logger = logging.getLogger(__name__)

_MARKET_CLOSE = pd.Timestamp("15:00").time()


def _intraday_dir(cfg: Config) -> Path:
    root = Path(str(cfg.get("data.intraday_dir", "data/intraday")))
    if not root.is_absolute():
        from ..cli import ROOT  # noqa: PLC0415 — avoid a hard import cycle

        root = ROOT / root
    return root


def fetch_symbol_minutes(cfg: Config, symbol: str, total_minutes: int = 40_000) -> pd.DataFrame:
    """Pull ``symbol``'s minute bars walking backwards in ≤10000-bar windows.

    Idempotent: a cached parquet short-circuits the fetch (incremental daily
    updates simply delete the cache for a symbol or re-run with a small
    ``total_minutes`` and merge upstream).
    """
    out_dir = _intraday_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{symbol.replace('.', '_')}.parquet"
    if cache_path.is_file():
        try:
            cached = pd.read_parquet(cache_path)
            if len(cached):
                return cached
        except Exception:  # noqa: BLE001
            pass

    adapter = AlphaFeedAdapter(api_key=str(cfg.get("data.alphafeed.api_key", "")))
    frames = []
    pulled = 0
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    while pulled < total_minutes:
        count = min(10_000, total_minutes - pulled)
        # 10000 trading minutes ≈ 41 trading days ≈ 58 calendar days (×1.4 margin)
        start = end - pd.Timedelta(days=int(count / 240 * 1.4))
        batch = adapter.fetch_minute_klines([symbol], period="1m", count=count, start=start, end=end)
        df = batch.get(symbol)
        if df is None or df.empty:
            break
        frames.append(df)
        pulled += count
        # walk the window backwards by the calendar span we just asked for
        end = start
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "timestamp" in out.columns and pd.api.types.is_numeric_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    elif "trade_time" in out.columns:
        out["timestamp"] = pd.to_datetime(out["trade_time"])
    elif "trade_date" in out.columns:
        out["timestamp"] = pd.to_datetime(out["trade_date"])
    else:
        raise ValueError(f"minute bars for {symbol} have no timestamp column")
    out = out.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    out.to_parquet(cache_path)
    return out


def _symbol_minutes(cfg: Config, symbol: str) -> pd.DataFrame:
    cache_path = _intraday_dir(cfg) / f"{symbol.replace('.', '_')}.parquet"
    if cache_path.is_file():
        try:
            df = pd.read_parquet(cache_path)
            if len(df):
                return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("intraday cache %s corrupt: %s", cache_path, exc)
    return pd.DataFrame()


def _daily_features(minutes: pd.DataFrame) -> pd.DataFrame:
    """One symbol's minute bars → per-day intraday aggregates."""
    df = minutes.copy()
    df["date"] = df["timestamp"].dt.normalize()
    df["typ"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["dollar"] = df["typ"] * df["volume"]
    day = df.groupby("date").agg(
        open=("open", "first"),
        close=("close", "last"),
        volume=("volume", "sum"),
        dollar=("dollar", "sum"),
    )
    day["vwap"] = day["dollar"] / day["volume"].replace(0, np.nan)
    day["vwap_gap"] = day["close"] / day["vwap"] - 1.0
    day["gap"] = day["open"] / day["close"].shift(1) - 1.0
    day["tail_vol"] = (
        df[df["timestamp"].dt.time >= pd.Timestamp("14:30").time()]
        .groupby("date")["volume"].sum()
        .reindex(day.index)
        / day["volume"].replace(0, np.nan)
    )
    # realized volatility: sqrt(sum of squared 1m log returns per day)
    df["logret"] = np.log(df["close"] / df["close"].shift(1))
    day["rv"] = np.sqrt((df["logret"] ** 2).groupby(df["date"]).sum())
    # open-30min return / intraday range / last-30min return
    t = df["timestamp"].dt.time
    open30 = df[t <= pd.Timestamp("10:00").time()].groupby("date")["close"].last()
    day["open30"] = open30.reindex(day.index) / day["open"] - 1.0
    day["range"] = (df.groupby("date")["high"].max() - df.groupby("date")["low"].min()) / day["vwap"]
    aft = df[t >= pd.Timestamp("14:30").time()].groupby("date")["close"].last()
    day["afternoon"] = day["close"] / aft.reindex(day.index) - 1.0
    out = day[["vwap_gap", "rv", "tail_vol", "gap", "open30", "range", "afternoon", "vwap", "close"]].copy()
    out.index.name = "date"
    return out


_FEATURES = ("vwap_gap", "rv", "tail_vol", "gap", "open30", "range", "afternoon", "vwap")


def build_intraday_frames(cfg: Config, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Per-symbol daily intraday features → wide ``date × symbol`` frames."""
    series: dict[str, dict[str, pd.Series]] = {c: {} for c in _FEATURES}
    for sym in symbols:
        minutes = _symbol_minutes(cfg, sym)
        if minutes.empty:
            continue
        d = _daily_features(minutes)
        for col in _FEATURES:
            series[col][sym] = d[col]
    wide: dict[str, pd.DataFrame] = {}
    for col in _FEATURES:
        if series[col]:
            wide[col] = pd.concat(series[col], axis=1).sort_index().sort_index(axis=1)
    return wide


def load_intraday_frames(cfg: Config, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load the cached rollup parquet (fetch first with ``scripts/fetch_intraday.py``)."""
    path = _intraday_dir(cfg) / "daily_features.parquet"
    if not path.is_file():
        return {}
    store = pd.read_parquet(path)
    out: dict[str, pd.DataFrame] = {}
    cols0 = store.columns.get_level_values(0).unique() if isinstance(store.columns, pd.MultiIndex) else []
    for key in _FEATURES:
        if key not in cols0:
            continue
        wide = store[key].sort_index()
        if symbols:
            wide = wide.reindex(columns=[s for s in symbols if s in wide.columns])
        out[key] = wide
    return out

__all__ = [
    "build_intraday_frames",
    "fetch_symbol_minutes",
    "load_intraday_frames",
]
