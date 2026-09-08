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


def make_minute_provider(cfg: Config):
    """Lazy per-symbol minute-bar loader: ``(date, symbol) → bars of that day``.

    Bars after 15:00 are excluded (the close bar belongs to the close rebalance,
    not the intraday sweep). Per-symbol parquets are cached in memory on first
    access — a pullback book touches only its held names.
    """
    cache: dict[str, pd.DataFrame | None] = {}

    def provider(date, symbol: str) -> pd.DataFrame | None:
        if symbol not in cache:
            df = _symbol_minutes(cfg, symbol)
            cache[symbol] = df if len(df) else None
        df = cache.get(symbol)
        if df is None or df.empty:
            return None
        d = pd.Timestamp(date).normalize()
        day = df[df["timestamp"].dt.normalize() == d]
        if len(day) == 0:
            return day
        return day[day["timestamp"].dt.time < pd.Timestamp("15:00").time()]

    return provider


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


# --------------------------------------------------------------------------- #
# incremental refresh (the 15:30 scheduler job + the 17:30 run's self-heal)
# --------------------------------------------------------------------------- #

def _normalize_minute_bars(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns and pd.api.types.is_numeric_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
        )
    elif "trade_time" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["trade_time"])
    return df


def _merge_symbol_cache(cfg: Config, symbol: str, fresh: pd.DataFrame) -> None:
    """Merge freshly fetched bars into the per-symbol cache (dedupe by timestamp)."""
    cache_path = _intraday_dir(cfg) / f"{symbol.replace('.', '_')}.parquet"
    frames = [fresh]
    if cache_path.is_file():
        try:
            old = pd.read_parquet(cache_path)
            if len(old):
                frames.insert(0, old)
        except Exception:  # noqa: BLE001 — corrupt cache is replaced by the fresh bars
            pass
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    out.to_parquet(cache_path)


def refresh_intraday(
    cfg: Config,
    symbols: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    batch: int = 25,
) -> dict:
    """Incremental minute-bar fetch for ``[start, end]`` + rollup rebuild.

    Symbol batches (one API call per batch per window) with retries, merged into
    the per-symbol caches, then a local rebuild of ``daily_features.parquet``.
    Returns ``{calls, symbols_updated, rollup_days, rollup_symbols}``.
    """
    import time as _time

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
    span_days = max(1, (end_ts - start_ts).days)
    # ~240 bars per trading day, ~5/7 trading days per calendar day
    count = min(10_000, int(span_days * 240 * 5 / 7) + 60)
    adapter = AlphaFeedAdapter(api_key=str(cfg.get("data.alphafeed.api_key", "")))
    calls = 0
    updated = 0
    for i in range(0, len(symbols), batch):
        chunk = list(symbols)[i : i + batch]
        got = None
        last = None
        for attempt in range(4):
            try:
                got = adapter.fetch_minute_klines(
                    chunk, period="1m", count=count, start=start_ts, end=end_ts
                )
                calls += 1
                break
            except Exception as exc:  # noqa: BLE001 — SSL EOF etc., retry with backoff
                last = exc
                _time.sleep(4 * (attempt + 1))
        if got is None:
            logger.warning("intraday refresh batch failed: %s", last)
            continue
        for sym in chunk:
            df = got.get(sym)
            if df is None or df.empty:
                continue
            _merge_symbol_cache(cfg, sym, _normalize_minute_bars(df))
            updated += 1

    wide = build_intraday_frames(cfg, list(symbols))
    out_path = _intraday_dir(cfg) / "daily_features.parquet"
    rollup = pd.concat({k: v for k, v in wide.items()}, axis=1)
    rollup.columns.names = ["feature", "symbol"]
    rollup.to_parquet(out_path)
    return {
        "calls": calls,
        "symbols_updated": updated,
        "rollup_days": int(len(rollup)),
        "rollup_symbols": int(rollup.shape[1] // len(_FEATURES)) if rollup.shape[1] else 0,
        "first_day": str(rollup.index.min().date()) if len(rollup) else None,
        "last_day": str(rollup.index.max().date()) if len(rollup) else None,
    }


def _resolve_end_date(now: pd.Timestamp, today: pd.Timestamp) -> pd.Timestamp:
    """Inclusive last day an incremental minute fetch should cover.

    The current day's bars are only COMPLETE after the 15:00 close, so a run
    before 15:00 must stop at the previous day — a partial day would poison
    ``tail_vol``/``rv``/``range`` (and the tail-volume entry gate would read a
    half-day volume share as if it were final). An explicitly requested PAST
    day is long closed and is always included, whatever the wall clock says
    (the 17:30 self-heal replays an earlier ``date``).
    """
    now = pd.Timestamp(now)
    today = pd.Timestamp(today).normalize()
    if now.normalize() > today:
        return today
    if now.time() >= _MARKET_CLOSE:
        return today
    return today - pd.Timedelta(days=1)


def refresh_intraday_daily(
    cfg: Config, symbols: list[str] | None = None, date: str | pd.Timestamp | None = None
) -> dict:
    """After-close incremental refresh: today's minute bars + rollup rebuild.

    Called by the PAICC 15:02 job (after the 15:00 close, so the current day is
    complete) and as the 17:30 run's self-heal. Before 15:00 the current day is
    NOT included — a partial day would poison tail_vol/rv/range — so the fetch
    ends at yesterday (see :func:`_resolve_end_date`); an explicitly requested
    historical ``date`` is always included.
    """
    from ..paper.shadow import resolve_shadow_universe  # noqa: PLC0415

    if symbols is None:
        symbols = resolve_shadow_universe(cfg, "hs300_500")
    now = pd.Timestamp.now()
    today = (pd.Timestamp(date) if date is not None else now).normalize()
    end = _resolve_end_date(now, today)
    start = today - pd.Timedelta(days=5)  # covers weekends/holidays before ``today``
    return refresh_intraday(cfg, list(symbols), start=start, end=end)


def ensure_intraday_current(
    cfg: Config, symbols: list[str], date: str | pd.Timestamp
) -> dict:
    """Guarantee the rollup covers ``date`` (fetch if missing) — point-in-time.

    The tail-volume entry gate treats a missing day as FAIL (no entries), so the
    daily loop calls this right after the market build: if today's row is absent
    the latest bars are fetched and the rollup rebuilt. Returns
    ``{covered, refreshed}`` — ``covered`` is the best-effort final state.
    """
    d = pd.Timestamp(date).normalize()
    frames = load_intraday_frames(cfg, symbols)
    first = frames.get("vwap_gap")
    if first is not None and d in first.index:
        return {"covered": True, "refreshed": False}
    summary = refresh_intraday_daily(cfg, symbols, date=d)
    frames = load_intraday_frames(cfg, symbols)
    first = frames.get("vwap_gap")
    return {
        "covered": bool(first is not None and d in first.index),
        "refreshed": True,
        "summary": summary,
    }

__all__ = [
    "build_intraday_frames",
    "ensure_intraday_current",
    "fetch_symbol_minutes",
    "load_intraday_frames",
    "refresh_intraday",
    "refresh_intraday_daily",
]
