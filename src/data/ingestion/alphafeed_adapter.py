"""AlphaFeed — the primary, paid (unlimited) A-share OHLCV source.

The official client is wrapped lazily (first network call instantiates it) and
rate-limited politely even though the quota is effectively unlimited. Batches of
symbols are the unit of work; the ingestor splits the universe into
``batch_size``-sized chunks.

Adjustment contract (ADR-0002): klines are fetched with ``adjust="none"`` so
``raw_close`` is the unadjusted price, and AlphaFeed's per-event ``ex_factor``
rows are combined (reverse-cumulative product of inverses) to derive the
backward-adjusted ``close``. This keeps both columns from a single klines call
and lets the B3 check audit real ex-dividend behaviour rather than the
tautology ``close == raw_close * factor``.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from ..schema.rate_limiter import RateLimiters


def _to_ms(date: str | pd.Timestamp) -> int:
    """AlphaFeed takes epoch milliseconds."""
    return int(pd.Timestamp(date).value // 1_000_000)


class AlphaFeedAdapter:
    """Rate-limited facade over the official ``alphafeed`` client."""

    def __init__(self, api_key: str, config=None) -> None:
        self.api_key = api_key
        self._client = None
        RateLimiters.configure(config)

    @property
    def client(self):
        """Lazy client — no import/network cost until the first fetch."""
        if self._client is None:
            from alphafeed import AlphaFeed

            self._client = AlphaFeed(api_key=self.api_key)
        return self._client

    def fetch_klines(
        self, symbols, start: str | pd.Timestamp, end: str | pd.Timestamp, adjust: str = "none"
    ) -> Dict[str, pd.DataFrame]:
        """Batch daily klines in ``[start, end]``; returns ``{symbol: OHLCV frame}``.

        The API silently caps each batch at 100 bars when ``count`` is unset —
        a full year is ~240 sessions, the whole 2010-2025 backfill ~3900 — so we
        pass an explicit ``count`` derived from the window span (trading days are
        always ≤ calendar days, plus a margin for the tail).
        """
        count = max(100, int((pd.Timestamp(end) - pd.Timestamp(start)).days) + 120)
        with RateLimiters.alphafeed_daily_batch:
            return self.client.klines.batch(
                symbols=list(symbols),
                period="1d",
                start_time=_to_ms(start),
                end_time=_to_ms(end),
                adjust=adjust,
                count=count,
                to_dataframe=True,
            )

    def fetch_ex_factors(self, symbols) -> Dict[str, pd.DataFrame]:
        """Cumulative backward-adjustment factors; returns ``{symbol: frame}``.

        ``ex_factors(..., to_dataframe=True)`` returns ONE long frame
        ``[symbol, timestamp, trade_date, ex_factor]``, not a per-symbol dict —
        normalise it to the ``{symbol: frame}`` shape :func:`to_price_records`
        consumes.
        """
        with RateLimiters.alphafeed_adjust:
            df = self.client.klines.ex_factors(list(symbols), to_dataframe=True)
        if df is None or df.empty:
            return {}
        return {symbol: group.reset_index(drop=True) for symbol, group in df.groupby("symbol", sort=False)}

    def fetch_quotes(self, universes: str = "CN_Stock") -> pd.DataFrame:
        """Full-market snapshot (实时快照): price/amount per symbol, one call."""
        with RateLimiters.alphafeed_quote:
            return self.client.quotes.get(universes=universes, to_dataframe=True)

    # -- intraday family (分钟K线 / 日内走势 / 盘口), rate-limited at 60/min ---

    def fetch_minute_klines(
        self, symbols, period: str = "1m", count: int = 240,
        start=None, end=None,
    ) -> Dict[str, pd.DataFrame]:
        """Batch minute klines (近一年, ≤10000 bars/symbol/call), optional window."""
        kwargs: dict = {
            "symbols": list(symbols),
            "period": period,
            "count": int(count),
            "to_dataframe": True,
        }
        if start is not None:
            kwargs["start_time"] = _to_ms(start)
        if end is not None:
            kwargs["end_time"] = _to_ms(end)
        with RateLimiters.alphafeed_minute_batch:
            return self.client.klines.batch(**kwargs)

    def fetch_intraday(self, symbols, period: str = "1m", count: int = 240) -> Dict[str, pd.DataFrame]:
        """Batch 日内走势 (intraday trend per symbol)."""
        with RateLimiters.alphafeed_minute_batch:
            return self.client.klines.intraday_batch(
                symbols=list(symbols), period=period, count=int(count),
                to_dataframe=True,
            )

    def fetch_depth(self, symbols) -> Dict[str, object]:
        """Batch 盘口 (market depth) snapshots."""
        with RateLimiters.alphafeed_depth_batch:
            return self.client.depth.batch(list(symbols))


def _factor_columns(d: pd.DataFrame, default_date_col: str) -> str:
    if "trade_date" in d.columns:
        return "trade_date"
    if "timestamp" in d.columns:
        return "timestamp"
    if default_date_col in d.columns:
        return default_date_col
    raise ValueError("klines frame has no trade_date/timestamp/date column")


def to_price_records(klines: Dict[str, pd.DataFrame], factors: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
    """Merge batch klines + ex_factors → a long dual-column price frame.

    * ``raw_close`` — unadjusted close;
    * ``adjust_factor`` — backward-adjustment factor anchored at the newest bar
      (1.0 there), equal to the product of inverse *per-event* ex_factors for
      events strictly after the bar's date;
    * ``close`` — backward-adjusted ``raw_close * adjust_factor``.

    AlphaFeed's ``ex_factor`` is a PER-EVENT ratio (one row per ex-dividend /
    split), NOT a cumulative factor: e.g. 601318 shows ``2.0125`` for its 2015
    10转10 split then ``1.0061`` — a cumulative factor can never fall. The
    backward-adjustment factor at date *t* is therefore the product of
    ``1/ex_factor`` over all events with date > *t* (the ex-date bar already
    trades ex-dividend). Bars with no events after them (or no factor data at
    all) get 1.0, so the newest bars satisfy adjusted == raw.
    """
    frames = []
    for symbol, df in klines.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        if "symbol" not in d.columns:
            d["symbol"] = symbol
        d["date"] = pd.to_datetime(d[_factor_columns(d, "date")])
        d["raw_close"] = d["close"]
        keep = ["symbol", "date", "name", "open", "high", "low", "raw_close", "volume", "amount"]
        frames.append(d[[c for c in keep if c in d.columns]])
    if not frames:
        return pd.DataFrame()
    # reset_index: sort_values leaves a permuted index; without it the label-based
    # assignment below would misalign with merged's fresh RangeIndex and swap
    # factors between symbols. merge(how="left") preserves this row order.
    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)

    out["adjust_factor"] = 1.0
    if factors:
        f_frames = []
        for symbol, ff in factors.items():
            if ff is None or ff.empty:
                continue
            g = ff.copy()
            if "symbol" not in g.columns:
                g["symbol"] = symbol
            g["date"] = pd.to_datetime(g[_factor_columns(g, "date")])
            f_frames.append(g[["symbol", "date", "ex_factor"]])
        if f_frames:
            events = pd.concat(f_frames, ignore_index=True).sort_values(["symbol", "date"])
            # Per-event factors → backward factor = reverse-cumprod of inverses,
            # looked up per bar as the first event STRICTLY after the bar's date.
            events["suffix"] = (1.0 / events["ex_factor"]).groupby(
                events["symbol"]
            ).transform(lambda s: s[::-1].cumprod()[::-1])
            # merge_asof needs `date` globally sorted (symbols' ranges interleave),
            # so carry a row-id to restore out's (symbol, date) order afterwards.
            bars = out.sort_values(["symbol", "date"]).reset_index(drop=True)
            bars["_row"] = bars.index.to_numpy()
            matched = pd.merge_asof(
                bars.sort_values("date"),
                events[["symbol", "date", "suffix"]].sort_values("date"),
                on="date",
                by="symbol",
                direction="forward",
                allow_exact_matches=False,
            ).sort_values("_row")
            out["adjust_factor"] = matched["suffix"].fillna(1.0).to_numpy()
    out["close"] = (out["raw_close"] * out["adjust_factor"]).round(4)
    return out.reset_index(drop=True)
