"""龙虎榜 (dragon-tiger list) daily listings — PIT ingestion + factors.

The exchanges publish each day's dragon-tiger list after the close, so a
listing dated ``d`` is visible from ``d+1`` (same closed-interval convention
as margin balances, ADR-0001). :func:`to_pit_records` encodes that.

**Look-ahead strip**: the upstream endpoint embeds forward-return columns
(上榜后1/2/5/10日 — returns realised AFTER the listing date). Ingesting them
would poison every downstream PIT query, so the normaliser drops them by
construction and keeps only the listing-day facts.

Multiple listings of one symbol on one day (different 上榜原因) are aggregated
to one record per ``(symbol, date)`` — the PIT primary key cannot hold two
facts for the same symbol on the same ``valid_from``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .schema.retry import retry_call
from .schema.symbols import from_bare_code

RECORD_TYPE = "lhb"

NET_BUY = "net_buy"            # 龙虎榜净买额 (元)
BUY = "buy"                    # 龙虎榜买入额
SELL = "sell"                  # 龙虎榜卖出额
LHB_AMOUNT = "lhb_amount"      # 龙虎榜成交额
MARKET_AMOUNT = "market_amount"  # 市场总成交额
NET_RATIO = "net_ratio"        # 净买额占总成交额比
TURNOVER = "turnover"          # 换手率
FLOAT_MV = "float_mv"          # 流通市值
REASON = "reason"              # 上榜原因 (aggregated)

#: upstream columns that leak the future — never stored
_FORWARD_COLS = ("上榜后1日", "上榜后2日", "上榜后5日", "上榜后10日")


def _tolerant_code(value: object) -> Optional[str]:
    try:
        return from_bare_code(value)
    except Exception:  # noqa: BLE001
        return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise one fetch batch: strip look-ahead columns, aggregate per (symbol, date)."""
    if df is None or df.empty:
        return pd.DataFrame()
    drop = [c for c in _FORWARD_COLS if c in df.columns]
    df = df.drop(columns=drop)

    out = pd.DataFrame()
    out["symbol"] = df["代码"].map(_tolerant_code)
    out["date"] = pd.to_datetime(df["上榜日"], errors="coerce")
    for col, key in (
        ("龙虎榜净买额", NET_BUY), ("龙虎榜买入额", BUY), ("龙虎榜卖出额", SELL),
        ("龙虎榜成交额", LHB_AMOUNT), ("市场总成交额", MARKET_AMOUNT),
        ("净买额占总成交额比", NET_RATIO), ("换手率", TURNOVER), ("流通市值", FLOAT_MV),
    ):
        out[key] = pd.to_numeric(df.get(col), errors="coerce")
    out[REASON] = df.get("上榜原因", "")
    out = out.dropna(subset=["symbol", "date"])
    if out.empty:
        return out

    def agg(g: pd.DataFrame) -> pd.Series:
        row = {}
        for c in (NET_BUY, BUY, SELL, LHB_AMOUNT, MARKET_AMOUNT, NET_RATIO, TURNOVER, FLOAT_MV):
            v = g[c].dropna()
            row[c] = float(v.sum()) if len(v) else np.nan
        reasons = " / ".join(sorted({str(r) for r in g[REASON].dropna() if str(r) != "nan"}))
        row[REASON] = reasons
        return pd.Series(row)

    return (
        out.groupby(["date", "symbol"], as_index=False).apply(agg, include_groups=False)
        .reset_index(drop=True)
    )


def fetch_lhb_akshare(start: str = "2022-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Fetch daily dragon-tiger listings for a date range (one call per range)."""
    import akshare as ak

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    df = retry_call(
        lambda: ak.stock_lhb_detail_em(start_date=s, end_date=e),
        retries=2, base_delay=1.0, backoff=2.0, on_error=lambda exc: pd.DataFrame(),
    )
    out = _normalize(df)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def to_pit_records(lhb: pd.DataFrame) -> pd.DataFrame:
    """Convert the normalised lhb frame into PIT records (visible next day).

    ``valid_to`` closes each listing at the next listing of the SAME symbol
    (listings are sparse, not daily snapshots) — a stale listing must not look
    live forever.
    """
    if lhb.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["symbol"] = lhb["symbol"]
    out["valid_from"] = pd.to_datetime(lhb["date"]) + pd.to_timedelta("1D")
    for col in (NET_BUY, BUY, SELL, LHB_AMOUNT, MARKET_AMOUNT, NET_RATIO, TURNOVER, FLOAT_MV, REASON):
        out[col] = lhb[col] if col in lhb.columns else None
    out = out.sort_values(["symbol", "valid_from"])
    out["valid_to"] = out.groupby("symbol")["valid_from"].shift(-1)
    out["record_type"] = RECORD_TYPE
    return out.reset_index(drop=True)


def upsert_lhb(store, lhb: pd.DataFrame) -> dict[str, int]:
    """Store lhb records into a PIT store, closing superseded listings."""
    recs = to_pit_records(lhb)
    if recs.empty:
        return {"records": 0}
    try:
        old = store.snapshot(RECORD_TYPE)
    except Exception:  # noqa: BLE001
        old = pd.DataFrame()
    closes: list[pd.DataFrame] = []
    if not old.empty and {"symbol", "valid_from", "valid_to"} <= set(old.columns):
        old = old.copy()
        old["valid_from"] = pd.to_datetime(old["valid_from"])
        for sym, grp in recs.groupby("symbol"):
            new_min = grp["valid_from"].min()
            mask = (old["symbol"] == sym) & old["valid_to"].isna() & (old["valid_from"] < new_min)
            if mask.any():
                sub = old[mask].copy()
                sub["valid_to"] = new_min
                closes.append(sub)
    if closes:
        recs = pd.concat([recs, *closes], ignore_index=True)
    store.upsert(recs)
    return {"records": int(len(recs)), "closed": int(sum(len(c) for c in closes))}


__all__ = [
    "RECORD_TYPE", "NET_BUY", "BUY", "SELL", "LHB_AMOUNT", "MARKET_AMOUNT",
    "NET_RATIO", "TURNOVER", "FLOAT_MV", "REASON",
    "fetch_lhb_akshare", "to_pit_records", "upsert_lhb",
]
