"""融资融券 (margin trading) daily balances — PIT ingestion + factors.

SSE/SZSE publish each trading day's margin balances *after* the close
(typically that evening). A decision made at close of day ``d`` may therefore
only use balances published at or before ``d``: the balance of day ``b``
becomes visible on ``b+1`` (calendar day, the closed-interval convention —
ADR-0001). :func:`to_pit_records` encodes exactly that: ``valid_from = b + 1D``.

Records are stored with ``record_type="margin"`` in the same ``pit_records``
table as price/universe/fundamental records (ADR-0003), so every existing
query path is anti-look-ahead by construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .schema.retry import retry_call
from .schema.symbols import from_bare_code

RECORD_TYPE = "margin"

#: payload keys (raw CNY balances / volumes from the exchanges)
FIN_BALANCE = "fin_balance"       # 融资余额 (元)
FIN_BUY = "fin_buy"               # 融资买入额 (元)
SL_BALANCE = "sl_balance"         # 融券余额 (元)
SL_VOLUME = "sl_volume"           # 融券余量 (股)
SL_SELL_VOLUME = "sl_sell_volume"  # 融券卖出量 (股)


def _tolerant_code(value: object) -> Optional[str]:
    """Map a bare exchange code to a FQA symbol, dropping what can't be mapped.

    The SSE detail feed includes ETF codes (5xxxxx) that the stock-only symbol
    inference rejects — ETFs are outside the research universe, so they are
    dropped rather than crashing the whole backfill.
    """
    try:
        return from_bare_code(value)
    except Exception:  # noqa: BLE001 — SymbolError for ETFs / malformed codes
        return None


def _norm_sse(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Normalise ``stock_margin_detail_sse`` output (per-symbol daily detail).

    SSE columns: 信用交易日期 / 标的证券代码 / 标的证券简称 / 融资余额 /
    融资买入额 / 融资偿还额 / 融券余量 / 融券卖出量 / 融券偿还量.
    (No 融券余额 on SSE — short *balance* is SZSE-only.)
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["symbol"] = df["标的证券代码"].map(_tolerant_code)
    out["date"] = pd.to_datetime(date)
    out[FIN_BALANCE] = pd.to_numeric(df.get("融资余额"), errors="coerce")
    out[FIN_BUY] = pd.to_numeric(df.get("融资买入额"), errors="coerce")
    out[SL_BALANCE] = np.nan
    out[SL_VOLUME] = pd.to_numeric(df.get("融券余量"), errors="coerce")
    out[SL_SELL_VOLUME] = pd.to_numeric(df.get("融券卖出量"), errors="coerce")
    return out.dropna(subset=["symbol"])


def _norm_szse(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Normalise ``stock_margin_detail_szse`` output (per-symbol daily detail).

    SZSE columns: 证券代码 / 证券简称 / 融资买入额 / 融资余额 / 融券卖出量 /
    融券余额 / 融券余量 / 融资融券余额 — note 融资买入额 precedes 融资余额.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["symbol"] = df["证券代码"].map(_tolerant_code)
    out["date"] = pd.to_datetime(date)
    out[FIN_BALANCE] = pd.to_numeric(df.get("融资余额"), errors="coerce")
    out[FIN_BUY] = pd.to_numeric(df.get("融资买入额"), errors="coerce")
    out[SL_BALANCE] = pd.to_numeric(df.get("融券余额"), errors="coerce")
    out[SL_VOLUME] = pd.to_numeric(df.get("融券余量"), errors="coerce")
    out[SL_SELL_VOLUME] = pd.to_numeric(df.get("融券卖出量"), errors="coerce")
    return out.dropna(subset=["symbol"])


_DEFAULT_CACHE_DIR = "data/margin"


def load_margin_cache(cache_dir: str | Path = _DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Load all cached monthly parquets — offline, no network, no new fetches."""
    cache = Path(cache_dir)
    if not cache.is_dir():
        return pd.DataFrame()
    files = sorted(cache.glob("margin_*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def fetch_margin_akshare(
    start: str = "2024-01-01",
    end: Optional[str] = None,
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Fetch per-symbol daily margin balances (SSE + SZSE) via AKShare.

    Both exchanges expose per-day detail endpoints only, so the history is
    fetched one trading date at a time (2 calls/day) and checkpointed into
    monthly parquet files under ``cache_dir`` — re-runs skip cached months and
    the daily shadow loop only ever pays for the newest date.
    """
    import akshare as ak

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start, end)

    # fetch/checkpoint month by month: every finished month is a parquet file,
    # so a re-run only pays for unfinished months and the final assembly just
    # reads the files back.
    acc: dict[str, list[pd.DataFrame]] = {}
    for i, d in enumerate(dates):
        key = d.strftime("%Y%m")
        if key in acc and acc[key] is None:  # month finished from cache
            continue
        if key not in acc and (cache / f"margin_{key}.parquet").is_file():
            acc[key] = None
            continue
        date = d.strftime("%Y%m%d")
        sse = retry_call(
            lambda: ak.stock_margin_detail_sse(date=date),
            retries=2, base_delay=1.0, backoff=2.0, on_error=lambda exc: pd.DataFrame(),
        )
        szse = retry_call(
            lambda: ak.stock_margin_detail_szse(date=date),
            retries=2, base_delay=1.0, backoff=2.0, on_error=lambda exc: pd.DataFrame(),
        )
        acc.setdefault(key, []).append(
            pd.concat([_norm_sse(sse, date), _norm_szse(szse, date)], ignore_index=True)
        )
        is_last = i == len(dates) - 1
        month_ends = not is_last and dates[i + 1].strftime("%Y%m") != key
        if month_ends or is_last:
            frame = pd.concat(acc[key], ignore_index=True)
            frame.to_parquet(cache / f"margin_{key}.parquet", index=False)
            acc[key] = None

    # one file per DISTINCT month — the date generator repeats each month key
    # ~20 times, which would read every parquet ~20 times
    months = sorted({d.strftime("%Y%m") for d in dates})
    frames = [
        pd.read_parquet(cache / f"margin_{m}.parquet")
        for m in months
        if (cache / f"margin_{m}.parquet").is_file()
    ]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def to_pit_records(margin: pd.DataFrame) -> pd.DataFrame:
    """Convert the normalised margin frame into PIT records.

    ``valid_from = date + 1D`` (published after close → visible next day).
    Within the batch each record's ``valid_to`` is the *next* snapshot's
    ``valid_from`` per symbol (NaT for the newest) — a snapshot series must be
    a closed interval, unlike the open-ended single-fact model, or an old
    balance would look valid forever.
    """
    if margin.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["symbol"] = margin["symbol"]
    out["valid_from"] = pd.to_datetime(margin["date"]) + pd.to_timedelta("1D")
    for col in (FIN_BALANCE, FIN_BUY, SL_BALANCE, SL_VOLUME, SL_SELL_VOLUME):
        out[col] = margin[col] if col in margin.columns else None
    out = out.sort_values(["symbol", "valid_from"])
    out["valid_to"] = out.groupby("symbol")["valid_from"].shift(-1)
    out["record_type"] = RECORD_TYPE
    return out.reset_index(drop=True)


def upsert_margin(store, margin: pd.DataFrame) -> dict[str, int]:
    """Store margin records into a PIT store, closing superseded snapshots.

    Records superseding previously-ingested open-ended records (incremental
    daily ingestion) must *close* them: the old rows are re-upserted with
    ``valid_to`` = the new batch's first ``valid_from`` per symbol, so a PIT
    query never sees two live snapshots for one symbol.
    """
    recs = to_pit_records(margin)
    if recs.empty:
        return {"records": 0}
    try:
        old = store.snapshot(RECORD_TYPE)
    except Exception:  # noqa: BLE001 — a backend without snapshot support is fine
        old = pd.DataFrame()
    closes: list[pd.DataFrame] = []
    if not old.empty and {"symbol", "valid_from", "valid_to"} <= set(old.columns):
        old = old.copy()
        old["valid_from"] = pd.to_datetime(old["valid_from"])
        for sym, grp in recs.groupby("symbol"):
            new_min = grp["valid_from"].min()
            mask = (
                (old["symbol"] == sym)
                & old["valid_to"].isna()
                & (old["valid_from"] < new_min)
            )
            if mask.any():
                sub = old[mask].copy()
                sub["valid_to"] = new_min
                closes.append(sub)
    if closes:
        recs = pd.concat([recs, *closes], ignore_index=True)
    store.upsert(recs)
    return {"records": int(len(recs)), "closed": int(sum(len(c) for c in closes))}


# --------------------------------------------------------------------------- #
# factors (built on the PIT store — anti-look-ahead by construction)
# --------------------------------------------------------------------------- #

def margin_factors(margin_long: pd.DataFrame) -> pd.DataFrame:
    """Leverage-structure factors from a ``margin_long`` frame.

    * ``fin_growth`` — 20-day financing-balance growth (retail leverage trend);
    * ``fin_buy_growth`` — 20-day financing-buy growth (leverage inflow);
    * ``sl_growth`` — 20-day short-sale balance growth;
    * ``sl_mix`` — short balance / financing balance (short interest mix).

    ``margin_long`` carries ``date`` + ``symbol`` columns (the normalised
    fetch output or a PIT snapshot's valid_from/symbol re-projection). Pure
    panel arithmetic — PIT safety comes from the caller using an as-of slice.
    """
    need = {"date", "symbol"}
    if not need <= set(margin_long.columns):
        raise ValueError(f"margin_long needs {sorted(need)} columns")
    mh = margin_long.set_index(["date", "symbol"]).sort_index()
    mh = mh[~mh.index.duplicated(keep="last")]
    wide = mh.unstack()  # date × (feature, symbol)
    cols: dict[str, pd.Series] = {}
    if FIN_BALANCE in mh.columns:
        cols["fin_growth"] = wide[FIN_BALANCE].pct_change(20, fill_method=None).stack(dropna=False).rename("fin_growth")
    if FIN_BUY in mh.columns:
        cols["fin_buy_growth"] = wide[FIN_BUY].pct_change(20, fill_method=None).stack(dropna=False).rename("fin_buy_growth")
    if SL_BALANCE in mh.columns:
        cols["sl_growth"] = wide[SL_BALANCE].pct_change(20, fill_method=None).stack(dropna=False).rename("sl_growth")
    if FIN_BALANCE in mh.columns and SL_BALANCE in mh.columns:
        mix = wide[SL_BALANCE] / wide[FIN_BALANCE].replace(0.0, np.nan)
        cols["sl_mix"] = mix.stack(dropna=False).rename("sl_mix")
    out = pd.concat(cols.values(), axis=1) if cols else pd.DataFrame(
        columns=["fin_growth", "fin_buy_growth", "sl_growth", "sl_mix"]
    )
    out.index.names = ["date", "symbol"]
    # plain float64 — a pd.NA (NAType) here would break numpy/LightGBM downstream
    return out.astype(np.float64)


def margin_ml_features(margin_long: pd.DataFrame) -> dict[str, pd.Series]:
    """ML-ready feature series: raw crowding factors + reversed (short-crowding).

    The full-history gate showed the leverage-crowding direction is consistently
    NEGATIVE (long the crowded names underperforms) but too weak to pass the
    0.015 single-factor gate alone — exactly the case where ML combination can
    help, so BOTH directions are exposed as features.
    """
    fac = margin_factors(margin_long)
    out: dict[str, pd.Series] = {}
    for col in fac.columns:
        s = fac[col].rename(f"x_margin_{col}")
        out[f"x_margin_{col}"] = s
        out[f"x_margin_{col}_rev"] = (-s).rename(f"x_margin_{col}_rev")
    return out


def load_margin_extras_from_store(cfg) -> dict[str, pd.Series]:
    """Margin crowding factors (raw + reversed) from the configured PIT store.

    The ML-book bridge needs this without importing scripts/: same logic as
    ``scripts/ml_common.load_margin_extras`` — factors are computed on the
    ``valid_from`` grid (PIT-safe), trailing 20-day growth.
    """
    from .point_in_time_loader import from_url

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
    return margin_ml_features(long)


__all__ = [
    "RECORD_TYPE",
    "FIN_BALANCE",
    "FIN_BUY",
    "SL_BALANCE",
    "SL_VOLUME",
    "SL_SELL_VOLUME",
    "fetch_margin_akshare",
    "load_margin_cache",
    "to_pit_records",
    "upsert_margin",
    "margin_factors",
    "margin_ml_features",
    "load_margin_extras_from_store",
]
