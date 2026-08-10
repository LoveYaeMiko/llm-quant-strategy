"""Historical quarterly profit fetch + cache for PEAD (Phase 9.2).

PEAD (post-earnings announcement drift) needs a **point-in-time** history of
quarterly earnings with the true *publication* date, so the signal can never
leak into a backtest before it existed. Two free sources were rejected in the
Phase 9.2 design review (2026-08-10):

* ``akshare.stock_yjbb_em`` — one call per report period, but its
  ``最新公告日期`` is Eastmoney's *dataset-refresh* date, not the original
  announcement (600519 Q1-2024 showed 2025-04-30 instead of 2024-04-25) — a
  one-year leak, unusable for PIT;
* ``akshare.stock_financial_abstract`` — wide-format, no per-report announce
  date at all.

Baostock's ``query_profit_data`` is the one free source whose ``pubDate`` is
the true publication date (verified: 600519 annual 2023 -> pubDate 2024-04-03).
It costs one call per (symbol, year, quarter) — HS300 x 2020-2025 = ~7.2k calls,
one-time, cached per symbol under ``data/financials/``.

The cached panel drives :mod:`src.factors.pead`: SUE uses the **cumulative**
EPS (netProfit / totalShare, matching the statement convention) and compares
against the same quarter one year earlier — the seasonal difference cancels the
cumulative part without deriving single-quarter EPS.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

from .ingestion.baostock_adapter import BaostockAdapter

logger = logging.getLogger(__name__)

# System symbol form is "600519.SH"; baostock uses "sh.600519".
_EXCHANGE_EXTS = {"SH": "sh", "SZ": "sz", "BJ": "bj"}

QUARTER_END = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}


def symbol_to_baostock(code: str) -> str:
    """Convert a system symbol ("600519.SH") to baostock form ("sh.600519")."""
    num, _, ex = code.partition(".")
    ext = _EXCHANGE_EXTS.get((ex or "").upper(), "sh")
    return f"{ext}.{num}"


def _symbol_to_system(bs_code: str) -> str:
    """Convert baostock form ("sh.600519") back to a system symbol ("600519.SH")."""
    ex, _, num = bs_code.partition(".")
    ext = "SH" if ex == "sh" else "SZ" if ex == "sz" else "BJ"
    return f"{num}.{ext}"


def fetch_symbol_profit(
    symbol: str,
    years: list[int],
    adapter: BaostockAdapter | None = None,
    pause: float = 0.0,
) -> pd.DataFrame:
    """Quarterly profit for one symbol over ``years`` (statDate, pubDate, eps_cum).

    eps_cum = netProfit / totalShare (statement convention: cumulative within a
    fiscal year). Rows are sorted by pubDate (announcement time — the PIT axis).
    """
    adapter = adapter or BaostockAdapter(enabled=True)
    bs_code = symbol_to_baostock(symbol)
    rows: list[dict] = []
    for year in years:
        for quarter in (1, 2, 3, 4):
            try:
                df = adapter._query(
                    "query_profit_data", code=bs_code, year=year, quarter=quarter
                )
            except Exception as exc:  # noqa: BLE001 — a flaky quarter must not kill the symbol
                logger.warning("profit %s %dQ%d failed: %s", symbol, year, quarter, exc)
                continue
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                net = r.get("netProfit")
                total = r.get("totalShare")
                try:
                    eps_cum = float(net) / float(total) if net and total else float("nan")
                except (TypeError, ValueError):
                    eps_cum = float("nan")
                rows.append(
                    {
                        "symbol": symbol,
                        "statDate": str(r.get("statDate", "")),
                        "pubDate": str(r.get("pubDate", "")),
                        "netProfit": net,
                        "totalShare": total,
                        "eps_cum": eps_cum,
                    }
                )
            if pause:
                time.sleep(pause)
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "statDate", "pubDate", "netProfit", "totalShare", "eps_cum"]
        )
    out = pd.DataFrame(rows).dropna(subset=["pubDate"])
    out["pubDate"] = pd.to_datetime(out["pubDate"])
    out["statDate"] = pd.to_datetime(out["statDate"])
    out = out.drop_duplicates(subset=["statDate", "pubDate"], keep="last")
    return out.sort_values("pubDate").reset_index(drop=True)


def _cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"profit_{symbol.replace('.', '_')}.csv"


def ensure_profit_panel(
    symbols: list[str],
    years: list[int],
    cache_dir: str | Path | None = None,
    force: bool = False,
    max_symbols: int | None = None,
) -> pd.DataFrame:
    """Fetch (or load cached) quarterly profit for ``symbols`` into one panel.

    Per-symbol CSV cache under ``cache_dir`` (default ``data/financials``); a
    symbol already cached and complete for ``years`` is loaded, not re-fetched.
    """
    cache_dir = Path(cache_dir or os.environ.get("FQA_FIN_CACHE", "data/financials"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    adapter = BaostockAdapter(enabled=True)
    n = len(symbols)
    for i, symbol in enumerate(symbols, 1):
        path = _cache_path(cache_dir, symbol)
        if not force and path.is_file():
            try:
                frames.append(pd.read_csv(path))
                continue
            except Exception:  # noqa: BLE001 — corrupt cache -> refetch
                logger.warning("corrupt cache %s, refetching", path)
        df = fetch_symbol_profit(symbol, years, adapter=adapter)
        df.to_csv(path, index=False)
        frames.append(df)
        if i % 25 == 0 or i == n:
            logger.info("profit fetch %d/%d symbols", i, n)
        if max_symbols and i >= max_symbols:
            break
    if not frames:
        return pd.DataFrame(
            columns=["symbol", "statDate", "pubDate", "netProfit", "totalShare", "eps_cum"]
        )
    panel = pd.concat(frames, ignore_index=True)
    panel["pubDate"] = pd.to_datetime(panel["pubDate"])
    panel["statDate"] = pd.to_datetime(panel["statDate"])
    return panel
