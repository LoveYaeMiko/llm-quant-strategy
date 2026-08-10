"""A-share price-limit lock detection — LIMIT_DOWN_BLUEPRINT, 方案 B.

The engine's portfolio Sharpe / max-drawdown are computed on *tradeable*
forward returns: a ``(date, symbol)`` forward-return bar is excluded when the
symbol is price-limit locked at entry (can't buy / can't short the auction) or
at exit (can't sell / can't cover the close), because realising a -10% limit
continuation is not possible when the stock is locked at the limit. Rank IC
deliberately stays on the *raw* forward returns — ranking information is
untouched (blueprint §3.1: "IC 计算仍然使用原始 forward_returns").

Detection: a bar is locked when its open or close breaches the limit band
relative to the previous close. Boards carry different bands — 主板 10%,
创业板 10% until 2020-08-24 then 20%, 科创板 20%, 北交所 30% — so the
effective threshold is board- and date-aware (``dynamic_threshold``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 创业板 switched from a 10% to a 20% daily limit on this date.
CHINEXT_20PCT_SINCE = pd.Timestamp("2020-08-24")


def _board_limit(symbol: str, date: pd.Timestamp, dynamic: bool) -> float:
    """Limit band (as a return) for ``symbol`` on ``date``."""
    if not dynamic:
        return 0.095
    sym = str(symbol)
    if sym.startswith(("300", "301")):          # 创业板
        return 0.195 if date >= CHINEXT_20PCT_SINCE else 0.095
    if sym.startswith(("688", "689")):          # 科创板
        return 0.195
    if sym.startswith(("8", "4")):              # 北交所
        return 0.295
    return 0.095                                # 主板


def limit_lock_mask(
    long: pd.DataFrame,
    *,
    base_threshold: float = 0.095,
    dynamic_threshold: bool = True,
) -> pd.Series:
    """Boolean ``(date, symbol)`` series: True where that bar is limit-locked.

    ``long`` is the ``(date, symbol)`` OHLCV panel built by the CLI market
    (see ``src.cli._market_from_records``). A bar is locked when its **close**
    breaches the limit band relative to the previous close:

    * detection is **close-only**. The portfolio executes at the close, so what
      makes a bar untradeable is the close sitting on the limit (you cannot buy
      or sell at a locked close) — touching the limit intraday and recovering is
      not a lock. It is also the only basis-consistent choice on this panel:
      ``open``/``high``/``low`` are raw prices while ``close`` is
      adjustment-scaled (baostock-style 前复权), so mixing them would flag every
      bar whose raw open lies outside the adjusted-close band (a real, measured
      ~86% false-positive rate before this fix).

    The first day of a symbol's history has no previous close and is never
    locked.
    """
    idx = long.index
    close = long["close"].astype(float)
    prev = close.groupby(level=1).shift(1)          # previous close per symbol

    thr = pd.Series(base_threshold, index=idx)
    if dynamic_threshold:
        dates = idx.get_level_values(0)
        after = np.asarray(dates) >= np.datetime64(CHINEXT_20PCT_SINCE)
        is_ce = pd.Series(idx.get_level_values(1)).str.startswith(("300", "301")).to_numpy()
        is_ke = pd.Series(idx.get_level_values(1)).str.startswith(("688", "689")).to_numpy()
        is_bj = pd.Series(idx.get_level_values(1)).str.startswith(("8", "4")).to_numpy()
        thr = thr.copy()
        thr.iloc[np.where(is_ce)[0]] = np.where(after, 0.195, base_threshold)[is_ce]
        thr.iloc[np.where(is_ke)[0]] = 0.195
        thr.iloc[np.where(is_bj)[0]] = 0.295

    lo = prev * (1 - thr)
    hi = prev * (1 + thr)
    locked = (close <= lo) | (close >= hi)
    locked = locked & prev.notna()
    return locked.astype(bool)


def tradeable_forward_returns(
    forward: pd.Series,
    long: pd.DataFrame,
    *,
    base_threshold: float = 0.095,
    dynamic_threshold: bool = True,
) -> pd.Series:
    """``forward`` with limit-locked bars masked to NaN (untradeable).

    A bar ``(t, s)`` is masked when ``s`` is locked at entry ``t`` (the
    position could not be opened at close) or at exit ``t+1`` (the position
    could not be closed at the next close). Everything else is untouched, so
    IC computed on the raw ``forward`` is identical — only the portfolio
    returns that feed Sharpe / max-drawdown change.
    """
    locked = limit_lock_mask(
        long, base_threshold=base_threshold, dynamic_threshold=dynamic_threshold
    )
    entry = locked.reindex(forward.index, fill_value=False)
    # locked shifted one trading day forward per symbol: the exit-day lock.
    exit_ = locked.groupby(level=1).shift(-1).reindex(forward.index, fill_value=False)
    untradeable = entry | exit_
    return forward.mask(untradeable)


__all__ = ["limit_lock_mask", "tradeable_forward_returns"]
