"""LIMIT_DOWN blueprint 方案 B — price-limit lock detection and tradeable returns.

The engine's portfolio Sharpe / max-drawdown run on forward returns with
price-limit-locked bars masked to NaN (a -10% continuation you cannot actually
transact), while rank IC stays on the raw series. These tests pin the mask
identification (incl. 2015-08-31-style all-limit-down days), the board/date-aware
thresholds, and the eval-agent integration (raw IC, tradable portfolio).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.agents.base_agent import AgentContext
from src.agents.eval_agent import EvalAgent
from src.backtest.limit_locked import limit_lock_mask, tradeable_forward_returns


def _long_from_closes(closes_by_symbol: dict[str, list[float]], dates) -> pd.DataFrame:
    rows = []
    for sym, closes in closes_by_symbol.items():
        for d, c in zip(dates, closes):
            if c is None:
                continue
            rows.append(
                {
                    "date": d, "symbol": sym,
                    "open": c, "high": c, "low": c, "close": c, "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows).set_index(["date", "symbol"]).sort_index()


def _forward(long: pd.DataFrame) -> pd.Series:
    close_wide = long["close"].unstack()
    return close_wide.pct_change(fill_method=None).shift(-1).stack().rename("fwd")


def test_limit_lock_mask_identifies_limit_down_and_up():
    dates = pd.bdate_range("2021-01-01", periods=5)
    # B limits down at day 2 (-10% vs prev); C limits up at day 3 (+10% vs prev).
    long = _long_from_closes(
        {
            "A": [100, 101, 100, 101, 102],
            "B": [100, 101, 90.9, 91, 91],   # day2 close 90.9 = 101 * 0.90
            "C": [100, 99, 100, 110, 110],   # day3 close 110 = 100 * 1.10
        },
        dates,
    )
    mask = limit_lock_mask(long)
    assert mask.loc[(dates[2], "B")]          # limit-down bar is locked
    assert not mask.loc[(dates[1], "B")]
    assert mask.loc[(dates[3], "C")]          # limit-up bar is locked
    assert not mask.loc[(dates[4], "A")]      # normal bar untouched
    assert not mask.loc[(dates[0], "A")]      # no previous close -> never locked


def test_tradeable_forward_masks_entry_and_exit_locks_only():
    dates = pd.bdate_range("2021-01-01", periods=5)
    long = _long_from_closes(
        {"A": [100, 101, 100, 101, 102], "B": [100, 101, 90.9, 91, 91]}, dates
    )
    fwd = _forward(long)
    tradable = tradeable_forward_returns(fwd, long)

    # B's limit-down day is d2.  The forward bar INTO d2 (exit lock, can't sell
    # at the locked close) and the bar OUT of d2 (entry lock, can't buy at the
    # locked close) are both masked; surrounding bars are untouched.
    assert np.isnan(tradable.loc[(dates[1], "B")])   # exit into the limit-down
    assert np.isnan(tradable.loc[(dates[2], "B")])   # entry at the limit-down
    assert not np.isnan(tradable.loc[(dates[0], "B")])
    assert not np.isnan(tradable.loc[(dates[3], "B")])
    assert tradable.loc[(dates[0], "B")] == fwd.loc[(dates[0], "B")]

    # A (no limit days) is entirely untouched.
    assert (tradable.xs("A", level=1) == fwd.xs("A", level=1)).all()


def test_2015_crash_style_all_limit_down_day_is_fully_masked():
    # A "2015-08-31" style day: ~75% of names close at limit-down. Every forward
    # bar touching such a name that day must be masked.
    dates = pd.bdate_range("2015-08-28", periods=3)
    closes = {f"S{i:02d}": [100.0, 90.0, 90.9] for i in range(8)}  # 8 limit-down
    closes["H01"] = [100.0, 99.5, 100.0]                            # 1 normal
    long = _long_from_closes(closes, dates)
    tradable = tradeable_forward_returns(_forward(long), long)
    limit_day = dates[1]  # 2015-08-31
    for i in range(8):
        assert np.isnan(tradable.loc[(limit_day, f"S{i:02d}")])       # entry lock
        prev = tradable.xs("S%02d" % i, level=1)
        assert np.isnan(prev.loc[dates[0]])                           # exit lock
    assert not np.isnan(tradable.loc[(limit_day, "H01")])


def test_board_dynamic_threshold():
    pre = [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    post = [pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")]
    # -11% on a ChiNext name: locked under the pre-2020 10% band, normal under 20%.
    assert limit_lock_mask(_long_from_closes({"300001.SZ": [100.0, 89.0]}, pre)).loc[
        (pre[1], "300001.SZ")
    ]
    assert not limit_lock_mask(_long_from_closes({"300001.SZ": [100.0, 89.0]}, post)).loc[
        (post[1], "300001.SZ")
    ]
    # -19.6% is locked post-2020 on ChiNext, and -11% is locked on 主板 at any time.
    assert limit_lock_mask(_long_from_closes({"300001.SZ": [100.0, 80.4]}, post)).loc[
        (post[1], "300001.SZ")
    ]
    assert limit_lock_mask(_long_from_closes({"600000.SH": [100.0, 89.0]}, post)).loc[
        (post[1], "600000.SH")
    ]
    # dynamic_threshold=False applies the single 0.095 band everywhere.
    assert limit_lock_mask(
        _long_from_closes({"300001.SZ": [100.0, 89.0]}, post), dynamic_threshold=False
    ).loc[(post[1], "300001.SZ")]


def test_mixed_basis_open_does_not_falsely_lock():
    # The PIT panel is baostock-style mixed-basis: open/high/low are raw while
    # close is adjustment-scaled (前复权). Before the close-only fix, comparing
    # the raw open against the adjusted-close band flagged ~86% of bars as
    # limit-locked. A bar whose raw open is huge but whose adjusted close is
    # flat must NOT be locked.
    dates = pd.bdate_range("2010-01-01", periods=3)
    long = pd.DataFrame(
        [
            {"date": dates[0], "symbol": "000001.SZ", "open": 23.75, "high": 23.90,
             "low": 22.75, "close": 6.0184, "volume": 556499.0},
            {"date": dates[1], "symbol": "000001.SZ", "open": 6.10, "high": 6.15,
             "low": 6.00, "close": 6.1042, "volume": 500000.0},
            {"date": dates[2], "symbol": "000001.SZ", "open": 6.20, "high": 6.22,
             "low": 6.05, "close": 6.1560, "volume": 500000.0},
        ]
    ).set_index(["date", "symbol"])
    mask = limit_lock_mask(long)
    assert not mask.any()          # +1.4% / +0.8% closes are plainly tradeable


def test_eval_agent_keeps_raw_ic_and_tradable_portfolio():
    dates = pd.bdate_range("2021-01-01", periods=8)
    # A collapses with two consecutive limit-down closes; B-F drift normally.
    closes = {
        "A": [100, 101, 102, 103, 104, 105, 94.5, 85.05],
        "B": [100, 100.5, 101, 101.5, 102, 102.5, 103, 103.5],
        "C": [100, 101, 100, 101, 100, 101, 100, 101],
        "D": [100, 99.5, 100, 99.5, 100, 99.5, 100, 99.5],
        "E": [100, 102, 101, 103, 102, 104, 103, 105],
        "F": [100, 100, 100, 100, 100, 100, 100, 100],
    }
    long = _long_from_closes(closes, dates)
    fwd = _forward(long)
    tradable = tradeable_forward_returns(fwd, long)
    assert int(fwd.notna().sum()) > int(tradable.notna().sum())  # bars were masked

    # A factor that systematically longs the crashing name.
    scores = pd.Series(1.0, index=fwd.index).where(
        fwd.index.get_level_values(1) != "A", 2.0
    ).rename("sig")

    agent = EvalAgent(config=None)
    ctx = AgentContext(as_of=dates[-1], symbols=list(closes), config=None)
    m_raw = agent.evaluate(ctx, scores, fwd)
    m_trd = agent.evaluate(ctx, scores, fwd, forward_tradable=tradable)

    # IC is computed on the raw series in both paths — identical.
    assert m_raw["rank_ic"] == m_trd["rank_ic"]
    # The portfolio no longer realises the untradeable -10% limit-down bars.
    assert m_trd["max_drawdown"] <= m_raw["max_drawdown"]
    assert m_trd["total_return"] > m_raw["total_return"]
