"""The forward-period RISK gate — the replacement for the capability gate.

Why (2026-09-09 audit, items 1.1-1.3). A forward window cannot prove alpha.
With ``SE(Sharpe) = sqrt(252/N)`` the t-statistic is ``SR × sqrt(N/252)``, so
proving ``SR = 1.0`` at t = 2 needs ~4.0 years and at t = 2.5 needs ~6.3 years;
``SR = 0.5`` needs ~16 / 25 years. Regime matching decays in 1-3 years, so by the
time the window has power, the parameters no longer belong to the regime. Judging
the strategy on forward Sharpe therefore measures noise and calls it evidence.

What a forward window CAN answer are the high-signal-to-noise questions about the
pipeline itself:

* ``pipeline_tracking_error`` — does the deployed assembly reproduce the frozen
  specification day after day (data drift, non-determinism, silent code change)?
* ``cost_model_calibration`` — is the charged cost consistent with the
  pre-registered cost spec and with the realized execution prices?
* ``operational_reliability`` — availability, data freshness, legal-constraint
  violations, symbol coverage.

:func:`evaluate_gate` turns those measurements into a hard/soft verdict. HARD
failures mean "the pipeline is not trustworthy" (stop and fix); they say nothing
about whether the strategy makes money. SOFT metrics (Sharpe, drawdown, excess
return) are recorded and never gate anything — the window has no power for them.

Every function here is pure (no file/DB/network I/O) so the arithmetic is
unit-testable; :mod:`scripts.forward_health` does the I/O and calls in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

#: Trading days per year used by the Sharpe standard error (A-share ~243, but the
#: standard formula the project already uses is 252; keep ONE convention).
TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# 1.1 power: what the window can and cannot prove
# --------------------------------------------------------------------------- #
def years_for_t(sharpe: float, t: float = 2.0) -> float:
    """Calendar years needed for ``|t| = t`` given a true Sharpe (``N = 252(t/SR)²``)."""
    if sharpe <= 0:
        return float("inf")
    n_days = TRADING_DAYS * (t / sharpe) ** 2
    return n_days / TRADING_DAYS


def power_table(sharpe: float, t_values: Sequence[float] = (2.0, 2.5)) -> dict:
    """``{t: years}`` for one Sharpe — the numbers quoted in the protocol doc."""
    return {f"t_{t:g}".replace(".", "_"): round(years_for_t(sharpe, t), 1) for t in t_values}


def sharpe_standard_error(n_days: int) -> float:
    """``sqrt(252/N)`` — the sampling error of an annualized Sharpe."""
    return math.sqrt(TRADING_DAYS / n_days) if n_days > 0 else float("inf")


def binom_two_sided_p(n_pos: int, n_neg: int) -> float:
    """Exact two-sided binomial p-value for ``n_pos`` of ``n_pos+n_neg`` successes.

    Used for the sign-bias test: under "no systematic bias" the daily tracking
    error flips sign like a fair coin. A tiny p-value means the pipeline is
    systematically one-sided (e.g. always fills worse than the reference), which
    an absolute-mean threshold alone cannot see.
    """
    n = int(n_pos) + int(n_neg)
    if n <= 0:
        return 1.0
    k = min(int(n_pos), int(n_neg))
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


# --------------------------------------------------------------------------- #
# 1.2/1.3 metric computations
# --------------------------------------------------------------------------- #
def tracking_error(
    recorded: pd.Series,
    replay: pd.Series,
    *,
    exclude_dates: Optional[Iterable] = None,
) -> dict:
    """Daily tracking error of the deployed book vs its replay, in pp/day.

    ``recorded`` and ``replay`` are DAILY RETURNS of the same book over the same
    dates, one produced by the live pipeline and one by re-running the same rules
    on the data as it exists now. A large value means the pipeline's output is
    not reproducible — the data moved (adjustment-anchor drift, a revised bar) or
    the code changed silently.

    ``exclude_dates`` drops days whose outcome the replay CANNOT reproduce by
    construction: a date on which the real-time layer executed a fill at a live
    print (``source="live"``) is an external market event, not a deterministic
    function of the bars. Leaving those days in would measure "live execution vs
    bar replay" — a known, intended difference — instead of pipeline fidelity.
    The excluded days are reported separately so the difference stays visible.

    Returns ``{n_days, mean_signed_pp, mean_abs_pp, max_abs_pp, rmse_pp,
    n_pos, n_neg, sign_bias_p, worst_date, n_excluded, excluded_mean_abs_pp}``.
    """
    a = pd.Series(recorded).dropna()
    b = pd.Series(replay).dropna()
    idx = a.index.intersection(b.index)
    diff_all = (a.loc[idx] - b.loc[idx]).astype(float) * 100.0 if len(idx) else pd.Series(dtype=float)
    excluded = {pd.Timestamp(d).normalize() for d in (exclude_dates or [])}
    if len(diff_all) and excluded:
        mask = ~pd.Index([pd.Timestamp(i).normalize() for i in diff_all.index]).isin(excluded)
        mask = np.asarray(mask)
        diff = diff_all[mask]
        excl_diff = diff_all[~mask]
    else:
        diff = diff_all
        excl_diff = pd.Series(dtype=float)
    if len(diff) == 0:
        return {"n_days": 0, "mean_signed_pp": 0.0, "mean_abs_pp": 0.0, "max_abs_pp": 0.0,
                "rmse_pp": 0.0, "n_pos": 0, "n_neg": 0, "sign_bias_p": 1.0, "worst_date": None,
                "n_excluded": int(len(excl_diff)),
                "excluded_mean_abs_pp": (round(float(excl_diff.abs().mean()), 4)
                                         if len(excl_diff) else None)}
    n_pos = int((diff > 0).sum())
    n_neg = int((diff < 0).sum())
    return {
        "n_days": int(len(diff)),
        "mean_signed_pp": round(float(diff.mean()), 4),
        "mean_abs_pp": round(float(diff.abs().mean()), 4),
        "max_abs_pp": round(float(diff.abs().max()), 4),
        "rmse_pp": round(float(np.sqrt((diff ** 2).mean())), 4),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "sign_bias_p": round(binom_two_sided_p(n_pos, n_neg), 4),
        "worst_date": str(pd.Timestamp(diff.abs().idxmax()).date()) if len(diff) else None,
        "n_excluded": int(len(excl_diff)),
        "excluded_mean_abs_pp": (round(float(excl_diff.abs().mean()), 4)
                                 if len(excl_diff) else None),
    }


#: Fill provenance whose recorded price is a MODEL, not a market print: a
#: replayed intraday stop is booked at ``min(stop, bar_close)``, so comparing it
#: with the bar close would measure the model's conservatism, not execution
#: quality. Every other source — including the legacy rows whose ``source`` is
#: empty but whose ``time`` field tells us which reference to use — is checkable.
PRICE_UNCHECKABLE_SOURCES: tuple[str, ...] = ("replay",)

#: Kept for backwards compatibility with callers that imported the old name.
PRICE_CHECKABLE_SOURCES: tuple[str, ...] = ("live", "auction", "close", "unlabelled")


def cost_deviation(
    fills: pd.DataFrame,
    cost_spec: Mapping[str, float],
    *,
    reference_prices: Optional[Mapping[tuple[str, str], float]] = None,
    modeled_slippage_bps: float = 2.0,
    price_integrity_bps_max: float = 2.0,
) -> dict:
    """Cost-model calibration on REAL fills.

    A paper account has no broker statement, so only two cost questions are
    answerable — and the audit's complaint was that the old red line answered
    neither (it compared the ledger against the very model that produced it):

    1. ``fee_deviation_pct`` — the charged fee vs the fee the PRE-REGISTERED spec
       implies for each fill's own notional/side. Non-zero means the cost model
       drifted from the spec (config edit, inconsistent min-commission floor).
    2. ``price_integrity_*`` — the recorded fill price vs a market reference (the
       day's adjusted close for close/auction fills, the same-minute print for
       live fills), over :data:`PRICE_CHECKABLE_SOURCES` only. A deviation means
       the ledger holds a price the market never printed (stale bar, basis error,
       fabricated fill).

    The market-impact term (``modeled_slippage_bps``) is NOT measurable without
    real broker fills; it is reported under ``unmeasured`` and must be calibrated
    before any real-money deployment — see ``docs/FORWARD_PROTOCOL.md`` §1.3.
    """
    empty = {"n_fills": 0, "charged_total": 0.0, "expected_total": 0.0,
             "fee_deviation_pct": 0.0, "price_integrity_bps": None,
             "price_integrity_abs_bps": None, "price_integrity_max_bps": None,
             "n_price_checked": 0, "n_price_skipped": 0, "n_price_unlabelled": 0,
             "price_check_sources": [],
             "slippage_model_bps": float(modeled_slippage_bps),
             "price_integrity_bps_max": float(price_integrity_bps_max),
             "unmeasured": ["market_impact_bps"], "by_source": {}}
    if fills is None or len(fills) == 0:
        return empty
    df = fills.copy()
    if "source" not in df.columns:
        df["source"] = ""
    df["source"] = df["source"].fillna("").replace("", "unlabelled")
    comm_bps = float(cost_spec.get("commission_bps", 0.0))
    transfer_bps = float(cost_spec.get("transfer_fee_bps", 0.0))
    stamp_bps = float(cost_spec.get("stamp_tax_sell_bps", 0.0))
    min_comm = float(cost_spec.get("min_commission", 0.0))

    def _expected(notional: float, side: str) -> float:
        commission = max(min_comm, notional * comm_bps / 10_000.0)
        transfer = notional * transfer_bps / 10_000.0
        stamp = notional * stamp_bps / 10_000.0 if str(side).lower().startswith("s") else 0.0
        return commission + transfer + stamp

    charged = float(pd.to_numeric(df["commission"], errors="coerce").fillna(0.0).sum())
    expected = float(sum(_expected(abs(float(r.notional)), str(r.side)) for r in df.itertuples()))
    denom = expected if expected > 0 else charged
    fee_dev = 0.0 if denom == 0 else (charged - expected) / denom * 100.0

    # --- price integrity on the market-price fills ---------------------------
    signed: list[float] = []
    skipped = 0
    n_unlabelled = 0
    checked_sources: set[str] = set()
    if reference_prices:
        for r in df.itertuples():
            src = str(getattr(r, "source", "") or "unlabelled")
            if src in PRICE_UNCHECKABLE_SOURCES:
                skipped += 1
                continue
            if src == "unlabelled":
                # legacy rows predate the source column; their ``time`` field still
                # says which reference applies (empty → the day's close), so they
                # ARE checkable and must not be waved through unchecked
                n_unlabelled += 1
            key = (str(pd.Timestamp(r.date).date()), str(r.symbol))
            ref = reference_prices.get(key)
            if ref is None or not np.isfinite(ref) or ref <= 0:
                skipped += 1
                continue
            px = float(r.price)
            side = 1.0 if float(r.shares) > 0 else -1.0
            # signed: buying above the reference (or selling below) is a cost
            signed.append(side * (px / float(ref) - 1.0) * 10_000.0)
            checked_sources.add(src)
    integrity = round(float(np.mean(signed)), 3) if signed else None
    integrity_abs = round(float(np.mean(np.abs(signed))), 3) if signed else None
    integrity_max = round(float(np.max(np.abs(signed))), 3) if signed else None

    by_source: dict[str, dict] = {}
    for src, grp in df.groupby("source"):
        by_source[str(src)] = {
            "n_fills": int(len(grp)),
            "charged": round(float(pd.to_numeric(grp["commission"], errors="coerce").fillna(0.0).sum()), 2),
            "notional": round(float(pd.to_numeric(grp["notional"], errors="coerce").fillna(0.0).sum()), 2),
            "price_checkable": bool(str(src) in PRICE_CHECKABLE_SOURCES),
        }
    return {
        "n_fills": int(len(df)),
        "charged_total": round(charged, 2),
        "expected_total": round(expected, 2),
        "fee_deviation_pct": round(fee_dev, 2),
        "price_integrity_bps": integrity,
        "price_integrity_abs_bps": integrity_abs,
        "price_integrity_max_bps": integrity_max,
        "n_price_checked": len(signed),
        "n_price_skipped": skipped,
        "n_price_unlabelled": n_unlabelled,
        "price_check_sources": sorted(checked_sources),
        "slippage_model_bps": float(modeled_slippage_bps),
        "price_integrity_bps_max": float(price_integrity_bps_max),
        "unmeasured": ["market_impact_bps"],
        "by_source": by_source,
    }


def fill_violations(
    fills: pd.DataFrame,
    price_panel: Optional[pd.DataFrame],
    *,
    limit_fn=None,
    tick: float = 0.01,
    lot: int = 100,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> dict:
    """Legal-constraint violations on the recorded fills (each must be 0).

    * ``t_plus_1`` — a sell on the same date as the buy that opened the lot;
    * ``lot_size`` — shares not a multiple of 100 (A-share board lot);
    * ``tick_size`` — price off the 0.01 grid;
    * ``limit_locked`` — buying into a limit-up / selling into a limit-down
      (direction-aware: the opposite is legal and must NOT be flagged);
    * ``suspension`` — a fill on a date with no bar for that symbol.

    ``price_panel=None`` skips the bar-dependent checks (``suspension``,
    ``limit_locked``) and lists them under ``unmeasured`` — an unmeasured
    constraint is not a satisfied constraint, and the gate fails on it.

    ``since``/``until`` restrict which violations are COUNTED (the full fill
    history must still be passed in: a position opened before the window and sold
    inside it is legal, and truncating the history would flag it as a T+1 breach).
    """
    out = {"t_plus_1": 0, "lot_size": 0, "tick_size": 0, "limit_locked": 0,
           "suspension": 0, "total": 0, "unmeasured": [], "examples": []}
    if fills is None or len(fills) == 0:
        return out
    has_panel = price_panel is not None and len(price_panel) > 0
    if not has_panel:
        out["unmeasured"] = ["suspension", "limit_locked"]
    df = fills.copy()
    if "seq" in df.columns:
        df = df.sort_values(["date", "seq"])
    else:
        df = df.sort_values("date")
    rets = price_panel.pct_change(fill_method=None) if has_panel else None
    lo = pd.Timestamp(since) if since else None
    hi = pd.Timestamp(until) if until else None

    def _counted(day_ts: pd.Timestamp) -> bool:
        if lo is not None and day_ts < lo:
            return False
        return not (hi is not None and day_ts > hi)

    def _ex(kind: str, row, day_ts: pd.Timestamp) -> None:
        if not _counted(day_ts):
            return
        out[kind] += 1
        if len(out["examples"]) < 8:
            out["examples"].append({"kind": kind, "date": str(row.get("date")),
                                    "symbol": str(row.get("symbol")), "side": row.get("side"),
                                    "shares": float(row.get("shares", 0)),
                                    "price": float(row.get("price", 0))})

    opens: dict[str, list[str]] = {}
    for row in df.itertuples():
        r = row._asdict()
        sym = str(r["symbol"])
        day_ts = pd.Timestamp(r["date"])
        day = str(day_ts.date())
        shares = float(r["shares"])
        px = float(r["price"])
        if abs(shares) % lot != 0:
            _ex("lot_size", r, day_ts)
        if abs(px / tick - round(px / tick)) > 1e-6:
            _ex("tick_size", r, day_ts)
        if has_panel:
            try:
                bar = price_panel.loc[day_ts, sym]
            except KeyError:
                bar = None
            if bar is None or not np.isfinite(bar):
                _ex("suspension", r, day_ts)
        if shares > 0:
            opens.setdefault(sym, []).append(day)
        else:
            held = opens.get(sym, [])
            if not held or held[0] == day:
                _ex("t_plus_1", r, day_ts)
            elif held:
                opens[sym] = held[1:]
        if has_panel and limit_fn is not None:
            try:
                lv = float(rets.loc[day_ts, sym])
            except (KeyError, TypeError):
                lv = float("nan")
            if np.isfinite(lv):
                lim = float(limit_fn(sym, day_ts, True)) - 0.005
                illegal = (shares > 0 and lv >= lim) or (shares < 0 and lv <= -lim)
                if illegal:
                    _ex("limit_locked", r, day_ts)
    out["total"] = int(out["t_plus_1"] + out["lot_size"] + out["tick_size"]
                       + out["limit_locked"] + out["suspension"])
    return out


#: The A-share decision window is TWO segments: the lunch break (11:30-13:00) is
#: not downtime. Treating 09:30-15:00 as one interval counted the 90-minute break
#: as an outage and reported ~62% availability on a perfectly healthy session
#: (found 2026-09-10 by the forward-sample tool, whose first row showed a
#: "91-minute gap" that was exactly the break).
DECISION_WINDOWS: tuple[tuple[str, str], ...] = (("09:30", "11:30"), ("13:00", "15:00"))


def _window_bounds(day: pd.Timestamp, start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    def _at(hhmm: str) -> pd.Timestamp:
        return day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))

    return _at(start), _at(end)


def availability(heartbeats: pd.DataFrame, trading_days: Sequence[pd.Timestamp],
                 *,
                 windows: Sequence[tuple[str, str]] = DECISION_WINDOWS,
                 max_gap_minutes: int = 5) -> dict:
    """Share of decision-window minutes with a fresh heartbeat.

    ``heartbeats`` needs ``ts`` (parsed timestamps) — one row per live-layer poll.
    A trading day with no heartbeat at all counts as a total outage; a gap longer
    than ``max_gap_minutes`` inside a decision segment counts as downtime. The
    lunch break between the segments is NOT downtime.
    """
    days = [pd.Timestamp(d).normalize() for d in trading_days]
    hb = pd.DataFrame(heartbeats) if heartbeats is not None else pd.DataFrame()
    if len(hb) and "ts" in hb.columns:
        hb = hb.assign(ts=pd.to_datetime(hb["ts"], errors="coerce")).dropna(subset=["ts"])
    if not days:
        # An EMPTY window is unmeasured, not perfect: returning 1.0 here would let
        # a window that has not started yet report full availability.
        return {"availability": None, "unmeasured": True, "n_days": 0,
                "n_days_down": None, "down_minutes": None, "window_minutes": None,
                "worst_day": None,
                "note": "no trading day in the window yet — availability unmeasured"}
    if not len(hb):
        # No heartbeat record at all: the mechanism did not exist for this window,
        # so availability is UNMEASURED — reporting 0.0 would claim we observed an
        # outage, reporting 1.0 would claim we observed uptime. Neither is true.
        return {"availability": None, "unmeasured": True, "n_days": len(days),
                "n_days_down": None, "down_minutes": None, "window_minutes": None,
                "worst_day": None,
                "note": "no live heartbeat recorded for this window (outputs/live_<acct>.jsonl)"}
    window_minutes = 0
    down_minutes = 0
    n_down_days = 0
    worst_day, worst_down = None, -1
    for d in days:
        for start_s, end_s in windows:
            start, end = _window_bounds(d, start_s, end_s)
            minutes = int((end - start).total_seconds() // 60) + 1
            window_minutes += minutes
            stamps = hb.loc[(hb["ts"] >= start) & (hb["ts"] <= end), "ts"].sort_values().tolist()
            if not stamps:
                down = minutes
            else:
                down = 0
                cursor = start
                for s in stamps:
                    gap = (s - cursor).total_seconds() / 60.0
                    if gap > max_gap_minutes:
                        down += int(gap - max_gap_minutes)
                    cursor = max(cursor, s)
                tail = (end - cursor).total_seconds() / 60.0
                if tail > max_gap_minutes:
                    down += int(tail - max_gap_minutes)
            down_minutes += down
            if down > 0 and worst_day != str(d.date()):
                n_down_days += 1
            if down > worst_down:
                worst_down, worst_day = down, str(d.date())
    return {
        "availability": round(1.0 - down_minutes / window_minutes, 4) if window_minutes else 1.0,
        "n_days": len(days),
        "n_days_down": n_down_days,
        "down_minutes": int(down_minutes),
        "window_minutes": int(window_minutes),
        "worst_day": worst_day,
        "max_gap_minutes": int(max_gap_minutes),
        "windows": [list(w) for w in windows],
    }


def data_freshness(bar_max_date: str | pd.Timestamp, last_trading_day: str | pd.Timestamp) -> dict:
    """Calendar days the data lags the last trading day (must be <= 1)."""
    bar = pd.Timestamp(bar_max_date).normalize()
    last = pd.Timestamp(last_trading_day).normalize()
    lag = int((last - bar).days)
    return {"bar_max_date": str(bar.date()), "last_trading_day": str(last.date()),
            "lag_days": lag}


def symbol_minute_coverage(
    frames: Mapping[str, pd.DataFrame],
    price_panel: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    threshold: float = 0.95,
) -> dict:
    """Per-symbol minute coverage measured on each symbol's OWN tradable days.

    Same definition as ``scripts/d_oos.py``: a suspension or a pre-listing period
    is not a hole, a missing feature on a day the name actually traded is.
    """
    tail = (frames or {}).get("tail_vol")
    if tail is None or not len(tail):
        return {"min_coverage": None, "n_symbols": 0, "below_threshold": [],
                "threshold": threshold, "unmeasured": True,
                "note": "no intraday feature frames loaded — coverage unmeasured"}
    pwin = price_panel.loc[(price_panel.index >= pd.Timestamp(start))
                           & (price_panel.index <= pd.Timestamp(end))]
    cov: dict[str, float] = {}
    for sym in tail.columns:
        if sym not in pwin.columns:
            continue
        tradable = pwin[sym].notna()
        n_tr = int(tradable.sum())
        if n_tr == 0:
            continue
        have = tail[sym].reindex(pwin.index).notna() & tradable
        cov[sym] = float(have.sum()) / n_tr
    if not cov:
        # no symbol had a tradable day inside the window (e.g. the window has not
        # started): report UNMEASURED rather than a 0% coverage that would read
        # like a data hole
        return {"min_coverage": None, "n_symbols": 0, "below_threshold": [],
                "threshold": threshold, "unmeasured": True,
                "note": "no tradable day inside the window — coverage unmeasured"}
    below = sorted([s for s, v in cov.items() if v < threshold], key=lambda s: cov[s])
    return {
        "min_coverage": round(min(cov.values()), 4),
        "n_symbols": len(cov),
        "below_threshold": below[:50],
        "n_below_threshold": len(below),
        "threshold": threshold,
    }


def panel_universe_health(
    price_panel: pd.DataFrame,
    *,
    as_of: Optional[str | pd.Timestamp] = None,
    warm_windows: Sequence[int] = (20, 60),
    lookback_days: int = 120,
) -> dict:
    """Is the price panel actually the universe the spec claims?

    Found 2026-09-10, on the first forward morning: the D track is declared as an
    800-name (``hs300_500``) book, but the 2026 PIT price panel carried only **301
    symbols** per day (the incremental daily ingest had covered ~300 names since
    January), so every 2026 shadow/IS day ran a ~300-name cross-section while the
    artifact recorded ``universe_size = 800``. The daily ingest widened to 800 on
    2026-09-08, but those 499 names have an eight-month gap and therefore no warm
    indicators.

    Three counts, all measured on ``as_of`` (default: the panel's last day):

    * ``n_columns`` — symbols the panel knows about at all;
    * ``n_with_price`` — symbols with a bar on ``as_of``;
    * ``n_warm_k`` — symbols with at least ``k`` non-NaN closes inside the
      trailing ``lookback_days`` (a name needs ~20 bars for ATR20 and ~60 for
      EMA50 before any rule can rank or stop it).

    ``effective_ratio`` is ``n_warm_20 / n_columns``: the share of the declared
    universe that can actually be traded. A book whose cross-section silently
    shrank must not be reported as the full universe.
    """
    out: dict = {"n_columns": int(price_panel.shape[1]) if len(price_panel.columns) else 0,
                 "as_of": None, "n_with_price": 0, "effective_ratio": None,
                 "universe_size": None}
    if price_panel is None or not len(price_panel):
        return out
    day = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(price_panel.index.max())
    out["as_of"] = str(day.date())
    hist = price_panel.loc[:day].tail(int(lookback_days))
    if not len(hist):
        return out
    out["n_with_price"] = int(hist.iloc[-1].notna().sum())
    for k in warm_windows:
        out[f"n_warm_{int(k)}"] = int((hist.notna().sum(axis=0) >= int(k)).sum())
    base = out.get("n_warm_20") or 0
    out["effective_ratio"] = round(base / out["n_columns"], 4) if out["n_columns"] else None
    out["lookback_days"] = int(lookback_days)
    return out


def soft_metrics(equity: pd.Series, benchmark: Optional[pd.Series] = None, *,
                 fills: Optional[pd.DataFrame] = None) -> dict:
    """Sharpe / maxDD / excess / hit-rate / turnover — RECORDED ONLY.

    No forward-window power (see the module docstring), so none of these gate
    anything; they are computed because a reviewer reading the artifact should not
    have to re-derive them, and because the policy file declares them.
    """
    eq = pd.Series(equity).dropna().astype(float)
    if len(eq) < 2:
        return {"sharpe": None, "max_drawdown": None, "total_return": None,
                "excess_return": None, "hit_rate": None, "n_fills": None,
                "turnover": None, "n_days": int(len(eq))}
    rets = eq.pct_change().dropna()
    vol = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / vol * math.sqrt(TRADING_DAYS)) if vol > 0 else 0.0
    dd = float((eq / eq.cummax() - 1.0).min())
    n_fills = int(len(fills)) if fills is not None else None
    turnover = None
    if fills is not None and len(fills):
        notional = float(pd.to_numeric(fills.get("notional"), errors="coerce").fillna(0.0).sum()) \
            if "notional" in getattr(fills, "columns", []) else 0.0
        mean_eq = float(eq.mean())
        turnover = round(notional / mean_eq, 4) if mean_eq > 0 else None
    out = {
        "sharpe": round(sharpe, 3),
        "sharpe_standard_error": round(sharpe_standard_error(len(eq)), 3),
        "max_drawdown": round(dd, 4),
        "total_return": round(float(eq.iloc[-1] / eq.iloc[0] - 1.0), 4),
        "excess_return": None,
        "hit_rate": round(float((rets > 0).mean()), 4) if len(rets) else None,
        "n_fills": n_fills,
        "turnover": turnover,
        "n_days": int(len(eq)),
    }
    if benchmark is not None:
        bm = pd.Series(benchmark).dropna().astype(float)
        idx = eq.index.intersection(bm.index)
        if len(idx) > 1:
            out["excess_return"] = round(
                float(eq.loc[idx].iloc[-1] / eq.loc[idx].iloc[0]
                      - bm.loc[idx].iloc[-1] / bm.loc[idx].iloc[0]), 4
            )
    return out


# --------------------------------------------------------------------------- #
# 1.3 gate evaluation
# --------------------------------------------------------------------------- #
@dataclass
class GateThresholds:
    """Hard/soft thresholds, loaded from ``configs/forward_policy.yaml``."""

    tracking_error_daily_pp_max: float = 0.2
    tracking_error_sign_bias_p_min: float = 0.05
    tracking_error_min_days: int = 5
    cost_deviation_max: float = 0.20
    price_integrity_bps_max: float = 2.0
    violations_max: int = 0
    availability_min: float = 0.99
    data_freshness_days_max: int = 1
    symbol_minute_coverage_min: float = 0.95
    effective_universe_min: float = 0.95
    soft_record_only: tuple[str, ...] = ("sharpe", "max_drawdown", "excess_return",
                                         "hit_rate", "n_fills", "turnover")

    @classmethod
    def from_config(cls, cfg) -> "GateThresholds":
        """Load the hard/soft thresholds, REFUSING unknown keys.

        A misspelled key used to be filtered away silently, so
        ``tracking_eror_daily_pp_max`` produced a gate that quietly ran on the
        default 0.2 while the config looked like it had set something else. For a
        gate whose whole purpose is "no undeclared threshold", an unrecognised
        key must stop the run.
        """
        hard = dict(cfg.get("forward.risk_gate.hard", {}) or {})
        soft = cfg.get("forward.risk_gate.soft.record_only", None)
        known = set(cls.__dataclass_fields__) - {"soft_record_only"}
        unknown = sorted(set(hard) - known)
        if unknown:
            raise ValueError(
                f"unknown forward gate threshold key(s): {unknown} — "
                f"known keys are {sorted(known)} (a typo must not silently fall back "
                "to the built-in default)"
            )
        kwargs = {k: v for k, v in hard.items() if k in known}
        if soft:
            kwargs["soft_record_only"] = tuple(str(x) for x in soft)
        return cls(**kwargs)


def evaluate_gate(metrics: Mapping[str, Any], thresholds: Optional[GateThresholds] = None) -> dict:
    """Turn a metric bundle into ``{hard: {...}, soft: {...}, verdict, failed}``.

    ``verdict`` is ``pass`` only when EVERY hard check passes. A missing metric
    fails its check (an unmeasured gate is not a passed gate) — the whole point
    is that "we could not measure it" must never read as "it is fine".
    """
    th = thresholds or GateThresholds()
    te = dict(metrics.get("tracking_error") or {})
    cost = dict(metrics.get("cost") or {})
    viol = dict(metrics.get("violations") or {})
    avail = dict(metrics.get("availability") or {})
    fresh = dict(metrics.get("data_freshness") or {})
    cov = dict(metrics.get("symbol_coverage") or {})
    prereg = dict(metrics.get("prereg") or {})

    te_days = int(te.get("n_days", 0) or 0)
    # A configurable floor must never be able to switch the measurement off:
    # ``tracking_error_min_days: 0`` plus ``--no-replay`` used to yield
    # ``n_days=0, mean_abs_pp=0.0, p=1.0`` → every tracking-error gate passed on
    # EMPTY series, which is precisely the "unmeasured read as fine" failure this
    # gate exists to prevent. One measurable day is now an unconditional floor.
    min_days = max(1, int(th.tracking_error_min_days))
    te_abs = te.get("mean_abs_pp")
    te_p = te.get("sign_bias_p")
    cost_fee = cost.get("fee_deviation_pct")
    cost_integrity = cost.get("price_integrity_abs_bps")
    cost_integrity_max = float(cost.get("price_integrity_bps_max", 2.0) or 2.0)
    viol_total = viol.get("total")
    avail_v = avail.get("availability")
    fresh_lag = fresh.get("lag_days")
    cov_min = cov.get("min_coverage")

    hard = {
        # The evaluation must be BOUND to a frozen pre-registration: an unbound
        # number is produced by whatever the thresholds happened to be, so it is
        # not evidence (see src.forward.prereg.prereg_gate).
        "prereg_binding": {
            "ok": bool(prereg.get("ok")),
            "rule_id": prereg.get("rule_id"),
            "frozen_at": prereg.get("frozen_at"),
            "window_match": prereg.get("window_match"),
            "frozen_before_window": prereg.get("frozen_before_window"),
            "policy_sha256_match": prereg.get("policy_sha256_match"),
            "code_commit_match": prereg.get("code_commit_match"),
            "waived": bool(prereg.get("waived")),
            "waiver_reason": prereg.get("waiver_reason"),
            "issues": list(prereg.get("issues") or []),
        },
        "tracking_error_daily_pp": {
            "value": te_abs, "max": th.tracking_error_daily_pp_max,
            "n_days": te_days, "min_days": min_days,
            "n_excluded": te.get("n_excluded"),
            "ok": bool(te_days >= min_days and te_abs is not None
                       and te_abs <= th.tracking_error_daily_pp_max),
        },
        "tracking_error_sign_bias": {
            "value": te_p, "min_p": th.tracking_error_sign_bias_p_min,
            "n_pos": te.get("n_pos"), "n_neg": te.get("n_neg"),
            "ok": bool(te_days >= min_days and te_p is not None
                       and te_p >= th.tracking_error_sign_bias_p_min),
        },
        "cost_fee_deviation": {
            "value_pct": cost_fee, "max_pct": th.cost_deviation_max * 100.0,
            "ok": bool(cost_fee is not None and abs(cost_fee) <= th.cost_deviation_max * 100.0),
        },
        "cost_price_integrity": {
            "mean_abs_bps": cost_integrity, "max_abs_bps": cost.get("price_integrity_max_bps"),
            "limit_bps": cost_integrity_max,
            "n_price_checked": cost.get("n_price_checked"),
            "n_price_skipped": cost.get("n_price_skipped"),
            "unmeasured": list(cost.get("unmeasured") or []),
            # an unchecked price is not a verified price: no reference prices ⇒ fail
            "ok": bool(cost_integrity is not None and cost_integrity <= cost_integrity_max
                       and (cost.get("n_price_checked") or 0) > 0),
        },
        "violations": {
            "value": viol_total, "max": th.violations_max,
            "unmeasured": list(viol.get("unmeasured") or []),
            "ok": bool(viol_total is not None and viol_total <= th.violations_max
                       and not viol.get("unmeasured")),
        },
        "availability": {
            "value": avail_v, "min": th.availability_min,
            "unmeasured": bool(avail.get("unmeasured")),
            "ok": bool(avail_v is not None and not avail.get("unmeasured")
                       and avail_v >= th.availability_min),
        },
        "data_freshness": {
            "value_days": fresh_lag, "max_days": th.data_freshness_days_max,
            "ok": bool(fresh_lag is not None and fresh_lag <= th.data_freshness_days_max),
        },
        "symbol_minute_coverage": {
            "value": cov_min, "min": th.symbol_minute_coverage_min,
            "ok": bool(cov_min is not None and cov_min >= th.symbol_minute_coverage_min),
        },
        "effective_universe": {
            "value": (metrics.get("universe") or {}).get("effective_ratio"),
            "n_columns": (metrics.get("universe") or {}).get("n_columns"),
            "n_with_price": (metrics.get("universe") or {}).get("n_with_price"),
            "n_warm_20": (metrics.get("universe") or {}).get("n_warm_20"),
            "n_warm_60": (metrics.get("universe") or {}).get("n_warm_60"),
            "min": th.effective_universe_min,
            # a book that silently runs on a shrunken cross-section is not the
            # universe its pre-registration claims — measured, not assumed
            "ok": bool(((metrics.get("universe") or {}).get("effective_ratio") is not None)
                       and float((metrics.get("universe") or {})["effective_ratio"])
                       >= th.effective_universe_min),
        },
    }
    # A hard gate may be expressed either as a mapping (``{"ok": ...}``) or as a
    # plain bool. Filtering on ``isinstance(v, Mapping)`` alone silently DROPPED
    # every bool gate — so a future "this must be true" check would never fail.
    def _failed(v: Any) -> bool:
        if isinstance(v, Mapping):
            return not v.get("ok", False)
        if isinstance(v, bool):
            return not v
        return True  # an unrecognisable gate entry is a failure, not a pass

    failed = [k for k, v in hard.items() if _failed(v)]
    soft = dict(metrics.get("soft") or {})
    return {
        "hard": hard,
        "soft": {"record_only": list(th.soft_record_only), "values": soft,
                 "note": "no forward-window power: never gates the verdict"},
        "failed": failed,
        "verdict": "pass" if not failed else "fail",
        "thresholds": {
            "tracking_error_daily_pp_max": th.tracking_error_daily_pp_max,
            "tracking_error_sign_bias_p_min": th.tracking_error_sign_bias_p_min,
            "tracking_error_min_days": min_days,
            "cost_deviation_max": th.cost_deviation_max,
            "price_integrity_bps_max": th.price_integrity_bps_max,
            "violations_max": th.violations_max,
            "availability_min": th.availability_min,
            "data_freshness_days_max": th.data_freshness_days_max,
            "symbol_minute_coverage_min": th.symbol_minute_coverage_min,
            "effective_universe_min": th.effective_universe_min,
        },
    }


# --------------------------------------------------------------------------- #
# 2 paired candidate comparison
# --------------------------------------------------------------------------- #
def paired_comparison(
    incumbent: pd.Series,
    candidate: pd.Series,
    *,
    window_days: int = 120,
    t_min: float = 1.5,
    diff_gt: float = 0.0,
) -> dict:
    """Paired (same-day) comparison of two books — the only valid 2-book test.

    Two stop-width variants share ~84% of their daily returns, so an unpaired
    comparison of two 120-day Sharpes is dominated by the common factor. The
    paired difference series removes it: the question becomes "on the days they
    differ, which is better, and is the average difference distinguishable from
    zero". The switch rule is pre-registered: ``mean(diff) > diff_gt`` AND
    ``t > t_min`` over ``window_days``; otherwise HOLD — and "never separates" is
    an acceptable outcome, not a reason to keep looking.
    """
    a = pd.Series(incumbent).dropna().astype(float)
    b = pd.Series(candidate).dropna().astype(float)
    idx = a.index.intersection(b.index)
    if len(idx) > window_days:
        idx = idx[-window_days:]
    if len(idx) < 2:
        # same key set as the full result: a consumer must not need a special
        # case for "not enough days yet" (that is the normal state on day 1)
        return {"n_days": int(len(idx)), "ready": False, "reason": "insufficient paired days",
                "switch": False, "t_stat": None, "mean_diff_pp": None, "corr": None,
                "cum_diff_pp": None, "annualized_diff_pp": None, "sd_diff_pp": None,
                "hit_rate": None, "days_needed_for_t": None, "verdict": "hold",
                "window_days": window_days, "t_min": t_min, "diff_gt": diff_gt,
                "note": "hold is a legitimate permanent outcome — the two books may never separate"}
    ra, rb = a.loc[idx], b.loc[idx]
    diff = rb - ra
    n = len(diff)
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    t_stat = float(diff.mean() / (sd / math.sqrt(n))) if sd > 0 else 0.0
    corr = float(ra.corr(rb)) if ra.std() > 0 and rb.std() > 0 else None
    switch = bool(diff.mean() > diff_gt and t_stat > t_min)
    # how much more data would be needed at the observed effect size for t=t_min
    need = None
    if sd > 0 and abs(diff.mean()) > 0:
        need = int(math.ceil((t_min * sd / abs(diff.mean())) ** 2))
    return {
        "n_days": n,
        "ready": bool(n >= window_days),
        "window_days": window_days,
        "t_min": t_min,
        "diff_gt": diff_gt,
        "mean_diff_pp": round(float(diff.mean()) * 100, 4),
        "cum_diff_pp": round(float((1.0 + diff).prod() - 1.0) * 100, 4),
        "annualized_diff_pp": round(float(diff.mean()) * TRADING_DAYS * 100, 2),
        "sd_diff_pp": round(sd * 100, 4),
        "t_stat": round(t_stat, 3),
        "corr": round(corr, 4) if corr is not None else None,
        "hit_rate": round(float((diff > 0).mean()), 4),
        "days_needed_for_t": need,
        "switch": switch,
        "verdict": "switch" if switch else "hold",
        "note": "hold is a legitimate permanent outcome — the two books may never separate",
    }


__all__ = [
    "DECISION_WINDOWS",
    "GateThresholds",
    "PRICE_CHECKABLE_SOURCES",
    "TRADING_DAYS",
    "availability",
    "binom_two_sided_p",
    "cost_deviation",
    "data_freshness",
    "evaluate_gate",
    "fill_violations",
    "paired_comparison",
    "panel_universe_health",
    "power_table",
    "sharpe_standard_error",
    "soft_metrics",
    "symbol_minute_coverage",
    "tracking_error",
    "years_for_t",
]
