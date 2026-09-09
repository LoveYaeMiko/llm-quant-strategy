#!/usr/bin/env python
"""Survivorship-bias stress test with a blocking verdict for the alpha-layer loop.

Design question (2026-09-09): before enabling any self-evolution loop over the
alpha layer, prove that the survivorship bias in the data foundation is smaller
than ``threshold_ratio × target_alpha`` (default ``0.3 × 8pp = 2.4pp/yr``). If the
bias is larger, an evolution loop would learn the bias as if it were alpha.

What this script measures, all four required legs:

1. **Missing-name quantification** — universe symbols with ZERO price bars, how
   many sit inside the D-track shadow universe (800 names) and inside the stored
   ``universe`` snapshots, their ``valid_from`` span and names.
2. **Break-even drag model** — annual drag ``entries_per_year · x · (1/k) · L``
   over ``x ∈ {0.001 … 0.05}`` and ``L ∈ {-0.30,-0.50,-0.80}``, plus the
   break-even ``x`` at the threshold and at the full target alpha.
3. **Empirical hazard bound** — annual disappearance rate of names from the
   ``universe`` snapshots (full stored universe and the D universe), turned into
   ``x_upper_bound``; plus the ST / low-price / low-liquidity proxies that can be
   computed from price data.
4. **De-biased-subset run** — the D book run twice with the PRODUCTION assembly
   (``src.cli._shadow_cycle``) into fresh scratch ledgers: baseline (full 800-name
   shadow universe) vs de-biased (universe minus the objectively riskiest
   segment). One market is built and reused through ``market_override``.

Isolation follows ``scripts/d_oos.py`` exactly: fresh scratch ledger per run under
``outputs/_bias_scratch/``, ``write_artifacts=False``, and a production-config
fingerprint probe to assert ``params_match_production``. The production ledger
``outputs/shadow_ledger_D_5W.sqlite`` is opened READ-ONLY (``mode=ro``) and never
written; no production status/report artifact is touched.

Usage::

    python scripts/bias_stress_test.py --label stress_v1
    python scripts/bias_stress_test.py --label stress_v1 --skip-runs
    python scripts/bias_stress_test.py --label stress_v1 --target-alpha 0.08 \
        --window-start 2025-09-01 --window-end 2025-12-31
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: F401 — pandas parquet/DB paths expect numpy present
import pandas as pd

# --------------------------------------------------------------------------- #
# constants — the design question's fixed parameters
# --------------------------------------------------------------------------- #
#: target alpha of the D book, annualised, as a fraction of equity
TARGET_ALPHA_DEFAULT = 0.08
#: bias must be below this fraction of the target alpha to allow self-evolution
THRESHOLD_RATIO = 0.3
#: D book shape (configs/master_config.yaml → shadow.accounts[D_5W])
K_SLOTS = 6
MAX_HOLD_DAYS = 40
TRADING_DAYS = 252
#: per-entry probability of landing on a doomed name
X_GRID: tuple[float, ...] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
#: loss given the name is doomed (fraction of the position's value)
LOSS_GRID: tuple[float, ...] = (-0.30, -0.50, -0.80)
LOSS_MID = -0.50
#: de-bias rule thresholds
PRICE_FLOOR_CNY = 3.0
AMOUNT_DECILE = 0.10

SCRATCH_DIR = ROOT / "outputs" / "_bias_scratch"
PROD_LEDGER = ROOT / "outputs" / "shadow_ledger_D_5W.sqlite"
D_OOS_REFERENCE = ROOT / "outputs" / "d_oos_oos_2025h2_v3.json"


def _log(msg: str) -> None:
    print(f"[bias] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# pure helpers (offline-testable: no DB, no network, no filesystem)
# --------------------------------------------------------------------------- #
def trading_year_fraction(days: float, trading_days: int = TRADING_DAYS) -> float:
    """Calendar length of ``days`` trading days, expressed in years."""
    if trading_days <= 0:
        raise ValueError("trading_days must be positive")
    return float(days) / float(trading_days)


def entries_per_year(n_entries: float, n_days: float, trading_days: int = TRADING_DAYS) -> float:
    """Annualise an entry count observed over ``n_days`` trading days."""
    if n_days is None or float(n_days) <= 0:
        raise ValueError("n_days must be positive to annualise entries")
    return float(n_entries) * float(trading_days) / float(n_days)


def annual_drag_frac(
    entries_per_year: float, x: float, k: int = K_SLOTS, loss: float = LOSS_MID
) -> float:
    """Annual drag as a fraction of equity (SIGNED: ``loss`` < 0 → negative).

    ``entries_per_year · x · (1/k) · L``: each entry risks one slot's weight
    (``1/k`` of equity), a fraction ``x`` of entries lands on a doomed name, and
    such a name costs ``L`` of that slot.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    return float(entries_per_year) * float(x) * (1.0 / float(k)) * float(loss)


def annual_drag_pp(
    entries_per_year: float, x: float, k: int = K_SLOTS, loss: float = LOSS_MID
) -> float:
    """``annual_drag_frac`` in percentage points (signed)."""
    return annual_drag_frac(entries_per_year, x, k=k, loss=loss) * 100.0


def drag_grid(
    epy: float,
    k: int = K_SLOTS,
    xs: Sequence[float] = X_GRID,
    losses: Sequence[float] = LOSS_GRID,
) -> dict[str, Any]:
    """Drag table in pp/yr, keyed by loss then by x (both as strings for JSON)."""
    return {
        "x_grid": [float(x) for x in xs],
        "loss_grid": [float(v) for v in losses],
        "drag_pp": {
            f"{float(loss):.2f}": {
                f"{float(x):g}": round(annual_drag_pp(epy, x, k=k, loss=loss), 4) for x in xs
            }
            for loss in losses
        },
    }


def breakeven_x(
    epy: float,
    k: int = K_SLOTS,
    loss: float = LOSS_MID,
    threshold_frac: float = THRESHOLD_RATIO * TARGET_ALPHA_DEFAULT,
) -> Optional[float]:
    """Per-entry doom probability at which the drag magnitude equals ``threshold_frac``.

    Returns ``None`` when the drag cannot reach the threshold (no entries, no
    loss) — an explicit "unmeasured/undefined" instead of an invented number.
    """
    denom = abs(annual_drag_frac(epy, 1.0, k=k, loss=loss))
    if denom <= 0:
        return None
    return float(threshold_frac) / denom


def annual_disappearance_rate(n_present: int, n_absent: int, years: float) -> float:
    """Geometric annual rate at which names leave the stored universe.

    ``n_present`` names existed at snapshot *t*; ``n_absent`` of them are gone
    from the newest snapshot ``years`` later. The surviving fraction
    ``1 - n_absent/n_present`` is compounded backwards to a per-year rate.
    """
    if n_present <= 0:
        raise ValueError("n_present must be positive")
    if years <= 0:
        raise ValueError("years must be positive")
    ratio = min(1.0, max(0.0, float(n_absent) / float(n_present)))
    return 1.0 - (1.0 - ratio) ** (1.0 / float(years))


def rule_of_three_upper_rate(n_present: int, years: float, events: int = 0) -> Optional[float]:
    """95% one-sided upper bound on the annual rate when ZERO events were seen.

    The rule of three: with ``events = 0`` observed over ``n`` independent
    trials, the 95% upper bound on the event probability is ``3/n``. Here the
    trials are name-years (``n_present · years``), so the bound is per year.
    """
    if n_present <= 0 or years <= 0:
        return None
    if events != 0:
        return None
    return 3.0 / (float(n_present) * float(years))


def x_upper_bound_from_hazard(
    h_annual: float,
    hold_days: int = MAX_HOLD_DAYS,
    trading_days: int = TRADING_DAYS,
    pick_prob: float = 1.0,
) -> float:
    """Upper bound on the per-entry doom probability ``x``.

    A name that disappears at rate ``h`` per year disappears inside a holding
    window of ``hold_days`` trading days with probability ``h · hold/252``
    (first-order, ``h`` small). ``pick_prob = 1`` is the worst case: the book may
    pick any name in the pool, including a doomed one.
    """
    return float(h_annual) * trading_year_fraction(hold_days, trading_days) * float(pick_prob)


def snapshot_sets(universe_rows: pd.DataFrame) -> dict[str, set[str]]:
    """``{ISO snapshot date: set(symbol)}`` from universe records (``valid_from``)."""
    if universe_rows is None or universe_rows.empty:
        return {}
    df = universe_rows.copy()
    df["valid_from"] = pd.to_datetime(df["valid_from"])
    out: dict[str, set[str]] = {}
    for ts, grp in df.groupby("valid_from"):
        out[str(pd.Timestamp(ts).date())] = set(grp["symbol"].astype(str))
    return out


def snapshots_covering(
    universe_rows: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> list[str]:
    """Snapshot dates whose validity interval intersects ``[start, end]``.

    A universe snapshot is visible at ``t`` iff ``valid_from <= t < valid_to``
    (``NaT`` = still valid). A snapshot "covers" the window when it is visible at
    some date inside it.
    """
    if universe_rows is None or universe_rows.empty:
        return []
    df = universe_rows.copy()
    df["valid_from"] = pd.to_datetime(df["valid_from"])
    df["valid_to"] = pd.to_datetime(df.get("valid_to"), errors="coerce")
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    out: list[str] = []
    for _, row in df.drop_duplicates(subset=["valid_from"]).iterrows():
        vf = row["valid_from"]
        vt = row["valid_to"]
        end_eff = hi if pd.isna(vt) else min(hi, vt - pd.Timedelta(days=1))
        if vf <= end_eff and (pd.isna(vt) or vt > lo):
            out.append(str(vf.date()))
    return sorted(set(out))


def missing_names(universe_rows: pd.DataFrame, price_symbols: Iterable[str]) -> pd.DataFrame:
    """Universe records whose symbol has ZERO price bars (the survivorship hole)."""
    if universe_rows is None or universe_rows.empty:
        return pd.DataFrame(columns=list(universe_rows.columns) if universe_rows is not None else [])
    have = {str(s) for s in price_symbols}
    df = universe_rows.copy()
    df["symbol"] = df["symbol"].astype(str)
    return df[~df["symbol"].isin(have)].reset_index(drop=True)


def missing_summary(
    universe_rows: pd.DataFrame,
    price_symbols: Iterable[str],
    d_universe: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    name_column: str = "name",
    max_samples: int = 20,
) -> dict[str, Any]:
    """Quantify the zero-price-bar universe symbols (deliverable 1)."""
    have = {str(s) for s in price_symbols}
    uni_syms = set(universe_rows["symbol"].astype(str)) if not universe_rows.empty else set()
    miss = missing_names(universe_rows, have)
    miss_syms = set(miss["symbol"]) if not miss.empty else set()
    d_set = {str(s) for s in d_universe}
    snaps = snapshot_sets(universe_rows)
    covering = snapshots_covering(universe_rows, start, end)
    newest_date = max(snaps) if snaps else None
    newest = snaps.get(newest_date, set()) if newest_date else set()
    oldest_date = min(snaps) if snaps else None
    oldest = snaps.get(oldest_date, set()) if oldest_date else set()

    names: list[dict[str, Any]] = []
    if not miss.empty:
        cols = [c for c in ("symbol", "valid_from", name_column) if c in miss.columns]
        for row in miss.sort_values(["valid_from", "symbol"]).head(max_samples)[cols].itertuples(index=False):
            rec = dict(zip(cols, row))
            rec["valid_from"] = str(pd.Timestamp(rec["valid_from"]).date())
            names.append(rec)

    return {
        "n_universe_symbols": int(len(uni_syms)),
        "n_universe_with_price_bars": int(len(uni_syms & have)),
        "n_missing_zero_price_bars": int(len(miss_syms)),
        "share_of_universe_missing": round(len(miss_syms) / len(uni_syms), 4) if uni_syms else None,
        "n_missing_in_d_universe": int(len(miss_syms & d_set)),
        "d_universe_size": int(len(d_set)),
        "n_missing_in_window_snapshots": int(
            len(miss_syms & set().union(*[snaps[d] for d in covering]) if covering else set())
        ),
        "window_snapshot_dates": covering,
        "snapshot_dates": sorted(snaps),
        "snapshot_sizes": {d: len(v) for d, v in sorted(snaps.items())},
        "n_missing_present_in_newest_snapshot": int(len(miss_syms & newest)),
        "n_missing_absent_from_newest_snapshot": int(len(miss_syms - newest)),
        "newest_snapshot": newest_date,
        "oldest_snapshot": oldest_date,
        "earliest_missing_valid_from": (
            str(pd.to_datetime(miss["valid_from"]).min().date()) if not miss.empty else None
        ),
        "latest_missing_valid_from": (
            str(pd.to_datetime(miss["valid_from"]).max().date()) if not miss.empty else None
        ),
        "missing_name_samples": names,
        "n_missing_name_samples_with_delist_tag": int(
            sum(
                1
                for n in miss.get(name_column, pd.Series(dtype=str)).astype(str)
                if ("退" in n) or ("ST" in n.upper())
            )
        ),
        "missing_classification": {
            "delisted_like": (
                "present in the oldest snapshot but ABSENT from the newest snapshot — these are "
                "the names that left the market and whose entire price history is missing"
            ),
            "new_listing_like": (
                "present in the newest snapshot but with no price bars at all — recently listed "
                "names whose bars were never ingested (NOT a survivorship hole; if anything they "
                "bias the opposite way by being absent from historical cross-sections)"
            ),
        },
        "missing_symbols": sorted(miss_syms),
    }


def hazard_summary(
    universe_rows: pd.DataFrame,
    d_universe: Iterable[str],
    *,
    n_d_universe_in_oldest: Optional[int] = None,
) -> dict[str, Any]:
    """Annual disappearance rate from the oldest to the newest snapshot.

    (a) the full stored universe; (b) the D (800-name) universe — names that
    appear in the oldest snapshot AND in the D universe, checked for absence in
    the newest snapshot.
    """
    snaps = snapshot_sets(universe_rows)
    dates = sorted(snaps)
    if len(dates) < 2:
        return {"measured": False, "reason": "fewer than two universe snapshots"}
    oldest, newest = dates[0], dates[-1]
    years = (pd.Timestamp(newest) - pd.Timestamp(oldest)).days / 365.25
    past, present = snaps[oldest], snaps[newest]
    d_set = {str(s) for s in d_universe}

    gone_full = past - present
    d_past = past & d_set
    gone_d = d_past - present
    n_d_past = len(d_past) if n_d_universe_in_oldest is None else int(n_d_universe_in_oldest)

    out = {
        "measured": True,
        "oldest_snapshot": oldest,
        "newest_snapshot": newest,
        "years": round(float(years), 4),
        "full_universe": {
            "n_present_at_oldest": int(len(past)),
            "n_absent_from_newest": int(len(gone_full)),
            "annual_disappearance_rate": round(
                annual_disappearance_rate(len(past), len(gone_full), years), 6
            ),
            "share_gone_over_period": round(len(gone_full) / len(past), 4) if past else None,
        },
        "d_universe": {
            "n_present_at_oldest_and_in_d": int(n_d_past),
            "n_absent_from_newest": int(len(gone_d)),
            "annual_disappearance_rate": round(
                annual_disappearance_rate(n_d_past, len(gone_d), years), 6
            )
            if n_d_past > 0
            else None,
            "rule_of_three_upper_rate": (
                None
                if n_d_past <= 0
                else _round_or_none(rule_of_three_upper_rate(n_d_past, years, events=len(gone_d)))
            ),
        },
    }
    # names that left the stored universe — useful context for the doc
    gone_names = []
    if "name" in universe_rows.columns:
        name_map = dict(
            zip(universe_rows["symbol"].astype(str), universe_rows.get("name", "").astype(str))
        )
        gone_names = [{"symbol": s, "name": name_map.get(s, "")} for s in sorted(gone_full)[:10]]
    out["gone_name_samples"] = gone_names
    return out


def risk_segment(
    latest_raw_close: pd.Series,
    median_amount: pd.Series,
    *,
    price_floor: float = PRICE_FLOOR_CNY,
    decile: float = AMOUNT_DECILE,
) -> dict[str, Any]:
    """Deterministic exclusion set: penny names OR bottom-decile liquidity.

    * ``latest_raw_close < price_floor`` (unadjusted close on the last bar at or
      before the window end) → excluded;
    * median daily ``amount`` over the window ``<=`` the universe's
      ``decile``-quantile (linear interpolation) → excluded; a symbol with NO
      window amount is treated as the least liquid and excluded.
    """
    idx = pd.Index(sorted(set(map(str, latest_raw_close.index)) | set(map(str, median_amount.index))))
    px = pd.to_numeric(latest_raw_close, errors="coerce").reindex(idx)
    amt = pd.to_numeric(median_amount, errors="coerce").reindex(idx)
    low_price = sorted(px.index[px.notna() & (px < float(price_floor))].astype(str))
    valid = amt.dropna()
    threshold = float(valid.quantile(float(decile))) if len(valid) else None
    low_amount = sorted(
        amt.index[(amt.isna() | (amt <= threshold if threshold is not None else False))].astype(str)
    )
    excluded = sorted(set(low_price) | set(low_amount))
    n = int(len(idx))
    return {
        "n_symbols": n,
        "price_floor_cny": float(price_floor),
        "amount_decile": float(decile),
        "amount_decile_threshold_cny": None if threshold is None else round(threshold, 2),
        "n_low_price": len(low_price),
        "n_low_amount": len(low_amount),
        "n_excluded": len(excluded),
        "share_excluded": round(len(excluded) / n, 4) if n else None,
        "n_missing_amount_data": int(amt.isna().sum()),
        "low_price_symbols": low_price,
        "low_amount_symbols": low_amount,
        "excluded_symbols": excluded,
        "share_low_price": round(len(low_price) / n, 4) if n else None,
        "share_bottom_decile_amount": round(len(low_amount) / n, 4) if n else None,
    }


def metrics_block(equity: dict[str, Any], fills: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """Normalise the shadow ``status['equity']`` block (+ entry count from fills)."""
    eq = dict(equity or {})
    n_entries = None
    n_buys = n_sells = None
    if fills is not None and len(fills):
        shares = pd.to_numeric(fills["shares"], errors="coerce").fillna(0.0)
        n_buys = int((shares > 0).sum())
        n_sells = int((shares < 0).sum())
        n_entries = n_buys
    return {
        "cumulative_return": _f(eq.get("total_return")),
        "annualized_return": _f(eq.get("annualized_return")),
        "sharpe": _f(eq.get("sharpe")),
        "max_drawdown": _f(eq.get("max_drawdown")),
        "n_days": _i(eq.get("n_days")),
        "n_fills": _i(eq.get("n_fills")),
        "n_entries": n_entries,
        "n_exits": n_sells,
        "total_commission": _f(eq.get("total_commission")),
        "final_equity": _f(eq.get("latest")),
        "fills_by_source": dict(eq.get("fills_by_source") or {}),
    }


def metric_delta(base: dict[str, Any], debiased: dict[str, Any]) -> dict[str, Any]:
    """De-biased minus baseline (pp for returns/drawdown, raw for Sharpe/counts)."""
    def _d(key: str, scale: float) -> Optional[float]:
        a, b = base.get(key), debiased.get(key)
        if a is None or b is None:
            return None
        return round((float(b) - float(a)) * scale, 6)

    def _i(key: str) -> Optional[int]:
        a, b = base.get(key), debiased.get(key)
        if a is None or b is None:
            return None
        return int(b) - int(a)

    return {
        "annualized_return_pp": _d("annualized_return", 100.0),
        "cumulative_return_pp": _d("cumulative_return", 100.0),
        "sharpe": _d("sharpe", 1.0),
        "max_drawdown_pp": _d("max_drawdown", 100.0),
        "n_fills": _i("n_fills"),
        "n_entries": _i("n_entries"),
    }


def verdict_block(
    *,
    target_alpha: float,
    threshold_ratio: float,
    x_upper_bound: float,
    epy: float,
    k: int = K_SLOTS,
    loss_mid: float = LOSS_MID,
    h_annual: Optional[float] = None,
    breakeven: Optional[float] = None,
    breakeven_h: Optional[float] = None,
) -> dict[str, Any]:
    """The exact verdict keys the evolution-loop gate consumes.

    ``drag_at_upper_bound_pp`` is the annual return LOSS in percentage points
    (positive magnitude) so the documented test ``> threshold_pp`` reads
    directly.
    """
    threshold_pp = float(threshold_ratio) * float(target_alpha) * 100.0
    drag_pp = abs(annual_drag_pp(epy, x_upper_bound, k=k, loss=loss_mid))
    ratio = drag_pp / (float(target_alpha) * 100.0)
    blocking = bool(drag_pp > threshold_pp)
    return {
        "target_alpha_annual": float(target_alpha),
        "threshold_ratio": float(threshold_ratio),
        "threshold_pp": round(threshold_pp, 4),
        "x_upper_bound": float(x_upper_bound),
        "drag_at_upper_bound_pp": round(drag_pp, 4),
        "bias_vs_alpha_ratio": round(ratio, 4),
        "bias_blocking_evolution": blocking,
        "verdict_text": verdict_sentence(
            blocking=blocking,
            drag_pp=drag_pp,
            threshold_pp=threshold_pp,
            ratio=ratio,
            x_upper_bound=x_upper_bound,
            h_annual=h_annual,
            epy=epy,
            breakeven=breakeven,
            breakeven_h=breakeven_h,
        ),
    }


def verdict_sentence(
    *,
    blocking: bool,
    drag_pp: float,
    threshold_pp: float,
    ratio: float,
    x_upper_bound: float,
    h_annual: Optional[float],
    epy: float,
    breakeven: Optional[float],
    breakeven_h: Optional[float] = None,
) -> str:
    """One sentence stating the gate decision and the numbers behind it."""
    h_txt = f"{h_annual:.2%}/yr" if h_annual is not None else "n/a"
    be_txt = f"{breakeven:.4%}" if breakeven is not None else "n/a"
    head = "BLOCKING" if blocking else "NOT BLOCKING"
    margin = ""
    if h_annual is not None and breakeven_h:
        margin = (
            f" The measured hazard is {h_annual / breakeven_h:.0%} of the break-even hazard "
            f"{breakeven_h:.2%}/yr."
        )
    return (
        f"{head}: at the measured annual name-disappearance rate {h_txt} the per-entry "
        f"doom probability bound x<={x_upper_bound:.4%} implies a {drag_pp:.2f}pp/yr drag on the "
        f"D book ({epy:.0f} entries/yr, k={K_SLOTS} slots, L={LOSS_MID:.0%}), which is "
        f"{ratio:.2f}x the target alpha and {'above' if blocking else 'below'} the "
        f"{threshold_pp:.1f}pp threshold (break-even x={be_txt}).{margin}"
    )


def _round_or_none(value: Optional[float], digits: int = 6) -> Optional[float]:
    """Round, preserving ``None`` (never turn an unmeasured value into 0.0)."""
    return None if value is None else round(float(value), digits)


def _f(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        return None if (math.isnan(out) or math.isinf(out)) else out
    except (TypeError, ValueError):
        return None


def _i(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# PIT database access
# --------------------------------------------------------------------------- #
def _pit_url(cfg) -> str:
    url = str(cfg.get("data.pit_database_url") or "").strip()
    if not url:
        raise SystemExit("data.pit_database_url is empty — cannot measure the data foundation")
    return url


def _connect(url: str):
    if url.startswith("postgresql"):
        import psycopg2

        return psycopg2.connect(url)
    if url.startswith("sqlite:///"):
        return sqlite3.connect(url[len("sqlite:///"):])
    return sqlite3.connect(url)


def _read_sql(sql: str, con, params: Any = None) -> pd.DataFrame:
    """``pd.read_sql_query`` without the noisy DBAPI2 connection warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*SQLAlchemy connectable.*")
        if params is None:
            return pd.read_sql_query(sql, con)
        return pd.read_sql_query(sql, con, params=params)


def load_universe_rows(url: str) -> pd.DataFrame:
    """All ``universe`` records: symbol, valid_from, valid_to, name."""
    con = _connect(url)
    try:
        if url.startswith("postgresql"):
            df = _read_sql(
                "SELECT symbol, valid_from, valid_to, payload->>'name' AS name "
                "FROM pit_records WHERE record_type='universe' ORDER BY valid_from, symbol",
                con,
            )
        else:
            df = _read_sql(
                "SELECT symbol, valid_from, valid_to, json_extract(payload,'$.name') AS name "
                "FROM pit_records WHERE record_type='universe' ORDER BY valid_from, symbol",
                con,
            )
    finally:
        con.close()
    df["valid_from"] = pd.to_datetime(df["valid_from"])
    df["valid_to"] = pd.to_datetime(df["valid_to"], errors="coerce")
    return df


def load_price_symbols(url: str, universe_symbols: Sequence[str]) -> list[str]:
    """Universe symbols that DO have at least one ``price`` bar.

    Uses an indexed anti-join (the PK is ``(symbol, valid_from, record_type)``):
    a full ``SELECT DISTINCT symbol FROM pit_records`` costs ~96s on the 12.5M-row
    price table, while this probe answers in ~1s and carries the same fact.
    """
    syms = [str(s) for s in universe_symbols]
    if not syms:
        return []
    con = _connect(url)
    try:
        if url.startswith("postgresql"):
            df = _read_sql(
                "SELECT DISTINCT u.symbol FROM (SELECT DISTINCT symbol FROM pit_records "
                "WHERE record_type='universe') u WHERE EXISTS (SELECT 1 FROM pit_records p "
                "WHERE p.record_type='price' AND p.symbol=u.symbol)",
                con,
            )
        else:
            df = _read_sql(
                "SELECT DISTINCT u.symbol FROM (SELECT DISTINCT symbol FROM pit_records "
                "WHERE record_type='universe') u WHERE EXISTS (SELECT 1 FROM pit_records p "
                "WHERE p.record_type='price' AND p.symbol=u.symbol)",
                con,
            )
    finally:
        con.close()
    have = set(df["symbol"].astype(str))
    return sorted(have & set(syms))


def load_window_price_stats(
    url: str,
    symbols: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Per-symbol price facts for the de-bias rule.

    Columns: ``median_amount`` (median daily turnover inside the window),
    ``n_bars_window``, ``last_raw_close`` / ``last_close`` / ``last_bar`` (the
    last bar at or before ``end``). The window slice is fetched once (≈420k rows
    for a 4-month window) and aggregated in pandas — per-symbol
    ``percentile_cont`` over 800 index probes took minutes on the 12.5M-row table.
    """
    syms = {str(s) for s in symbols}
    empty = pd.DataFrame(
        columns=["median_amount", "n_bars_window", "last_raw_close", "last_close", "last_bar"]
    )
    if not syms:
        return empty
    con = _connect(url)
    try:
        if url.startswith("postgresql"):
            sql = (
                "SELECT symbol, valid_from, (payload->>'amount')::float AS amount, "
                "(payload->>'raw_close')::float AS raw_close, (payload->>'close')::float AS close "
                "FROM pit_records WHERE record_type='price' "
                "AND valid_from >= %s AND valid_from <= %s"
            )
            raw = _read_sql(sql, con, params=(start, end))
        else:
            sql = (
                "SELECT symbol, valid_from, CAST(json_extract(payload,'$.amount') AS REAL) AS amount, "
                "CAST(json_extract(payload,'$.raw_close') AS REAL) AS raw_close, "
                "CAST(json_extract(payload,'$.close') AS REAL) AS close "
                "FROM pit_records WHERE record_type='price' "
                "AND valid_from >= ? AND valid_from <= ?"
            )
            raw = _read_sql(sql, con, params=(start, end))
    finally:
        con.close()
    if raw.empty:
        return empty
    raw["symbol"] = raw["symbol"].astype(str)
    raw = raw[raw["symbol"].isin(syms)].copy()
    if raw.empty:
        return empty
    raw["valid_from"] = pd.to_datetime(raw["valid_from"])
    raw["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    raw = raw.sort_values(["symbol", "valid_from"])
    med = raw.groupby("symbol")["amount"].median().rename("median_amount")
    n = raw.groupby("symbol")["symbol"].size().rename("n_bars_window")
    last = raw.groupby("symbol").tail(1).set_index("symbol")[
        ["raw_close", "close", "valid_from"]
    ].rename(columns={"raw_close": "last_raw_close", "close": "last_close",
                      "valid_from": "last_bar"})
    return med.to_frame().join(n).join(last, how="outer")


def read_production_ledger_stats(path: Path, start: Optional[str] = None,
                                 end: Optional[str] = None) -> dict[str, Any]:
    """READ-ONLY entry counts from the production ledger (never written)."""
    if not path.is_file():
        return {"available": False, "reason": f"{path.name} not found"}
    uri = f"file:{path.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        fills = pd.read_sql_query("SELECT date, shares FROM fills", con)
        days = pd.read_sql_query("SELECT date FROM daily_state", con)
    except Exception as exc:  # noqa: BLE001 — a schema drift must not crash the audit
        con.close()
        return {"available": False, "reason": f"cannot read ledger: {exc}"}
    con.close()
    fills["date"] = pd.to_datetime(fills["date"], errors="coerce")
    days["date"] = pd.to_datetime(days["date"], errors="coerce")
    window = fills
    window_days = days
    if start is not None:
        window = window[window["date"] >= pd.Timestamp(start)]
        window_days = window_days[window_days["date"] >= pd.Timestamp(start)]
    if end is not None:
        window = window[window["date"] <= pd.Timestamp(end)]
        window_days = window_days[window_days["date"] <= pd.Timestamp(end)]
    n_entries = int((pd.to_numeric(window["shares"], errors="coerce") > 0).sum())
    n_days = int(window_days["date"].nunique())
    return {
        "available": True,
        "path": str(path),
        "read_only": True,
        "window_start": start,
        "window_end": end,
        "n_entries_in_window": n_entries,
        "n_days_in_window": n_days,
        "n_fills_in_window": int(len(window)),
        "ledger_span": [
            str(fills["date"].min().date()) if len(fills) else None,
            str(fills["date"].max().date()) if len(fills) else None,
        ],
        "ledger_n_days": int(days["date"].nunique()),
        "entries_per_year": (
            round(entries_per_year(n_entries, n_days), 2) if n_days > 0 else None
        ),
    }


def read_d_oos_reference(path: Path) -> dict[str, Any]:
    """Fill counts from the citable OOS artifact (fallback entry estimate)."""
    if not path.is_file():
        return {"available": False, "reason": f"{path.name} not found"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return {"available": False, "reason": f"unreadable: {exc}"}
    return {
        "available": True,
        "path": str(path),
        "n_days": data.get("n_days"),
        "n_fills": data.get("n_fills"),
        "n_intraday_fills": data.get("n_intraday_fills"),
        "n_close_fills": data.get("n_close_fills"),
        "citable": data.get("citable"),
        "metrics": data.get("metrics", {}),
    }


# --------------------------------------------------------------------------- #
# the two production-assembly runs
# --------------------------------------------------------------------------- #
def run_book(
    cfg,
    market,
    symbols: list[str],
    account: dict,
    control,
    start: str,
    end: str,
    seed: int,
    ledger_path: Path,
    label: str,
) -> dict[str, Any]:
    """One isolated production-assembly run into a FRESH scratch ledger."""
    from src.cli import _shadow_cycle
    from src.paper.ledger import PaperLedger

    ledger_existed_before = ledger_path.exists()
    ledger_path.unlink(missing_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    probe: dict = {}
    t0 = time.time()
    status, ledger_out = _shadow_cycle(
        cfg, symbols, start, end, seed, skip_refresh=True,
        control_scale=control.gross_scale, account=account,
        ledger_override=str(ledger_path), write_artifacts=False,
        probe=probe, market_override=market,
    )
    elapsed = time.time() - t0

    ledger = PaperLedger(str(ledger_path))
    try:
        fills = ledger.fills()
        equity_curve = ledger.curve() if hasattr(ledger, "curve") else ledger.equity_curve()
        ledger_cost = float(ledger.total_commission())
    finally:
        ledger.close()

    metrics = metrics_block(status.get("equity", {}) or {}, fills)
    n_days = int(len(equity_curve))
    cost_from_metrics = float((status.get("equity", {}) or {}).get("total_commission", 0.0))
    cost_sum = float(pd.to_numeric(fills.get("commission", pd.Series(dtype=float)),
                                   errors="coerce").fillna(0.0).sum()) if len(fills) else 0.0
    checks = {
        "ledger_was_fresh": bool(not ledger_existed_before),
        "no_fill_after_window_end": bool(
            not len(fills) or pd.to_datetime(fills["date"]).max() <= pd.Timestamp(end)
        ),
        "cost_model_consistent": bool(
            abs(cost_sum - cost_from_metrics) < 0.01 and abs(cost_sum - ledger_cost) < 0.01
        ),
    }
    _log(
        f"run {label}: n={len(symbols)} days={n_days} fills={metrics['n_fills']} "
        f"entries={metrics['n_entries']} cum={metrics['cumulative_return']:+.2%} "
        f"ann={metrics['annualized_return']:+.2%} sharpe={metrics['sharpe']:+.2f} "
        f"maxDD={metrics['max_drawdown']:.2%} in {elapsed:.0f}s"
    )
    return {
        "label": label,
        "universe_size": len(symbols),
        "ledger": str(ledger_path),
        "ledger_existed_before": bool(ledger_existed_before),
        "n_days": n_days,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": metrics,
        "checks": checks,
        "fingerprint": probe,
    }


def production_fingerprint(cfg, market, symbols: list[str], account: dict, control) -> dict:
    """Fingerprint of the CONFIG account (no overrides) — the production reference."""
    from src.cli import _build_account_portfolio, _book_fingerprint
    from src.paper.ledger import PaperLedger
    from src.paper.shadow import paper_runner_kwargs

    probe_ledger_path = SCRATCH_DIR / "_prod_probe.sqlite"
    probe_ledger_path.unlink(missing_ok=True)
    ledger = PaperLedger(str(probe_ledger_path))
    try:
        book, _ = _build_account_portfolio(
            cfg, market, symbols, account, control.gross_scale, ledger=ledger
        )
        kwargs = paper_runner_kwargs(cfg)
        kwargs.update({
            "cash": float(account.get("cash", kwargs["cash"])),
            "notional_floor": float(account.get("notional_floor", 0.0)),
            "band_frac": float(account.get("band_frac", 0.0)),
            "rebalance_days": int(account.get("rebalance_days", 1)),
            "max_position_pct": float(account.get("max_position_pct", 0.05)),
        })
        return _book_fingerprint(book, runner_kwargs=kwargs, universe=symbols)
    finally:
        ledger.close()
        probe_ledger_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Survivorship-bias stress test (blocking verdict)")
    ap.add_argument("--label", required=True, help="output label → outputs/bias_stress_<label>.json")
    ap.add_argument("--target-alpha", type=float, default=TARGET_ALPHA_DEFAULT)
    ap.add_argument("--window-start", default="2025-09-01")
    ap.add_argument("--window-end", default="2025-12-31")
    ap.add_argument("--skip-runs", action="store_true", help="skip the two production runs")
    ap.add_argument(
        "--reuse-runs",
        default=None,
        metavar="LABEL",
        help="re-emit the JSON from a previous bias_stress_<LABEL>.json without re-running the book",
    )
    ap.add_argument("--account", default="D_5W")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    t_start = time.time()
    unmeasured: list[str] = []
    limitations: list[str] = []

    from src.config import load_config

    cfg = load_config()
    url = _pit_url(cfg)
    account = next(
        (a for a in (cfg.get("shadow.accounts") or []) if str(a.get("name")) == args.account), None
    )
    if account is None:
        raise SystemExit(f"account {args.account!r} not in shadow.accounts")
    account = dict(account)
    k_slots = int(account.get("pb_k", K_SLOTS))

    from src.paper.shadow import resolve_shadow_universe

    shadow_symbols = resolve_shadow_universe(cfg, account.get("universe"))
    _log(f"window=[{args.window_start}, {args.window_end}] account={args.account} "
         f"shadow universe={len(shadow_symbols)} k={k_slots}")

    # ---- 1. missing-name quantification ------------------------------------
    _log("loading universe records + price symbol list …")
    universe_rows = load_universe_rows(url)
    price_symbols = load_price_symbols(url, universe_rows["symbol"].astype(str).unique())
    miss = missing_summary(
        universe_rows, price_symbols, shadow_symbols, args.window_start, args.window_end
    )
    _log(f"missing: {miss['n_missing_zero_price_bars']}/{miss['n_universe_symbols']} universe "
         f"symbols have zero price bars; in D universe={miss['n_missing_in_d_universe']}; "
         f"in window snapshots={miss['n_missing_in_window_snapshots']}")

    # PIT-store semantics for the record (universe_as_of / delisted_symbols)
    store_notes: dict[str, Any] = {}
    try:
        from src.data.point_in_time_loader import from_url

        store = from_url(url)
        for tag, when in (("window_start", args.window_start), ("window_end", args.window_end),
                          ("oldest_snapshot", miss.get("oldest_snapshot"))):
            if not when:
                continue
            store_notes[f"universe_as_of_{tag}"] = {
                "as_of": str(when),
                "n": len(store.universe_as_of(when, "universe")),
            }
        store_notes["delisted_symbols_at_window_start"] = len(
            store.delisted_symbols(args.window_start, "universe")
        )
        if miss.get("oldest_snapshot"):
            store_notes["delisted_symbols_at_oldest_snapshot"] = len(
                store.delisted_symbols(miss["oldest_snapshot"], "universe")
            )
        if hasattr(store, "close"):
            store.close()
    except Exception as exc:  # noqa: BLE001 — the raw queries above already carry the facts
        unmeasured.append(f"PIT store universe_as_of/delisted_symbols probe failed: {exc}")
    miss["pit_store_probe"] = store_notes

    # ---- 3. empirical hazard bound -----------------------------------------
    hazard = hazard_summary(universe_rows, shadow_symbols)
    if hazard.get("measured"):
        h_full = hazard["full_universe"]["annual_disappearance_rate"]
        h_d = hazard["d_universe"]["annual_disappearance_rate"]
        _log(f"hazard: full universe {h_full:.4%}/yr "
             f"({hazard['full_universe']['n_absent_from_newest']}/"
             f"{hazard['full_universe']['n_present_at_oldest']} gone over "
             f"{hazard['years']:.2f}y); D universe {h_d}")
    else:
        h_full = None
        unmeasured.append("annual disappearance rate: fewer than two universe snapshots")

    # ---- price-based risk proxies + de-bias rule ---------------------------
    _log("loading window price stats (median amount, last raw_close) …")
    stats = load_window_price_stats(url, shadow_symbols, args.window_start, args.window_end)
    in_panel = [s for s in shadow_symbols if s in stats.index]
    seg = risk_segment(
        stats.loc[in_panel, "last_raw_close"], stats.loc[in_panel, "median_amount"]
    )
    excluded = set(seg["excluded_symbols"])
    debiased_symbols = [s for s in shadow_symbols if s not in excluded]
    debias_rule = {
        "price_floor_cny": float(PRICE_FLOOR_CNY),
        "amount_decile": float(AMOUNT_DECILE),
        "amount_decile_threshold_cny": seg["amount_decile_threshold_cny"],
        "latest_price_basis": (
            "raw_close on the last price bar at or before window_end "
            f"({args.window_end}) — unadjusted, PIT-visible at the window's end"
        ),
        "median_amount_basis": (
            f"median daily amount over [{args.window_start}, {args.window_end}]"
        ),
        "rule": (
            "exclude a shadow-universe symbol iff its last raw_close (on/before window_end) "
            f"< {PRICE_FLOOR_CNY:.2f} CNY OR its median daily amount over the window is <= the "
            f"{AMOUNT_DECILE:.0%} quantile of the shadow universe's median daily amounts; a symbol "
            "with no window amount is treated as least liquid and excluded"
        ),
        "n_shadow_universe": len(shadow_symbols),
        "n_with_price_stats": len(in_panel),
        "n_excluded": len(excluded),
        "n_remaining": len(debiased_symbols),
        "excluded_symbols": sorted(excluded),
    }
    _log(f"de-bias rule: exclude {len(excluded)} of {len(shadow_symbols)} names "
         f"({seg['n_low_price']} penny + {seg['n_low_amount']} illiquid, "
         f"amount threshold {seg['amount_decile_threshold_cny']:,.0f} CNY) "
         f"→ {len(debiased_symbols)} names remain")

    # ---- 4. the two runs ---------------------------------------------------
    runs: dict[str, Any] = {"skipped": bool(args.skip_runs), "reused_from": None}
    entries_source: dict[str, Any] = {}
    epy: Optional[float] = None
    debiased_run: dict[str, Any] = {}

    if args.reuse_runs:
        src = ROOT / "outputs" / f"bias_stress_{args.reuse_runs}.json"
        if not src.is_file():
            raise SystemExit(f"--reuse-runs: {src} not found")
        prev = json.loads(src.read_text(encoding="utf-8"))
        prev_runs = prev.get("runs") or {}
        if not (prev_runs.get("baseline") and prev_runs.get("debias")):
            raise SystemExit(f"--reuse-runs: {src.name} carries no baseline/debias run blocks")
        runs = dict(prev_runs)
        runs.pop("reused_from_elapsed_seconds", None)  # legacy key from earlier re-emits
        runs["skipped"] = False
        runs["reused_from"] = src.name
        runs["reused_at"] = datetime.now().isoformat(timespec="seconds")
        #: keep the FIRST invocation's wall clock across repeated re-emits (a
        #: re-emit's own ``elapsed_seconds`` is seconds, not the run's cost)
        first_elapsed = prev_runs.get("first_invocation_seconds")
        if first_elapsed is None and not prev_runs.get("reused_from"):
            first_elapsed = prev.get("elapsed_seconds")
        if first_elapsed is not None:
            runs["first_invocation_seconds"] = first_elapsed
        else:
            runs.pop("first_invocation_seconds", None)
        baseline, debi = runs["baseline"], runs["debias"]
        debiased_run = prev.get("debiased_run") or {
            "baseline": baseline["metrics"],
            "debiased": debi["metrics"],
            "delta_debiased_minus_baseline": metric_delta(baseline["metrics"], debi["metrics"]),
        }
        prev_excl = set((prev.get("debias_rule") or {}).get("excluded_symbols") or [])
        if prev_excl != excluded:
            limitations.append(
                "reused runs came from a different de-bias exclusion set "
                f"({len(prev_excl)} vs {len(excluded)} symbols) — the delta does not describe "
                "this run's de-bias rule"
            )
        if baseline.get("n_days"):
            epy = entries_per_year(baseline["metrics"]["n_entries"] or 0, baseline["n_days"])
        entries_source = {
            "source": f"reused baseline run from {src.name} (production assembly)",
            "n_entries": baseline["metrics"].get("n_entries"),
            "n_days": baseline.get("n_days"),
        }
        _log(f"reusing runs from {src.name}: baseline ann="
             f"{baseline['metrics']['annualized_return']:+.2%} debias ann="
             f"{debi['metrics']['annualized_return']:+.2%}")
    elif args.skip_runs:
        runs["reason"] = "--skip-runs"
        unmeasured.append("production-assembly baseline/de-biased runs (--skip-runs)")
    else:
        from src.autopilot.state import ControlState
        from src.cli import _build_market_for_paper

        state_path = str(
            ROOT / str(cfg.get("autopilot.state_file", "outputs/autopilot_state.json"))
        ).replace(".json", f"_{args.account}.json")
        control = ControlState.load(state_path)
        _log(f"kill-switch: mode={control.mode} gross={control.gross_scale:g}")

        _log(f"building market once for the {len(shadow_symbols)}-name shadow universe …")
        t0 = time.time()
        market = _build_market_for_paper(cfg, shadow_symbols, args.window_start, None, seed=args.seed)
        market_symbols = [s for s in shadow_symbols if s in market.price_panel.columns]
        _log(f"market built in {time.time() - t0:.0f}s: "
             f"{len(market_symbols)}/{len(shadow_symbols)} names in the price panel, "
             f"bars={len(market.price_panel.index)}")
        runs["market"] = {
            "n_symbols_requested": len(shadow_symbols),
            "n_symbols_in_panel": len(market_symbols),
            "n_bars": int(len(market.price_panel.index)),
            "first_bar": str(pd.Timestamp(market.price_panel.index.min()).date()),
            "last_bar": str(pd.Timestamp(market.price_panel.index.max()).date()),
            "build_seconds": round(time.time() - t0, 1),
        }
        if len(market_symbols) < len(shadow_symbols):
            runs["market"]["dropped_symbols"] = sorted(set(shadow_symbols) - set(market_symbols))

        prod_probe = production_fingerprint(cfg, market, market_symbols, account, control)
        runs["production_fingerprint"] = prod_probe

        debiased_in_panel = [s for s in market_symbols if s not in excluded]
        baseline = run_book(
            cfg, market, market_symbols, account, control, args.window_start, args.window_end,
            args.seed, SCRATCH_DIR / f"{args.label}_baseline.sqlite", "baseline",
        )
        debi = run_book(
            cfg, market, debiased_in_panel, account, control, args.window_start, args.window_end,
            args.seed, SCRATCH_DIR / f"{args.label}_debias.sqlite", "debias",
        )
        for run in (baseline, debi):
            run["params_match_production"] = bool(
                run["fingerprint"].get("params_hash")
                and run["fingerprint"].get("params_hash") == prod_probe.get("params_hash")
            )
            run["all_checks_passed"] = bool(
                all(run["checks"].values()) and run["params_match_production"]
            )
        if not (baseline["params_match_production"] and debi["params_match_production"]):
            limitations.append(
                "params_match_production is False for at least one run — its numbers are NOT "
                "the deployed configuration"
            )
        runs["baseline"] = baseline
        runs["debias"] = debi
        debiased_run = {
            "baseline": baseline["metrics"],
            "debiased": debi["metrics"],
            "delta_debiased_minus_baseline": metric_delta(baseline["metrics"], debi["metrics"]),
            "baseline_universe_size": len(market_symbols),
            "debiased_universe_size": len(debiased_in_panel),
            "n_excluded": len(excluded),
        }
        epy = entries_per_year(
            baseline["metrics"]["n_entries"] or 0, baseline["n_days"]
        ) if baseline["n_days"] else None
        entries_source = {
            "source": "baseline run of this window (production assembly)",
            "n_entries": baseline["metrics"]["n_entries"],
            "n_days": baseline["n_days"],
        }

    if epy is None:
        ledger_stats = read_production_ledger_stats(PROD_LEDGER, args.window_start, args.window_end)
        if not ledger_stats.get("n_days_in_window"):
            ledger_stats = read_production_ledger_stats(PROD_LEDGER)
        ref = read_d_oos_reference(D_OOS_REFERENCE)
        if ledger_stats.get("available") and ledger_stats.get("n_days_in_window"):
            epy = float(ledger_stats["entries_per_year"])
            entries_source = {
                "source": "production ledger (read-only)",
                "detail": ledger_stats,
            }
        elif ref.get("available") and ref.get("n_days") and ref.get("n_fills"):
            est_entries = float(ref["n_fills"]) * 0.5
            epy = entries_per_year(est_entries, float(ref["n_days"]))
            entries_source = {
                "source": "outputs/d_oos_oos_2025h2_v3.json (n_fills/2 as the buy share)",
                "detail": ref,
                "assumption": "half of all fills are entries",
            }
            limitations.append(
                "entries_per_year came from the OOS artifact assuming half the fills are buys"
            )
        else:
            epy = None
            unmeasured.append("entries_per_year (no run, no ledger, no reference artifact)")
    if epy is not None and "detail" not in entries_source:
        entries_source["entries_per_year"] = round(epy, 2)
    _log(f"entries_per_year = {epy if epy is None else round(epy, 2)} "
         f"({entries_source.get('source')})")

    # ---- 2. drag model + break-even ----------------------------------------
    drag: dict[str, Any] = {"k_slots": k_slots, "loss_mid": LOSS_MID}
    if epy is not None:
        drag["entries_per_year"] = round(epy, 2)
        drag["grid"] = drag_grid(epy, k=k_slots)
        drag["breakeven_x_at_threshold"] = {
            f"{float(loss):.2f}": (
                None if breakeven_x(epy, k=k_slots, loss=loss,
                                    threshold_frac=THRESHOLD_RATIO * args.target_alpha) is None
                else round(breakeven_x(epy, k=k_slots, loss=loss,
                                       threshold_frac=THRESHOLD_RATIO * args.target_alpha), 8)
            )
            for loss in LOSS_GRID
        }
        drag["breakeven_x_at_alpha"] = {
            f"{float(loss):.2f}": (
                None if breakeven_x(epy, k=k_slots, loss=loss,
                                    threshold_frac=args.target_alpha) is None
                else round(breakeven_x(epy, k=k_slots, loss=loss,
                                       threshold_frac=args.target_alpha), 8)
            )
            for loss in LOSS_GRID
        }
        drag["drag_pp_at_threshold_breakeven"] = round(
            abs(annual_drag_pp(epy, drag["breakeven_x_at_threshold"][f"{LOSS_MID:.2f}"] or 0.0,
                               k=k_slots, loss=LOSS_MID)), 4
        )
        _log(f"drag grid: x=0.001 → {drag['grid']['drag_pp'][f'{LOSS_MID:.2f}']['0.001']:+.3f}pp/yr, "
             f"x=0.05 → {drag['grid']['drag_pp'][f'{LOSS_MID:.2f}']['0.05']:+.3f}pp/yr; "
             f"breakeven x @ threshold (L={LOSS_MID}) = "
             f"{drag['breakeven_x_at_threshold'][f'{LOSS_MID:.2f}']:.4%}")
    else:
        unmeasured.append("drag grid / break-even x (entries_per_year unavailable)")

    # ---- verdict ------------------------------------------------------------
    x_ub = None
    if h_full is not None:
        x_ub = x_upper_bound_from_hazard(h_full, hold_days=int(account.get("pb_max_hold", MAX_HOLD_DAYS)))
    #: market-wide disappearance rate at which the drag reaches the threshold
    be_h: Optional[float] = None
    if epy:
        denom_be = epy * (1.0 / k_slots) * abs(LOSS_MID) * trading_year_fraction(
            int(account.get("pb_max_hold", MAX_HOLD_DAYS))
        )
        be_h = (THRESHOLD_RATIO * args.target_alpha) / denom_be if denom_be > 0 else None
    if x_ub is not None and epy is not None:
        breakeven = breakeven_x(epy, k=k_slots, loss=LOSS_MID,
                                threshold_frac=THRESHOLD_RATIO * args.target_alpha)
        verdict = verdict_block(
            target_alpha=args.target_alpha, threshold_ratio=THRESHOLD_RATIO,
            x_upper_bound=x_ub, epy=epy, k=k_slots, loss_mid=LOSS_MID,
            h_annual=h_full, breakeven=breakeven, breakeven_h=be_h,
        )
    else:
        verdict = {
            "target_alpha_annual": float(args.target_alpha),
            "threshold_ratio": float(THRESHOLD_RATIO),
            "threshold_pp": round(THRESHOLD_RATIO * args.target_alpha * 100.0, 4),
            "x_upper_bound": None,
            "drag_at_upper_bound_pp": None,
            "bias_vs_alpha_ratio": None,
            "bias_blocking_evolution": None,
            "verdict_text": (
                "UNMEASURED: cannot bound x — the empirical hazard and/or entries_per_year "
                "could not be measured; treat the gate as closed until they are."
            ),
        }
        unmeasured.append("verdict (x_upper_bound or entries_per_year unmeasured)")

    # sensitivity: the same verdict under alternative hazard / loss assumptions
    sensitivity: dict[str, Any] = {}
    if epy is not None:
        variants: dict[str, Any] = {}
        if h_full is not None:
            variants["market_wide_h_exposure_scaled"] = {
                "h_annual": h_full,
                "x": x_ub,
                "note": "PRIMARY: h × hold/252, pick_prob = 1",
            }
            variants["market_wide_h_no_exposure_scaling"] = {
                "h_annual": h_full,
                "x": float(h_full),
                "note": "over-conservative: every entry faces a full year of hazard",
            }
        d_rule = (hazard.get("d_universe", {}) or {}).get("rule_of_three_upper_rate") if hazard.get("measured") else None
        if d_rule:
            variants["d_universe_rule_of_three"] = {
                "h_annual": d_rule,
                "x": x_upper_bound_from_hazard(d_rule, hold_days=int(account.get("pb_max_hold", MAX_HOLD_DAYS))),
                "note": "95% upper bound with zero observed D-universe disappearances",
            }
        for name, spec in variants.items():
            x = spec.get("x")
            if x is None:
                continue
            spec["drag_pp_by_loss"] = {
                f"{float(loss):.2f}": round(abs(annual_drag_pp(epy, x, k=k_slots, loss=loss)), 4)
                for loss in LOSS_GRID
            }
            spec["blocking_at_mid_loss"] = bool(
                spec["drag_pp_by_loss"][f"{LOSS_MID:.2f}"] > THRESHOLD_RATIO * args.target_alpha * 100.0
            )
        sensitivity = variants
        _log("sensitivity: " + ", ".join(
            f"{k}={v['drag_pp_by_loss'][f'{LOSS_MID:.2f}']:.2f}pp"
            f"({'BLOCK' if v['blocking_at_mid_loss'] else 'pass'})" for k, v in variants.items()
        ))

    # ---- assemble -----------------------------------------------------------
    verdict_context = {
        "drag_sign_convention": (
            "drag_at_upper_bound_pp is the annual return LOSS in percentage points (positive = "
            "loss); the drag_model grid is SIGNED per the literal formula entries·x·(1/k)·L"
        ),
        "x_upper_bound_construction": (
            "x = h_annual × (max_hold / 252) × pick_prob, with h_annual = the MARKET-WIDE annual "
            "name-disappearance rate and pick_prob = 1 (worst case). The exposure factor converts "
            "a per-year hazard into the per-entry probability that a name dies while it is held."
        ),
        "direct_channel_is_zero": (
            f"{miss['n_missing_in_d_universe']} of the {len(shadow_symbols)} D-track pool names have "
            "zero price bars, so the direct missing-bars channel is 0 by construction; the bound "
            "is the market-wide STRESS case (as if the book picked from the full listed pool)."
        ),
        "breakeven_h_annual": _round_or_none(be_h),
        "breakeven_h_note": (
            "the market-wide disappearance rate above which the drag exceeds the threshold"
        ),
        "measured_h_annual": h_full,
    }
    if h_full is not None and be_h is not None:
        verdict_context["measured_h_vs_breakeven_h"] = round(h_full / be_h, 4) if be_h else None
    unmeasured.append(
        "index-membership look-ahead channel: resolve_shadow_universe() returns TODAY's "
        "HS300+ZZ500 list applied to the 2025 window, and the PIT store holds no historical "
        "index membership — this channel is unmeasurable here and is NOT covered by "
        "x_upper_bound"
    )
    limitations.extend([
        "the empirical hazard rests on TWO stored universe snapshots (2015-01-05 and 2026-08-07); "
        "the annual rate is the geometric average over that 11.6-year span and cannot see "
        "year-to-year variation in delisting intensity",
        "a name absent from the newest snapshot may have been delisted, merged, renamed or "
        "re-coded — only the delisted subset carries the modelled negative loss",
        "the index-membership channel is NOT measurable here: resolve_shadow_universe() returns "
        "today's HS300+ZZ500 list, applied to a 2025 window, and no historical index membership "
        "is stored",
        "entries_per_year is a point estimate from one window/ledger span; the drag scales "
        "linearly with it",
        "the drag model treats a doomed entry as a single -L hit on one slot and ignores "
        "correlated doom, suspension illiquidity and the exit path",
    ])
    result: dict[str, Any] = {
        "label": args.label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).name),
        "window": {"start": args.window_start, "end": args.window_end},
        "window_note": (
            f"full window {args.window_start}..{args.window_end}"
            if not (args.skip_runs or args.reuse_runs) else "runs not executed in this invocation"
        ),
        "account": args.account,
        "k_slots": k_slots,
        "target_alpha_annual": float(args.target_alpha),
        "threshold_ratio": float(THRESHOLD_RATIO),
        "verdict": verdict,
        "verdict_context": verdict_context,
        "missing_names": miss,
        "hazard": hazard,
        "risk_proxies": {
            "price_floor_cny": float(PRICE_FLOOR_CNY),
            "amount_decile": float(AMOUNT_DECILE),
            "share_low_price": seg["share_low_price"],
            "share_bottom_decile_amount": seg["share_bottom_decile_amount"],
            "share_excluded": seg["share_excluded"],
            "n_low_price": seg["n_low_price"],
            "n_low_amount": seg["n_low_amount"],
            "amount_decile_threshold_cny": seg["amount_decile_threshold_cny"],
            "n_shadow_universe": len(shadow_symbols),
        },
        "debias_rule": debias_rule,
        "drag_model": drag,
        "entries_source": entries_source,
        "runs": runs,
        "debiased_run": debiased_run,
        "sensitivity": sensitivity,
        "unmeasured": unmeasured,
        "limitations": limitations,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    if args.skip_runs:
        result["window_note"] = "runs skipped (--skip-runs); drag model uses the ledger/artifact entry estimate"
    elif args.reuse_runs:
        first_elapsed = runs.get("first_invocation_seconds")
        result["window_note"] = (
            f"full window {args.window_start}..{args.window_end} — no shrink needed: the one "
            f"market build + two production-assembly runs fitted the 10-minute budget"
            + (f" ({first_elapsed}s in the first invocation)" if first_elapsed else "")
            + f"; this JSON re-emits the analysis and reuses the runs recorded in "
            f"bias_stress_{args.reuse_runs}.json"
        )

    out_path = ROOT / "outputs" / f"bias_stress_{args.label}.json"
    # Provenance contract (audit P-6): this artifact is the ADMISSION GATE for the
    # alpha-layer evolution loop, so it must state its slice, its convention and
    # the code/data cut-off behind the verdict.
    from src.provenance import stamp_artifact

    window = result.get("window") or {"start": args.window_start, "end": args.window_end}
    result = stamp_artifact(
        result, window=window,
        convention=(
            "survivorship-bias stress test on the PIT universe/price records; "
            "drag model entries_per_year * x * (1/k) * L; de-biased run uses the "
            "production D-track assembly on a re-ranked subset"
        ),
        data_as_of=str(window.get("end") if isinstance(window, dict) else args.window_end),
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    v = result["verdict"]
    _log("=" * 78)
    _log(f"missing names: {miss['n_missing_zero_price_bars']} of {miss['n_universe_symbols']} "
         f"universe symbols have ZERO price bars "
         f"({miss['n_missing_absent_from_newest_snapshot']} delisted-like, "
         f"{miss['n_missing_present_in_newest_snapshot']} new-listing-like); "
         f"inside the D universe: {miss['n_missing_in_d_universe']}")
    _log(f"hazard: full universe {hazard.get('full_universe', {}).get('annual_disappearance_rate')}"
         f"/yr over {hazard.get('years')}y; D universe "
         f"{hazard.get('d_universe', {}).get('annual_disappearance_rate')}")
    _log(f"entries_per_year={epy if epy is None else round(epy, 1)}  "
         f"x_upper_bound={v['x_upper_bound']}")
    _log(f"drag_at_upper_bound_pp={v['drag_at_upper_bound_pp']}  "
         f"ratio={v['bias_vs_alpha_ratio']}  threshold_pp={v['threshold_pp']}")
    _log(f"bias_blocking_evolution={v['bias_blocking_evolution']}")
    _log(v["verdict_text"])
    if debiased_run:
        d = debiased_run["delta_debiased_minus_baseline"]
        _log(f"de-biased vs baseline: ann {d['annualized_return_pp']:+.2f}pp/yr, "
             f"cum {d['cumulative_return_pp']:+.2f}pp, sharpe {d['sharpe']:+.2f}, "
             f"maxDD {d['max_drawdown_pp']:+.2f}pp, fills {d['n_fills']:+d}, "
             f"entries {d['n_entries']:+d}")
    for item in unmeasured:
        _log(f"UNMEASURED: {item}")
    _log(f"wrote {out_path} in {result['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
