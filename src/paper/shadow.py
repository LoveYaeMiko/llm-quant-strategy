"""Shadow mode — daily no-order simulation, status/report emission, red lines.

The "影子模式" runs the *same* three-layer portfolio and execution loop as the
simulated paper trade (:class:`~src.paper.runner.PaperRunner`), but its purpose
is **monitoring** rather than validation: no real orders are placed, and every
run appends to the same resumable ledger (``outputs/paper_ledger.sqlite``) so
PAICC can watch the book evolve day by day from ``shadow.start_date`` (2026-01-01,
out-of-sample vs ``test_end`` 2025-12-31) forward.

What this module owns (kept free of ``cli`` imports so it stays testable):

* ``resolve_shadow_universe`` — HS300 constituents for the shadow book;
* ``paper_runner_kwargs`` — the ``PaperRunner`` cost/ledger parameters read from
  the ``paper`` config section (including the real A-share cost model);
* ``refresh_price`` / ``refresh_pead`` / ``refresh_sentiment`` — the three
  incremental data refreshes the user chose ("行情+财报+研报舆情 完整");
* ``build_shadow_status`` — the ``outputs/shadow_status.json`` payload PAICC
  consumes (equity / positions / §7 params / four canonical red lines);
* ``render_shadow_report`` — the Chinese markdown daily report for email.

Market construction and portfolio assembly live in ``src.cli`` (they reuse the
same ``_market_data`` / ``_slice_market`` / factor-pool logic as ``cmd_paper``);
the calibration sweeps in :mod:`src.calibration` import these helpers from here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# universe + cost params
# --------------------------------------------------------------------------- #
def resolve_shadow_universe(cfg, name=None) -> list[str]:
    """Shadow book universe — cached constituent list(s) under ``data.universe_dir``.

    ``name`` (``shadow.universe``, default ``hs300``) may be ``hs300``,
    ``hs300_500`` (union of HS300 + ZZ500 — the research cross-section the ML
    artifacts were trained on), or any other cached ``<name>.json``.
    """
    from ..config import Config

    name = name or str(cfg.get("shadow.universe", "hs300"))
    universe_dir = Path(str(cfg.get("data.universe_dir", "data/universe")))
    if name == "hs300_500":
        out: list[str] = []
        for sub in ("hs300", "zz500"):
            path = universe_dir / f"{sub}.json"
            if not path.is_file():
                raise SystemExit(f"universe file {path} not found — run `python -m src.cli ingest` first")
            out.extend(json.loads(path.read_text(encoding="utf-8")))
        return sorted(set(out))
    path = universe_dir / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"universe file {path} not found — run `python -m src.cli ingest` first")
    return sorted(json.loads(path.read_text(encoding="utf-8")))


def paper_runner_kwargs(cfg) -> dict[str, Any]:
    """``PaperRunner`` kwargs from the ``paper`` config section (cost model included)."""
    pcfg = cfg.section("paper")
    return {
        "cash": float(pcfg.get("initial_cash", 100_000.0)),
        "slippage_bps": float(pcfg.get("slippage_bps", 2.0)),
        "commission_bps": float(pcfg.get("commission_bps", 5.0)),
        "min_commission": float(pcfg.get("min_commission", 1.0)),
        "stamp_tax_sell_bps": float(pcfg.get("stamp_tax_sell_bps", 0.0)),
        "transfer_fee_bps": float(pcfg.get("transfer_fee_bps", 0.0)),
        "max_position_pct": float(pcfg.get("max_position_pct", 0.05)),
        "rebalance_days": int(pcfg.get("rebalance_days", 1)),
        "pit_strict": bool(pcfg.get("pit_strict", True)),
    }


def real_cost_model(cfg) -> dict[str, float]:
    """The real A-share cost structure (``s7_calibration.cost_model``)."""
    cm = cfg.section("s7_calibration").get("cost_model", {}) or {}
    return {
        "stamp_tax_sell_bps": float(cm.get("stamp_tax_sell_bps", 5.0)),
        "transfer_fee_bps": float(cm.get("transfer_fee_bps", 0.1)),
        "commission_bps": float(cm.get("commission_bps", 2.5)),
        "min_commission": float(cm.get("min_commission", 5.0)),
        "slippage_bps": float(cm.get("slippage_bps", 2.0)),
    }


# --------------------------------------------------------------------------- #
# data refresh (daily, incremental)
# --------------------------------------------------------------------------- #
def refresh_price(cfg, symbols: list[str]) -> dict[str, Any]:
    """Incremental price ingest for ``symbols`` (bars after the newest stored bar)."""
    from ..data.ingestion.ingestor import Ingestor

    ing = Ingestor(cfg)
    stats = ing.ingest(symbols=symbols, resume=True)
    return {"bars": getattr(stats, "price_records", None) or getattr(stats, "price_bars", None),
            "errors": list(getattr(stats, "errors", []) or [])}


def refresh_pead(cfg, symbols: list[str], current_year: int | None = None) -> int:
    """Fetch the current year's quarterly profit for symbols missing it, merging
    into the per-symbol cache (keeps the 2020-2025 history intact). Returns the
    number of symbols (re-)fetched.

    ``ensure_profit_panel`` loads a cached symbol as-is (never re-fetches), so a
    fresh calendar year would otherwise never enter the PEAD panel — this closes
    that gap with a targeted current-year fetch.
    """
    from ..data.financials import fetch_symbol_profit

    current_year = current_year or pd.Timestamp.today().year
    cache_dir = Path(str(cfg.get("pead", {}).get("cache_dir", "data/financials")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for symbol in symbols:
        path = cache_dir / f"profit_{symbol.replace('.', '_')}.csv"
        has_current = False
        if path.is_file():
            try:
                old = pd.read_csv(path)
                if "pubDate" in old.columns and not old.empty:
                    years = set(pd.to_datetime(old["pubDate"], errors="coerce").dt.year.dropna())
                    has_current = current_year in years
            except Exception:  # noqa: BLE001 — corrupt cache -> refetch current year
                has_current = False
        if has_current:
            continue
        df = fetch_symbol_profit(symbol, [current_year])
        if df is None or df.empty:
            continue
        if path.is_file():
            try:
                old = pd.read_csv(path)
                merged = pd.concat([old, df], ignore_index=True).drop_duplicates(
                    subset=["statDate", "pubDate"], keep="last"
                )
                merged.to_csv(path, index=False)
            except Exception:  # noqa: BLE001
                df.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)
        fetched += 1
    return fetched


def refresh_sentiment(cfg, symbols: list[str]) -> dict[str, int]:
    """Re-fetch the research-report history for ``symbols`` (full per-symbol
    history — the endpoint has no date param). Heavy (~15-20 min full HS300).

    Symbols whose cache parquet was written *today* are skipped: the deployed
    daily scheduler runs once per day, so this keeps the loop incremental while
    a full re-fetch stays available as a manual one-off (delete the parquet).
    """
    from ..sentiment.ingestion import ReportIngestor

    report_dir = str(cfg.get("sentiment", {}).get("report_dir", "data/reports"))
    ingestor = ReportIngestor(report_dir)
    want = set(symbols)
    today = datetime.today().date()
    fresh: set[str] = set()
    for p in ingestor.data_dir.glob("*.parquet"):
        if p.stat().st_size <= 0:
            continue
        try:
            if datetime.fromtimestamp(p.stat().st_mtime).date() >= today:
                fresh.add(p.stem.replace("_", "."))
        except Exception:  # noqa: BLE001
            continue
    todo = [s for s in symbols if s not in fresh]
    if not todo:
        return {"symbols": 0, "rows": 0, "fresh_skipped": len(fresh & want)}
    counts = ingestor.collect(todo, pause=0.2, force=True)
    return {"symbols": len(counts), "rows": sum(counts.values()), "fresh_skipped": len(fresh & want)}


# --------------------------------------------------------------------------- #
# benchmark (HS300 index, 000300.SH)
# --------------------------------------------------------------------------- #
def _benchmark_cache(cfg) -> Path:
    return Path(str(cfg.section("shadow").get("benchmark_cache", "data/benchmark/hs300_index.csv")))


def fetch_benchmark_index(
    cfg,
    start_date: str = "20150101",
    end_date: str | None = None,
) -> pd.Series | None:
    """HS300 index (000300.SH / ``sh000300``) daily close → ``date → close`` Series.

    Fetched via AKShare's Sina feed (``stock_zh_index_daily``) rather than the
    Eastmoney endpoint: this machine's Python TLS stack gets ``RemoteDisconnected``
    from ``*.eastmoney.com`` API hosts (curl works, requests/urllib3 do not), while
    the Sina feed is reachable. Network-dependent and best-effort — any failure
    returns ``None`` so the shadow run never breaks on a down feed.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.warning("akshare not installed; HS300 benchmark unavailable")
        return None
    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
    except Exception as exc:  # noqa: BLE001
        logger.warning("HS300 index fetch failed: %s", exc)
        return None
    if df is None or df.empty:
        return None
    date_col = "date" if "date" in df.columns else df.columns[0]
    close_col = "close" if "close" in df.columns else df.columns[1]
    s = pd.Series(
        df[close_col].to_numpy(dtype=float),
        index=pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d").tolist(),
    )
    s = s.dropna().sort_index()
    lo = _fmt_date(start_date)
    if lo:
        s = s[s.index >= lo]
    hi = _fmt_date(end_date)
    if hi:
        s = s[s.index <= hi]
    return s


def _fmt_date(d: str | None) -> str | None:
    """Normalize an 8-digit ``YYYYMMDD`` (or ISO) date to ``YYYY-MM-DD``."""
    if not d:
        return None
    d = str(d)
    if len(d) == 8 and d.isdigit():
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return d


def refresh_benchmark(cfg) -> dict[str, Any]:
    """Fetch HS300 index and overwrite the cached CSV (idempotent)."""
    s = fetch_benchmark_index(cfg)
    if s is None or s.empty:
        return {"error": "HS300 index fetch failed (kept previous cache if any)"}
    path = _benchmark_cache(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.rename("close").to_csv(path, encoding="utf-8")
    return {"path": str(path), "rows": int(len(s))}


def load_benchmark_index(cfg) -> pd.Series | None:
    """Read the cached HS300 index CSV (``date → close``), or ``None``."""
    path = _benchmark_cache(cfg)
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, index_col=0)
        s = df.iloc[:, 0] if df.shape[1] else pd.Series(dtype=float)
        s.index = s.index.astype(str)
        return s.astype(float).sort_index()
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to load benchmark cache %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------- #
# red lines
# --------------------------------------------------------------------------- #
def _real_fee(notional: float, side: str, cost: dict[str, float]) -> float:
    commission = max(cost["min_commission"], notional * cost["commission_bps"] / 10_000.0)
    transfer = notional * cost["transfer_fee_bps"] / 10_000.0
    stamp = (notional * cost["stamp_tax_sell_bps"] / 10_000.0) if side == "sell" else 0.0
    return commission + transfer + stamp


def compute_cost_deviation(
    fills: pd.DataFrame,
    real_params: dict[str, float],
) -> dict[str, Any]:
    """Compare the ledger's charged cost (current model) vs the real A-share cost.

    The "current" total is read straight from the ``commission`` column already
    persisted on each fill; only the real side is recomputed. Returns
    ``{current_total, real_total, deviation_pct}`` where deviation is
    ``(real - current) / current * 100`` (positive = current model under-charges).
    """
    if fills.empty:
        return {"current_total": 0.0, "real_total": 0.0, "deviation_pct": 0.0}
    current_total = float(fills["commission"].sum())
    real_total = float(
        sum(_real_fee(float(r.notional), r.side, real_params) for r in fills.itertuples())
    )
    denom = current_total if current_total > 0 else real_total
    deviation = 0.0 if denom == 0 else (real_total - current_total) / denom * 100.0
    return {"current_total": round(current_total, 2), "real_total": round(real_total, 2),
            "deviation_pct": round(deviation, 2)}


def _market_trend_pct(price_panel: pd.DataFrame, days: int) -> float:
    """Equal-weight market ``days``-day compounded return (PIT-safe), latest value."""
    daily = price_panel.pct_change(fill_method=None).mean(axis=1)
    trend = (1.0 + daily).rolling(int(days)).apply(lambda x: x.prod() - 1.0, raw=True)
    vals = trend.dropna()
    return float(vals.iloc[-1] * 100.0) if not vals.empty else 0.0


def _threshold_level(value: float, threshold: float, critical: float) -> str:
    """Map a numeric red-line value to ``ok`` / ``warning`` / ``critical``."""
    if value >= critical:
        return "critical"
    if value >= threshold:
        return "warning"
    return "ok"


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Running drawdown (``equity / cummax − 1``) for a daily equity series."""
    peak = equity.cummax()
    return equity / peak - 1.0


def enrich_positions(
    fills: pd.DataFrame,
    positions: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Per-symbol cost basis via moving average cost over the execution ledger.

    ``ledger.fills()`` already carries the sign in ``shares`` (sells negative,
    buys positive). Fills in the direction of the open position add ``qty * px``
    to basis; fills that *reduce* the position remove ``closed_qty * avg_cost``
    (not the trade price), so ``entry_price = basis / shares`` is the running
    average cost — correct for both long and short. Returns
    ``{symbol: {entry_price, pnl_basis, first_date}}``.
    """
    if fills is None or fills.empty:
        return {}
    basis: dict[str, float] = {}
    signed_shares: dict[str, float] = {}
    first_date: dict[str, str] = {}
    for r in fills.itertuples(index=False):
        sym = str(r.symbol)
        qty = float(r.shares)
        px = float(r.price)
        first_date.setdefault(sym, str(r.date))
        s = signed_shares.get(sym, 0.0)
        b = basis.get(sym, 0.0)
        if s == 0.0:
            # Open a fresh position at the trade price.
            signed_shares[sym] = qty
            basis[sym] = qty * px
            continue
        avg = b / s  # positive: b and s always share a sign
        if (qty > 0) == (s > 0):
            # Same direction: add to the position at the trade price.
            signed_shares[sym] = s + qty
            basis[sym] = b + qty * px
        else:
            # Reducing/reversing: remove closed shares at current avg cost.
            closing = min(abs(qty), abs(s))
            sign = 1.0 if s > 0 else -1.0
            new_b = b - sign * closing * avg
            new_s = s + qty
            remaining = abs(qty) - closing
            if remaining > 0:
                # Reversed through flat into the opposite side at trade price.
                open_sign = 1.0 if qty > 0 else -1.0
                new_s = open_sign * remaining
                new_b = open_sign * remaining * px
            signed_shares[sym] = new_s
            basis[sym] = new_b

    out: dict[str, dict[str, Any]] = {}
    for sym, shares in positions.items():
        if shares == 0.0:
            continue
        b = basis.get(sym, 0.0)
        entry = b / shares
        out[sym] = {
            "entry_price": round(float(entry), 4),
            "pnl_basis": round(float(abs(b)), 2),
            "first_date": first_date.get(sym),
        }
    return out


# --------------------------------------------------------------------------- #
# status + report
# --------------------------------------------------------------------------- #
def build_shadow_status(
    cfg,
    ledger,
    market,
    result: dict[str, Any],
    overlays: dict[str, Any],
    meta: dict[str, Any],
    benchmark: pd.Series | None = None,
    book_long_pct: float | None = None,
    book_short_pct: float | None = None,
) -> dict[str, Any]:
    """Build the ``outputs/shadow_status.json`` payload PAICC consumes."""
    last_date, cash, positions = ledger.latest_state()
    eq = ledger.equity_curve()
    metrics = result.get("metrics", {}) or {}
    pcfg = cfg.section("paper")
    acfg = cfg.section("alpha_core")
    s7 = cfg.section("s7_calibration")
    rcfg = cfg.section("risk_overlay")
    tcfg = cfg.section("seasonal_tilt")
    rlc = cfg.section("red_lines")

    # data freshness: days since the newest stored price bar
    latest_price = pd.Timestamp(market.price_panel.index.max())
    freshness_days = max(0, (pd.Timestamp.today().normalize() - latest_price).days)

    # current §7 params (as configured)
    s7_params = {
        "amplitude": float(tcfg.get("amplitude", 0.20)),
        "zscore_threshold": float(rcfg.get("zscore_threshold", -2.5)),
        "position_cut": float(rcfg.get("position_cut", 0.50)),
        "freeze_days": int(rcfg.get("freeze_days", 5)),
        "slippage_bps": float(pcfg.get("slippage_bps", 2.0)),
        "commission_bps": float(pcfg.get("commission_bps", 5.0)),
        "min_commission": float(pcfg.get("min_commission", 1.0)),
        "stamp_tax_sell_bps": float(pcfg.get("stamp_tax_sell_bps", 0.0)),
        "transfer_fee_bps": float(pcfg.get("transfer_fee_bps", 0.0)),
    }

    # positions (shares -> weights + cost basis) at the latest recorded day
    positions_out: list[dict[str, Any]] = []
    latest_equity = float(eq.iloc[-1]) if len(eq) else float(pcfg.get("initial_cash", 100_000.0))
    enriched = enrich_positions(ledger.fills(), positions)
    if last_date is not None and positions:
        try:
            close = market.price_panel.loc[pd.Timestamp(last_date)]
        except KeyError:
            close = market.price_panel.iloc[-1]
        for sym, shares in positions.items():
            px = float(close.get(sym, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            weight = shares * px / latest_equity if latest_equity else 0.0
            info = enriched.get(sym, {})
            entry = info.get("entry_price")
            basis = info.get("pnl_basis", 0.0)
            pnl = (px - entry) * shares if entry is not None else None
            pnl_pct = (pnl / basis) if (pnl is not None and basis > 0) else None
            days_held = None
            first_date = info.get("first_date")
            if first_date is not None and last_date is not None:
                days_held = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days
            positions_out.append(
                {"symbol": sym, "shares": round(float(shares), 0),
                 "weight": round(float(weight), 6),
                 "side": "long" if shares > 0 else "short",
                 "entry_price": entry,
                 "last_price": round(float(px), 4),
                 "pnl": round(float(pnl), 2) if pnl is not None else None,
                 "pnl_pct": round(float(pnl_pct), 6) if pnl_pct is not None else None,
                 "days_held": days_held}
            )
    positions_out.sort(key=lambda p: -abs(p["weight"]))

    # --- daily equity / benchmark / excess curves ----------------------------
    initial_cash = float(pcfg.get("initial_cash", 100_000.0))
    equity_curve_out: list[dict[str, Any]] = []
    benchmark_out: list[dict[str, Any]] = []
    excess_out: list[dict[str, Any]] = []
    if len(eq):
        dd = _drawdown_series(eq)
        equity_curve_out = [
            {"date": str(d), "equity": round(float(v), 2),
             "drawdown": round(float(dd.loc[d]), 6)}
            for d, v in eq.items()
        ]
        if benchmark is not None and not benchmark.empty and eq.iloc[0] > 0:
            bench = benchmark.reindex(eq.index).ffill().dropna()
            if not bench.empty and bench.iloc[0] > 0:
                port_nav = eq / eq.iloc[0]      # both normalized to 1.0 at shadow start
                bench_nav = bench / bench.iloc[0]
                bench_dd = _drawdown_series(bench_nav * initial_cash)
                for d in bench.index:
                    benchmark_out.append({
                        "date": str(d),
                        "equity": round(float(bench_nav.loc[d] * initial_cash), 2),
                        "drawdown": round(float(bench_dd.loc[d]), 6),
                    })
                    excess_out.append({
                        "date": str(d),
                        "excess": round(float(port_nav.loc[d] - bench_nav.loc[d]), 6),
                    })

    # --- four canonical red lines -------------------------------------------
    fills_df = ledger.fills()
    real = real_cost_model(cfg)
    cost_dev = compute_cost_deviation(fills_df, real)

    # short-leg imbalance: |long_gross - short_gross| / gross. The regime-adaptive
    # short leg deliberately scales shorts to ``short_scale`` in strong uptrends,
    # and long-only books (short_pct=0) are 100% long by design — the expected
    # imbalance is |long - ss*short| / (long + ss*short); flag only the EXCESS
    # over that design baseline, otherwise every uptrend day fires a spurious
    # critical line.
    trend_gate = float(acfg.get("trend_gate", 0.03))
    trend_pct = _market_trend_pct(market.price_panel, int(acfg.get("trend_days", 60)))
    regime_triggered = trend_pct > trend_gate * 100.0
    book_long = float(book_long_pct if book_long_pct is not None else acfg.get("long_pct", 0.10))
    book_short = float(book_short_pct if book_short_pct is not None else acfg.get("short_pct", 0.10))
    short_scale_cfg = float(acfg.get("short_scale", 0.5)) if regime_triggered else 1.0
    if book_short > 0:
        expected_imb = abs(book_long - short_scale_cfg * book_short) / (
            book_long + short_scale_cfg * book_short
        ) * 100.0
    else:
        expected_imb = 100.0
    short_dev = 0.0
    if last_date is not None and positions:
        try:
            close = market.price_panel.loc[pd.Timestamp(last_date)]
        except KeyError:
            close = market.price_panel.iloc[-1]
        long_gross = 0.0
        short_gross = 0.0
        for sym, s in positions.items():
            px = close.get(sym, np.nan)
            if not np.isfinite(px):  # suspended name — no mark today
                continue
            if s > 0:
                long_gross += s * float(px)
            else:
                short_gross += -s * float(px)
        gross = long_gross + short_gross
        if gross > 0:
            raw_imb = abs(long_gross - short_gross) / gross * 100.0
            short_dev = max(0.0, raw_imb - expected_imb)

    # pead anomaly: tilt expected but no valid PEAD data
    pead = overlays.get("pead")
    pead_ok = pead is not None and len(getattr(pead, "symbols", []) or []) > 0
    pead_anomaly = (not pead_ok) and bool(cfg.get("pead", {}).get("enabled", True))

    # red-line thresholds from config (was hardcoded 10/20 — now tunable)
    cost_threshold = float(rlc.get("cost_deviation_threshold", 10.0))
    cost_critical = float(rlc.get("cost_deviation_critical", 20.0))
    short_threshold = float(rlc.get("short_leg_threshold", 10.0))
    short_critical = float(rlc.get("short_leg_critical", 20.0))

    red_lines: list[dict[str, Any]] = [
        {
            "name": "cost_deviation",
            "label": "成本模型偏差",
            "value": cost_dev["deviation_pct"],
            "level": _threshold_level(abs(cost_dev["deviation_pct"]), cost_threshold, cost_critical),
            "threshold": cost_threshold,
            "critical": cost_critical,
            "detail": (f"固定成本 {cost_dev['current_total']:.2f} vs 真实 {cost_dev['real_total']:.2f} "
                       f"(累计偏差 {cost_dev['deviation_pct']:+.1f}%)"),
        },
        {
            "name": "short_leg_deviation",
            "label": "多空敞口失衡",
            "value": round(short_dev, 2),
            "level": _threshold_level(short_dev, short_threshold, short_critical),
            "threshold": short_threshold,
            "critical": short_critical,
            "detail": (f"多头/空头敞口超出 regime 基准 {short_dev:.1f}%"
                       f"（regime 基准 {expected_imb:.1f}%）"),
        },
        {
            "name": "regime_switch",
            "label": "趋势切换",
            "value": round(trend_pct, 2),
            "level": "warning" if regime_triggered else "ok",
            "threshold": round(trend_gate * 100.0, 2),
            "critical": round(trend_gate * 100.0, 2),
            "detail": (f"60日等权市场趋势 {trend_pct:+.2f}% (门限 {trend_gate * 100:.0f}%)"
                       + (" → 空腿收缩已触发" if regime_triggered and book_short > 0 else "")
                       + (" → 上行趋势（长多簿形无空腿）" if regime_triggered and book_short <= 0 else "")
                       + ("，未触发" if not regime_triggered else "")),
        },
        {
            "name": "pead_anomaly",
            "label": "PEAD 覆盖异常",
            "value": bool(pead_anomaly),
            "level": "warning" if pead_anomaly else "ok",
            "detail": ("PEAD 因子无覆盖，倾斜未生效" if pead_anomaly
                       else f"PEAD 倾斜正常 ({len(pead.symbols)} 只覆盖)" if pead_ok
                       else "PEAD 未启用"),
        },
    ]

    return {
        "as_of": str(pd.Timestamp.today().date()),
        "last_run": datetime.now().isoformat(timespec="seconds"),
        "last_trading_date": last_date,
        "data_freshness_days": int(freshness_days),
        "equity": {
            "latest": round(float(metrics.get("final_equity", latest_equity)), 2),
            "final_cash": round(float(metrics.get("final_cash", cash or 0.0)), 2),
            "total_return": metrics.get("total_return", 0.0),
            "annualized_return": metrics.get("annualized_return", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "n_days": metrics.get("n_days", 0),
            "n_fills": metrics.get("n_fills", 0),
            "total_commission": metrics.get("total_commission", 0.0),
        },
        "positions": positions_out[:30],  # Top-30 by |weight| for the dashboard
        "equity_curve": equity_curve_out,
        "benchmark": benchmark_out,
        "excess_curve": excess_out,
        "s7_params": s7_params,
        "refreshed": meta,
        "red_lines": red_lines,
    }


def render_shadow_report(status: dict[str, Any], trades: list[dict[str, Any]] | None = None) -> str:
    """Chinese markdown daily report (email body / ``shadow_report.md``).

    ``trades`` (optional) = the latest trading day's fills — the report gains a
    "当日成交" section so PAICC's daily email carries the trade log.
    """
    eq = status.get("equity", {})
    account = status.get("account_name")
    lines: list[str] = [
        "# FQA 影子模式日报" + (f" — {account}" if account else ""),
        "",
        f"- 观察日期（数据截至）: **{status.get('last_trading_date') or status.get('as_of')}**",
        f"- 上次运行: {status.get('last_run')}",
        f"- 数据新鲜度: {status.get('data_freshness_days')} 天",
        "",
        "## 净值",
        "",
        f"- 最新净值: **{eq.get('latest', 0):,.2f}**（现金 {eq.get('final_cash', 0):,.2f}）",
        f"- 累计收益: {eq.get('total_return', 0):.2%}　年化: {eq.get('annualized_return', 0):.2%}",
        f"- Sharpe: {eq.get('sharpe', 0):.2f}　最大回撤: {eq.get('max_drawdown', 0):.2%}",
        f"- 交易日: {eq.get('n_days', 0)}　成交笔数: {eq.get('n_fills', 0)}　累计成本: {eq.get('total_commission', 0):,.2f}",
        "",
        "## 当日成交",
        "",
    ]
    if trades:
        lines.append("| 时间 | 代码 | 方向 | 股数 | 价格 | 佣金 | 金额 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for t in trades[:50]:
            lines.append(
                f"| {t.get('date', '')} | {t.get('symbol', '')} | {t.get('side', '')} | "
                f"{float(t.get('shares', 0)):+,.0f} | {float(t.get('price', 0)):.2f} | "
                f"{float(t.get('commission', 0)):.2f} | {float(t.get('notional', 0)):,.0f} |"
            )
        if len(trades) > 50:
            lines.append(f"（另有 {len(trades) - 50} 笔，详见账本）")
    else:
        lines.append("（今日无成交）")
    lines += [
        "",
        "## 当日目标持仓 (Top 10)",
        "",
    ]
    pos = status.get("positions", [])
    if pos:
        lines.append("| 代码 | 方向 | 权重 |")
        lines.append("| --- | --- | --- |")
        for p in pos[:10]:
            lines.append(f"| {p['symbol']} | {p['side']} | {p['weight']:.3%} |")
    else:
        lines.append("（无持仓）")
    lines += [
        "",
        "## §7 当前参数",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
    ]
    s7 = status.get("s7_params", {})
    lines.append(f"| PEAD 倾斜幅度 | {s7.get('amplitude', 0):.2f} |")
    lines.append(f"| 舆情阈值 z | {s7.get('zscore_threshold', 0):.2f} (cut {s7.get('position_cut', 0):.0%}, freeze {s7.get('freeze_days', 0)}d) |")
    lines.append(f"| 佣金 / 最低 | {s7.get('commission_bps', 0):.2f} bps / {s7.get('min_commission', 0):.2f} |")
    lines.append(f"| 印花税(卖) / 过户 | {s7.get('stamp_tax_sell_bps', 0):.2f} / {s7.get('transfer_fee_bps', 0):.2f} bps |")
    lines.append(f"| 冲击成本 | {s7.get('slippage_bps', 0):.2f} bps |")
    lines += [
        "",
        "## 红线",
        "",
        "| 红线 | 状态 | 值 | 说明 |",
        "| --- | --- | --- | --- |",
    ]
    for rl in status.get("red_lines", []):
        lines.append(f"| {rl.get('label', rl.get('name'))} | {rl.get('level')} | {rl.get('value')} | {rl.get('detail')} |")
    return "\n".join(lines) + "\n"
