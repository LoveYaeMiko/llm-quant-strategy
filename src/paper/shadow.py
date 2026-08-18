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
def resolve_shadow_universe(cfg) -> list[str]:
    """HS300 constituents (``data/universe/hs300.json``) — the shadow book universe."""
    from ..config import Config

    universe_dir = Path(str(cfg.get("data.universe_dir", "data/universe")))
    path = universe_dir / "hs300.json"
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
    history — the endpoint has no date param). Heavy (~15-20 min full HS300)."""
    from ..sentiment.ingestion import ReportIngestor

    report_dir = str(cfg.get("sentiment", {}).get("report_dir", "data/reports"))
    ingestor = ReportIngestor(report_dir)
    counts = ingestor.collect(symbols, pause=0.2, force=True)
    return {"symbols": len(counts), "rows": sum(counts.values())}


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

    # positions (shares -> weights) at the latest recorded day
    positions_out: list[dict[str, Any]] = []
    latest_equity = float(eq.iloc[-1]) if len(eq) else float(pcfg.get("initial_cash", 100_000.0))
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
            positions_out.append(
                {"symbol": sym, "shares": round(float(shares), 0),
                 "weight": round(float(weight), 6),
                 "side": "long" if shares > 0 else "short"}
            )
    positions_out.sort(key=lambda p: -abs(p["weight"]))

    # --- four canonical red lines -------------------------------------------
    fills_df = ledger.fills()
    real = real_cost_model(cfg)
    cost_dev = compute_cost_deviation(fills_df, real)

    # short-leg imbalance: |long_gross - short_gross| / gross
    short_dev = 0.0
    if last_date is not None and positions:
        try:
            close = market.price_panel.loc[pd.Timestamp(last_date)]
        except KeyError:
            close = market.price_panel.iloc[-1]
        long_gross = sum(s * float(close.get(sym, 0.0)) for sym, s in positions.items() if s > 0)
        short_gross = sum(-s * float(close.get(sym, 0.0)) for sym, s in positions.items() if s < 0)
        gross = long_gross + short_gross
        short_dev = 0.0 if gross <= 0 else abs(long_gross - short_gross) / gross * 100.0

    # regime switch: current 60-day equal-weight trend vs gate
    trend_gate = float(acfg.get("trend_gate", 0.03))
    trend_pct = _market_trend_pct(market.price_panel, int(acfg.get("trend_days", 60)))
    regime_triggered = trend_pct > trend_gate * 100.0

    # pead anomaly: tilt expected but no valid PEAD data
    pead = overlays.get("pead")
    pead_ok = pead is not None and len(getattr(pead, "symbols", []) or []) > 0
    pead_anomaly = (not pead_ok) and bool(cfg.get("pead", {}).get("enabled", True))

    red_lines: list[dict[str, Any]] = [
        {
            "name": "cost_deviation",
            "label": "成本模型偏差",
            "value": cost_dev["deviation_pct"],
            "level": _threshold_level(abs(cost_dev["deviation_pct"]), 10.0, 20.0),
            "threshold": 10.0,
            "critical": 20.0,
            "detail": (f"固定成本 {cost_dev['current_total']:.2f} vs 真实 {cost_dev['real_total']:.2f} "
                       f"(累计偏差 {cost_dev['deviation_pct']:+.1f}%)"),
        },
        {
            "name": "short_leg_deviation",
            "label": "多空敞口失衡",
            "value": round(short_dev, 2),
            "level": _threshold_level(short_dev, 10.0, 20.0),
            "threshold": 10.0,
            "critical": 20.0,
            "detail": f"多头/空头总敞口失衡 {short_dev:.1f}%",
        },
        {
            "name": "regime_switch",
            "label": "趋势切换",
            "value": round(trend_pct, 2),
            "level": "warning" if regime_triggered else "ok",
            "detail": (f"60日等权市场趋势 {trend_pct:+.2f}% (门限 {trend_gate * 100:.0f}%)"
                       + (" → 空腿收缩已触发" if regime_triggered else "，未触发")),
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
        "s7_params": s7_params,
        "refreshed": meta,
        "red_lines": red_lines,
    }


def render_shadow_report(status: dict[str, Any]) -> str:
    """Chinese markdown daily report (email body / ``shadow_report.md``)."""
    eq = status.get("equity", {})
    lines: list[str] = [
        "# FQA 影子模式日报",
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
