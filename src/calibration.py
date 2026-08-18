"""§7 三项回校 — PEAD 倾斜幅度 / 舆情阈值 / 交易成本模型.

The three knobs tuned on the *accumulated point-in-time data* once the shadow
mode has run long enough:

* **成本模型** — recompute the ledger's fills under the real A-share cost
  structure (stamp tax / transfer fee / commission min 5 元) and compare to the
  fixed ``paper`` model; the recommendation is simply the real structure (the
  ``slippage_bps`` impact term stays a fixed estimate — it needs order-book data
  to calibrate, which the free EOD feeds cannot provide).
* **PEAD 倾斜幅度** — sweep ``seasonal_tilt.amplitude`` over ``amplitude_grid``,
  re-running the execution-aware ``PaperRunner`` (alpha + tilt, risk off) on the
  ``window_start``..``window_end`` sample; pick the amplitude with the best
  Sharpe (max-drawdown reported alongside for the risk trade-off).
* **舆情阈值** — sweep ``risk_overlay.zscore_threshold`` × ``freeze_days`` over
  ``zscore_grid`` × ``freeze_grid`` (alpha + risk, tilt off); pick the best
  Sharpe.

:func:`apply_to_config` writes the recommendations back to
``configs/master_config.yaml`` by **line edit** (only the value on the key line
changes; keys, indentation and trailing comments are preserved — no full-file
``yaml.dump`` that would strip the extensive hand-written comments).
"""

from __future__ import annotations

import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ROOT

# Top-level section pattern: a YAML mapping key at column 0.
_SECTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # render integral floats without the trailing `.0` (5.0 -> "5")
        return str(int(value)) if value.is_integer() else str(round(value, 6))
    if isinstance(value, int):
        return str(value)
    return str(value)


def _same_value(old: str, value: Any) -> bool:
    """True when the on-disk ``old`` already equals ``value`` (numeric-tolerant).

    Compares numerically when both sides parse as numbers so a formatting-only
    difference (``2.0`` vs ``2.00``, or ``0.20`` vs ``0.2``) is not rewritten.
    """
    try:
        return float(old) == float(value)
    except (TypeError, ValueError):
        return str(old) == str(value)


# --------------------------------------------------------------------------- #
# sweep helper
# --------------------------------------------------------------------------- #
def _run_sweep(cfg, market, symbols, portfolio) -> dict[str, Any]:
    """Run ``portfolio`` through the execution-aware runner on a fresh ledger."""
    from .paper.ledger import PaperLedger
    from .paper.runner import PaperRunner
    from .paper.shadow import paper_runner_kwargs

    kw = paper_runner_kwargs(cfg)
    with tempfile.TemporaryDirectory() as td:
        ledger = PaperLedger(Path(td) / "sweep.sqlite")
        runner = PaperRunner(portfolio, market, ledger, symbols=symbols, seed=0, **kw)
        result = runner.run()
        metrics = result.get("metrics", {})
        ledger.close()
    return metrics


# --------------------------------------------------------------------------- #
# (c) cost model
# --------------------------------------------------------------------------- #
def calibrate_cost(
    fills: pd.DataFrame,
    current_cost: dict[str, float],
    real_cost: dict[str, float],
) -> dict[str, Any]:
    """Recompute the ledger's fills under the real cost model vs the fixed one."""
    from .paper.shadow import compute_cost_deviation

    dev = compute_cost_deviation(fills, real_cost)
    return {
        "current": current_cost,
        "recommended": {
            "commission_bps": real_cost["commission_bps"],
            "min_commission": real_cost["min_commission"],
            "stamp_tax_sell_bps": real_cost["stamp_tax_sell_bps"],
            "transfer_fee_bps": real_cost["transfer_fee_bps"],
            "slippage_bps": real_cost["slippage_bps"],
        },
        "n_fills": int(len(fills)),
        "current_total": dev["current_total"],
        "real_total": dev["real_total"],
        "deviation_pct": dev["deviation_pct"],
        "note": ("佣金/印花/过户取自 A 股真实费率；slippage 为冲击成本估计，"
                 "无法由 EOD 成交数据校准，保持不变。"),
    }


# --------------------------------------------------------------------------- #
# (a) PEAD amplitude
# --------------------------------------------------------------------------- #
def calibrate_amplitude(
    cfg,
    market,
    symbols: list[str],
    alpha,
    pead,
    grid: list[float],
) -> dict[str, Any]:
    from .portfolio.layer_integration import ThreeLayerPortfolio
    from .portfolio.seasonal_tilt import PEADSeasonalTilt

    tcfg = cfg.section("seasonal_tilt")
    min_weight = float(tcfg.get("min_weight", 0.015))
    months = tuple(int(m) for m in tcfg.get("months", [1, 2, 4, 8, 10]))
    current = float(tcfg.get("amplitude", 0.20))

    results: list[dict[str, Any]] = []
    for a in grid:
        tilt = PEADSeasonalTilt(pead, universe=symbols, amplitude=float(a),
                                min_weight=min_weight, months=months)
        m = _run_sweep(cfg, market, symbols, ThreeLayerPortfolio(alpha, tilt=tilt, risk=None))
        results.append({
            "amplitude": float(a),
            "sharpe": round(float(m.get("sharpe", 0.0)), 4),
            "max_drawdown": round(float(m.get("max_drawdown", 0.0)), 4),
            "total_return": round(float(m.get("total_return", 0.0)), 4),
            "annualized_return": round(float(m.get("annualized_return", 0.0)), 4),
        })

    best = max(results, key=lambda r: r["sharpe"]) if results else None
    return {
        "current": current,
        "grid": [float(x) for x in grid],
        "results": results,
        "best": best,
        "recommended": float(best["amplitude"]) if best else current,
    }


# --------------------------------------------------------------------------- #
# (b) sentiment thresholds
# --------------------------------------------------------------------------- #
def calibrate_sentiment(
    cfg,
    market,
    symbols: list[str],
    alpha,
    sig: pd.Series,
    zscore_grid: list[float],
    freeze_grid: list[int],
) -> dict[str, Any]:
    from .portfolio.layer_integration import ThreeLayerPortfolio
    from .portfolio.risk_overlay import SentimentRiskOverlay

    rcfg = cfg.section("risk_overlay")
    position_cut = float(rcfg.get("position_cut", 0.50))
    min_samples = int(rcfg.get("min_trigger_samples", 20))
    current = {
        "zscore_threshold": float(rcfg.get("zscore_threshold", -2.5)),
        "freeze_days": int(rcfg.get("freeze_days", 5)),
    }

    results: list[dict[str, Any]] = []
    for z in zscore_grid:
        for f in freeze_grid:
            risk = SentimentRiskOverlay(
                sig, zscore_threshold=float(z), position_cut=position_cut,
                freeze_days=int(f), min_trigger_samples=min_samples,
            )
            m = _run_sweep(cfg, market, symbols, ThreeLayerPortfolio(alpha, tilt=None, risk=risk))
            results.append({
                "zscore_threshold": float(z),
                "freeze_days": int(f),
                "sharpe": round(float(m.get("sharpe", 0.0)), 4),
                "max_drawdown": round(float(m.get("max_drawdown", 0.0)), 4),
                "total_return": round(float(m.get("total_return", 0.0)), 4),
            })

    best = max(results, key=lambda r: r["sharpe"]) if results else None
    return {
        "current": current,
        "zscore_grid": [float(x) for x in zscore_grid],
        "freeze_grid": [int(x) for x in freeze_grid],
        "results": results,
        "best": best,
        "recommended": {"zscore_threshold": float(best["zscore_threshold"]),
                        "freeze_days": int(best["freeze_days"])} if best else dict(current),
    }


# --------------------------------------------------------------------------- #
# auto-apply (line-edit master_config.yaml)
# --------------------------------------------------------------------------- #
def apply_to_config(recommendations: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Write dotted-path -> value recommendations into ``master_config.yaml``.

    Only the value on each key line changes; indentation and trailing comments
    are preserved. Returns ``{changed: {path: {old, new}}, unchanged: [...]}``.
    """
    path = Path(path or ROOT / "configs" / "master_config.yaml")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    # group recommendations by top-level section
    grouped: dict[str, list[tuple[str, Any]]] = {}
    for dotted, value in recommendations.items():
        section, _, key = dotted.partition(".")
        grouped.setdefault(section, []).append((key, value))

    # top-level section -> (start, end) line range
    section_starts: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        m = _SECTION_RE.match(line)
        if m:
            section_starts.append((m.group(1), i))
    ranges: dict[str, tuple[int, int]] = {}
    for idx, (name, start) in enumerate(section_starts):
        end = section_starts[idx + 1][1] if idx + 1 < len(section_starts) else len(lines)
        ranges[name] = (start, end)

    changed: dict[str, dict[str, str]] = {}
    unchanged: list[str] = []
    for section, edits in grouped.items():
        rng = ranges.get(section)
        if rng is None:
            unchanged.append(section)
            continue
        start, end = rng
        for key, value in edits:
            pattern = re.compile(
                r"^(\s*" + re.escape(key) + r"\s*:\s*)([^#]*?)(\s*(?:#.*)?)$"
            )
            found = False
            for i in range(start, end):
                m = pattern.match(lines[i].rstrip("\n"))
                if not m:
                    continue
                old = m.group(2).strip()
                new = _fmt(value)
                if _same_value(old, value):
                    # value already matches (up to formatting) — leave the line
                    # untouched so an unchanged run doesn't churn the file.
                    found = True
                    break
                lines[i] = m.group(1) + new + m.group(3) + ("\n" if lines[i].endswith("\n") else "")
                changed[f"{section}.{key}"] = {"old": old, "new": new}
                found = True
                break
            if not found:
                unchanged.append(f"{section}.{key}")

    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return {"changed": changed, "unchanged": unchanged}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def calibrate_s7(
    cfg,
    market,
    symbols: list[str],
    alpha,
    pead,
    sig: pd.Series,
    fills: pd.DataFrame,
    *,
    auto_apply: bool | None = None,
) -> dict[str, Any]:
    """Run all three §7 calibrations and (optionally) write the results back."""
    from .paper.shadow import real_cost_model

    s7 = cfg.section("s7_calibration")
    auto_apply = s7.get("auto_apply", True) if auto_apply is None else auto_apply

    # cost
    pcfg = cfg.section("paper")
    current_cost = {
        "commission_bps": float(pcfg.get("commission_bps", 5.0)),
        "min_commission": float(pcfg.get("min_commission", 1.0)),
        "stamp_tax_sell_bps": float(pcfg.get("stamp_tax_sell_bps", 0.0)),
        "transfer_fee_bps": float(pcfg.get("transfer_fee_bps", 0.0)),
        "slippage_bps": float(pcfg.get("slippage_bps", 2.0)),
    }
    cost = calibrate_cost(fills, current_cost, real_cost_model(cfg))

    # amplitude
    if pead is None:
        amplitude = {
            "current": float(cfg.section("seasonal_tilt").get("amplitude", 0.20)),
            "grid": [float(x) for x in s7.get("amplitude_grid", [0.10, 0.15, 0.20, 0.25, 0.30])],
            "results": [], "best": None,
            "recommended": float(cfg.section("seasonal_tilt").get("amplitude", 0.20)),
            "note": "PEAD 因子数据缺失，幅度未校准（保持现值）",
        }
    else:
        amplitude = calibrate_amplitude(
            cfg, market, symbols, alpha, pead,
            [float(x) for x in s7.get("amplitude_grid", [0.10, 0.15, 0.20, 0.25, 0.30])],
        )

    # sentiment
    if sig is None:
        sentiment = {
            "current": {"zscore_threshold": float(cfg.section("risk_overlay").get("zscore_threshold", -2.5)),
                        "freeze_days": int(cfg.section("risk_overlay").get("freeze_days", 5))},
            "zscore_grid": [float(x) for x in s7.get("zscore_grid", [-2.0, -2.5, -3.0])],
            "freeze_grid": [int(x) for x in s7.get("freeze_grid", [3, 5, 7])],
            "results": [], "best": None,
            "recommended": {"zscore_threshold": float(cfg.section("risk_overlay").get("zscore_threshold", -2.5)),
                            "freeze_days": int(cfg.section("risk_overlay").get("freeze_days", 5))},
            "note": "舆情信号数据缺失，阈值未校准（保持现值）",
        }
    else:
        sentiment = calibrate_sentiment(
            cfg, market, symbols, alpha, sig,
            [float(x) for x in s7.get("zscore_grid", [-2.0, -2.5, -3.0])],
            [int(x) for x in s7.get("freeze_grid", [3, 5, 7])],
        )

    recommendations: dict[str, Any] = {
        "seasonal_tilt.amplitude": amplitude["recommended"],
        "risk_overlay.zscore_threshold": sentiment["recommended"]["zscore_threshold"],
        "risk_overlay.freeze_days": sentiment["recommended"]["freeze_days"],
        "paper.commission_bps": cost["recommended"]["commission_bps"],
        "paper.min_commission": cost["recommended"]["min_commission"],
        "paper.stamp_tax_sell_bps": cost["recommended"]["stamp_tax_sell_bps"],
        "paper.transfer_fee_bps": cost["recommended"]["transfer_fee_bps"],
        "paper.slippage_bps": cost["recommended"]["slippage_bps"],
    }

    applied: dict[str, Any] = {}
    if auto_apply:
        applied = apply_to_config(recommendations)

    return {
        "as_of": str(pd.Timestamp.today().date()),
        "last_run": datetime.now().isoformat(timespec="seconds"),
        "window": {
            "start": str(s7.get("window_start", "2020-01-01")),
            "end": str(s7.get("window_end", "2025-12-31")),
        },
        "cost": cost,
        "amplitude": amplitude,
        "sentiment": sentiment,
        "recommendations": recommendations,
        "auto_apply": bool(auto_apply),
        "applied": applied,
    }


def render_calibration_report(result: dict[str, Any]) -> str:
    """Chinese markdown report for the §7 calibration results."""
    amp = result.get("amplitude", {})
    sent = result.get("sentiment", {})
    cost = result.get("cost", {})
    lines: list[str] = [
        "# FQA §7 三项回校报告",
        "",
        f"- 回校窗口: {result.get('window', {}).get('start')} ~ {result.get('window', {}).get('end')}",
        f"- 运行时间: {result.get('last_run')}",
        f"- 自动写回: {'是' if result.get('auto_apply') else '否'}",
        "",
        "## (a) PEAD 倾斜幅度",
        "",
    ]
    amp_note = amp.get("note")
    if amp_note:
        lines.append(f"> {amp_note}")
    else:
        lines.append(f"- 当前幅度: {amp.get('current')} → 推荐: **{amp.get('recommended')}**")
        lines.append("")
        lines.append("| 幅度 | Sharpe | 最大回撤 | 累计收益 |")
        lines.append("| --- | --- | --- | --- |")
        for r in amp.get("results", []):
            mark = " ← 最优" if r["amplitude"] == amp.get("recommended") else ""
            lines.append(f"| {r['amplitude']:.2f} | {r['sharpe']:.3f} | {r['max_drawdown']:.2%} | {r['total_return']:.2%}{mark} |")

    lines += ["", "## (b) 舆情阈值", ""]
    sent_note = sent.get("note")
    if sent_note:
        lines.append(f"> {sent_note}")
    else:
        rec = sent.get("recommended", {})
        lines.append(f"- 当前: z={sent.get('current', {}).get('zscore_threshold')}, freeze={sent.get('current', {}).get('freeze_days')} → "
                     f"推荐: **z={rec.get('zscore_threshold')}, freeze={rec.get('freeze_days')}**")
        lines.append("")
        lines.append("| z | freeze | Sharpe | 最大回撤 | 累计收益 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in sent.get("results", []):
            best = (r["zscore_threshold"] == rec.get("zscore_threshold")
                    and r["freeze_days"] == rec.get("freeze_days"))
            mark = " ← 最优" if best else ""
            lines.append(f"| {r['zscore_threshold']:.1f} | {r['freeze_days']} | {r['sharpe']:.3f} | "
                         f"{r['max_drawdown']:.2%} | {r['total_return']:.2%}{mark} |")

    lines += ["", "## (c) 交易成本模型", ""]
    lines.append(f"- 成交笔数: {cost.get('n_fills', 0)}")
    lines.append(f"- 固定模型累计成本: {cost.get('current_total', 0):,.2f} vs 真实: {cost.get('real_total', 0):,.2f} "
                 f"（偏差 {cost.get('deviation_pct', 0):+.1f}%）")
    rec = cost.get("recommended", {})
    lines.append(f"- 推荐佣金 {rec.get('commission_bps')} bps（最低 {rec.get('min_commission')}）、"
                 f"印花税(卖) {rec.get('stamp_tax_sell_bps')} bps、过户 {rec.get('transfer_fee_bps')} bps、"
                 f"冲击 {rec.get('slippage_bps')} bps")
    lines.append(f"- 说明: {cost.get('note', '')}")
    lines += [
        "",
        "## 写回结果",
        "",
    ]
    applied = result.get("applied", {})
    changed = applied.get("changed", {})
    if changed:
        lines.append("| 配置项 | 旧值 | 新值 |")
        lines.append("| --- | --- | --- |")
        for path, kv in changed.items():
            lines.append(f"| {path} | {kv.get('old')} | {kv.get('new')} |")
    else:
        lines.append("（未写回任何配置项）")
    return "\n".join(lines) + "\n"
