"""Portfolio-level tradeability gate for the short-period zoo candidates.

The 32 "reliable" zoo survivors (8 price/volume + 24 alpha158) pass the IC
gate on raw returns — but Phase 8.1's lesson is that short-horizon reversal
bleeds on LIMIT-LOCKED (untradeable) returns in crisis windows, which is why
production enforces ``min_lookback: 60``. This script re-runs the PRODUCTION
risk gate (portfolio on tradeable returns, abs drawdown ≤ 15%, crisis-window
drawdown ≤ 20%) for every survivor, on BOTH the test window and the full
sample (for 2015/2018 crisis coverage) — the honest test of whether the
min_lookback policy is over-conservative or earned.

Usage:  python scripts/gate_short_candidates.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.agents.base_agent import AgentContext  # noqa: E402
from src.cli import (  # noqa: E402
    _make_scores_fn,
    _market_data,
    _market_returns,
    _pipeline,
    _slice_market,
    _tradable,
    load_config,
)
from src.exploration.run import _eval_full  # noqa: E402
from src.factors.code_generator import FactorContext  # noqa: E402


def _load_candidates() -> list[str]:
    out: list[str] = []
    for name in ("scan_results.json", "scan_results_alpha158.json"):
        p = ROOT / "paper" / "factor_zoo" / name
        if not p.is_file():
            continue
        scan = json.loads(p.read_text(encoding="utf-8"))
        for fqa, r in scan["results"].items():
            if r.get("reliable"):
                out.append(fqa)
    return list(dict.fromkeys(out))


def _gate_window(market, pipeline, cfg, fqa: str) -> dict:
    long = market.long
    fctx = FactorContext(long)
    scores = _make_scores_fn(fctx)(fqa)
    forward = market.forward_returns
    tradable = _tradable(market)
    context = AgentContext(
        as_of=long.index.get_level_values(0).max(),
        symbols=long.index.get_level_values(1).unique().tolist(),
        config=cfg,
        data=long,
    )
    metrics = pipeline["eval_agent"].evaluate(
        context, scores, forward, n_trials=1, forward_tradable=tradable, benchmark=None
    )
    risk = pipeline["risk_agent"].validate(
        context, scores, forward, metrics, n_trials=1,
        market_returns=_market_returns(market),
    )
    return {
        "verdict": metrics.get("verdict"),
        "sharpe": metrics.get("sharpe"),
        "max_drawdown": metrics.get("max_drawdown"),
        "risk_passed": bool(risk.get("passed")) if isinstance(risk, dict) else False,
        "risk": {k: v for k, v in risk.items() if k != "checks"} if isinstance(risk, dict) else risk,
    }


def main() -> int:
    cfg = load_config()
    t0 = time.time()
    market = _market_data(cfg, seed=1)
    print(f"market loaded [{time.time()-t0:.0f}s]", flush=True)
    pipeline = _pipeline(cfg)

    candidates = _load_candidates()
    print(f"candidates: {len(candidates)}", flush=True)

    test_slice = _slice_market(market, "2022-01-01", "2025-12-31")

    results = {}
    for i, fqa in enumerate(candidates):
        t1 = time.time()
        entry = {"fqa": fqa}
        try:
            entry["test"] = _gate_window(test_slice, pipeline, cfg, fqa)
        except Exception as exc:  # noqa: BLE001
            entry["test"] = {"error": f"{type(exc).__name__}: {str(exc)[:100]}"}
        try:
            entry["full"] = _gate_window(market, pipeline, cfg, fqa)
        except Exception as exc:  # noqa: BLE001
            entry["full"] = {"error": f"{type(exc).__name__}: {str(exc)[:100]}"}
        results[fqa] = entry
        passed = entry.get("test", {}).get("risk_passed") and entry.get("full", {}).get("risk_passed")
        print(f"[{i+1}/{len(candidates)}] risk_gate={'PASS' if passed else 'FAIL'} "
              f"test_sharpe={entry.get('test', {}).get('sharpe')} "
              f"full_dd={entry.get('full', {}).get('max_drawdown')} "
              f"[{time.time()-t1:.0f}s] :: {fqa[:70]}", flush=True)

    out = ROOT / "paper" / "factor_zoo" / "gate_short_candidates.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    n_pass = sum(
        1 for r in results.values()
        if r.get("test", {}).get("risk_passed") and r.get("full", {}).get("risk_passed")
    )
    print(f"\ngate result: {n_pass}/{len(candidates)} pass the production risk gate "
          f"(tradeable returns, abs_dd<=15%, crisis_dd<=20%)", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
