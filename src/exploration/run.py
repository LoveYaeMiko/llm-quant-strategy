"""Exploration track orchestrator + CLI entry.

Runs a NON-INVASIVE, parallel research pass over the new exploration-tier
operators (+ optional re-opened LLM hypotheses). It loads the production data
pipeline, evaluates every candidate walk-forward across train/val/test, applies
the LOOSE exploration gate (risk acceptance), then flags which survivors also
pass the HARD production gates for potential promotion.

The production pool (``outputs/factors.json``) is never touched here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..agents.base_agent import AgentContext
from ..cli import (
    _make_scores_fn,
    _market_data,
    _market_returns,
    _pipeline,
    _slice_market,
    _tradable,
)
from ..config import Config, load_config
from ..factors.code_generator import FactorContext
from .journal import render_report, write_journal
from .sweep import build_sweep_formulas, exploration_verdict, is_reliable


def _eval_full(market_slice: Any, pipeline: dict, prod_cfg: Config, formula: str) -> dict:
    """Evaluate one formula on one window: raw metrics + production risk report."""
    long = market_slice.long
    fctx = FactorContext(long)
    scores_fn = _make_scores_fn(fctx)
    try:
        scores = scores_fn(formula)
    except Exception as exc:  # noqa: BLE001 — a broken formula must not kill the run
        return {"error": str(exc), "metrics": {"verdict": "error"}, "risk": {"passed": False}}
    if not hasattr(scores, "groupby"):  # a literal scalar — not a tradable signal
        return {"error": "scalar", "metrics": {"verdict": "error"}, "risk": {"passed": False}}
    forward = market_slice.forward_returns
    tradable = _tradable(market_slice)
    context = AgentContext(
        as_of=long.index.get_level_values(0).max(),
        symbols=long.index.get_level_values(1).unique().tolist(),
        config=prod_cfg,
        data=long,
    )
    metrics = pipeline["eval_agent"].evaluate(
        context, scores, forward, n_trials=1, forward_tradable=tradable, benchmark=None
    )
    risk: dict = {"passed": False, "checks": {}}
    if metrics.get("verdict") in ("keep", "good"):
        risk = pipeline["risk_agent"].validate(
            context, scores, forward, metrics, n_trials=1,
            market_returns=_market_returns(market_slice),
        )
    return {"metrics": metrics, "risk": risk}


def _pick(metrics: dict, *keys: str) -> dict:
    return {k: metrics.get(k, 0.0) for k in keys}


def _load_exploration_candidates(pipeline: dict, prod_cfg: Config, ecfg: dict, args: Any) -> list[dict]:
    """Sweep candidates (deterministic) + optional controlled-LLM candidates."""
    candidates: list[dict] = []
    sweep_cfg = ecfg.get("sweep", {}) or {}
    if sweep_cfg.get("enabled", True) and getattr(args, "sweep", True):
        cands = build_sweep_formulas(sweep_cfg)
        if getattr(args, "operators", None):
            want = set(args.operators)
            cands = [c for c in cands if c["operator"] in want]
        if getattr(args, "limit", None):
            cands = cands[: args.limit]
        candidates.extend(cands)

    if getattr(args, "llm", True) and pipeline.get("backend") is not None:
        from .llm import propose_formulas

        llm_cfg = ecfg.get("llm", {}) or {}
        n = int(llm_cfg.get("hypotheses_per_round", 16))
        rounds = int(llm_cfg.get("rounds", 1))
        min_lb = int(ecfg.get("min_lookback", 20))
        max_lb = int(ecfg.get("max_lookback", 240))
        backend = pipeline["backend"]
        for r in range(rounds):
            print(f"[llm] round {r + 1}/{rounds}: proposing {n} hypotheses...", flush=True)
            formulas = propose_formulas(
                backend, n,
                temperature=float(llm_cfg.get("temperature_hypotheses", 0.8)),
                min_lookback=min_lb, max_lookback=max_lb,
            )
            print(f"[llm] round {r + 1}/{rounds}: {len(formulas)} formulas accepted", flush=True)
            for formula in formulas:
                candidates.append(
                    {
                        "name": f"llm_{len(candidates)}",
                        "formula": formula,
                        "meaning": "controlled-LLM hypothesis",
                        "operator": "",
                        "source": "llm",
                    }
                )
    return candidates


def run_exploration(args: Any) -> int:
    prod = load_config()  # production config (master + thresholds + routing), untouched
    exp = load_config(Path(__file__).resolve().parent.parent.parent / "configs" / "exploration.yaml")
    ecfg = exp.get("exploration", {}) or {}

    pipeline = _pipeline(prod)
    market = _market_data(prod, seed=args.seed, symbols=args.symbols)

    windows = list(ecfg.get("windows", ["train", "val", "test"]))
    sliced: dict[str, Any] = {}
    for w in windows:
        start = prod.get(f"research.{w}_start")
        end = prod.get(f"research.{w}_end")
        sliced[w] = _slice_market(market, start, end)

    candidates = _load_exploration_candidates(pipeline, prod, ecfg, args)
    gates = ecfg.get("gates", {}) or {}

    print(
        f"exploration: {len(candidates)} candidates x {len(windows)} windows | "
        f"llm={pipeline['backend'].model if pipeline['backend'] else 'OFFLINE'} | "
        f"gates={gates}",
        flush=True,
    )

    results: list[dict] = []
    for idx, cand in enumerate(candidates):
        # Log BEFORE evaluating so a native crash (segfault) leaves the culprit
        # formula in the output — a segfault can't be caught by a try/except.
        print(f"[{idx + 1}/{len(candidates)}] {cand['formula']}", flush=True)
        row: dict = dict(cand)
        row["windows"] = {}
        for w in windows:
            ev = _eval_full(sliced[w], pipeline, prod, cand["formula"])
            metrics = ev["metrics"]
            row["windows"][w] = {
                **_pick(metrics, "rank_ic", "icir", "ic", "sharpe", "max_drawdown", "n_days"),
                "exploration": exploration_verdict(metrics, gates),
                "production_verdict": metrics.get("verdict", "?"),
                "production_risk_passed": bool(ev["risk"].get("passed", False)),
                "error": ev.get("error"),
            }
        train = row["windows"].get("train", {})
        test = row["windows"].get("test", {})
        row["reliable"] = is_reliable(train, test, gates)
        row["promotable"] = bool(
            row["reliable"]
            and test.get("production_verdict") in ("keep", "good")
            and test.get("production_risk_passed")
        )
        results.append(row)

    survivors = [r for r in results if r["reliable"]]
    promoted = [r for r in results if r["promotable"]]

    # -- persist (outputs/exploration/ only — production pool untouched) -----
    out_dir = Path(ecfg.get("output_dir", "outputs/exploration"))
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = Path(ecfg.get("journal", "outputs/exploration/journal.json"))
    survivors_path = Path(ecfg.get("survivors", "outputs/exploration/survivors.json"))
    promoted_path = Path(ecfg.get("promoted", "outputs/exploration/promoted.json"))
    report_path = Path(ecfg.get("report", "outputs/exploration/report.md"))

    write_journal(journal_path, {"results": results, "survivors": survivors, "promoted": promoted})
    write_journal(survivors_path, survivors)
    write_journal(promoted_path, promoted)
    report_path.write_text(render_report(results, survivors, promoted), encoding="utf-8")

    # -- summary -------------------------------------------------------------
    print(f"\n=== exploration summary ===")
    print(f"candidates: {len(results)}  reliable: {len(survivors)}  promotable: {len(promoted)}")
    for r in promoted:
        t = r["windows"]["test"]
        print(
            f"  [PROMOTE] {r['formula']}  rank_ic_test={t['rank_ic']:.4f} "
            f"dd={t['max_drawdown']:.2%} sharpe={t['sharpe']:.2f}"
        )
    if not promoted and survivors:
        print("  (no candidate passed the hard production gates; see survivors.json)")
    print(f"\njournal: {journal_path}")
    print(f"report:  {report_path}")
    return 0


def cmd_explore(args: Any) -> int:
    return run_exploration(args)
