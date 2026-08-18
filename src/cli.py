"""llm-quant — command-line entry point for the whole pipeline.

Subcommands:
    mine     Run the multi-agent factor-mining loop (signal -> code -> eval ->
             risk), persist the accepted pool, and emit an audit record.
    evolve   One EvoQuant self-evolution round around a base factor.
    backtest Backtest a single formula (or the accepted pool) on PIT data.
    export   Compile a formula into the deterministic online artifact (JSON).
    verify   Run the blueprint verification checklist (PIT / FinCAD / diversity /
             cost).
    sentiment-ingest  Phase 9.1 live news (real-time forward, HS300).
    report-ingest     Phase 9.1 research-report history (backfillable).
    sentiment-factor  Phase 9.1 gate: report-title sentiment IC, 2022-2025.

Every subcommand works fully offline when no ``DEEPSEEK_API_KEY`` is set (the
agents fall back to deterministic implementations); with the key present the
DeepSeek ``deepseek-v4-flash`` backend is wired through the FinCAD wrapper and
its usage is recorded against the $500/month budget gate.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

from .agents.base_agent import AgentContext
from .agents.code_agent import CodeAgent
from .agents.debate_agent import DebateAgent
from .agents.dynamic_router import DynamicRouter
from .agents.eval_agent import EvalAgent
from .agents.risk_agent import RiskAgent
from .agents.signal_agent import SignalAgent
from .audit import ExperimentAuditor
from .backtest.engine import BacktestConfig, PointInTimeBacktest
from .backtest.limit_locked import tradeable_forward_returns
from .bias_control.context_decoder import FinCADWrapper
from .checklist import run_all
from .config import Config, load_config
from .cost_tracker import CostTracker
from .data.point_in_time_loader import PointInTimeStore
from .data.synthetic import make_synthetic_market
from .evolver import EvoQuant
from .factors.code_generator import CodeGenerator, FormulaError, default_formula_for, eval_expression
from .factors.memory_manager import MemoryManager
from .factors.semantic_space import SchemaPlan, SemanticSpace
from .llm_client import build_llm_backend
from .online.signal_calculator import compile_factor
from .reporting import build_digest, build_notifier

ROOT = Path(__file__).resolve().parent.parent


def _out_dir() -> Path:
    """Resolved lazily so tests can point outputs elsewhere via env var."""
    return Path(os.environ.get("LLM_QUANT_OUTPUTS", ROOT / "outputs"))


def _market_data(config: Config, seed: int = 1, symbols=None, bound_to_universe: bool = True):
    """Real PIT store when data exists, else the synthetic market (offline).

    Detects data presence via ``store.snapshot("price")`` rather than a query
    probe: under closed-interval bars (ADR-0001) every bar has expired by the
    full-history probe date, so ``query()`` would always come back empty.

    ``symbols`` overrides ``research.universe`` (Q1-B: research runs on the
    bounded universe; ``--symbols`` is the escape hatch for full-A validation).
    The returned bundle also carries an ``audit_store`` with price + universe
    records so the B1-B5 checklist can see the survivorship snapshots.
    """
    url = config.get("data.pit_database_url")
    if not url:
        print("WARNING: PIT_DATABASE_URL not set — falling back to synthetic data", file=sys.stderr)
        return make_synthetic_market(seed=seed)
    try:
        from .data.point_in_time_loader import from_url

        store = from_url(url)
        recs = store.snapshot("price")
    except Exception as exc:
        if url.startswith("postgresql"):
            print(f"ERROR: cannot reach PIT database ({url.split('@')[-1]}): {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        return make_synthetic_market(seed=seed)
    if recs is None or recs.empty:
        return make_synthetic_market(seed=seed)
    if bound_to_universe:
        recs = _bound_to_universe(config, recs, store, symbols)
    market = _market_from_records(recs)
    _attach_audit_store(market, store, recs)
    return _attach_tradable_forward(market, config)


def _bound_to_universe(config: Config, recs: pd.DataFrame, store, symbols) -> pd.DataFrame:
    """Keep only price records inside ``research.universe`` (Q1-B).

    ``symbols`` (an explicit ``--symbols`` list) wins over the config universe;
    an unresolvable universe degrades to the full stored panel with a warning.
    """
    if symbols:
        from .data.schema.symbols import normalize_symbol

        want = {normalize_symbol(s) for s in symbols}
    else:
        name = str(config.get("research.universe", "hs300_500"))
        if name == "all":
            return recs
        try:
            from .data.ingestion.ingestor import resolve_research_universe

            want = set(resolve_research_universe(config, store))
        except FileNotFoundError as exc:
            print(f"WARNING: {exc} — running on the full stored universe", file=sys.stderr)
            return recs
    if not want:
        return recs
    return recs[recs["symbol"].isin(want)]


def _attach_audit_store(market, store, price_recs: pd.DataFrame) -> None:
    """Seed ``market.audit_store`` with price + universe records for the checklist."""
    try:
        uni = store.snapshot("universe")
    except Exception:  # noqa: BLE001 — a loader without universe support is fine
        uni = None
    if uni is not None and not uni.empty:
        from .data.point_in_time_loader import PointInTimeStore

        audit = PointInTimeStore()
        audit.upsert(pd.concat([price_recs, uni], ignore_index=True))
        market.audit_store = audit


def _market_from_records(records: pd.DataFrame):
    """Build a usable SyntheticMarket-like bundle from PIT price records."""
    from .data.synthetic import SyntheticMarket

    store = PointInTimeStore()
    store.upsert(records)
    rec = store.records.copy()
    rec["date"] = pd.to_datetime(rec["valid_from"])
    for col in ("open", "high", "low", "close", "volume"):
        if col not in rec.columns:
            rec[col] = 0.0
    long = rec.set_index(["date", "symbol"])[["open", "high", "low", "close", "volume"]].sort_index()
    close_wide = long["close"].unstack()
    # fill_method=None: a forward return is only valid between two *consecutive*
    # trading days of the same name. The default fill_method='pad' forward-fills
    # close across suspension/resumption gaps and fabricates multi-hundred-percent
    # returns that dominate return-based metrics (Sharpe/maxDD) while rank-IC
    # stays unaffected — the systematic `reject_high_risk` cause in Phase 8.1.
    fwd = close_wide.pct_change(fill_method=None).shift(-1).stack().rename("fwd")
    return SyntheticMarket(
        records=store.records,
        long=long,
        price_panel=close_wide,
        forward_returns=fwd,
        pit_store=store,
        n_symbols=len(close_wide.columns),
        n_days=len(close_wide),
    )


def _attach_tradable_forward(market, config=None) -> SyntheticMarket:
    """Set ``market.forward_returns_tradable`` (LIMIT_DOWN blueprint 方案 B).

    Portfolio Sharpe / max-drawdown must be computed on forward returns with
    price-limit-locked bars masked (a -10% continuation you cannot actually
    transact), while rank IC stays on the raw series. Thresholds come from
    ``evaluation.portfolio``; defaults match the blueprint (0.095 / dynamic).
    Called by ``_market_data`` and ``_slice_market``; the synthetic fallback
    carries no price-limit structure so its tradable series equals raw.
    """
    port = {}
    if config is not None:
        port = dict(config.get("evaluation.portfolio", {}) or {})
    exclude = bool(port.get("exclude_limit_locked", True))
    thr = float(port.get("limit_threshold", getattr(market, "limit_threshold", 0.095)))
    dyn = bool(port.get("dynamic_threshold", getattr(market, "limit_dynamic", True)))
    market.limit_threshold = thr
    market.limit_dynamic = dyn
    market.forward_returns_tradable = (
        tradeable_forward_returns(
            market.forward_returns, market.long, base_threshold=thr, dynamic_threshold=dyn
        )
        if exclude
        else market.forward_returns
    )
    return market


def _tradable(market) -> pd.Series:
    """The portfolio-evaluation forward series for ``market`` (falls back to raw)."""
    t = getattr(market, "forward_returns_tradable", None)
    return t if t is not None else market.forward_returns


# ---------------------------------------------------------------------------
# shared pipeline
# ---------------------------------------------------------------------------


def _pipeline(config: Config):
    """Assemble agents, memory, cost tracker, auditor, backend."""
    cfg = config
    memory = MemoryManager()
    costs = CostTracker(monthly_budget_usd=float(cfg.get("budget.monthly_llm_cost_usd", 500)))
    backend = build_llm_backend(cfg, cost_tracker=costs)
    fincad = FinCADWrapper(backend) if backend is not None else None
    space = SemanticSpace()
    generator = CodeGenerator()

    signal_agent = SignalAgent(
        space=space,
        memory=memory,
        llm=backend,
        config=cfg,
        seed=0,
        rejection_history_path=str(_out_dir() / "rejection_history.json"),
        feedback_enabled=bool(cfg.get("factor_mining.enable_mining_feedback", True)),
        feedback_rounds=int(cfg.get("factor_mining.feedback_rounds", 3)),
    )
    code_agent = CodeAgent(generator=generator, memory=memory, llm=backend, config=cfg)
    eval_agent = EvalAgent(memory=memory, llm=backend, config=cfg)
    risk_agent = RiskAgent(memory=memory, llm=backend, config=cfg, seed=0)
    debate_agent = DebateAgent(llm=backend, config=cfg)
    router = DynamicRouter(
        {"signal": signal_agent, "code": code_agent, "eval": eval_agent, "risk": risk_agent},
        market_state="sideways",
    )
    auditor = ExperimentAuditor()
    return {
        "memory": memory,
        "costs": costs,
        "backend": backend,
        "fincad": fincad,
        "space": space,
        "generator": generator,
        "agents": {"signal": signal_agent, "code": code_agent, "eval": eval_agent, "risk": risk_agent},
        "router": router,
        "auditor": auditor,
        "eval_agent": eval_agent,
        "risk_agent": risk_agent,
        "code_agent": code_agent,
        "debate_agent": debate_agent,
    }


def _make_scores_fn(ctx):
    def fn(formula: str) -> pd.Series:
        return eval_expression(formula, ctx)

    return fn


def _market_returns(market) -> pd.Series:
    """Daily equal-weighted market return series (for regime classification)."""
    # Same fill_method=None as the forward-return builder — no pad-fabricated gaps.
    return market.price_panel.pct_change(fill_method=None).mean(axis=1).dropna()


def _benchmark_returns(market, config) -> pd.Series:
    """Daily benchmark return series for the excess-drawdown gate (§3.3).

    validation_BLUEPRINT wants HS300 (``000300.SH``), but the PIT store is
    stock-only — index *bars* were never ingested (only index constituents, for
    the universe). The resolver therefore prefers the configured index when its
    bars happen to exist, otherwise falls back to the equal-weighted market
    return (``_market_returns``) — the natural benchmark for a long-short neutral
    portfolio on a stock-only store.
    """
    name = str((config.get("risk_management") or {}).get("benchmark", "") or "")
    if name:
        try:
            syms = set(market.pit_store.symbols("price"))
        except Exception:  # noqa: BLE001 — any store hiccup degrades to EW market
            syms = set()
        if name in syms:
            try:
                rec = market.pit_store.history(name, "price")
                closes = rec.set_index("valid_from")["close"].sort_index()
                closes = closes[~closes.index.duplicated(keep="last")]
                ret = closes.pct_change(fill_method=None).dropna()
                if len(ret) > 10:
                    print(f"benchmark: {name} (from PIT store, {len(ret)} bars)")
                    return ret
            except Exception:  # noqa: BLE001
                pass
    bench = _market_returns(market)
    print(
        f"benchmark: equal-weighted market ({len(bench)} days) "
        f"({'configured index ' + name + ' not in store' if name else 'no benchmark configured'})"
    )
    return bench


# ---------------------------------------------------------------------------
# walk-forward window slicing (requirements.md C1-C2)
# ---------------------------------------------------------------------------


def _resolve_window(config: Config, args, default: str = "train") -> tuple:
    """Resolve ``(start, end)`` from ``--window`` / ``--start`` / ``--end``.

    ``--start/--end`` win outright; otherwise the named research window
    (``train``/``val``/``test``/``all``); ``--window all`` returns ``(None, None)``
    meaning the full market.
    """
    if args.start or args.end:
        start = args.start or config.get(f"research.{default}_start")
        end = args.end or config.get(f"research.{default}_end")
        return start, end
    name = args.window
    if name == "all":
        return None, None
    start = config.get(f"research.{name}_start")
    end = config.get(f"research.{name}_end")
    if not start:
        raise ValueError(f"unknown research window {name!r} (train|val|test|all)")
    return start, end


def _slice_market(market, start, end):
    """Rebuild the market limited to ``[start, end]`` for a walk-forward window."""
    if not start and not end:
        return market
    rec = market.pit_store.records.copy()
    rec["date"] = pd.to_datetime(rec["valid_from"])
    if start:
        rec = rec[rec["date"] >= pd.Timestamp(start)]
    if end:
        rec = rec[rec["date"] <= pd.Timestamp(end)]
    if rec.empty:
        print(f"WARNING: window [{start} .. {end}] is empty — using the full market", file=sys.stderr)
        return market
    out = _market_from_records(rec)
    # carry the limit-lock thresholds + recompute the tradable forward on the slice
    out.limit_threshold = getattr(market, "limit_threshold", 0.095)
    out.limit_dynamic = getattr(market, "limit_dynamic", True)
    _attach_tradable_forward(out)
    # carry the price+universe audit store through the slice so B4 still sees the
    # survivorship snapshots inside the window
    if market.audit_store is not None:
        arec = market.audit_store.records.copy()
        arec["date"] = pd.to_datetime(arec["valid_from"])
        if start:
            arec = arec[arec["date"] >= pd.Timestamp(start)]
        if end:
            arec = arec[arec["date"] <= pd.Timestamp(end)]
        from .data.point_in_time_loader import PointInTimeStore

        audit = PointInTimeStore()
        audit.upsert(arec)
        out.audit_store = audit
    return out


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------


def cmd_mine(args) -> int:
    cfg = load_config()
    p = _pipeline(cfg)
    market = _market_data(cfg, seed=args.seed, symbols=args.symbols)
    start, end = _resolve_window(cfg, args, default="train")
    market = _slice_market(market, start, end)
    # the checklist audits price + universe (B4), so prefer the audit store
    store: PointInTimeStore = market.audit_store or market.pit_store
    long = market.long
    forward = market.forward_returns
    context = AgentContext(
        as_of=long.index.get_level_values(0).max(),
        symbols=long.index.get_level_values(1).unique().tolist(),
        config=cfg,
        data=long,
    )

    from .factors.code_generator import FactorContext

    fctx = FactorContext(long)
    scores_fn = _make_scores_fn(fctx)

    record = p["auditor"].begin("mine")
    p["auditor"].snapshot_config(record, cfg)
    p["auditor"].snapshot_routing(record, cfg)
    p["auditor"].set_pit_window(
        record,
        str(long.index.get_level_values(0).min().date()),
        str(long.index.get_level_values(0).max().date()),
        market.n_symbols,
    )
    p["auditor"].set_checklist(record, llm_enabled=p["backend"] is not None)

    out = _out_dir()
    out.mkdir(parents=True, exist_ok=True)
    n_iter = args.iterations if args.iterations is not None else int(
        cfg.get("research.mining.iterations", 3)
    )
    n_hyps = args.hypotheses
    accepted: list[dict] = []
    report_rows: list[dict] = []
    tradable = _tradable(market)  # LIMIT_DOWN blueprint 方案 B: portfolio eval series
    # validation_BLUEPRINT §2.1/§3.2: per-iteration candidate pool = free (LLM,
    # through the code-layer firewall) + combination-template (deterministic
    # dual-factor equal-weight) slots. Defaults free=2, template=2 -> 20 factors
    # across the 5-iteration validation, matching the 6/20 (>=30%) success gate.
    free_slots = int(cfg.get("factor_mining.free_generation_slots", 0) or 0)
    tpl_slots = int(cfg.get("factor_mining.template_slots", 0) or 0)
    slots_configured = bool(free_slots or tpl_slots)
    free_slots = free_slots if slots_configured else n_hyps
    tpl_slots = tpl_slots if slots_configured else 0
    # validation_BLUEPRINT §3.3's excess-drawdown gate was reverted to the absolute
    # 15% gate (2026-08): the excess metric is structurally unpassable for a
    # market-neutral book (a pure-cash position scores excess_dd ≈ 1.03 vs the
    # 0.25 limit). Only thread a benchmark when max_excess_drawdown is explicitly
    # re-enabled in config — the engine then computes excess_max_drawdown and the
    # eval/risk gates gate on it; with the key absent they fall back to the
    # absolute drawdown limits (sharpe.max_drawdown_limit / max_drawdown_in_crisis).
    rm = cfg.get("risk_management") or {}
    benchmark = _benchmark_returns(market, cfg) if rm.get("max_excess_drawdown") is not None else None

    # CLI overrides (LIMIT_DOWN blueprint §6): --enable-feedback /
    # --feedback-rounds / --crisis-test
    signal = p["agents"]["signal"]
    if getattr(args, "enable_feedback", None) is not None:
        signal.feedback_enabled = args.enable_feedback
    if getattr(args, "feedback_rounds", None):
        signal.feedback_rounds = args.feedback_rounds
    if getattr(args, "crisis_test", None):
        crisis = cfg.raw.setdefault("risk_management", {}).setdefault("crisis_test", {})
        crisis["enabled"] = True

    print(
        f"mining: {n_iter} iterations x {free_slots + tpl_slots} candidates "
        f"(free={free_slots}, template={tpl_slots}) | "
        f"llm={'deepseek-v4-flash' if p['backend'] else 'OFFLINE'} | "
        f"feedback={signal.feedback_enabled} | "
        f"limit-locked-excluded={(cfg.get('evaluation.portfolio') or {}).get('exclude_limit_locked', True)}"
    )
    # run6 diagnosis (2026-08-10): ``rejection_history.json`` PERSISTS across
    # cmd_mine invocations (95 entries / 43 unique accumulated from run5+run6).
    # The agent loads it at construction (signal_agent.py) and the template
    # generator blocks every entry against the bounded 24-formula pool — so at
    # run6's start 20/24 pool members were already locked by STALE prior-run
    # rejections, the pool exhausted by iter2, template slots returned 0 in
    # iters 2/4, and the 20-candidate contract degraded to 14 (5/20 = 25%,
    # below the 6/20 >=30% validation gate). Rejection history must be scoped
    # to the current run only: the mining-feedback prompt and the template-pool
    # block both then reflect just this run's rejections (record_rejection
    # rewrites the file on each call, so the on-disk history self-heals too).
    signal.rejection_history = []
    # Accepted formulas accumulate so template slots never re-draw a winner —
    # accepted formulas are NOT in ``rejection_history`` (blueprint §3.2 leak:
    # run4's iter4 template slot re-drew iter0's accepted free formula verbatim,
    # an exact duplicate that failed the diversity checklist). Blocking ALL
    # free-path formulas instead (a previous attempt) exhausted the bounded
    # template pool, so only the small accepted set is added to ``skip``.
    accepted_ever: set[str] = set()
    for it in range(n_iter):
        candidates: list[dict] = []
        # 1. free slots — LLM (or offline semantic-space) proposals, each passed
        #    through the code-layer firewall (sanitize_formula) in CodeAgent.
        for plan in signal.generate_hypotheses(context, n=free_slots):
            gf = p["agents"]["code"].translate(context, plan)
            candidates.append(
                {"gf": gf, "source": "free", "plan": plan}
            )
        # 2. template slots — deterministic combination templates, evaluated as-is.
        #    ``skip`` dedups against the free path (a sanitized free formula can
        #    otherwise coincide with a template — blueprint §3.2 merge-dedup) and
        #    against formulas already accepted in earlier rounds.
        free_formulas = {c["gf"].formula for c in candidates}
        for formula in signal.generate_template_formulas(
            n=tpl_slots, skip=free_formulas | accepted_ever
        ):
            gf = p["generator"].generate(
                formula,
                name=f"combo{abs(hash(formula)) % 10**9:09d}",
                meaning="双因子等权组合模板（validation_BLUEPRINT §3.1）",
                category="combination_template",
            )
            candidates.append({"gf": gf, "source": "combination_template", "plan": None})

        for cand in candidates:
            gf = cand["gf"]
            plan = cand["plan"]
            try:
                scores = scores_fn(gf.formula)
                metrics = p["eval_agent"].evaluate(
                    context, scores, forward, n_trials=args.trials,
                    forward_tradable=tradable, benchmark=benchmark,
                )
            except Exception as exc:
                metrics = {"verdict": "error", "error": str(exc)}
                scores = None
            risk = {"passed": False, "checks": {}}
            if scores is not None and metrics.get("verdict") in ("keep", "good"):
                risk = p["risk_agent"].validate(
                    context, scores, forward, metrics,
                    n_trials=args.trials,
                    market_returns=_market_returns(market),
                )
            passed = risk["passed"]
            # 辩论评审 (Bull/Bear adversarial review) — config-gated so the default
            # acceptance behaviour is unchanged. When enabled, a factor that the
            # bear case defeats is rejected even if the risk gate passed.
            if passed and cfg.get("debate.require_pass", False):
                debate = p["debate_agent"].debate(
                    context, gf.formula, metrics, risk_report=risk
                )
                metrics["debate_margin"] = debate["margin"]
                metrics["debate_verdict"] = debate["verdict"]
                if not debate["passed"]:
                    passed = False
                    metrics["verdict"] = "reject_debate"
            row = {
                "iteration": it,
                "schema": plan.key() if plan else "",
                "formula": gf.formula,
                "source": cand["source"],
                "verdict": metrics.get("verdict", "?"),
                "risk_passed": passed,
                "rank_ic": round(metrics.get("rank_ic", 0.0), 4),
                "icir": round(metrics.get("icir", 0.0), 3),
                "sharpe": round(metrics.get("sharpe", 0.0), 2),
                "excess_dd": round(metrics.get("excess_max_drawdown", 0.0), 4),
            }
            report_rows.append(row)
            if passed:
                if gf.formula in accepted_ever:
                    # exact duplicate of an already-accepted formula (run5
                    # regression: a free slot re-drew an earlier template-slot
                    # winner verbatim). The verification checklist measures
                    # pairwise AST distance over the accepted pool — an exact
                    # duplicate contributes 0.00 and fails the diversity gate.
                    # Track the acceptance but never let the same formula into
                    # the pool twice.
                    accepted_ever.add(gf.formula)
                    continue
                accepted_ever.add(gf.formula)
                record.add_factor(gf.to_dict(), metrics, "accepted")
                accepted.append(
                    {"factor": gf.to_dict(), "metrics": metrics, "risk": risk, "source": cand["source"]}
                )
                p["memory"].record_result(
                    iteration=it,
                    schema=plan.to_dict() if plan else {"source": cand["source"], "formula": gf.formula},
                    formula=gf.formula,
                    metrics=metrics,
                )
            else:
                record.add_factor(gf.to_dict(), metrics, f"rejected:{metrics.get('verdict','?')}")
                # LIMIT_DOWN blueprint 方案 C: feed the rejection back to the miner
                signal.record_rejection(
                    {
                        "formula": gf.formula,
                        "source": cand["source"],
                        "verdict": metrics.get("verdict", "?"),
                        "reason": metrics.get("verdict", "rejected"),
                        "ic": metrics.get("ic", 0.0),
                        "rank_ic": metrics.get("rank_ic", 0.0),
                        "sharpe": metrics.get("sharpe", 0.0),
                        "max_drawdown": metrics.get("max_drawdown", 0.0),
                        "excess_max_drawdown": metrics.get("excess_max_drawdown", 0.0),
                    }
                )

    # ---- summary -----------------------------------------------------------
    df = pd.DataFrame(report_rows)
    pd.set_option("display.width", 160)
    print("\n=== per-factor results ===")
    print(df.to_string(index=False) if not df.empty else "(no factors)")
    print(f"\naccepted {len(accepted)}/{len(report_rows)} factors")

    if accepted:
        top = sorted(accepted, key=lambda a: a["metrics"].get("rank_ic", 0.0), reverse=True)[:5]
        print("\n=== top accepted by rank_ic ===")
        for a in top:
            print(
                f"  {a['factor']['formula']:<55} rank_ic={a['metrics'].get('rank_ic',0):.4f} "
                f"icir={a['metrics'].get('icir',0):.3f}"
            )

    # ---- blueprint checklist ----------------------------------------------
    checks = run_all(
        formulas=[a["factor"]["formula"] for a in accepted] or None,
        store=store,
        tracker=p["costs"],
        config=cfg,
        factor_data=long,
    )
    p["auditor"].set_checklist(
        record, **{c.name: c.passed for c in checks},
        **{f"{c.name}_detail": c.detail for c in checks},
    )

    # ---- persist -----------------------------------------------------------
    p["memory"].save(out / "memory.json")
    run_id = record.run_id
    record.write(out / f"audit_{run_id}.json")
    with open(out / "factors.json", "w", encoding="utf-8") as fh:
        json.dump(accepted, fh, ensure_ascii=False, indent=2, default=str)

    print("\n=== verification checklist ===")
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name:10s} {c.detail}")
    print(f"\ncost: {p['costs'].snapshot()}")

    # ---- C2: structured daily digest + webhook notification -----------------
    digest = build_digest(
        run_id=run_id,
        accepted=accepted,
        checklist=checks,
        cost_snapshot=p["costs"].snapshot(),
    )
    digest_path = out / f"digest_{run_id}.md"
    digest_path.write_text(digest, encoding="utf-8")
    sent = build_notifier(cfg).send_digest(digest)
    print(f"\ndigest: {digest_path} (webhook {'sent' if sent else 'not configured'})")

    print(f"artifacts: {out / 'memory.json'}, {out / f'audit_{run_id}.json'}, {out / 'factors.json'}")
    return 0 if all(c.passed for c in checks) else 1


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


def cmd_backtest(args) -> int:
    cfg = load_config()
    market = _market_data(cfg, seed=args.seed, symbols=args.symbols)
    start, end = _resolve_window(cfg, args, default="test")
    market = _slice_market(market, start, end)
    forward = market.forward_returns
    # LIMIT_DOWN blueprint 方案 B: portfolio Sharpe/maxDD use the limit-locked
    # masked series; rank-IC (below) keeps the raw series.
    tradable = _tradable(market)
    from .factors.code_generator import FactorContext

    fctx = FactorContext(market.long)
    bt = PointInTimeBacktest(
        BacktestConfig(
            long_pct=0.10,
            short_pct=0.10,
            max_position_pct=float(cfg.get("online_execution.max_position_pct", 0.05)),
        )
    )

    # Phase 8.3 — multi-factor combination backtest over a managed factor pool.
    if args.factor_pool:
        from .pool import combination_backtest, load_pool, write_json

        pool = load_pool(args.factor_pool)
        if args.neutralize and args.neutralize not in ("none", "market"):
            print(
                f"WARNING: --neutralize {args.neutralize} needs industry/size fundamentals "
                "(not ingested yet); the composite is cross-sectionally z-scored "
                "(market-level neutral) instead.",
                file=sys.stderr,
            )
        weights = "icir" if args.weights == "icir_weighted" else args.weights
        res = combination_backtest(
            fctx, forward, pool, weights=weights, n_trials=args.trials,
            bt_config=bt.config, forward_tradable=tradable,
        )
        dest = Path(args.output) if args.output else _out_dir() / f"backtest_{weights}.json"
        write_json(dest, res)
        comp = res.get("composite", {})
        print(f"combination ({weights}) across {res.get('n_factors', 0)} factors")
        print(
            f"  sharpe={comp.get('sharpe', 0):.2f} ann_ret={comp.get('annualized_return', 0):.1%} "
            f"maxdd={comp.get('max_drawdown', 0):.1%} t={comp.get('t_stat', 0):.2f} "
            f"turnover={comp.get('turnover', 0):.2f}"
        )
        for f, m in res.get("per_factor", {}).items():
            if "error" in m:
                print(f"  {f:<50} ERROR")
            else:
                print(
                    f"  {f:<50} sharpe={m.get('sharpe', 0):.2f} "
                    f"maxdd={m.get('max_drawdown', 0):.1%}"
                )
        print(f"wrote {dest}")
        return 0

    formulas = args.formulas
    if not formulas:
        # fall back to the accepted pool from the last mining run
        pool_file = _out_dir() / "factors.json"
        if pool_file.exists():
            with open(pool_file, "r", encoding="utf-8") as fh:
                pool = json.load(fh)
            formulas = [a["factor"]["formula"] for a in pool][:10]
        if not formulas:
            formulas = ["Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))"]
    results = {}
    for f in formulas:
        try:
            scores = eval_expression(f, fctx)
            res = bt.run(scores, tradable)
            m = res.metrics
            m["rank_ic"] = _rank_ic(scores, forward)
            results[f] = m
            print(
                f"{f:<55} sharpe={m['sharpe']:.2f} rank_ic={m['rank_ic']:.4f} "
                f"dd={m['max_drawdown']:.3f} t={m['t_stat']:.2f}"
            )
        except Exception as exc:
            print(f"{f:<55} ERROR: {exc}")
            results[f] = {"error": str(exc)}
    return 0


def _rank_ic(scores, forward) -> float:
    df = pd.concat([scores.rename("s"), forward.rename("f")], axis=1).dropna()
    if len(df) < 3 or df["s"].nunique() < 2 or df["f"].nunique() < 2:
        return 0.0
    # rank-based Pearson — numpy/pandas only, no scipy dependency
    return float(df["s"].rank().corr(df["f"].rank(), method="pearson"))


# ---------------------------------------------------------------------------
# pead — Phase 9.2 single-factor validation
# ---------------------------------------------------------------------------


def _cached_universe_json(name: str, config) -> list[str]:
    """Load a cached constituent list (e.g. ``hs300.json``) as system symbols."""
    from pathlib import Path as _P

    d = _P(str(config.get("data.universe_dir", "data/universe")))
    p = d / f"{name}.json"
    if not p.is_file():
        raise SystemExit(f"universe file {p} not found — run `python -m src.cli ingest` first")
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_pead(args) -> int:
    """PEAD single-factor gate (PHASE9 §8.1): SUE rank-IC > 0.015 on 2022-2025.

    Builds the PIT-correct seasonal SUE signal from the cached Baostock profit
    panel, aligns it to the test-window forward-return panel, and reports the
    factor_eval bundle (ic / rank_ic / icir / sharpe / maxdd) plus the full
    portfolio backtest. Cross-sectional data-driven factor — NOT a formula, so
    it lives outside the string-based --factor-pool path.
    """
    cfg = load_config()
    pead_cfg = cfg.get("pead") or {}
    symbols = list(args.symbols) if args.symbols else _cached_universe_json("hs300", cfg)
    years = [int(y) for y in range(2020, 2026)]
    cache = str(pead_cfg.get("cache_dir", "data/financials"))

    from .backtest.metrics import factor_eval
    from .data.financials import ensure_profit_panel
    from .factors.pead import PEADFactor

    panel = ensure_profit_panel(symbols, years, cache_dir=cache)
    if panel.empty:
        print("ERROR: empty profit panel — run the financials fetch first", file=sys.stderr)
        return 1
    factor = PEADFactor(
        panel,
        signal_expiry_days=int(pead_cfg.get("signal_expiry_days", 60)),
        min_eps_history=int(pead_cfg.get("min_eps_history", 8)),
    )
    print(f"pead: {factor.symbols.__len__()} symbols with quarterly EPS, "
          f"panel rows={len(panel)}")

    market = _market_data(cfg, seed=args.seed, symbols=symbols)
    start, end = _resolve_window(cfg, args, default="test")
    market = _slice_market(market, start, end)
    forward = market.forward_returns
    tradable = _tradable(market)
    dates = sorted(forward.index.get_level_values(0).unique())
    print(f"window: {dates[0].date()} -> {dates[-1].date()} | {len(dates)} days | "
          f"{len(forward.index.get_level_values(1).unique())} symbols")

    scores = factor.score_panel(dates, symbols)
    if scores.dropna().empty:
        print("ERROR: no PIT-valid SUE signal in the window (expiry/history filters)", file=sys.stderr)
        return 1
    direction = getattr(args, "direction", "drift")
    if direction == "reversal":
        # 2022-2025 diagnosis: high-SUE stocks UNDERperform after earnings
        # (Q4-Q0 fwd20 = -2.6%), so the reversal factor longs low SUE. Inverting
        # the percentile rank is equivalent to flipping the long/short legs.
        scores = 1.0 - scores
    m = factor_eval(scores, forward, n_trials=args.trials)
    bt = PointInTimeBacktest(
        BacktestConfig(
            long_pct=0.10, short_pct=0.10,
            max_position_pct=float(cfg.get("online_execution.max_position_pct", 0.05)),
        )
    )
    pm = bt.run(scores, tradable).metrics
    gate = float(pead_cfg.get("ic_gate", 0.015))
    top_ic = max(m["ic"], m["rank_ic"])
    print(f"  rank_ic={m['rank_ic']:.4f}  ic={m['ic']:.4f}  icir={m['icir']:.3f}  "
          f"n_days={m['n_days']}  significant={m['significant']}")
    print(f"  portfolio sharpe={pm['sharpe']:.2f}  maxdd={pm['max_drawdown']:.3f}  "
          f"t={pm['t_stat']:.2f}")

    passed = top_ic >= gate
    print(f"\n=== PEAD gate ===  max(ic, rank_ic)={top_ic:.4f} >= {gate} -> "
          f"{'PASS' if passed else 'FAIL'}")

    from .pool import write_json

    out = _out_dir()
    factor_label = "earnings-reversal" if direction == "reversal" else "PEAD (seasonal SUE)"
    result = {
        "factor": factor_label,
        "window": [str(dates[0]), str(dates[-1])],
        "n_symbols": len(symbols),
        "panel_rows": int(len(panel)),
        "metrics": m,
        "portfolio": {k: pm.get(k) for k in ("sharpe", "max_drawdown", "annualized_return", "t_stat", "turnover")},
        "gate": {"ic_threshold": gate, "passed": passed},
    }
    write_json(out / "pead_result.json", result)
    print(f"artifacts: {out / 'pead_result.json'}")
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# sentiment — Phase 9.1 (news forward ingestion + research-report factor gate)
# ---------------------------------------------------------------------------


def cmd_sentiment_ingest(args) -> int:
    """Real-time forward news collection for the HS300 (Phase 9.1 live channel).

    AKShare news feeds have no historical pagination (verified), so this grows
    the store forward one sweep at a time, deduped on article URL with resume
    state. Returns non-zero only on hard failure; per-symbol fetch errors are
    logged and skipped.
    """
    from .sentiment.ingestion import NewsIngestor

    cfg = load_config()
    symbols = list(args.symbols) if args.symbols else _cached_universe_json("hs300", cfg)
    sent_cfg = cfg.get("sentiment") or {}
    ingestor = NewsIngestor(str(sent_cfg.get("data_dir", "data/news")))
    counts = ingestor.collect(symbols, pause=args.pause)
    fresh = sum(counts.values())
    print(f"sentiment-ingest: {len(symbols)} symbols, {fresh} fresh articles "
          f"(dedup on URL), last_run={ingestor._state.get('last_run')}")
    cov = ingestor.coverage()
    if not cov.empty:
        print(f"  coverage: {len(cov)} shard-days, "
              f"{int(cov['articles'].sum())} total articles "
              f"({cov['articles'].iloc[-1]} on {cov['date'].iloc[-1]})")
    return 0


def cmd_report_ingest(args) -> int:
    """Fetch every HS300 symbol's full research-report history (backfillable).

    ``stock_research_report_em(symbol)`` returns the complete per-symbol
    history (2017→present), which is what makes the 2022-2025 sentiment gate
    testable. One Parquet per symbol; ``--force`` re-fetches, ``--limit`` caps
    the sweep for quick smoke runs.
    """
    from .sentiment.ingestion import ReportIngestor

    cfg = load_config()
    symbols = list(args.symbols) if args.symbols else _cached_universe_json("hs300", cfg)
    if args.limit:
        symbols = symbols[: args.limit]
    ingestor = ReportIngestor(str((cfg.get("sentiment") or {}).get("report_dir", "data/reports")))
    fetched = ingestor.collect(symbols, pause=args.pause, force=args.force)
    have = ingestor.cached_symbols() & set(symbols)
    print(f"report-ingest: {len(fetched)} symbols fetched now "
          f"({sum(fetched.values())} rows), {len(have)}/{len(symbols)} cached total")
    if not fetched and not have:
        print("WARNING: no reports available — check AKShare or network", file=sys.stderr)
        return 1
    return 0


def cmd_sentiment_factor(args) -> int:
    """Phase 9.1 gate: research-report title-sentiment factor, 2022-2025 HS300.

    Builds a PIT carry-forward panel (report date ≤ t, decay window) of report
    title sentiment scored by the Chinese TriAgent (word→FinBERT_zh), then runs
    the standard factor_eval bundle + long-short portfolio. Gate per user
    decision: max(ic, rank_ic) >= 0.015. Report scores are cached by title.
    """
    from .llm_client import build_llm_backend
    from .sentiment.bert import ChineseBertSentiment
    from .sentiment.critic import DeepSeekCritic
    from .sentiment.ingestion import ReportIngestor
    from .sentiment.lexicon import ChineseFinancialLexicon
    from .sentiment.report_factor import ensure_report_scores, run_report_gate
    from .sentiment.triagent import TriAgentSentiment

    cfg = load_config()
    sent_cfg = cfg.get("sentiment") or {}
    symbols = list(args.symbols) if args.symbols else _cached_universe_json("hs300", cfg)
    report_dir = str(sent_cfg.get("report_dir", "data/reports"))
    score_cache = str(sent_cfg.get("score_cache", "data/reports/report_sentiment.parquet"))

    ingestor = ReportIngestor(report_dir)
    cached = ingestor.cached_symbols()
    missing = set(symbols) - cached
    if missing:
        print(f"ERROR: reports missing for {len(missing)} symbols "
              f"(e.g. {sorted(missing)[:3]}...) — run `python -m src.cli report-ingest` first",
              file=sys.stderr)
        return 2
    reports = ingestor.load(symbols=symbols)

    # B2: the critic is the synthesis/judgment node → deep tier; the generator
    # and code paths (built in _pipeline) stay on the cheap quick tier.
    backend = build_llm_backend(cfg, CostTracker(), tier="deep")
    agent = TriAgentSentiment(
        lexicon=ChineseFinancialLexicon(),
        bert=ChineseBertSentiment() if args.tier == "triagent" else None,
        critic=DeepSeekCritic(backend=backend),
    )
    if args.tier == "word":
        # Cheap lexicon-only read of the gate; no BERT load.
        scores = ensure_report_scores(reports, agent, score_cache, tier="word")
    else:
        scores = ensure_report_scores(reports, agent, score_cache, tier="triagent",
                                      workers=args.workers)
    print(f"sentiment-factor: {len(reports)} reports, {len(scores)} unique titles scored "
          f"(tier={args.tier}, decay={args.decay}d)")

    market = _market_data(cfg, seed=args.seed, symbols=symbols)
    start, end = _resolve_window(cfg, args, default="test")
    market = _slice_market(market, start, end)

    res = run_report_gate(reports, scores, market, symbols,
                          decay_days=args.decay, gate=float(sent_cfg.get("ic_gate", 0.015)))
    m = res["metrics"]
    print(f"  rank_ic={m['rank_ic']:.4f}  ic={m['ic']:.4f}  icir={m['icir']:.3f}  "
          f"n_days={m['n_days']}  significant={m['significant']}")
    print(f"  portfolio sharpe={res['portfolio']['sharpe']:.2f}  "
          f"maxdd={res['portfolio']['max_drawdown']:.3f}  t={res['portfolio']['t_stat']:.2f}")
    print(f"\n=== sentiment gate ===  max(ic, rank_ic)={res['gate']['max_ic']:.4f} >= "
          f"{res['gate']['ic_threshold']} -> "
          f"{'PASS' if res['gate']['passed'] else 'FAIL'}")

    from .pool import write_json

    res["factor"] = "news-sentiment (research-report titles, Chinese TriAgent)"
    res["window"] = [start, end]
    res["n_symbols"] = len(symbols)
    res["decay_days"] = args.decay
    out = _out_dir()
    write_json(out / "sentiment_factor_result.json", res)
    print(f"artifacts: {out / 'sentiment_factor_result.json'}")
    return 0 if res["gate"]["passed"] else 1


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def cmd_export(args) -> int:
    cfg = load_config()
    formula = args.formula or "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))"
    compiled = compile_factor(formula, name=args.name or "exported_factor")
    out = _out_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{compiled.name}.compiled.json"
    path.write_text(compiled.to_json(), encoding="utf-8")
    print(f"compiled {compiled.formula}")
    print(f"  fields: {compiled.fields}")
    print(f"  lookbacks: {compiled.lookbacks}")
    print(f"  operators: {compiled.operators}")
    print(f"wrote {path}")
    return 0


# ---------------------------------------------------------------------------
# evolve
# ---------------------------------------------------------------------------


def cmd_evolve(args) -> int:
    cfg = load_config()
    p = _pipeline(cfg)
    market = _market_data(cfg, seed=args.seed, symbols=args.symbols)
    start, end = _resolve_window(cfg, args, default="train")
    market = _slice_market(market, start, end)
    from .factors.code_generator import FactorContext

    fctx = FactorContext(market.long)
    scores_fn = _make_scores_fn(fctx)
    context = AgentContext(
        as_of=market.long.index.get_level_values(0).max(),
        symbols=market.long.index.get_level_values(1).unique().tolist(),
        config=cfg,
        data=market.long,
    )
    space = p["space"]
    if args.base_plan:
        plan = SchemaPlan.from_dict(json.loads(args.base_plan))
    else:
        plan = space.sample(random.Random(args.seed))
    base_formula = args.formula or default_formula_for(plan)

    evo = EvoQuant(
        space=space,
        generator=p["generator"],
        memory=p["memory"],
        risk_agent=p["risk_agent"],
        llm=p["backend"],
        config=cfg,
    )
    result = evo.evolve(
        context,
        plan,
        base_formula,
        scores_fn,
        market.forward_returns,
        market_returns=_market_returns(market),
        n_trials=args.trials,
    )
    print("diagnosis:")
    for d in result.diagnosis:
        print(f"  - {d}")
    print(f"\nbase_rank_ic={result.metrics.get('base_rank_ic', 0.0):.4f} "
          f"best_rank_ic={result.metrics.get('best_rank_ic', 0.0):.4f}")
    if result.accepted:
        print("\naccepted edit:")
        print(f"  formula: {result.accepted['formula']}")
        print(f"  metrics: rank_ic={result.accepted['metrics'].get('rank_ic', 0.0):.4f} "
              f"sharpe={result.accepted['metrics'].get('sharpe', 0.0):.2f}")
    else:
        print("\nno edit passed the risk gates")
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def cmd_verify(args) -> int:
    cfg = load_config()
    # verify audits the whole store (not the research universe), and the
    # checklist needs the price + universe records B4 reads
    market = _market_data(cfg, seed=args.seed, bound_to_universe=False)
    # B5 (freshness): a historical backfill (ingest --end <past>) is stale by
    # construction — measure staleness against the backfill horizon instead of
    # today. "backfill" == the ingest run's end date; "live" == now().
    freshness_as_of = None
    if args.mode == "backfill":
        freshness_as_of = str(cfg.get("project.end_date", pd.Timestamp.today().normalize().date()))
    checks = run_all(
        store=market.audit_store or market.pit_store,
        tracker=None,
        config=cfg,
        freshness_as_of=freshness_as_of,
        real_data_audit=True,
    )
    print("=== blueprint verification checklist ===")
    for c in checks:
        print(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name:10s} {c.detail}")
    return 0 if all(c.passed for c in checks) else 1


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def cmd_ingest(args) -> int:
    cfg = load_config()
    from .data.ingestion.ingestor import Ingestor

    ing = Ingestor(cfg)
    # --symbols all (blueprint verbatim) == the full universe: omit the list so
    # the price pass resolves targets from the ingested universe snapshots.
    symbols = None if args.symbols == ["all"] else args.symbols
    stats = ing.ingest(
        symbols=symbols,
        start=args.start,
        end=args.end,
        fundamentals=args.fundamentals,
        news=args.news,
        resume=args.resume,
        limit=args.limit,
        universe_only=args.universe_only,
    )
    print(stats.summary())
    return 1 if stats.symbols_failed else 0


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------


def cmd_monitor(args) -> int:
    cfg = load_config()
    market = _market_data(cfg, seed=args.seed, symbols=args.symbols)
    start, end = _resolve_window(cfg, args, default="test")
    market = _slice_market(market, start, end)
    from .factors.code_generator import FactorContext, eval_expression
    from .monitoring.decay_tracker import DecayTracker

    fctx = FactorContext(market.long)

    # Phase 8.4 — decay watchlist across a whole factor pool.
    if args.watchlist:
        from .pool import load_pool, monitor_watchlist, write_json

        pool = load_pool(args.watchlist)
        res = monitor_watchlist(
            fctx,
            market.forward_returns,
            pool,
            window_days=args.window_days,
            icir_threshold=args.threshold,
        )
        dest = Path(args.output) if args.output else _out_dir() / "monitor_report.json"
        write_json(dest, {"results": res})
        n_decayed = sum(
            1 for r in res.values() if isinstance(r, dict) and r.get("decayed")
        )
        print(
            f"watchlist {len(res)} factors, {n_decayed} decayed "
            f"(ICIR floor {args.threshold}, window {args.window_days}d)"
        )
        for f, r in res.items():
            if "error" in r:
                print(f"  {f:<50} ERROR {r['error']}")
            else:
                print(
                    f"  {f:<50} recent_icir={r.get('recent_icir', 0):+.2f} "
                    f"{'DECAYED' if r.get('decayed') else 'ok'}"
                )
        print(f"wrote {dest}")
        return 0 if n_decayed == 0 else 1

    formula = args.formula or "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))"
    scores = eval_expression(formula, fctx)
    tracker = DecayTracker.from_config(cfg)
    res = tracker.monitor(scores, market.forward_returns)
    print(res.summary)
    print("\nrecent windows (end | ic | icir | days | state):")
    for w in res.windows[-8:]:
        print(
            f"  {w.end.date()} | {w.ic:+.4f} | {w.icir:+.2f} | {w.n_days:4d} | "
            f"{'DECAYED' if w.decayed else 'ok'}"
        )
    return 1 if res.decayed else 0


# ---------------------------------------------------------------------------
# pool — Phase 8 factor-pool management
# ---------------------------------------------------------------------------


def _pool_market(args, default: str = "val"):
    """Load the research-universe market sliced to a walk-forward window."""
    cfg = load_config()
    market = _market_data(cfg, seed=getattr(args, "seed", 1), symbols=getattr(args, "symbols", None))
    start, end = _resolve_window(cfg, args, default=default)
    market = _slice_market(market, start, end)
    from .factors.code_generator import FactorContext

    return cfg, market, FactorContext(market.long), market.forward_returns


def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.3rem; }} table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 10px; font-size: 0.85rem; text-align: right; }}
  th {{ background: #f5f5f5; }} td:first-child {{ text-align: left; }}
</style></head><body><h1>{title}</h1>{body}</body></html>"""


def _table_html(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    trs = ["<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows]
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def cmd_pool(args) -> int:
    from .pool import diversify_pool, evaluate_pool, filter_pool, load_pool, write_json

    out = _out_dir()
    src = Path(args.input) if getattr(args, "input", None) else out / "factors.json"
    if not src.exists():
        print(f"ERROR: input pool not found: {src}", file=sys.stderr)
        return 1
    pool = load_pool(src)
    action = args.pool_action

    if action == "filter":
        _, market, fctx, forward = _pool_market(args, default="val")
        formulas = [str(e.get("factor", {}).get("formula", "")) for e in pool]
        metrics = evaluate_pool(fctx, forward, formulas, n_trials=args.trials)
        kept = filter_pool(pool, metrics, min_ic=args.min_ic, min_icir=args.min_icir)
        dest = Path(args.output) if args.output else out / "factor_pool_filtered.json"
        write_json(
            dest,
            {
                "factors": kept,
                "filter": {"window": args.window, "min_ic": args.min_ic, "min_icir": args.min_icir,
                           "n_in": len(pool), "n_out": len(kept)},
            },
        )
        print(f"filter ({args.window} window): {len(pool)} -> {len(kept)} "
              f"(val IC>={args.min_ic}, ICIR>={args.min_icir})")
        for e in kept:
            m = e.get("val_metrics", {})
            print(f"  {e['factor']['formula']:<50} val_ic={m.get('ic', 0):.4f} "
                  f"val_icir={m.get('icir', 0):.2f} sharpe={m.get('sharpe', 0):.2f}")
        print(f"wrote {dest}")
        return 0

    if action == "diversify":
        kept = diversify_pool(pool, min_distance=args.min_distance, order_by="ic")
        dest = Path(args.output) if args.output else out / "factor_pool_diverse.json"
        write_json(
            dest,
            {"factors": kept, "diversify": {"min_ast_distance": args.min_distance,
                                            "n_in": len(pool), "n_out": len(kept)}},
        )
        print(f"diversify: {len(pool)} -> {len(kept)} (min AST distance {args.min_distance})")
        for e in kept:
            print(f"  {e['factor']['formula']}")
        print(f"wrote {dest}")
        return 0

    if action == "report":
        from .factors.code_generator import CodeGenerator, ast_distance

        gen = CodeGenerator()
        nodes = []
        for e in pool:
            try:
                nodes.append((e, gen.parse(e["factor"]["formula"])))
            except Exception:  # noqa: BLE001
                continue
        dists, below, pairs = [], 0, 0
        for i, (_, na) in enumerate(nodes):
            for (_, nb) in nodes[i + 1:]:
                pairs += 1
                d = ast_distance(na, nb)
                dists.append(d)
                if d < args.min_distance:
                    below += 1
        rows = []
        for e, _ in nodes:
            m = e.get("val_metrics", e.get("metrics", {}))
            rows.append([e["factor"]["formula"], f"{m.get('ic', 0):.4f}",
                         f"{m.get('icir', 0):.2f}", f"{m.get('sharpe', 0):.2f}",
                         f"{m.get('max_drawdown', 0):.2f}"])
        min_d = min(dists) if dists else float("nan")
        body = (f"<p>factors: {len(nodes)} · pairs: {pairs} · min AST distance: {min_d:.2f} "
                f"· below {args.min_distance}: {below}</p>"
                + _table_html(["formula", "IC", "ICIR", "sharpe", "maxDD"], rows))
        dest = Path(args.output) if args.output else out / "diversity_report.html"
        dest.write_text(_html_page("Diversity report", body), encoding="utf-8")
        print(f"report: {len(nodes)} factors, min AST distance {min_d:.2f}, "
              f"{below}/{pairs} pairs below {args.min_distance}")
        print(f"wrote {dest}")
        return 0

    if action == "promote":
        bt_path = Path(args.backtest)
        if not bt_path.exists():
            print(f"ERROR: backtest report not found: {bt_path}", file=sys.stderr)
            return 1
        bt = json.loads(bt_path.read_text(encoding="utf-8"))
        sharpe = float(bt.get("composite", {}).get("sharpe", 0.0))
        ok = sharpe >= args.min_sharpe
        promoted = []
        for e in pool:
            c = deepcopy(e)
            c["status"] = "deployable" if ok else "candidate"
            c["backtest_sharpe"] = sharpe
            promoted.append(c)
        dest = Path(args.output) if args.output else out / "factors_deployable.json"
        write_json(
            dest,
            {"factors": promoted, "promote": {"min_sharpe": args.min_sharpe,
                                              "composite_sharpe": sharpe, "deployable": ok}},
        )
        print(f"promote: composite sharpe {sharpe:.2f} vs floor {args.min_sharpe} -> "
              f"{'DEPLOYABLE' if ok else 'NOT (stays candidate)'}")
        print(f"wrote {dest}")
        return 0 if ok else 1

    if action == "flag":
        mon_path = Path(args.monitor)
        if not mon_path.exists():
            print(f"ERROR: monitor report not found: {mon_path}", file=sys.stderr)
            return 1
        mon = json.loads(mon_path.read_text(encoding="utf-8"))
        results = mon.get("results", mon)
        flagged = []
        for e in pool:
            c = deepcopy(e)
            r = results.get(str(e.get("factor", {}).get("formula", "")))
            c["decay"] = r if isinstance(r, dict) else {}
            c["status"] = "decayed" if (isinstance(r, dict) and r.get("decayed")) else c.get("status", "deployable")
            flagged.append(c)
        dest = Path(args.output) if args.output else out / "factors_with_decay.json"
        write_json(dest, {"factors": flagged,
                          "n_decayed": sum(1 for e in flagged if e.get("status") == "decayed")})
        print(f"flag: wrote {dest} "
              f"({sum(1 for e in flagged if e.get('status') == 'decayed')} decayed)")
        return 0

    return 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def cmd_paper(args) -> int:
    """Simulated paper trading — the resumable daily loop (simulated 模拟盘).

    Bridges the backtest → paper-trading gap: the Phase 10 three-layer
    portfolio is priced day-by-day through the order executor (slippage /
    commission / cash account) with a SQLite ledger that persists cash,
    positions and fills so the run can be resumed after a restart.

    Fully offline when no PIT database is set (synthetic market + a default
    factor); the PEAD tilt and sentiment risk overlay attach automatically when
    their cached data is present and are otherwise skipped with a warning.
    """
    cfg = load_config()
    pcfg = cfg.section("paper")

    start, end = args.start, args.end
    if not start and not end:
        start = cfg.get("research.test_start")
        end = cfg.get("research.test_end")

    # market (real PIT store when set, else synthetic) + 360d warmup so the
    # alpha lookbacks (120/240d) are filled before the first window date.
    market = _market_data(cfg, seed=args.seed, symbols=args.symbols)
    if start:
        warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=360)).date().isoformat()
        market = _slice_market(market, warmup_start, end)
    elif end:
        market = _slice_market(market, None, end)
    symbols = args.symbols or sorted(market.price_panel.columns)

    # alpha core — the Phase 8 low-vol/low-turnover pool (outputs/factors.json),
    # or a default factor offline.
    from .factors.code_generator import FactorContext
    from .portfolio.alpha_core import AlphaCore

    pool_file = _out_dir() / "factors.json"
    if pool_file.exists():
        with open(pool_file, "r", encoding="utf-8") as fh:
            formulas = [a.get("factor", {}).get("formula", a.get("formula")) for a in json.load(fh)]
        formulas = [f for f in formulas if isinstance(f, str) and f.strip()]
    else:
        formulas = []
    if not formulas:
        formulas = ["Rank(Close)"]
    fctx = FactorContext(market.long)
    alpha = AlphaCore(fctx, formulas, long_pct=0.10, short_pct=0.10,
                      max_position_pct=float(pcfg.get("max_position_pct", 0.05)))
    print(f"alpha core: {len(formulas)} factor(s), {alpha.composite.notna().sum()} non-NaN cells")

    # optional overlays (best-effort, real cached data only)
    tilt = risk = None
    # Overlays need real PEAD / report-sentiment caches keyed to real symbols —
    # when there is no PIT database the market is synthetic and every overlay
    # fetch would be a wasted (and invalid) network call, so skip them.
    if cfg.get("data.pit_database_url"):
        try:
            from .data.financials import ensure_profit_panel
            from .factors.pead import PEADFactor
            from .portfolio.seasonal_tilt import PEADSeasonalTilt

            pead_cfg = cfg.get("pead") or {}
            years = [int(y) for y in range(2020, 2026)]
            panel = ensure_profit_panel(symbols, years, cache_dir=str(pead_cfg.get("cache_dir", "data/financials")))
            pead = PEADFactor(panel, signal_expiry_days=int(pead_cfg.get("signal_expiry_days", 60)),
                              min_eps_history=int(pead_cfg.get("min_eps_history", 8)))
            tilt = PEADSeasonalTilt(pead, universe=symbols)
            print(f"pead tilt: {len(pead.symbols)} symbols with quarterly EPS")
        except Exception as exc:  # noqa: BLE001 — offline / no cached profit panel
            print(f"WARNING: PEAD tilt unavailable ({exc}) — alpha-only", file=sys.stderr)

        try:
            from .portfolio.risk_overlay import SentimentRiskOverlay
            from .sentiment.ingestion import ReportIngestor
            from .sentiment.triagent import build_report_signal

            sent_cfg = cfg.get("sentiment") or {}
            report_dir = str(sent_cfg.get("report_dir", "data/reports"))
            score_cache = str(sent_cfg.get("score_cache", "data/reports/report_sentiment.parquet"))
            decay = int(sent_cfg.get("decay_days", 10))
            ingestor = ReportIngestor(report_dir)
            have = set(symbols) & set(ingestor.cached_symbols())
            if have:
                reports = ingestor.load(symbols=sorted(have))
                scores = pd.read_parquet(score_cache)
                td = sorted(market.forward_returns.index.get_level_values(0).unique())
                sig = build_report_signal(reports, scores, td, sorted(have), decay_days=decay)
                rk = {k: v for k, v in (cfg.get("risk_overlay") or {}).items()
                      if k in {"zscore_threshold", "position_cut", "freeze_days", "min_trigger_samples"}}
                risk = SentimentRiskOverlay(sig, **rk)
                print(f"risk overlay: {int(sig.notna().sum())} sentiment cells")
            else:
                print("WARNING: no cached reports — risk overlay is a no-op", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — offline / no cached sentiment
            print(f"WARNING: risk overlay unavailable ({exc}) — no sentiment cut", file=sys.stderr)

    from .paper import PaperLedger, PaperRunner
    from .portfolio.layer_integration import ThreeLayerPortfolio

    ledger = PaperLedger(args.ledger or str(pcfg.get("ledger_db", "outputs/paper_ledger.sqlite")))
    runner = PaperRunner(
        ThreeLayerPortfolio(alpha, tilt=tilt, risk=risk),
        market,
        ledger,
        symbols=symbols,
        cash=float(pcfg.get("initial_cash", 100_000.0)),
        slippage_bps=float(pcfg.get("slippage_bps", 2.0)),
        commission_bps=float(pcfg.get("commission_bps", 5.0)),
        min_commission=float(pcfg.get("min_commission", 1.0)),
        max_position_pct=float(pcfg.get("max_position_pct", 0.05)),
        rebalance_days=int(pcfg.get("rebalance_days", 1)),
        pit_strict=bool(pcfg.get("pit_strict", True)),
        seed=args.seed,
    )
    result = runner.run(start=start, end=end)
    ledger.close()

    m = result.get("metrics", {})
    print("\n=== PAPER TRADING ===")
    print(f"  days={m.get('n_days', 0)}  fills={m.get('n_fills', 0)}  "
          f"commission={m.get('total_commission', 0)}")
    print(f"  total_ret={m.get('total_return', 0):.1%}  ann_ret={m.get('annualized_return', 0):.1%}  "
          f"sharpe={m.get('sharpe', 0):.2f}  maxDD={m.get('max_drawdown', 0):.1%}")
    print(f"  final_equity={m.get('final_equity', 0):,.2f}  final_cash={m.get('final_cash', 0):,.2f}")

    out = args.output or str(pcfg.get("output_json", "outputs/paper_run.json"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {out}  (ledger {ledger.db_path})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-quant",
        description="LLM-driven quantitative trading system (offline R&D / deterministic online).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_mine = sub.add_parser("mine", help="run the multi-agent factor-mining loop")
    p_mine.add_argument("--iterations", type=int, default=None, help="default from research.mining.iterations")
    p_mine.add_argument("--hypotheses", type=int, default=4)
    p_mine.add_argument("--trials", type=int, default=1)
    p_mine.add_argument("--seed", type=int, default=1)
    p_mine.add_argument("--window", choices=["train", "val", "test", "all"], default="train")
    p_mine.add_argument("--start", type=str, default=None)
    p_mine.add_argument("--end", type=str, default=None)
    p_mine.add_argument("--symbols", nargs="*", default=None,
                        help="override research.universe (e.g. full-A validation)")
    p_mine.add_argument("--enable-feedback", dest="enable_feedback", type=bool, default=None,
                        help="LIMIT_DOWN blueprint 方案 C: inject last-round rejection reasons "
                             "into the mining LLM prompt (default: config factor_mining.enable_mining_feedback)")
    p_mine.add_argument("--feedback-rounds", type=int, default=None,
                        help="how many recent rejections to replay (default: config)")
    p_mine.add_argument("--crisis-test", action="store_true",
                        help="LIMIT_DOWN blueprint: enable the 2015/2018/2024 crisis drawdown gate")
    p_mine.set_defaults(func=cmd_mine)

    p_bt = sub.add_parser("backtest", help="backtest formulas on PIT data")
    p_bt.add_argument("--formulas", nargs="*", default=[])
    p_bt.add_argument("--seed", type=int, default=1)
    p_bt.add_argument("--window", choices=["train", "val", "test", "all"], default="test")
    p_bt.add_argument("--start", type=str, default=None)
    p_bt.add_argument("--end", type=str, default=None)
    p_bt.add_argument("--symbols", nargs="*", default=None,
                      help="override research.universe (e.g. full-A validation)")
    p_bt.add_argument("--factor-pool", type=str, default=None,
                      help="path to a factor-pool JSON; run a combination backtest over its formulas")
    p_bt.add_argument("--weights", type=str, default="equal",
                      choices=["equal", "icir", "icir_weighted", "dynamic"],
                      help="combination weight scheme (default: equal)")
    p_bt.add_argument("--neutralize", type=str, default=None,
                      help="neutralization levels like industry,size — NOT yet supported; "
                           "the composite is cross-sectionally z-scored (market-level) instead")
    p_bt.add_argument("--trials", type=int, default=1,
                      help="bootstrap trials for the composite backtest")
    p_bt.add_argument("--output", type=str, default=None,
                      help="JSON output path (default: outputs/backtest_<weights>.json)")
    p_bt.set_defaults(func=cmd_backtest)

    p_pead = sub.add_parser("pead", help="Phase 9.2 PEAD single-factor validation gate")
    p_pead.add_argument("--seed", type=int, default=1)
    p_pead.add_argument("--window", choices=["train", "val", "test", "all"], default="test")
    p_pead.add_argument("--start", type=str, default=None)
    p_pead.add_argument("--end", type=str, default=None)
    p_pead.add_argument("--symbols", nargs="*", default=None,
                        help="override HS300 (default: data/universe/hs300.json)")
    p_pead.add_argument("--trials", type=int, default=1,
                        help="bootstrap trials for significance")
    p_pead.add_argument("--direction", choices=["drift", "reversal"], default="drift",
                        help="drift = classic PEAD (high SUE -> long); "
                             "reversal = earnings reversal (low SUE -> long, from 2022-2025 negative-PEAD diagnosis)")
    p_pead.set_defaults(func=cmd_pead)

    p_ns = sub.add_parser("sentiment-ingest",
                          help="Phase 9.1: real-time forward news collection (HS300)")
    p_ns.add_argument("--symbols", nargs="*", default=None,
                      help="override HS300 (default: data/universe/hs300.json)")
    p_ns.add_argument("--pause", type=float, default=0.0,
                      help="per-symbol sleep between fetches (politeness)")
    p_ns.set_defaults(func=cmd_sentiment_ingest)

    p_ri = sub.add_parser("report-ingest",
                          help="Phase 9.1: fetch HS300 research-report history (backfillable)")
    p_ri.add_argument("--symbols", nargs="*", default=None,
                      help="override HS300 (default: data/universe/hs300.json)")
    p_ri.add_argument("--limit", type=int, default=None,
                      help="cap the number of symbols (quick smoke runs)")
    p_ri.add_argument("--pause", type=float, default=0.2,
                      help="per-symbol sleep between fetches")
    p_ri.add_argument("--force", action="store_true",
                      help="re-fetch symbols that are already cached")
    p_ri.set_defaults(func=cmd_report_ingest)

    p_sf = sub.add_parser("sentiment-factor",
                          help="Phase 9.1 gate: report-title sentiment IC on 2022-2025 HS300")
    p_sf.add_argument("--seed", type=int, default=1)
    p_sf.add_argument("--window", choices=["train", "val", "test", "all"], default="test")
    p_sf.add_argument("--start", type=str, default=None)
    p_sf.add_argument("--end", type=str, default=None)
    p_sf.add_argument("--symbols", nargs="*", default=None,
                      help="override HS300 (default: data/universe/hs300.json)")
    p_sf.add_argument("--tier", choices=["triagent", "word"], default="triagent",
                      help="triagent = lexicon+FinBERT_zh (full); word = lexicon-only quick read")
    p_sf.add_argument("--decay", type=int, default=10,
                      help="report-signal carry-forward days (default: 10)")
    p_sf.add_argument("--workers", type=int, default=1,
                      help="parallel BERT scoring processes (CPU speedup; default: 1)")
    p_sf.set_defaults(func=cmd_sentiment_factor)

    p_exp = sub.add_parser("export", help="compile a formula for the online layer")
    p_exp.add_argument("--formula", type=str, default=None)
    p_exp.add_argument("--name", type=str, default=None)
    p_exp.set_defaults(func=cmd_export)

    p_ev = sub.add_parser("evolve", help="run one EvoQuant self-evolution round")
    p_ev.add_argument("--formula", type=str, default=None)
    p_ev.add_argument("--trials", type=int, default=1)
    p_ev.add_argument("--seed", type=int, default=1)
    p_ev.add_argument("--base-plan", type=str, default=None)
    p_ev.add_argument("--window", choices=["train", "val", "test", "all"], default="train")
    p_ev.add_argument("--start", type=str, default=None)
    p_ev.add_argument("--end", type=str, default=None)
    p_ev.add_argument("--symbols", nargs="*", default=None,
                      help="override research.universe (e.g. full-A validation)")
    p_ev.set_defaults(func=cmd_evolve)

    p_ver = sub.add_parser("verify", help="run the blueprint verification checklist")
    p_ver.add_argument("--seed", type=int, default=1)
    p_ver.add_argument(
        "--mode", choices=["backfill", "live"], default="live",
        help="backfill: B5 freshness vs project.end_date; live: vs now()",
    )
    p_ver.set_defaults(func=cmd_verify)

    p_ing = sub.add_parser("ingest", help="ingest real data (universe → prices → optional fundamentals/news)")
    p_ing.add_argument("--symbols", nargs="*", default=None, help="restrict price pass (default: full universe)")
    p_ing.add_argument("--start", type=str, default=None)
    p_ing.add_argument("--end", type=str, default=None)
    p_ing.add_argument("--fundamentals", action="store_true", help="Q4 pilot snapshot (akshare, ~50 symbols)")
    p_ing.add_argument("--news", action="store_true", help="Q3 watchlist news (akshare)")
    p_ing.add_argument("--resume", action="store_true", help="only fetch bars after the newest stored bar")
    p_ing.add_argument("--limit", type=int, default=None, help="cap the number of symbols in the price pass")
    p_ing.add_argument(
        "--universe-only", action="store_true",
        help="stop after universe snapshots + index caches (B4 prep); skip the price pass",
    )
    p_ing.set_defaults(func=cmd_ingest)

    p_mon = sub.add_parser("monitor", help="score a factor's IC/ICIR decay over rolling windows")
    p_mon.add_argument("--formula", type=str, default=None)
    p_mon.add_argument("--seed", type=int, default=1)
    p_mon.add_argument("--window", choices=["train", "val", "test", "all"], default="test")
    p_mon.add_argument("--start", type=str, default=None)
    p_mon.add_argument("--symbols", nargs="*", default=None,
                       help="override research.universe (e.g. full-A validation)")
    p_mon.add_argument("--end", type=str, default=None)
    p_mon.add_argument("--watchlist", type=str, default=None,
                       help="path to a factor-pool JSON; decay-monitor every formula in it")
    p_mon.add_argument("--window-days", type=int, default=90,
                       help="rolling window for recent-ICIR (default: 90)")
    p_mon.add_argument("--threshold", type=float, default=0.30,
                       help="recent-ICIR floor for decay flagging (default: 0.30)")
    p_mon.add_argument("--output", type=str, default=None,
                       help="JSON output path (default: outputs/monitor_report.json)")
    p_mon.set_defaults(func=cmd_monitor)

    p_pool = sub.add_parser("pool", help="Phase 8 factor-pool management")
    pool_sub = p_pool.add_subparsers(dest="pool_action", required=True)

    pf = pool_sub.add_parser("filter", help="re-score pool on a window, keep passing factors")
    pf.add_argument("--input", type=str, default=None,
                    help="pool JSON (default: outputs/factors.json)")
    pf.add_argument("--min-ic", type=float, default=0.02)
    pf.add_argument("--min-icir", type=float, default=0.30)
    pf.add_argument("--window", choices=["train", "val", "test", "all"], default="val")
    pf.add_argument("--trials", type=int, default=1)
    pf.add_argument("--seed", type=int, default=1)
    pf.add_argument("--start", type=str, default=None)
    pf.add_argument("--end", type=str, default=None)
    pf.add_argument("--symbols", nargs="*", default=None)
    pf.add_argument("--output", type=str, default=None,
                    help="output JSON (default: outputs/factor_pool_filtered.json)")
    pf.set_defaults(func=cmd_pool)

    pdv = pool_sub.add_parser("diversify", help="greedy AST-distance diversity screening")
    pdv.add_argument("--input", type=str, default=None)
    pdv.add_argument("--min-distance", type=float, default=0.40)
    pdv.add_argument("--output", type=str, default=None,
                     help="output JSON (default: outputs/factor_pool_diverse.json)")
    pdv.set_defaults(func=cmd_pool)

    pr = pool_sub.add_parser("report", help="diversity report as HTML")
    pr.add_argument("--input", type=str, default=None)
    pr.add_argument("--min-distance", type=float, default=0.40)
    pr.add_argument("--output", type=str, default=None,
                    help="output HTML (default: outputs/diversity_report.html)")
    pr.set_defaults(func=cmd_pool)

    pp = pool_sub.add_parser("promote", help="promote pool to deployable if composite Sharpe passes")
    pp.add_argument("--input", type=str, default=None)
    pp.add_argument("--backtest", type=str, required=True,
                    help="path to the combination backtest JSON")
    pp.add_argument("--min-sharpe", type=float, default=1.0)
    pp.add_argument("--output", type=str, default=None,
                    help="output JSON (default: outputs/factors_deployable.json)")
    pp.set_defaults(func=cmd_pool)

    pfl = pool_sub.add_parser("flag", help="flag decayed factors from a monitor report")
    pfl.add_argument("--input", type=str, default=None)
    pfl.add_argument("--monitor", type=str, required=True,
                     help="path to the monitor report JSON")
    pfl.add_argument("--output", type=str, default=None,
                     help="output JSON (default: outputs/factors_with_decay.json)")
    pfl.set_defaults(func=cmd_pool)

    p_paper = sub.add_parser("paper", help="simulated paper trading — resumable daily loop over the three-layer portfolio")
    p_paper.add_argument("--start", type=str, default=None)
    p_paper.add_argument("--end", type=str, default=None)
    p_paper.add_argument("--symbols", nargs="*", default=None,
                         help="override research.universe (e.g. explicit HS300 names)")
    p_paper.add_argument("--seed", type=int, default=1)
    p_paper.add_argument("--ledger", type=str, default=None,
                         help="SQLite ledger path (default: config paper.ledger_db)")
    p_paper.add_argument("--output", type=str, default=None,
                         help="JSON output path (default: config paper.output_json)")
    p_paper.set_defaults(func=cmd_paper)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
