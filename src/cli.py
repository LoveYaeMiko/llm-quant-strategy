"""llm-quant — command-line entry point for the whole pipeline.

Subcommands:
    mine     Run the multi-agent factor-mining loop (signal -> code -> eval ->
             risk), persist the accepted pool, and emit an audit record.
    evolve   One EvoQuant self-evolution round around a base factor.
    backtest Backtest a single formula (or the accepted pool) on PIT data.
    export   Compile a formula into the deterministic online artifact (JSON).
    verify   Run the blueprint verification checklist (PIT / FinCAD / diversity /
             cost).

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
from pathlib import Path

import pandas as pd

from .agents.base_agent import AgentContext
from .agents.code_agent import CodeAgent
from .agents.dynamic_router import DynamicRouter
from .agents.eval_agent import EvalAgent
from .agents.risk_agent import RiskAgent
from .agents.signal_agent import SignalAgent
from .audit import ExperimentAuditor
from .backtest.engine import BacktestConfig, PointInTimeBacktest
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
    return market


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
    fwd = close_wide.pct_change().shift(-1).stack().rename("fwd")
    return SyntheticMarket(
        records=store.records,
        long=long,
        price_panel=close_wide,
        forward_returns=fwd,
        pit_store=store,
        n_symbols=len(close_wide.columns),
        n_days=len(close_wide),
    )


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

    signal_agent = SignalAgent(space=space, memory=memory, llm=backend, config=cfg, seed=0)
    code_agent = CodeAgent(generator=generator, memory=memory, llm=backend, config=cfg)
    eval_agent = EvalAgent(memory=memory, llm=backend, config=cfg)
    risk_agent = RiskAgent(memory=memory, llm=backend, config=cfg, seed=0)
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
    }


def _make_scores_fn(ctx):
    def fn(formula: str) -> pd.Series:
        return eval_expression(formula, ctx)

    return fn


def _market_returns(market) -> pd.Series:
    """Daily equal-weighted market return series (for regime classification)."""
    return market.price_panel.pct_change().mean(axis=1).dropna()


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

    print(
        f"mining: {n_iter} iterations x {n_hyps} hypotheses | "
        f"llm={'deepseek-v4-flash' if p['backend'] else 'OFFLINE'}"
    )
    for it in range(n_iter):
        plans = p["agents"]["signal"].generate_hypotheses(context, n=n_hyps)
        for plan in plans:
            gf = p["agents"]["code"].translate(context, plan)
            try:
                scores = scores_fn(gf.formula)
                metrics = p["eval_agent"].evaluate(context, scores, forward, n_trials=args.trials)
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
            row = {
                "iteration": it,
                "schema": plan.key(),
                "formula": gf.formula,
                "verdict": metrics.get("verdict", "?"),
                "risk_passed": passed,
                "rank_ic": round(metrics.get("rank_ic", 0.0), 4),
                "icir": round(metrics.get("icir", 0.0), 3),
                "sharpe": round(metrics.get("sharpe", 0.0), 2),
            }
            report_rows.append(row)
            if passed:
                record.add_factor(gf.to_dict(), metrics, "accepted")
                accepted.append({"factor": gf.to_dict(), "metrics": metrics, "risk": risk})
                p["memory"].record_result(
                    iteration=it, schema=plan.to_dict(), formula=gf.formula, metrics=metrics
                )
            else:
                record.add_factor(gf.to_dict(), metrics, f"rejected:{metrics.get('verdict','?')}")

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
    from .factors.code_generator import FactorContext

    fctx = FactorContext(market.long)
    bt = PointInTimeBacktest(
        BacktestConfig(
            long_pct=0.10,
            short_pct=0.10,
            max_position_pct=float(cfg.get("online_execution.max_position_pct", 0.05)),
        )
    )
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
            res = bt.run(scores, forward)
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
    checks = run_all(store=market.audit_store or market.pit_store, tracker=None, config=cfg)
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
    stats = ing.ingest(
        symbols=args.symbols or None,
        start=args.start,
        end=args.end,
        fundamentals=args.fundamentals,
        news=args.news,
        resume=args.resume,
        limit=args.limit,
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
# main
# ---------------------------------------------------------------------------


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
    p_mine.set_defaults(func=cmd_mine)

    p_bt = sub.add_parser("backtest", help="backtest formulas on PIT data")
    p_bt.add_argument("--formulas", nargs="*", default=[])
    p_bt.add_argument("--seed", type=int, default=1)
    p_bt.add_argument("--window", choices=["train", "val", "test", "all"], default="test")
    p_bt.add_argument("--start", type=str, default=None)
    p_bt.add_argument("--end", type=str, default=None)
    p_bt.add_argument("--symbols", nargs="*", default=None,
                      help="override research.universe (e.g. full-A validation)")
    p_bt.set_defaults(func=cmd_backtest)

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
    p_ver.set_defaults(func=cmd_verify)

    p_ing = sub.add_parser("ingest", help="ingest real data (universe → prices → optional fundamentals/news)")
    p_ing.add_argument("--symbols", nargs="*", default=None, help="restrict price pass (default: full universe)")
    p_ing.add_argument("--start", type=str, default=None)
    p_ing.add_argument("--end", type=str, default=None)
    p_ing.add_argument("--fundamentals", action="store_true", help="Q4 pilot snapshot (akshare, ~50 symbols)")
    p_ing.add_argument("--news", action="store_true", help="Q3 watchlist news (akshare)")
    p_ing.add_argument("--resume", action="store_true", help="only fetch bars after the newest stored bar")
    p_ing.add_argument("--limit", type=int, default=None, help="cap the number of symbols in the price pass")
    p_ing.set_defaults(func=cmd_ingest)

    p_mon = sub.add_parser("monitor", help="score a factor's IC/ICIR decay over rolling windows")
    p_mon.add_argument("--formula", type=str, default=None)
    p_mon.add_argument("--seed", type=int, default=1)
    p_mon.add_argument("--window", choices=["train", "val", "test", "all"], default="test")
    p_mon.add_argument("--start", type=str, default=None)
    p_mon.add_argument("--symbols", nargs="*", default=None,
                       help="override research.universe (e.g. full-A validation)")
    p_mon.add_argument("--end", type=str, default=None)
    p_mon.set_defaults(func=cmd_monitor)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
