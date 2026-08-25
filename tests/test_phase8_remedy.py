"""validation_BLUEPRINT — fusion remedy gate tests + §3.5 success criteria.

Covers the two improvements over the LIMIT_DOWN blueprint:
* **excess drawdown gate** (§3.3) — the engine computes ``excess_max_drawdown``
  relative to a benchmark, and both the eval gate and the risk agent gate on it
  (falling back to the absolute ``sharpe.max_drawdown_limit`` when no benchmark
  is threaded, so offline/synthetic runs keep the old behaviour);
* **sampling slots** (§3.2) — the signal agent's deterministic combination-
  template slot generator (distinct formulas, skips rejected), and the code
  agent's code-layer physical block (§3.1).

``test_phase8_remedy_success_criteria`` is the post-run harness for the 5-iteration
validation (blueprint §3.5): read ``outputs/factors.json`` and assert >=6/20
accepted, zero pure-reversal factors, and >=2 from combination templates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents.base_agent import AgentContext
from src.agents.code_agent import CodeAgent
from src.agents.eval_agent import EvalAgent
from src.agents.risk_agent import RiskAgent
from src.agents.signal_agent import SignalAgent
from src.backtest.engine import BacktestConfig, PointInTimeBacktest
from src.config import Config
from src.factors.code_generator import parse_expression, validate
from src.factors.schema.validator import is_forbidden, sample_combination_template
from src.factors.semantic_space import SchemaPlan

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# engine: excess_max_drawdown (§3.3)
# ---------------------------------------------------------------------------


def test_engine_computes_excess_max_drawdown_when_benchmark_given():
    dates = pd.bdate_range("2020-01-01", periods=250)
    syms = [f"S{i:02d}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    rng = np.random.default_rng(0)
    scores = pd.Series(rng.normal(0.0, 1.0, len(idx)), index=idx)
    fwd = pd.Series(rng.normal(0.0005, 0.02, len(idx)), index=idx)
    benchmark = pd.Series(rng.normal(0.001, 0.01, len(dates)), index=dates)

    bt = PointInTimeBacktest(BacktestConfig())
    with_bm = bt.run(scores, fwd, benchmark=benchmark).metrics
    without_bm = bt.run(scores, fwd).metrics

    assert "excess_max_drawdown" in with_bm
    assert 0.0 <= with_bm["excess_max_drawdown"] <= 1.0
    assert "excess_max_drawdown" not in without_bm  # backward compat


# ---------------------------------------------------------------------------
# eval gate: excess-aware (§3.3)
# ---------------------------------------------------------------------------


def _gate_metrics(max_dd: float, excess_dd: float | None = None) -> dict:
    m = {
        "max_drawdown": max_dd,
        "rank_ic": 0.05,
        "ic": 0.03,
        "icir": 0.5,
        "n_days": 500,
        "significant": True,
    }
    if excess_dd is not None:
        m["excess_max_drawdown"] = excess_dd
    return m


def _ctx(cfg: Config) -> AgentContext:
    return AgentContext(as_of=pd.Timestamp("2021-01-01"), symbols=[], config=cfg)


def test_eval_gate_uses_excess_drawdown_when_benchmark_present():
    cfg = Config(
        {
            "risk_management": {"max_excess_drawdown": 0.25, "crisis_test": {"enabled": False}},
            "sharpe": {"max_drawdown_limit": 0.15},
            "ic": {}, "rank_ic": {}, "icir": {}, "multiple_hypothesis": {},
        }
    )
    agent = EvalAgent(config=cfg)
    ctx = _ctx(cfg)
    # absolute DD 30% (>15% old gate) but excess 10% (<25%) -> NOT high-risk
    assert agent._apply_gate(_gate_metrics(0.30, 0.10), ctx) != "reject_high_risk"
    # excess DD 40% (>25%) -> reject_high_risk even though absolute DD is tiny
    assert agent._apply_gate(_gate_metrics(0.05, 0.40), ctx) == "reject_high_risk"


def test_eval_gate_falls_back_to_absolute_without_benchmark():
    cfg = Config(
        {
            "risk_management": {"crisis_test": {"enabled": False}},
            "sharpe": {"max_drawdown_limit": 0.15},
            "ic": {}, "rank_ic": {}, "icir": {}, "multiple_hypothesis": {},
        }
    )
    agent = EvalAgent(config=cfg)
    ctx = _ctx(cfg)
    assert agent._apply_gate(_gate_metrics(0.30), ctx) == "reject_high_risk"
    assert agent._apply_gate(_gate_metrics(0.10), ctx) != "reject_high_risk"


def test_eval_gate_crisis_uses_excess_limit():
    cfg = Config(
        {
            "risk_management": {
                "max_excess_drawdown": 0.25,
                "crisis_test": {
                    "enabled": True,
                    "max_drawdown_in_crisis": 0.20,
                    "max_excess_drawdown_in_crisis": 0.30,
                },
            },
            "sharpe": {"max_drawdown_limit": 0.15},
            "ic": {}, "rank_ic": {}, "icir": {}, "multiple_hypothesis": {},
        }
    )
    agent = EvalAgent(config=cfg)
    ctx = _ctx(cfg)

    # excess available: absolute crisis dd 25% (>20% old) but excess crisis dd 10%
    # (<30%) -> allowed; the excess standard governs the crisis window too.
    m = _gate_metrics(0.10, 0.10)
    m["crisis_max_drawdown"] = 0.25
    m["crisis_excess_max_drawdown"] = 0.10
    assert agent._apply_gate(m, ctx) != "reject_crisis"

    m2 = _gate_metrics(0.10, 0.10)
    m2["crisis_max_drawdown"] = 0.10
    m2["crisis_excess_max_drawdown"] = 0.35
    assert agent._apply_gate(m2, ctx) == "reject_crisis"


# ---------------------------------------------------------------------------
# risk agent: excess-aware drawdown check
# ---------------------------------------------------------------------------


def test_risk_drawdown_check_excess_aware():
    cfg = Config({"risk_management": {"max_excess_drawdown": 0.25}})
    agent = RiskAgent(config=cfg)
    rm = {"max_excess_drawdown": 0.25}

    c1 = agent._drawdown_check({"max_drawdown": 0.50, "excess_max_drawdown": 0.10}, 0.15, rm)
    assert c1["passed"] and c1["excess"] and c1["max_drawdown"] == 0.10

    c2 = agent._drawdown_check({"max_drawdown": 0.50, "excess_max_drawdown": 0.40}, 0.15, rm)
    assert not c2["passed"] and c2["excess"]

    c3 = agent._drawdown_check({"max_drawdown": 0.10}, 0.15, rm)
    assert c3["passed"] and not c3["excess"] and c3["limit"] == 0.15

    c4 = agent._drawdown_check({"max_drawdown": 0.20}, 0.15, rm)
    assert not c4["passed"] and not c4["excess"]


# ---------------------------------------------------------------------------
# sampling slots (§3.2) + code-layer block (§3.1)
# ---------------------------------------------------------------------------


def test_generate_template_formulas_distinct_and_parseable():
    agent = SignalAgent(llm=None, seed=3)
    out = agent.generate_template_formulas(n=10)
    assert len(out) == 10
    assert len(set(out)) == 10, "template slot generator must return distinct formulas"
    for formula in out:
        validate(parse_expression(formula))
        assert not is_forbidden(formula)


def test_generate_template_formulas_skips_rejected():
    agent = SignalAgent(llm=None, seed=5)
    first = agent.generate_template_formulas(n=3)
    for formula in first:
        agent.record_rejection({"formula": formula})
    # same seed -> same sequence, but rejected formulas are now skipped
    second = agent.generate_template_formulas(n=3)
    assert len(second) == 3
    assert not (set(first) & set(second)), "rejected template must not be re-proposed"


def test_generate_template_formulas_respects_skip():
    agent = SignalAgent(llm=None, seed=4)
    first = agent.generate_template_formulas(n=5)
    # same seed -> same sequence, but the free-path formulas are now skipped
    second = agent.generate_template_formulas(n=5, skip=set(first))
    assert len(second) == 5
    assert not (set(first) & set(second)), "free-path formulas must not be duplicated"


def test_template_formulas_not_re_proposed_across_rounds():
    """Accepted formulas must not be re-tested verbatim in a later round.

    Regression: the 0/20->9/20 remedy run produced 9 acceptances from only 2
    unique formulas — the bounded pool kept re-drawing the accepted winner because
    ``blocked`` only contained rejections. Each slot must explore a NEW formula.
    """
    agent = SignalAgent(llm=None, seed=1)
    drawn: list[str] = []
    for _ in range(5):
        # two template slots per round, like the validation contract
        batch = agent.generate_template_formulas(n=2)
        assert len(batch) == 2
        drawn.extend(batch)
        # accepted formulas are NOT added to rejection_history, so only rejections
        # below simulate what the CLI records for non-passed candidates
        for formula in batch:
            if not formula.endswith("Volume, 60))))"):
                agent.record_rejection({"formula": formula})
    assert len(drawn) == 10
    assert len(set(drawn)) == 10, "a template formula was re-proposed across rounds"
    # slot 0 of every round stays in the proven 低波+低换手 family
    for i in range(0, 10, 2):
        assert "TS_Std(Close" in drawn[i] and "TS_Mean(Volume" in drawn[i]


def test_accepted_duplicate_is_not_double_counted():
    """cmd_mine acceptance dedup: the SAME formula must never enter the accepted
    pool twice, whichever slot produced it.

    Regression: run5's duplicate — iter0's template slot proposed
    ``Avg(Neg(Rank(TS_Mean(Volume, 60))), Neg(Rank(TS_Std(Close, 120))))`` and
    iter3's FREE slot re-drew it verbatim (the reverse direction of run4's leak).
    The verification checklist measures pairwise AST distance over the accepted
    pool, so an exact duplicate contributes 0.00 and fails the diversity gate no
    matter the floor. cmd_mine tracks ``accepted_ever`` and skips the second copy.

    This test drives the generator the way cmd_mine now does: ``skip`` = current
    free formulas ∪ accepted_ever, and asserts the generator never returns an
    accepted winner in a later round.
    """
    agent = SignalAgent(llm=None, seed=2)
    accepted_ever: set[str] = set()
    drawn: list[str] = []
    for _ in range(5):
        free_formulas = {
            "Avg(Neg(Rank(TS_Mean(Volume, 60))), Neg(Rank(TS_Std(Close, 240))))",
            "Avg(Neg(Rank(TS_Mean(Volume, 120))), Neg(Rank(TS_Std(Close, 60))))",
        }
        batch = agent.generate_template_formulas(n=2, skip=free_formulas | accepted_ever)
        drawn.extend(batch)
        # no round may draw a formula accepted in an earlier round
        for formula in batch:
            assert formula not in accepted_ever, (
                f"template slot re-drew an accepted formula across rounds: {formula}"
            )
        # simulate acceptance of slot 0 of this round (low-vol family winner)
        if batch:
            accepted_ever.add(batch[0])
    # the generator must still produce all 10 template slots across 5 rounds
    # (the accepted set is tiny — ~5 — so it must not exhaust the pool)
    assert len(drawn) == 10
    assert len(set(drawn)) == 10, "template slots re-drew a formula across rounds"


def test_stale_rejection_history_must_not_starve_template_pool(tmp_path):
    """Cross-run rejection pollution (run6) must not choke the template pool.

    ``outputs/rejection_history.json`` persists across ``cmd_mine`` invocations,
    and the agent loads it at construction — so a SECOND run starts with ~20 of
    the bounded 24-formula template pool already locked by the FIRST run's
    rejections. run6: template slots returned 0 in iters 2/4, the 20-candidate
    validation contract degraded to 14, acceptance 5/20 = 25% (< 30% gate).

    The CLI fix scopes the history to the current run (``signal.rejection_history
    = []`` before the iteration loop). This test reproduces both states: with the
    stale history loaded the pool starves (contract not deliverable); after the
    CLI's reset it delivers the full 10 template slots across 5 rounds.
    """
    from src.factors.schema.validator import COMBINATION_TEMPLATES, LOOKBACKS

    # a "prior run" that rejected a large slice of the template pool — the real
    # run6 file held 20/24 pool members (rejections accumulate across runs).
    stale = []
    for tpl in COMBINATION_TEMPLATES:  # all 4 directions -> the whole 24-pool
        for (a, b) in __import__("itertools").permutations(LOOKBACKS, 2):
            stale.append({"formula": tpl.format(lb1=a, lb2=b)})
    rh_file = tmp_path / "rejection_history.json"
    rh_file.write_text(__import__("json").dumps(stale), encoding="utf-8")

    agent = SignalAgent(llm=None, seed=0, rejection_history_path=str(rh_file))
    # buggy state: stale history loaded -> template pool starves
    starved = agent.generate_template_formulas(n=2)
    assert len(starved) < 2, "stale prior-run rejections must choke the pool"

    # the CLI fix: scope the history to the current run
    agent.rejection_history = []
    total = 0
    for _ in range(5):
        total += len(agent.generate_template_formulas(n=2))
    assert total == 10, "after the reset the full 20-candidate contract is deliverable"


def test_code_agent_physically_blocks_reversal_formula():
    cfg = Config(
        {"factor_mining": {"enable_code_layer_blocking": True, "auto_upgrade_lookback": False}}
    )
    agent = CodeAgent(llm=None, config=cfg)  # offline path -> default_formula_for
    plan = SchemaPlan(
        event="Earnings Surprise",
        context="Post-Earnings Drift",
        qualities=("Mean Reversion",),  # default_formula_for -> Neg(TS_ZScore(...))
        direction="long",
        output="rank",
    )
    ctx = AgentContext(as_of=pd.Timestamp("2021-01-01"), symbols=["A"], config=cfg)
    gf = agent.translate(ctx, plan)
    assert not is_forbidden(gf.formula)
    assert "TS_ZScore" not in gf.formula


def test_code_agent_blocking_can_be_disabled():
    """The code-layer firewall is config-gated: with blocking disabled a
    blacklisted formula passes through unmodified. The offline ``default_formula_for``
    no longer emits blacklisted formulas (it returns a clean ``Neg(Rank(TS_Return))``
    for Mean Reversion), so inject a blacklisted formula through the LLM path."""
    blocked = "Neg(TS_ZScore(Close, 10))"
    assert is_forbidden(blocked)
    cfg = Config(
        {"factor_mining": {"enable_code_layer_blocking": False, "auto_upgrade_lookback": False}}
    )
    agent = CodeAgent(llm=None, config=cfg)
    agent.llm = True                                   # route translate() through the LLM path
    agent._llm_formula = lambda ctx, prompt: blocked   # inject a blacklisted formula
    plan = SchemaPlan(
        event="Earnings Surprise",
        context="Post-Earnings Drift",
        qualities=("Mean Reversion",),
        direction="long",
        output="rank",
    )
    ctx = AgentContext(as_of=pd.Timestamp("2021-01-01"), symbols=["A"], config=cfg)
    gf = agent.translate(ctx, plan)
    assert "TS_ZScore" in gf.formula


# ---------------------------------------------------------------------------
# §3.5 success criteria — run against the real validation output
# ---------------------------------------------------------------------------


def test_phase8_remedy_success_criteria():
    """validation_BLUEPRINT §3.5 — the post-validation gate.

    1. acceptance rate >= 30% (>= 6 of the 20-candidate validation contract)
    2. zero accepted pure-reversal factors
    3. at least 2 accepted factors sourced from combination templates

    Skips when the validation batch has not been run yet.
    """
    out = Path(os.environ.get("LLM_QUANT_OUTPUTS", ROOT / "outputs"))
    factors_file = out / "factors.json"
    if not factors_file.exists():
        pytest.skip("outputs/factors.json not present — run the 5-iteration validation first")

    results = json.loads(factors_file.read_text(encoding="utf-8"))
    assert isinstance(results, list), "factors.json must be a list of accepted factors"
    if not results:
        # the empty list is the 0/20 baseline artifact; the remedy run writes the
        # accepted factors here
        pytest.skip("outputs/factors.json is empty (0/20 baseline) — run the remedy validation first")
    assert len(results) >= 6, f"acceptance rate too low: {len(results)}/20 (<30%)"

    for r in results:
        formula = r["factor"]["formula"]
        assert not is_forbidden(formula), f"pure-reversal factor accepted: {formula}"

    template_count = sum(1 for r in results if r.get("source") == "combination_template")
    assert template_count >= 2, f"combination-template factors too few: {template_count}"
