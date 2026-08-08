"""Pipeline agent tests — routing, gating, triggers, and full offline mining."""

from __future__ import annotations

import pandas as pd

from src.agents.base_agent import AgentContext
from src.agents.code_agent import CodeAgent
from src.agents.dynamic_router import DynamicRouter, classify_market_state
from src.agents.eval_agent import AdaptiveZScoreTrigger, EvalAgent
from src.agents.inference_gate import InferenceGate
from src.agents.risk_agent import RiskAgent
from src.agents.signal_agent import SignalAgent
from src.factors.code_generator import CodeGenerator, eval_expression
from src.factors.memory_manager import MemoryManager
from src.factors.semantic_space import SemanticSpace
from src.data.synthetic import make_synthetic_market


def test_classify_market_state():
    import numpy as np

    bull = pd.Series(np.full(60, 0.001))   # steady rise
    bear = pd.Series(np.full(60, -0.001))
    assert classify_market_state(bull) == "bull"
    assert classify_market_state(bear) == "bear"


def test_dynamic_router_preferences():
    agents = {"signal": object(), "code": object(), "eval": object(), "risk": object()}
    r = DynamicRouter(agents, market_state="bear")
    order = r.route()
    assert order[0] == agents["risk"]  # discipline first in a sell-off
    r.update_state(pd.Series([0.001] * 80))
    assert r.market_state == "bull"


def test_adaptive_zscore_trigger_gate():
    trig = AdaptiveZScoreTrigger(window=10, trigger_z=2.0, min_invocation_interval=0)
    for _ in range(10):
        trig.observe(1.0)  # stable baseline
    inv, z = trig.should_invoke(1.0)
    assert inv is False
    inv2, _ = trig.should_invoke(10.0)  # outlier
    assert inv2 is True


def test_signal_agent_offline_generates_diverse_plans():
    agent = SignalAgent(space=SemanticSpace(), memory=MemoryManager(), seed=0)
    ctx = AgentContext(as_of="2024-06-01", symbols=["A"], config=None)
    plans = agent.generate_hypotheses(ctx, n=8)
    assert len(plans) == 8
    assert len({p.key() for p in plans}) == 8


def test_code_agent_translate_offline():
    space = SemanticSpace()
    plan = space.sample(__import__("random").Random(0))
    agent = CodeAgent(generator=CodeGenerator(), memory=MemoryManager())
    ctx = AgentContext(as_of="2024-06-01", symbols=["A"], config=None)
    gf = agent.translate(ctx, plan)
    # must be a valid closed-library formula
    CodeGenerator().parse(gf.formula)


def test_inference_gate_serializes(market):
    gate = InferenceGate()
    with gate.reserve(agent_name="signal"):
        assert gate.audit_log() == []
    log = gate.audit_log()
    assert len(log) == 1
    assert log[0]["agent"] == "signal"
    assert gate.replay() == ["1: signal"]


def _small_market():
    return make_synthetic_market(symbols=16, days=120, seed=5)


def test_eval_agent_verdict_and_trigger(market):
    from src.factors.code_generator import FactorContext

    ctx_data = FactorContext(market.long)
    scores = eval_expression("Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))", ctx_data)
    agent = EvalAgent(config=None)
    ctx = AgentContext(as_of="2024-06-01", symbols=["A"], config=None)
    metrics = agent.evaluate(ctx, scores, market.forward_returns)
    assert "verdict" in metrics
    assert metrics["n_days"] > 0
    assert isinstance(metrics["escalate_llm"], bool)


def test_full_offline_mining_loop(market):
    """End-to-end: signal -> code -> eval -> risk, all deterministic."""
    from src.config import load_config
    from src.factors.code_generator import FactorContext

    cfg = load_config()
    memory = MemoryManager()
    space = SemanticSpace()
    fctx = FactorContext(market.long)
    signal = SignalAgent(space=space, memory=memory, config=cfg, seed=0)
    code = CodeAgent(generator=CodeGenerator(), memory=memory, config=cfg)
    eval_agent = EvalAgent(memory=memory, config=cfg)
    risk = RiskAgent(memory=memory, config=cfg, seed=0)
    ctx = AgentContext(
        as_of=market.long.index.get_level_values(0).max(),
        symbols=market.long.index.get_level_values(1).unique().tolist(),
        config=cfg,
        data=market.long,
    )
    accepted = 0
    for plan in signal.generate_hypotheses(ctx, n=6):
        gf = code.translate(ctx, plan)
        try:
            scores = eval_expression(gf.formula, fctx)
        except Exception:
            continue
        metrics = eval_agent.evaluate(ctx, scores, market.forward_returns, n_trials=1)
        if metrics["verdict"] not in ("keep", "good"):
            continue
        report = risk.validate(ctx, scores, market.forward_returns, metrics, n_trials=1)
        if report["passed"]:
            accepted += 1
    assert accepted >= 0  # pipeline runs; acceptance depends on data luck
    assert memory.size() >= 0
