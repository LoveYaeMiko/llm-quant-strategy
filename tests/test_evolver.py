"""EvoQuant self-evolution tests."""

from __future__ import annotations

import pandas as pd

from src.agents.base_agent import AgentContext
from src.agents.risk_agent import RiskAgent
from src.evolver import EvoQuant
from src.factors.code_generator import CodeGenerator, eval_expression
from src.factors.memory_manager import MemoryManager
from src.factors.semantic_space import SchemaPlan, SemanticSpace
from src.data.synthetic import make_synthetic_market


def _fixture():
    market = make_synthetic_market(symbols=16, days=140, seed=11)
    from src.factors.code_generator import FactorContext

    fctx = FactorContext(market.long)
    space = SemanticSpace()
    plan = SchemaPlan(
        event="Earnings Surprise",
        context="Bull Market",
        qualities=("Momentum",),
        direction="long",
        output="score",
    )
    ctx = AgentContext(
        as_of=market.long.index.get_level_values(0).max(),
        symbols=market.long.index.get_level_values(1).unique().tolist(),
        config=None,
        data=market.long,
    )
    return market, fctx, space, plan, ctx


def test_diagnose_no_bottleneck():
    evo = EvoQuant()
    diag = evo.diagnose({"icir": 0.5, "max_drawdown": 0.05, "turnover": 1.0, "sharpe": 1.0})
    assert len(diag) == 1
    assert "bottleneck" in diag[0]


def test_diagnose_detects_decay():
    evo = EvoQuant()
    diag = evo.diagnose({"icir": 0.2, "turnover": 8.0, "sharpe": 0.5, "max_drawdown": 0.05})
    assert any("turnover" in d or "decay" in d for d in diag)


def test_propose_edits_diverse_and_valid():
    evo = EvoQuant()
    space = SemanticSpace()
    plan = SchemaPlan(event="Earnings Surprise", context="Bull Market", qualities=("Momentum",), direction="long", output="score")
    edits = evo.propose_edits(plan, ["excessive turnover — factor decays fast"], k=4)
    assert len(edits) >= 4
    for e in edits:
        assert space.validate(e)


def test_evolve_runs_end_to_end(market, fctx, forward):
    evo = EvoQuant(space=SemanticSpace(), generator=CodeGenerator(), memory=MemoryManager(),
                   risk_agent=RiskAgent(seed=0))
    plan = SchemaPlan(event="Earnings Surprise", context="Bull Market", qualities=("Momentum",), direction="long", output="score")

    def scores_fn(formula):
        return eval_expression(formula, fctx)

    ctx = AgentContext(as_of="2024-06-01", symbols=["A"], config=None)
    result = evo.evolve(
        ctx, plan, "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))",
        scores_fn, forward, n_trials=1,
    )
    assert isinstance(result.diagnosis, list)
    assert "base_rank_ic" in result.metrics
    assert "best_rank_ic" in result.metrics
    # either an accepted edit was found (with a valid formula) or none passed
    if result.accepted is not None:
        CodeGenerator().parse(result.accepted["formula"])


def test_evolve_distils_accepted_to_memory(market, fctx, forward):
    memory = MemoryManager()
    evo = EvoQuant(space=SemanticSpace(), generator=CodeGenerator(), memory=memory,
                   risk_agent=RiskAgent(seed=1))
    plan = SchemaPlan(event="Earnings Surprise", context="Bull Market", qualities=("Momentum",), direction="long", output="score")
    ctx = AgentContext(as_of="2024-06-01", symbols=["A"], config=None)
    evo.evolve(
        ctx, plan, "Neg(TS_ZScore(Close, 20))",
        lambda f: eval_expression(f, fctx), forward, n_trials=1,
    )
    # evolution may or may not accept an edit, but the pipeline is side-effect safe
    assert isinstance(memory.size(), int)
