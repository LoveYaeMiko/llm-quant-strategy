"""LIMIT_DOWN blueprint 方案 C — mining rejection feedback loop.

Rejected factors (verdict, IC / Sharpe / drawdown, reason) are recorded to a
persistent JSON store and replayed into the next LLM proposal prompt together
with hard constraints — no naive reversal (A-share limit-down continuations kill
it), a 60-day minimum lookback, and the crisis-test self-check.
"""

from __future__ import annotations

import json

import pandas as pd

from src.agents.base_agent import AgentContext
from src.agents.signal_agent import SignalAgent


def _ctx() -> AgentContext:
    return AgentContext(
        as_of=pd.Timestamp("2026-01-05"), symbols=["600519.SH"], config=None
    )


def _rejected() -> dict:
    return {
        "formula": "Neg(TS_ZScore(Close, 10))",
        "verdict": "reject_high_risk",
        "reason": "股灾中 -10% 跌停连板导致组合回撤超限",
        "ic": 0.021,
        "rank_ic": 0.030,
        "sharpe": -1.5,
        "max_drawdown": 0.35,
    }


def test_record_rejection_persists_and_reloads(tmp_path):
    path = tmp_path / "rejection_history.json"
    agent = SignalAgent(rejection_history_path=str(path))
    assert not path.exists()
    agent.record_rejection(_rejected())
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[0]["formula"] == "Neg(TS_ZScore(Close, 10))"

    # A fresh agent loads the same history back off disk.
    fresh = SignalAgent(rejection_history_path=str(path))
    assert fresh.rejection_history == agent.rejection_history


def test_build_rejection_feedback_contains_reason_and_hard_constraints():
    agent = SignalAgent()
    agent.record_rejection(_rejected())
    fb = agent._build_rejection_feedback()
    assert "Neg(TS_ZScore(Close, 10))" in fb          # the rejected formula
    assert "拒绝原因" in fb and "股灾中" in fb          # the reason is surfaced
    assert "严禁" in fb                               # hard constraint no reversal
    assert "≥ 60" in fb                               # 60-day lookback floor
    assert "股灾压力测试" in fb                        # crisis self-check
    assert "IC=0.021" in fb                           # the metrics recap


def test_feedback_injected_into_llm_prompt_when_enabled():
    agent = SignalAgent(rejection_history_path=None, feedback_enabled=True)
    agent.record_rejection(_rejected())
    captured: dict = {}

    def fake_complete(prompt, context, *, temperature=0.0, max_tokens=1024):
        captured["prompt"] = prompt
        return json.dumps(
            [
                {
                    "event": "Trend Following",
                    "context": "Normal Regime",
                    "qualities": ["Momentum"],
                    "direction": "long",
                    "output": "score",
                }
            ]
        )

    agent._complete = fake_complete
    plans = agent._llm_proposed_plans(_ctx(), 1)
    assert len(plans) == 1
    assert "硬约束指令" in captured["prompt"]
    assert "Neg(TS_ZScore(Close, 10))" in captured["prompt"]
    assert "严禁" in captured["prompt"]


def test_feedback_omitted_when_disabled():
    agent = SignalAgent(rejection_history_path=None, feedback_enabled=False)
    agent.record_rejection(_rejected())
    captured: dict = {}

    def fake_complete(prompt, context, *, temperature=0.0, max_tokens=1024):
        captured["prompt"] = prompt
        return "[]"

    agent._complete = fake_complete
    plans = agent._llm_proposed_plans(_ctx(), 1)
    assert plans == []
    assert "硬约束指令" not in captured["prompt"]
