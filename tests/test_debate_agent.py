"""辩论评审 — Bull/Bear adversarial review before acceptance."""

from __future__ import annotations

from src.agents.base_agent import AgentContext
from src.agents.debate_agent import DebateAgent
from src.config import Config


def _ctx(config=None):
    return AgentContext(as_of="2024-01-01", symbols=["A"], config=config)


def _strong_metrics():
    return {
        "rank_ic": 0.06, "icir": 0.5, "sharpe": 1.5,
        "max_drawdown": 0.05, "tail_spread": 0.12, "deflated_sharpe": 1.2,
        "turnover": 1.0,
    }


def _weak_metrics():
    return {
        "rank_ic": 0.01, "icir": 0.1, "sharpe": -0.5,
        "max_drawdown": 0.2, "tail_spread": -0.05, "deflated_sharpe": 0.3,
        "turnover": 8.0,
    }


def _passing_risk():
    return {"passed": True, "checks": {"overfitting": {"passed": True}}}


def _failing_risk():
    return {"passed": False, "checks": {"overfitting": {"passed": False}}}


def test_bull_points_for_strong_factor():
    agent = DebateAgent()
    pts = agent.bull_points(_strong_metrics(), _passing_risk())
    assert len(pts) >= 4  # IC, tail spread, DSR, ICIR, overfit all fire


def test_bear_points_for_weak_factor():
    agent = DebateAgent()
    pts = agent.bear_points(_weak_metrics(), _failing_risk())
    assert len(pts) >= 4  # IC, level effect, DSR, turnover, overfit all fire


def test_debate_bull_wins_for_strong_factor():
    agent = DebateAgent()
    report = agent.debate(_ctx(), "Rank(Close)", _strong_metrics(), _passing_risk())
    assert report["verdict"] == "bull"
    assert report["passed"] is True
    assert report["margin"] > 0


def test_debate_bear_wins_for_weak_factor():
    agent = DebateAgent()
    report = agent.debate(_ctx(), "Rank(Close)", _weak_metrics(), _failing_risk())
    assert report["verdict"] == "bear"
    assert report["passed"] is False
    assert report["margin"] < 0


def test_debate_offline_synthesis_is_deterministic():
    agent = DebateAgent()  # no LLM
    report = agent.debate(_ctx(), "Rank(Close)", _weak_metrics(), _failing_risk())
    assert "bear" in report["synthesis"]


def test_debate_tie_passes_at_default_margin():
    # icir low (no bull) + turnover high (one bear) + overfit passed (one bull) -> 1-1
    metrics = {"rank_ic": 0.025, "icir": 0.1, "turnover": 6.0,
               "tail_spread": None, "deflated_sharpe": None}
    agent = DebateAgent()
    report = agent.debate(_ctx(), "F", metrics, _passing_risk())
    assert report["margin"] == 0
    assert report["verdict"] == "tie"
    assert report["passed"] is True  # min_margin 0 -> tie passes


def test_debate_respects_config_min_margin():
    metrics = {"rank_ic": 0.025, "icir": 0.1, "turnover": 6.0,
               "tail_spread": None, "deflated_sharpe": None}
    cfg = Config({"debate": {"min_margin": 1, "require_pass": True}})
    agent = DebateAgent(config=cfg)
    report = agent.debate(_ctx(config=cfg), "F", metrics, _passing_risk())
    assert report["margin"] == 0
    assert report["passed"] is False  # min_margin 1 -> tie fails
