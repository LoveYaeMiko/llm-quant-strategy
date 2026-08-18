"""Debate Agent — Bull/Bear adversarial review before a factor is accepted.

TradingAgents (arXiv:2412.20138) frames a trading decision as a Bull/Bear
debate with structured documents; AgenticAITA adds the adversarial check. FQA
adapts the *idea* — a candidate factor must survive an explicit bear case, not
just clear the (already multi-stage) risk gate — while keeping the verdict
deterministic so the pipeline stays offline-capable:

* the **bull** case cites the factor's positive evidence (strong RankIC, positive
  tail spread, significant Deflated Sharpe, stable ICIR, small in/out gap);
* the **bear** case cites the failure modes (weak/level-effect IC, sub-critical
  Deflated Sharpe, excessive turnover, large in/out gap);
* the **judge** nets the two into a ``margin = bull − bear``; ``passed`` when the
  margin clears ``debate.min_margin``.

An LLM (the *deep* tier, per B2) can enrich the synthesis with a natural-language
verdict, but the pass/fail call never depends on it.
"""

from __future__ import annotations

from typing import Optional

from ..config import Config
from .base_agent import AgentContext, AgentResult, BaseAgent


class DebateAgent(BaseAgent):
    name = "debate"

    def __init__(
        self,
        llm: Optional[object] = None,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(llm=llm, config=config)

    # -- deterministic bull / bear cases ------------------------------------

    def bull_points(self, metrics: dict, risk_report: Optional[dict] = None) -> list[str]:
        """Positive evidence for the factor's alpha being real."""
        pts: list[str] = []
        ric = float(metrics.get("rank_ic", 0.0))
        if ric > 0.03:
            pts.append(f"RankIC {ric:.3f} 显著为正")
        ts = metrics.get("tail_spread")
        if ts is not None and float(ts) > 0:
            pts.append(f"尾部多空价差 {float(ts):.3f} 为正（真可交易，非 level 效应）")
        dsr = metrics.get("deflated_sharpe")
        if dsr is not None and float(dsr) >= 0.95:
            pts.append(f"Deflated Sharpe {float(dsr):.2f} ≥ 0.95（显著超过运气基线）")
        if float(metrics.get("icir", 0.0)) > 0.30:
            pts.append("ICIR 稳定")
        of = (risk_report or {}).get("checks", {}).get("overfitting", {})
        if of.get("passed"):
            pts.append("样本内/外 IC 差距小（无过拟合迹象）")
        return pts

    def bear_points(self, metrics: dict, risk_report: Optional[dict] = None) -> list[str]:
        """Failure modes — why the factor is plausibly overfit or untradeable."""
        pts: list[str] = []
        ric = float(metrics.get("rank_ic", 0.0))
        if ric <= 0.02:
            pts.append(f"RankIC {ric:.3f} 未达初筛线")
        ts = metrics.get("tail_spread")
        if ts is not None and float(ts) < 0:
            pts.append(f"尾部多空价差 {float(ts):.3f} 为负（level 效应，不可交易）")
        dsr = metrics.get("deflated_sharpe")
        if dsr is not None and float(dsr) < 0.90:
            pts.append(f"Deflated Sharpe {float(dsr):.2f} < 0.90（大概率是搜出来的运气）")
        if float(metrics.get("turnover", 0.0)) > 5.0:
            pts.append("换手过高（因子快速衰减）")
        of = (risk_report or {}).get("checks", {}).get("overfitting", {})
        if not of.get("passed"):
            pts.append("样本内/外 IC 差距大（过拟合）")
        return pts

    # -- LLM synthesis (deep tier, optional) ---------------------------------

    def _llm_synthesis(
        self,
        context: AgentContext,
        formula: str,
        bull: list[str],
        bear: list[str],
        margin: int,
        verdict: str,
    ) -> Optional[str]:
        if self.llm is None:
            return None
        prompt = (
            "You are the judge of a factor-mining debate. Summarise in one or two "
            "Chinese sentences whether the bull or bear case is stronger, and why.\n\n"
            f"FACTOR: {formula}\n"
            f"BULL: {'; '.join(bull) or '（无）'}\n"
            f"BEAR: {'; '.join(bear) or '（无）'}\n"
            f"DETERMINISTIC NET MARGIN: {margin:+d} → verdict {verdict}\n"
        )
        try:
            return self._complete(prompt, context, temperature=0.2, max_tokens=256).strip()
        except Exception:
            return None

    # -- debate -------------------------------------------------------------

    def debate(
        self,
        context: AgentContext,
        formula: str,
        metrics: dict,
        risk_report: Optional[dict] = None,
    ) -> dict:
        """Run the bull/bear review and return a structured verdict."""
        cfg: Optional[Config] = context.config or self.config
        db = cfg.section("debate") if cfg else {}
        bull = self.bull_points(metrics, risk_report)
        bear = self.bear_points(metrics, risk_report)
        margin = len(bull) - len(bear)
        verdict = "bull" if margin > 0 else ("bear" if margin < 0 else "tie")
        passed = margin >= int(db.get("min_margin", 0))
        synthesis = self._llm_synthesis(context, formula, bull, bear, margin, verdict)
        return {
            "passed": bool(passed),
            "verdict": verdict,
            "margin": margin,
            "bull": bull,
            "bear": bear,
            "bull_text": "；".join(bull) or "（无）",
            "bear_text": "；".join(bear) or "（无）",
            "synthesis": synthesis or f"多空净得分 {margin:+d}，判定 {verdict}",
        }

    def run(
        self,
        context: AgentContext,
        factor,
        metrics: dict,
        risk_report: Optional[dict] = None,
    ) -> AgentResult:
        report = self.debate(context, factor, metrics, risk_report)
        return AgentResult(
            agent=self.name,
            summary=f"debate {report['verdict']} (margin {report['margin']:+d})",
            artifacts={"debate_report": report},
            context=context,
        )
