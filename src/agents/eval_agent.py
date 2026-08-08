"""Evaluation Agent — IC/RankIC evaluator with FinCAD awareness + gating.

Blueprinted as the quant who runs the backtest, computes IC/RankIC, applies the
layered thresholds and — via the **Adaptive Z-Score Trigger Engine** (AgenticAITA,
review.md §2.1) — decides whether a *statistical anomaly* justifies calling the
heavy LLM critic. When nothing anomalous is happening the pipeline stays cheap;
that gate is the difference between a $20/mo pipeline and a $5,000/mo one.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from ..backtest import metrics as M
from ..backtest.engine import BacktestConfig, PointInTimeBacktest
from ..bias_control.context_decoder import LLMBackend
from ..bias_control.look_ahead_detector import LookAheadDetector
from ..config import Config
from ..factors.memory_manager import MemoryManager
from .base_agent import AgentContext, AgentResult, BaseAgent


class AdaptiveZScoreTrigger:
    """AgenticAITA Adaptive Z-Score Trigger Engine.

    Maintains a rolling window of a market/factor metric. When the latest value
    deviates by more than ``trigger_z`` standard deviations AND enough time has
    passed since the last invocation, the heavy LLM path is allowed to run.
    """

    def __init__(
        self,
        window: int = 20,
        trigger_z: float = 2.0,
        min_invocation_interval: float = 60.0,
    ) -> None:
        self.window = max(5, int(window))
        self.trigger_z = float(trigger_z)
        self.min_invocation_interval = float(min_invocation_interval)
        self._history: deque[float] = deque(maxlen=self.window)
        self._last_invocation: Optional[float] = None

    def observe(self, value: float) -> float:
        """Feed a value; returns its z-score in the rolling window."""
        self._history.append(float(value))
        if len(self._history) < max(5, self.window // 2):
            return 0.0
        import numpy as np

        arr = np.asarray(self._history, dtype=float)
        sd = arr.std()
        if sd == 0:
            return 0.0
        return float((value - arr.mean()) / sd)

    def should_invoke(self, value: float, now: Optional[float] = None) -> tuple[bool, float]:
        """True when ``value`` is an anomaly and the gate permits a call."""
        z = self.observe(value)
        if abs(z) <= self.trigger_z:
            return False, z
        if self._last_invocation is not None and now is not None:
            if (now - self._last_invocation) < self.min_invocation_interval:
                return False, z
        self._last_invocation = now
        return True, z


class EvalAgent(BaseAgent):
    name = "eval"

    def __init__(
        self,
        backtester: Optional[PointInTimeBacktest] = None,
        detector: Optional[LookAheadDetector] = None,
        memory: Optional[MemoryManager] = None,
        llm: Optional[LLMBackend] = None,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(llm=llm, config=config)
        self.backtester = backtester or PointInTimeBacktest(
            BacktestConfig(
                max_position_pct=float(
                    config.get("online_execution.max_position_pct", 0.05) if config else 0.05
                )
            )
        )
        self.detector = detector
        self.memory = memory
        w = config.get("trigger.adaptive_zscore.window", 20) if config else 20
        z = config.get("trigger.adaptive_zscore.trigger_z", 2.0) if config else 2.0
        iv = config.get("trigger.adaptive_zscore.min_invocation_interval", 60) if config else 60
        self.trigger = AdaptiveZScoreTrigger(window=w, trigger_z=z, min_invocation_interval=iv)

    # -- core evaluation ----------------------------------------------------

    def evaluate(
        self,
        context: AgentContext,
        scores,
        forward,
        n_trials: int = 1,
    ) -> dict:
        bt = self.backtester.run(scores, forward)
        fe = M.factor_eval(scores, forward, n_trials=n_trials)
        metrics: dict = {**fe, **bt.metrics}
        # FinCAD bookkeeping: how many future-date references were suppressed
        # on the path that produced this factor (0 when no model involved).
        if self.fincad is not None:
            metrics["fincad_suppressions"] = self.fincad.total_suppressions
        metrics["verdict"] = self._apply_gate(metrics, context, n_trials=n_trials)
        metrics["escalate_llm"], metrics["trigger_z"] = self.trigger.should_invoke(
            fe.get("rank_ic", 0.0)
        )
        return metrics

    def _apply_gate(self, metrics: dict, context: AgentContext, n_trials: int = 1) -> str:
        """Layered gate from configs/factor_thresholds.yaml."""
        cfg: Optional[Config] = context.config or self.config
        ic = cfg.section("ic") if cfg else {}
        rk = cfg.section("rank_ic") if cfg else {}
        icir = cfg.section("icir") if cfg else {}
        sh = cfg.section("sharpe") if cfg else {}
        mh = cfg.section("multiple_hypothesis") if cfg else {}

        keep_ic = ic.get("keep_threshold", 0.02)
        good_ic = ic.get("good_threshold", 0.04)
        rk_keep = rk.get("keep_threshold", 0.035)
        icir_keep = icir.get("keep_threshold", 0.30)
        dd_limit = sh.get("max_drawdown_limit", 0.15)

        if metrics.get("max_drawdown", 0.0) > dd_limit:
            return "reject_high_risk"
        if n_trials > 1 and not metrics.get("significant", False):
            # FINSABER: significance after multiple-hypothesis correction
            if metrics.get("rank_ic", 0.0) < good_ic:
                return "reject_snooped"
        rank_ic = metrics.get("rank_ic", 0.0)
        icir_v = metrics.get("icir", 0.0)
        if rank_ic >= good_ic:
            return "good"
        if rank_ic >= rk_keep and icir_v >= icir_keep:
            return "keep"
        if metrics.get("ic", 0.0) >= keep_ic:
            return "keep"
        return "reject"

    def run(self, context: AgentContext, factor, scores, forward, n_trials: int = 1) -> AgentResult:
        metrics = self.evaluate(context, scores, forward, n_trials=n_trials)
        return AgentResult(
            agent=self.name,
            summary=(
                f"verdict={metrics['verdict']} rank_ic={metrics.get('rank_ic', 0.0):.4f} "
                f"icir={metrics.get('icir', 0.0):.3f} sharpe={metrics.get('sharpe', 0.0):.2f}"
            ),
            artifacts={
                "metrics": metrics,
                "factor_name": getattr(factor, "name", None),
                "escalate_llm": metrics.get("escalate_llm", False),
            },
            context=context,
        )
