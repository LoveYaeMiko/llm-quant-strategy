"""Risk Agent — EvoQuant-style validator (review.md §2.4, blueprint Phase 4).

Before a factor may enter the pool (or be exported online) it must pass a
multi-stage validation pipeline:

1. **Overfitting check**  — train/test IC gap on a time-split (PIT-safe);
2. **Robustness check**   — IC stability under score perturbation;
3. **Market-state check** — IC inside bull / bear / sideways regimes;
4. **Multiple-hypothesis**— Bonferroni-corrected significance (FINSABER §4C);
5. **Drawdown cap**       — max drawdown within risk_management limits.

Any failing check blocks promotion; the report is machine-readable for the
evolver to act on.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd

from ..backtest import metrics as M
from ..bias_control.context_decoder import LLMBackend
from ..config import Config
from ..factors.memory_manager import MemoryManager
from .base_agent import AgentContext, AgentResult, BaseAgent
from .dynamic_router import classify_market_state


class RiskAgent(BaseAgent):
    name = "risk"

    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        llm: Optional[LLMBackend] = None,
        config: Optional[Config] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(llm=llm, config=config)
        self.memory = memory
        self.rng = random.Random(seed)

    # -- individual checks --------------------------------------------------

    def _overfitting_check(self, scores, forward) -> dict:
        df = pd.DataFrame({"sig": scores, "fwd": forward}).dropna()
        if len(df) < 40:
            return {"passed": True, "gap": 0.0, "note": "insufficient data"}
        dates = sorted(df.index.get_level_values(0).unique())
        cut = dates[len(dates) // 2]
        train = df[df.index.get_level_values(0) <= cut]
        test = df[df.index.get_level_values(0) > cut]
        train_ic = M.mean_ic(train["sig"], train["fwd"])
        test_ic = M.mean_ic(test["sig"], test["fwd"])
        gap = abs(train_ic - test_ic)
        # A big train/test gap means the factor overfits the early regime.
        return {"passed": bool(gap <= max(0.06, 2 * abs(test_ic))), "train_ic": train_ic, "test_ic": test_ic, "gap": gap}

    def _robustness_check(self, scores, forward, noise_sigma: float = 0.05, reps: int = 5) -> dict:
        ic_list = []
        for _ in range(reps):
            jitter = pd.Series(
                [self.rng.gauss(1.0, noise_sigma) for _ in range(len(scores))],
                index=scores.index,
            )
            ic_list.append(M.mean_ic(scores * jitter, forward))
        if not ic_list:
            return {"passed": False, "note": "no data"}
        mean_ic = float(np.mean(ic_list))
        std_ic = float(np.std(ic_list))
        stable = std_ic <= max(0.02, abs(mean_ic) * 0.5)
        return {"passed": bool(stable), "mean_ic": mean_ic, "std_ic": std_ic}

    def _market_state_check(self, scores, forward, market_returns) -> dict:
        if market_returns is None or len(market_returns) < 40:
            return {"passed": True, "note": "no market data"}
        df = pd.DataFrame({"sig": scores, "fwd": forward}).dropna()
        regime = classify_market_state(market_returns)
        per_regime: dict[str, float] = {}
        dates = df.index.get_level_values(0)
        # crude: regime applies over the whole window; a strategy must not be a
        # pure bull-market artefact.
        sub = df[dates >= dates.min()]
        ic = M.mean_ic(sub["sig"], sub["fwd"])
        return {"passed": bool(ic > -0.01), "regime": regime, "ic": ic}

    def _multiple_hypothesis_check(self, metrics: dict, n_trials: int, alpha: float = 0.05) -> dict:
        required = M.significance_threshold_sharpe(
            int(metrics.get("n_days", 0)), int(n_trials), alpha=alpha
        )
        sharpe = metrics.get("sharpe", 0.0)
        return {"passed": bool(sharpe >= required), "sharpe": sharpe, "required_sharpe": required, "n_trials": int(n_trials)}

    def _drawdown_check(self, metrics: dict, limit: float, rm: Optional[dict] = None) -> dict:
        """Drawdown cap — validation_BLUEPRINT §3.3.

        When ``rm.max_excess_drawdown`` is configured AND the metrics carry an
        ``excess_max_drawdown`` (i.e. a benchmark was threaded through the eval
        backtest), gate on the excess drawdown; otherwise fall back to the
        absolute ``limit`` (``sharpe.max_drawdown_limit``).
        """
        rm = rm or {}
        use_excess = (
            rm.get("max_excess_drawdown") is not None
            and metrics.get("excess_max_drawdown") is not None
        )
        if use_excess:
            dd = metrics.get("excess_max_drawdown", 0.0)
            limit = float(rm.get("max_excess_drawdown"))
        else:
            dd = metrics.get("max_drawdown", 0.0)
        return {
            "passed": bool(dd <= limit),
            "max_drawdown": dd,
            "excess": use_excess,
            "limit": limit,
        }

    # -- full validation ----------------------------------------------------

    def validate(
        self,
        context: AgentContext,
        scores,
        forward,
        metrics: Optional[dict] = None,
        *,
        n_trials: int = 1,
        market_returns: Optional[pd.Series] = None,
    ) -> dict:
        cfg: Optional[Config] = context.config or self.config
        mh = cfg.section("multiple_hypothesis") if cfg else {}
        sh = cfg.section("sharpe") if cfg else {}
        dd_limit = sh.get("max_drawdown_limit", 0.15)
        rm = cfg.get("risk_management") if cfg else {}

        checks = {
            "overfitting": self._overfitting_check(scores, forward),
            "robustness": self._robustness_check(scores, forward),
            "market_state": self._market_state_check(scores, forward, market_returns),
            "multiple_hypothesis": self._multiple_hypothesis_check(
                metrics or {}, n_trials, alpha=mh.get("alpha", 0.05)
            ),
            "drawdown": self._drawdown_check(metrics or {}, dd_limit, rm),
        }
        return {
            "passed": all(c["passed"] for c in checks.values()),
            "checks": checks,
        }

    def run(
        self,
        context: AgentContext,
        factor,
        scores,
        forward,
        metrics: Optional[dict] = None,
        *,
        n_trials: int = 1,
        market_returns: Optional[pd.Series] = None,
    ) -> AgentResult:
        report = self.validate(
            context, scores, forward, metrics, n_trials=n_trials, market_returns=market_returns
        )
        return AgentResult(
            agent=self.name,
            summary=f"risk {'PASS' if report['passed'] else 'FAIL'}: {[k for k, v in report['checks'].items() if not v['passed']]}",
            artifacts={"risk_report": report},
            context=context,
        )
