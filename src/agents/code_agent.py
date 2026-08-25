"""Code Agent — translates a schema plan into executable factor code.

Blueprint Phase 3: the code agent runs at **temperature 0.0** and translates
through the *hardcoded* operator library (:mod:`src.factors.code_generator`), so
it cannot hallucinate operators. AlphaSchema's finding — implementation quality
is robust to which LLM translates — lets us run this step cheaply (review.md §2.2).
"""

from __future__ import annotations

import re
from typing import Optional

from ..bias_control.context_decoder import LLMBackend
from ..config import Config
from ..factors.code_generator import (
    CodeGenerator,
    FormulaError,
    GeneratedFactor,
    bump_lookbacks,
    default_formula_for,
)
from ..factors.memory_manager import MemoryManager
from ..factors.semantic_space import SchemaPlan
from .base_agent import AgentContext, AgentResult, BaseAgent


class CodeAgent(BaseAgent):
    name = "code"

    def __init__(
        self,
        generator: Optional[CodeGenerator] = None,
        memory: Optional[MemoryManager] = None,
        llm: Optional[LLMBackend] = None,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(llm=llm, config=config)
        self.generator = generator or CodeGenerator()
        self.memory = memory

    def translate(self, context: AgentContext, plan: SchemaPlan, name: Optional[str] = None) -> GeneratedFactor:
        formula: str
        if self.llm is not None:
            prompt = (
                "Translate this trading schema into a SINGLE factor formula. Use "
                "only operators from: add sub mul div avg abs sign log sqrt "
                "ts_mean ts_std ts_median ts_sum ts_max ts_min ts_skew ts_kurt "
                "ts_rank ts_return ts_delay ts_delta ts_zscore ts_quantile "
                "ts_ema ts_wma ts_corr ts_cov ts_beta ts_slope ts_mad ts_count_pos "
                "ts_decay_linear cs_rank cs_zscore cs_scale cs_tanh cs_neutralize "
                "rank_mul rank_add rank_sub rank_div cond greater less. Use "
                "fields: Close Open High Low Volume. TS_* lookback windows must "
                "be >= 60 days (medium/low frequency). Prefer cross-sectionally "
                "normalised signals (Rank / cs_zscore) over raw levels to keep "
                "single-factor drawdown under 15%; the proven low-vol + "
                "low-turnover family is the strongest direction. Return ONLY the "
                "formula string, e.g. Avg(Neg(Rank(TS_Std(Close, 120))), "
                "Neg(Rank(TS_Mean(Volume, 60)))).\n\n"
                f"SCHEMA: {plan.natural_language()}"
            )
            formula = self._llm_formula(context, prompt)
        else:
            formula = default_formula_for(plan)

        try:
            self.generator.parse(formula)
        except FormulaError:
            # the LLM drifted outside the closed library — fall back to the
            # deterministic mapping rather than propagating a hallucination
            formula = default_formula_for(plan)

        # LIMIT_DOWN blueprint 方案 D: auto-upgrade any short lookback the LLM
        # slipped in (a 5/10-day reversal is exactly the crash-continuation
        # family Phase 8.1 diagnosed). Config-driven; default on.
        if (self.config is not None
                and self.config.get("factor_mining.auto_upgrade_lookback", True)
                and isinstance(formula, str)):
            min_lb = int(self.config.get("factor_mining.min_lookback", 60))
            try:
                formula = bump_lookbacks(formula, min_lb)
            except FormulaError:
                pass

        # validation_BLUEPRINT §3.1 code-layer physical block: deepseek-v4-flash
        # ignores prompt feedback and re-proposes the banned reversal family
        # verbatim, so the code layer force-replaces any blacklisted formula with
        # a dual-factor equal-weight combination template — no error, no retry.
        # Config-driven; default on.
        if (self.config is not None
                and self.config.get("factor_mining.enable_code_layer_blocking", True)
                and isinstance(formula, str)):
            from ..factors.schema.validator import sanitize_formula

            formula = sanitize_formula(formula)

        return self.generator.generate(
            formula,
            name=name or plan.key()[:32],
            meaning=plan.natural_language(),
            category=plan.qualities[0],
        )

    def _llm_formula(self, context: AgentContext, prompt: str) -> str:
        return self._complete(prompt, context, temperature=0.0, max_tokens=512).strip()

    def run(self, context: AgentContext, plan: SchemaPlan) -> AgentResult:
        gf = self.translate(context, plan)
        return AgentResult(
            agent=self.name,
            summary=f"translated schema -> {gf.formula}",
            artifacts={
                "factor": gf.to_dict(),
                "python_code": gf.python_code,
                "operators_used": gf.operators_used,
                "lookback_periods": gf.lookback_periods,
            },
            context=context,
        )
