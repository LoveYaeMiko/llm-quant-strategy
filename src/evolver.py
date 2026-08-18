"""EvoQuant — verifier-guided strategy self-evolution (blueprint Phase 4).

EvoQuant (review.md §2.4) replaces hand-tuning with a loop:

1. **diagnose**   — the LLM (or a deterministic rule set) identifies the
                   bottleneck ("factor decays after 5 days");
2. **propose**    — semantic *candidate edits* are generated, not random
                   mutations (unlike plain genetic search);
3. **validate**   — a multi-stage pipeline (overfit → robustness → regime) picks
                   the best edit;
4. **distil**     — successful experiences are written back into the
                   :class:`~src.factors.memory_manager.MemoryManager`, which is
                   exactly how AlphaMemo compounds search knowledge.

An ``accepted`` edit changes the formula; the loop returns both the diagnosis and
the winning candidate so the caller can promote it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from .agents.base_agent import AgentContext
from .agents.risk_agent import RiskAgent
from .backtest import metrics as M
from .factors.code_generator import CodeGenerator, default_formula_for
from .factors.memory_manager import MemoryManager
from .factors.residual_memory import ResidualMemory, edit_motif
from .factors.semantic_space import SchemaPlan, SemanticSpace

ScoresFn = Callable[[str], pd.Series]  # formula -> factor scores (long panel)


@dataclass
class EvoResult:
    diagnosis: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    accepted: Optional[dict] = None
    metrics: dict = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        return self.accepted is not None


class EvoQuant:
    """One self-evolution round: diagnose -> propose -> validate -> distil."""

    def __init__(
        self,
        space: Optional[SemanticSpace] = None,
        generator: Optional[CodeGenerator] = None,
        memory: Optional[MemoryManager] = None,
        risk_agent: Optional[RiskAgent] = None,
        llm=None,
        config=None,
        residual_memory: Optional[ResidualMemory] = None,
    ) -> None:
        self.space = space or SemanticSpace()
        self.generator = generator or CodeGenerator()
        self.memory = memory or MemoryManager()
        self.risk = risk_agent or RiskAgent(llm=llm, config=config)
        self.llm = llm
        self.config = config
        self.residual_memory = residual_memory or ResidualMemory()

    # -- 1. diagnose --------------------------------------------------------

    def diagnose(
        self,
        metrics: dict,
        market_state: Optional[str] = None,
        current_plan: Optional[SchemaPlan] = None,
    ) -> list[str]:
        """Deterministic bottleneck rules; an LLM critic can enrich them."""
        findings: list[str] = []
        if metrics.get("icir", 0.0) < 0.3:
            findings.append("low information ratio — signal is noisy")
        if metrics.get("max_drawdown", 0.0) > 0.10:
            findings.append("drawdown too deep for the reward")
        if metrics.get("turnover", 0.0) > 5.0:
            findings.append("excessive turnover — factor decays fast")
        if metrics.get("sharpe", 0.0) < 0.0:
            findings.append("negative Sharpe on the backtest window")
        if market_state == "bear" and metrics.get("sharpe", 0.0) < 0.5:
            findings.append("fails in the current bear regime")
        if not findings:
            findings.append("no clear bottleneck — try a complementary quality")

        if self.llm is not None and current_plan is not None:
            prompt = (
                "A factor mining loop produced these metrics and schema. "
                "Diagnose the single most likely performance bottleneck and "
                "suggest ONE semantic edit (change event, context, quality, "
                "direction, or output). Return concise text.\n\n"
                f"SCHEMA: {current_plan.natural_language()}\n"
                f"METRICS: {metrics}"
            )
            try:
                llm_diag = self.llm.complete(prompt, temperature=0.2).strip()
                findings.append(f"LLM: {llm_diag[:200]}")
            except Exception:
                pass
        return findings

    # -- 2. propose ---------------------------------------------------------

    def propose_edits(self, plan: SchemaPlan, diagnosis: list[str], k: int = 4) -> list[SchemaPlan]:
        """Targeted semantic edits: try neighbours first, bias by diagnosis."""
        import random

        rng = random.Random(0)
        candidates = self.space.neighbors(plan, rng, k=k)
        # If the diagnosis points at decay/turnover, prefer a longer lookback
        # quality; if noise, prefer mean-reversion. Deterministic heuristic.
        joined = " ".join(diagnosis).lower()
        if "decay" in joined or "turnover" in joined:
            candidates.insert(0, SchemaPlan(plan.event, plan.context, ("Momentum",), plan.direction, plan.output))
        if "noisy" in joined or "information ratio" in joined:
            candidates.insert(0, SchemaPlan(plan.event, plan.context, ("Low Volatility",), plan.direction, plan.output))
        # dedupe preserving order; veto edit motifs the residual memory has
        # already proven to fail repeatedly for this category (memory loop).
        category = self.residual_memory.category_of(plan)
        vetoed = self.residual_memory.vetoed_motifs(category)
        seen, out = set(), []
        for c in candidates:
            if c.key() in seen:
                continue
            if edit_motif(plan, c) in vetoed:
                continue
            seen.add(c.key())
            out.append(c)
        return out[:max(k, 6)]

    # -- 3 + 4. validate & distil -------------------------------------------

    def evolve(
        self,
        context: AgentContext,
        plan: SchemaPlan,
        base_formula: str,
        scores_fn: ScoresFn,
        forward: pd.Series,
        *,
        base_metrics: Optional[dict] = None,
        market_returns: Optional[pd.Series] = None,
        n_trials: int = 1,
    ) -> EvoResult:
        base_metrics = base_metrics or M.factor_eval(scores_fn(base_formula), forward)
        from .agents.dynamic_router import classify_market_state

        state = (
            classify_market_state(market_returns)
            if market_returns is not None and len(market_returns) >= 20
            else "sideways"
        )
        diagnosis = self.diagnose(base_metrics, market_state=state, current_plan=plan)
        candidates = self.propose_edits(plan, diagnosis)

        best: Optional[dict] = None
        best_score = -float("inf")
        candidate_records: list[dict] = []
        scores_list: list[pd.Series] = []
        for idx, cand in enumerate(candidates):
            formula = default_formula_for(cand)
            try:
                scores = scores_fn(formula)
                metrics = M.factor_eval(scores, forward, n_trials=n_trials)
                scores_list.append(scores)
            except Exception as exc:  # an edit that does not evaluate is rejected
                candidate_records.append(
                    {"plan": cand.to_dict(), "formula": formula, "error": str(exc)}
                )
                continue
            risk = self.risk.validate(
                context, scores, forward, metrics, n_trials=n_trials,
                market_returns=market_returns,
            )
            record = {"plan": cand.to_dict(), "formula": formula, "metrics": metrics, "risk_passed": risk["passed"]}
            candidate_records.append(record)
            # memory loop (记忆回路): record this edit's residual vs the parent so
            # the search learns which *semantic edits* help and which repeatedly
            # fail (AlphaMemo residual memory), not just which flat formulas.
            self.residual_memory.update(
                category=self.residual_memory.category_of(plan),
                motif=edit_motif(plan, cand),
                child_quality=float(metrics.get("rank_ic", 0.0)),
                parent_quality=float(base_metrics.get("rank_ic", 0.0)),
                success=risk["passed"],
            )
            score = metrics.get("rank_ic", -1.0) if risk["passed"] else -2.0
            if score > best_score:
                best_score = score
                best = record

        # A3 neighbourhood selection — plateau + in/out IC correlation. Always
        # *recorded* so the caller can audit; the argmax *rejection* only fires
        # when config.neighborhood.reject_argmax is on (off by default, so a lone
        # candidate or a caller that wants plain argmax is unaffected).
        nb = self.config.section("neighborhood") if self.config else {}
        ics = [r["metrics"].get("rank_ic") for r in candidate_records if "metrics" in r]
        is_plateau, support = M.neighborhood_plateau(ics)
        r_inout = M.in_out_ic_correlation(scores_list, forward)
        if (
            best is not None
            and bool(nb.get("reject_argmax"))
            and len(ics) >= 3
            and support < float(nb.get("min_support_fraction", 0.40))
        ):
            best = None  # a lone spike — reject the argmax (A3)
        if (
            best is not None
            and bool(nb.get("reject_argmax"))
            and r_inout is not None
            and r_inout < float(nb.get("in_out_ic_min_r", -0.20))
        ):
            best = None  # ranking does not generalise out-of-sample (A3)

        result = EvoResult(diagnosis=diagnosis, candidates=candidate_records, accepted=best)
        result.metrics = {
            "base_rank_ic": base_metrics.get("rank_ic", 0.0),
            "best_rank_ic": best_score,
            "plateau_support": support,
            "is_plateau": is_plateau,
            "in_out_ic_r": r_inout,
        }

        if best is not None:
            # distil the winning experience back into structured memory
            self.memory.record_result(
                iteration=len(self.memory.trajectories),
                schema=best["plan"],
                formula=best["formula"],
                metrics=best["metrics"],
            )
        return result
