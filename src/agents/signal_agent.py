"""Signal Agent — generates N independent, complementary hypotheses.

Blueprinted as the researcher: proposes factor hypotheses using the AlphaSchema
semantic space. Independence comes from two levers (review.md §2.2):

* **schema diversity** — every plan is a distinct point in the semantic space;
* **frequent-subtree avoidance** — formulas whose AST subtrees already crowd the
  memory are skipped, so the pool does not collapse onto a few template shapes
  (AlphaJungle).
"""

from __future__ import annotations

import json
import random
import re
from typing import Optional

from ..bias_control.context_decoder import LLMBackend
from ..config import Config
from ..factors.code_generator import default_formula_for
from ..factors.memory_manager import MemoryManager
from ..factors.semantic_space import SchemaPlan, SemanticSpace
from .base_agent import AgentContext, AgentResult, BaseAgent

_CODE_FENCE = re.compile(r"```(?:json)?\s*|\s*```")


class SignalAgent(BaseAgent):
    name = "signal"

    def __init__(
        self,
        space: Optional[SemanticSpace] = None,
        memory: Optional[MemoryManager] = None,
        llm: Optional[LLMBackend] = None,
        config: Optional[Config] = None,
        n_hypotheses: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(llm=llm, config=config)
        self.space = space or SemanticSpace()
        self.memory = memory or MemoryManager()
        self.n_hypotheses = n_hypotheses
        self.rng = random.Random(seed)

    # -- generation ---------------------------------------------------------

    def _llm_proposed_plans(self, context: AgentContext, n: int) -> list[SchemaPlan]:
        prompt = (
            "You are a quantitative researcher generating independent trading "
            "hypotheses. Propose exactly "
            f"{n} DISTINCT schema plans. Each plan is a JSON object with keys "
            '"event", "context", "qualities" (array), "direction" ("long"/"short"/'
            '"long_short"), "output" ("score"/"signal"/"rank"). Return a JSON '
            "array of such objects, nothing else.\n\n"
            f"Universe horizon: {context.as_of.date()}. Available events/contexts/"
            "qualities are the standard AlphaSchema catalogs; be diverse."
        )
        text = self._complete(prompt, context, temperature=0.8, max_tokens=2048)
        text = _CODE_FENCE.sub("", text).strip()
        try:
            raw = json.loads(text)
            plans = [SchemaPlan.from_dict(d) for d in raw if isinstance(d, dict)]
            return [p for p in plans if self.space.validate(p)]
        except (json.JSONDecodeError, ValueError, TypeError):
            return []

    def generate_hypotheses(
        self, context: AgentContext, n: Optional[int] = None
    ) -> list[SchemaPlan]:
        n = n or self.n_hypotheses
        plans: list[SchemaPlan] = []
        if self.llm is not None:
            plans = self._llm_proposed_plans(context, n)

        while len(plans) < n:
            candidate = self.space.sample(self.rng)
            formula = default_formula_for(candidate)
            if self.memory.has_formula(formula):
                continue
            # frequent-subtree avoidance: skip re-explored structures with a
            # probability proportional to how often they already appear
            penalty = self.memory.avoidance_penalty(formula)
            if penalty > 0 and self.rng.random() < min(penalty, 0.95):
                continue
            plans.append(candidate)
        return plans[:n]

    def run(self, context: AgentContext, n: Optional[int] = None) -> AgentResult:
        plans = self.generate_hypotheses(context, n)
        return AgentResult(
            agent=self.name,
            summary=f"generated {len(plans)} independent hypotheses",
            artifacts={
                "n_hypotheses": len(plans),
                "plans": [p.to_dict() for p in plans],
            },
            context=context,
        )
