"""AlphaJungle-style MCTS over the AlphaSchema semantic space.

Navigating the Alpha Jungle (AAAI 2026, review.md §2.2) couples an LLM's
instruction-following with Monte-Carlo Tree Search: the tree lives in the
*semantic* space (schema plans), each node is a plan, each rollout evaluates the
plan's factor on real data and feeds the backtest feedback back up the tree.

This implementation keeps the **LLM optional**: when no backend is supplied the
explorer expands nodes from the catalog neighbourhood (rule-based) and evaluates
via an injected ``evaluator`` callable. That keeps the whole loop offline and
deterministic for tests, while the production path attaches the LLM to propose
refinements at expansion time (blueprint Phase 2 step 1-2).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from .memory_manager import MemoryManager
from .semantic_space import SchemaPlan, SemanticSpace

Evaluator = Callable[[SchemaPlan, str], float]  # (plan, formula) -> value (e.g. IC)


@dataclass
class MCTSNode:
    plan: SchemaPlan
    formula: str
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


@dataclass
class ScoredPlan:
    plan: SchemaPlan
    formula: str
    metrics: dict
    score: float

    def to_dict(self) -> dict:
        return {"schema": self.plan.to_dict(), "formula": self.formula, "metrics": self.metrics, "score": self.score}


class SchemaExplorer:
    """MCTS over the semantic space with LLM-guided (or rule-based) expansion."""

    def __init__(
        self,
        space: SemanticSpace,
        evaluator: Evaluator,
        memory: MemoryManager,
        llm: Optional[Callable[[str], str]] = None,
        *,
        n_iterations: int = 60,
        exploration_weight: float = 1.41,
        expand_breadth: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.space = space
        self.evaluator = evaluator
        self.memory = memory
        self.llm = llm
        self.n_iterations = n_iterations
        self.exploration_weight = exploration_weight
        self.expand_breadth = expand_breadth
        self.rng = random.Random(seed)

    # -- tree primitives ----------------------------------------------------

    def _uct(self, node: MCTSNode) -> float:
        if node.visits == 0:
            return math.inf
        exploit = node.mean_value
        explore = self.exploration_weight * math.sqrt(
            math.log(max(1, node.parent.visits)) / node.visits
        ) if node.parent else 0.0
        return exploit + explore

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            node = max(node.children, key=self._uct)
        return node

    def _formula_for(self, plan: SchemaPlan) -> str:
        """Map a schema plan to a candidate formula.

        With an LLM, ask it to translate the plan into an operator formula
        (AlphaSchema's decoupling — the translation is the cheap step). Without
        one, derive a deterministic formula from the schema's qualities.
        """
        if self.llm is not None:
            prompt = (
                "Translate this trading schema into a single factor formula "
                "using only operators from the library (add, sub, mul, div, "
                "ts_mean, ts_std, ts_rank, ts_return, cs_rank, cs_zscore, "
                "cs_neutralize, rank_mul, ...) and fields (Close, Open, High, "
                "Low, Volume). Return ONLY the formula string.\n\n"
                f"SCHEMA: {plan.natural_language()}"
            )
            try:
                text = self.llm(prompt).strip()
                # guard against LLM emitting prose around the formula
                for line in text.splitlines():
                    if "(" in line and ")" in line and not line.startswith(("#", "The", "```")):
                        return line.strip()
                return text
            except Exception:
                return self._fallback_formula(plan)
        return self._fallback_formula(plan)

    def _fallback_formula(self, plan: SchemaPlan) -> str:
        """Deterministic schema → formula mapping (offline / tests)."""
        q = plan.qualities[0]
        w = self.rng.choice([5, 10, 20, 30, 60])
        if q in ("Momentum", "Trend", "Carry"):
            return f"Rank_Mul(Rank(Close), Rank(TS_Return(Close, {w})))"
        if q in ("Mean Reversion", "Short-Term Reversal"):
            return f"Neg(TS_ZScore(Close, {w}))"
        if q == "Low Volatility":
            return f"Inv(TS_Std(Close, {w}))"
        if q == "Value":
            return f"Rank(Close)"
        if q in ("Liquidity", "Liquidity"):
            return f"Rank(Volume)"
        return f"TS_Rank(TS_Return(Close, {w}), {w})"

    def _expand(self, node: MCTSNode) -> list[MCTSNode]:
        candidates = self.space.neighbors(node.plan, self.rng, k=self.expand_breadth)
        new_nodes: list[MCTSNode] = []
        for plan in candidates:
            formula = self._formula_for(plan)
            child = MCTSNode(plan=plan, formula=formula, parent=node)
            node.children.append(child)
            new_nodes.append(child)
        return new_nodes

    def _simulate(self, node: MCTSNode) -> tuple[float, dict]:
        try:
            value = self.evaluator(node.plan, node.formula)
        except Exception:
            value = -1.0
        metrics = {"value": value}
        return value, metrics

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    # -- main loop ----------------------------------------------------------

    def explore(self, root_plan: Optional[SchemaPlan] = None) -> list[ScoredPlan]:
        """Run the MCTS loop and return the scored leaves, best first."""
        root_plan = root_plan or self.space.sample(self.rng)
        root = MCTSNode(plan=root_plan, formula=self._formula_for(root_plan))
        scored: dict[str, ScoredPlan] = {}

        for i in range(self.n_iterations):
            # Selection
            node = self._select(root)
            # Expansion
            if node.visits > 0 or node is root:
                self._expand(node)
            # Simulation on a leaf
            leaf = node.children[0] if node.children else node
            value, metrics = self._simulate(leaf)
            # Backpropagation
            self._backpropagate(leaf, value)
            # Persist to memory (frequent-subtree avoidance input)
            self.memory.record_result(
                iteration=i,
                schema=leaf.plan.to_dict(),
                formula=leaf.formula,
                metrics={"value": value, **metrics},
            )
            scored[leaf.formula] = ScoredPlan(
                plan=leaf.plan, formula=leaf.formula,
                metrics={"value": value}, score=value,
            )

        return sorted(scored.values(), key=lambda s: s.score, reverse=True)

    # -- accessors ----------------------------------------------------------

    @staticmethod
    def best_from_results(results: list[ScoredPlan], k: int = 5) -> list[ScoredPlan]:
        return results[:k]
