"""AlphaMemo — structured memory of the search process (review.md §2.2).

The miner is only as good as what it remembers: every schema + formula + its
backtest metrics is stored as a **trajectory**. Two AlphaJungle / AlphaMemo
mechanisms are reproduced:

* **Frequent Subtree Avoidance** — AST subtrees that keep recurring across the
  pool are penalised during candidate selection, forcing structurally diverse
  exploration (blueprint Phase 2 step 4);
* **Structured retrieval** — ``top_performers`` and ``diversity_screen`` give the
  explorer the pool's best and its coverage so it can steer away from crowded
  regions.

The manager is persistence-capable (JSON) so a mining run can be resumed.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from . import code_generator as cg


def _json_default(o: Any) -> Any:
    """Coerce numpy scalars (np.True_, np.float64, ...) to Python natives.

    Backtest metrics occasionally leak numpy scalar types — e.g. ``np.True_``
    from a numpy reflected comparison, whose ``__class__.__name__`` is ``"bool"``
    but which the standard encoder cannot serialize. This default keeps
    ``Trajectory.save`` robust no matter which metric field carries the scalar.
    """
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(
        f"Object of type {o.__class__.__name__} is not JSON serializable"
    )


@dataclass
class Trajectory:
    iteration: int
    schema: dict
    formula: str
    metrics: dict
    recorded_at: str = ""

    @property
    def canonical(self) -> str:
        try:
            node = cg.parse_expression(self.formula)
            return cg.canonical(node)
        except cg.FormulaError:
            return self.formula


class MemoryManager:
    """Structured search memory with frequent-subtree avoidance."""

    def __init__(self) -> None:
        self.trajectories: list[Trajectory] = []
        self._subtree_counts: Counter = Counter()

    # -- recording ----------------------------------------------------------

    def record(self, trajectory: Trajectory) -> None:
        self.trajectories.append(trajectory)
        try:
            node = cg.parse_expression(trajectory.formula)
            for subtree in cg.ast_subtrees(node):
                self._subtree_counts[subtree] += 1
        except cg.FormulaError:
            # invalid formulas are still recorded (for the failure audit) but
            # contribute no subtrees.
            pass

    def record_result(
        self,
        iteration: int,
        schema: dict,
        formula: str,
        metrics: dict,
        recorded_at: str = "",
    ) -> None:
        self.record(Trajectory(iteration, schema, formula, metrics, recorded_at))

    # -- frequent-subtree avoidance -----------------------------------------

    def subtree_frequency(self, formula: str) -> float:
        """Mean number of times the formula's subtrees have appeared before."""
        try:
            node = cg.parse_expression(formula)
            subs = cg.ast_subtrees(node)
        except cg.FormulaError:
            return 0.0
        if not subs:
            return 0.0
        return sum(self._subtree_counts[s] for s in subs) / len(subs)

    def avoidance_penalty(self, formula: str, strength: float = 0.15) -> float:
        """Penalty in [0, strength*N] proportional to subtree re-use."""
        return strength * max(0.0, self.subtree_frequency(formula) - 1.0)

    # -- retrieval ----------------------------------------------------------

    def top_performers(self, k: int = 10, metric: str = "rank_ic") -> list[Trajectory]:
        scored = [
            t for t in self.trajectories
            if isinstance(t.metrics, dict) and t.metrics.get(metric) is not None
        ]
        scored.sort(key=lambda t: float(t.metrics[metric]), reverse=True)
        return scored[:k]

    def diversity_screen(self, candidate: str, min_distance: float = 0.40) -> bool:
        """True if ``candidate`` is structurally far enough from every stored
        formula (AlphaAgent AST-principle gate)."""
        try:
            cand_node = cg.parse_expression(candidate)
        except cg.FormulaError:
            return False
        for t in self.trajectories:
            try:
                other = cg.parse_expression(t.formula)
            except cg.FormulaError:
                continue
            if cg.ast_distance(cand_node, other) < min_distance:
                return False
        return True

    def has_formula(self, formula: str) -> bool:
        try:
            cand = cg.canonical(cg.parse_expression(formula))
        except cg.FormulaError:
            return formula in {t.formula for t in self.trajectories}
        return any(t.canonical == cand for t in self.trajectories)

    def size(self) -> int:
        return len(self.trajectories)

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                [asdict(t) for t in self.trajectories], fh,
                ensure_ascii=False, indent=2, default=_json_default,
            )

    @classmethod
    def load(cls, path: str | Path) -> "MemoryManager":
        mm = cls()
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for item in raw:
            mm.record(Trajectory(**item))
        return mm
