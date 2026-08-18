"""Residual / veto memory — closes the writer→judge→feedback loop (记忆回路).

The search loop currently only feeds **negative** feedback back to the writer:
rejected factors and their reasons are replayed into the next prompt
(``SignalAgent.record_rejection``). What is missing is the **positive** side —
which *semantic edits* actually improved a factor — and a principled way to
**veto** edit motifs that repeatedly fail.

This module reproduces AlphaMemo's online residual memory
(``sspm/memory/residual.py``), adapted to FQA's :class:`SchemaPlan` semantics and
the ``rank_ic`` metric:

* every candidate edit is scored as a **residual** = child rank_ic − parent
  rank_ic (so a +0.02 means "this edit beat its parent by 2 points of IC");
* a **confidence** weight combines a count gate, an entropy-based certainty term
  and a variance penalty, so a motif is only trusted after several consistent
  observations;
* a **veto** set flags (category, motif) cells whose failure rate is high with
  confidence — the writer must skip them.

The memory is persistence-capable (JSON) so it compounds across mining runs,
exactly as AlphaMemo's does.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .semantic_space import SchemaPlan


def edit_motif(parent: SchemaPlan, child: SchemaPlan) -> str:
    """Describe the semantic edit from ``parent`` to ``child``.

    Neighbours change exactly one component (``SemanticSpace.neighbors``), so the
    motif is a single ``field: old → new`` string. Multiple simultaneous changes
    (possible only in hand-built plans) are joined with ``;``.
    """
    p, c = parent.to_dict(), child.to_dict()
    changes = []
    for field in ("event", "context", "direction", "output"):
        if p[field] != c[field]:
            changes.append(f"{field}:{p[field]}→{c[field]}")
    if tuple(p["qualities"]) != tuple(c["qualities"]):
        changes.append(f"quality:{'+'.join(p['qualities'])}→{'+'.join(c['qualities'])}")
    return ";".join(changes) if changes else "identity"


@dataclass
class ResidualCell:
    residuals: list[float] = field(default_factory=list)
    successes: int = 0
    failures: int = 0

    @property
    def n(self) -> int:
        return self.successes + self.failures

    @property
    def fail_rate(self) -> float:
        return self.failures / max(self.n, 1)

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.n, 1)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


class ResidualMemory:
    """Online residual memory over (factor category, edit motif)."""

    def __init__(self, n_conf: int = 12, veto_threshold: float = 0.80, min_observations: int = 5) -> None:
        self.n_conf = n_conf
        self.veto_threshold = veto_threshold
        self.min_observations = min_observations
        self.cells: dict[tuple[str, str], ResidualCell] = defaultdict(ResidualCell)
        self.parent_bucket_baselines: dict[str, list[float]] = defaultdict(list)

    # -- recording ----------------------------------------------------------

    def _parent_bucket(self, parent_quality: float) -> str:
        if parent_quality >= 0.06:
            return "high"
        if parent_quality >= 0.03:
            return "medium"
        return "low"

    def category_of(self, plan: SchemaPlan) -> str:
        """The factor 'category' a plan belongs to — its primary quality."""
        return plan.qualities[0] if plan.qualities else "?"

    def update(
        self,
        category: str,
        motif: str,
        child_quality: float,
        parent_quality: float,
        success: bool,
    ) -> float:
        """Record one edit outcome; return the residual (child − parent)."""
        bucket = self._parent_bucket(parent_quality)
        history = self.parent_bucket_baselines[bucket]
        baseline = _mean(history) if history else parent_quality
        residual = child_quality - baseline
        history.append(child_quality)

        cell = self.cells[(category, motif)]
        cell.residuals.append(float(residual))
        if success:
            cell.successes += 1
        else:
            cell.failures += 1
        return float(residual)

    # -- retrieval ----------------------------------------------------------

    def query(self, category: str, motif: str) -> tuple[float, float]:
        """Return ``(mean_residual, confidence)`` for a (category, motif) cell."""
        cell = self.cells.get((category, motif))
        if cell is None or cell.n < 2:
            return 0.0, 0.0
        mean_residual = _mean(cell.residuals)
        p = (cell.successes + 1.0) / (cell.n + 2.0)
        entropy = 0.0
        if 0.0 < p < 1.0:
            entropy = -(p * math.log(p, 2) + (1 - p) * math.log(1 - p, 2))
        certainty = max(0.0, 1.0 - entropy)
        count_gate = min(1.0, cell.n / max(self.n_conf, 1))
        if len(cell.residuals) > 1:
            variance_penalty = max(0.0, 1.0 - _std(cell.residuals) / (abs(mean_residual) + 0.03))
        else:
            variance_penalty = 1.0
        confidence = count_gate * (0.5 + 0.5 * certainty) * variance_penalty
        return float(mean_residual), float(max(0.0, min(1.0, confidence)))

    def top_cells(self, k: int = 8) -> list[dict]:
        """Highest-confidence *positive* residual motifs — steer the writer."""
        rows = []
        for (category, motif), cell in self.cells.items():
            if not cell.residuals:
                continue
            delta, conf = self.query(category, motif)
            if delta <= 0:
                continue
            rows.append({
                "category": category, "motif": motif, "n": cell.n,
                "successes": cell.successes, "failures": cell.failures,
                "mean_residual": round(delta, 4), "confidence": round(conf, 3),
            })
        return sorted(rows, key=lambda r: r["confidence"] * r["mean_residual"], reverse=True)[:k]

    # -- veto ---------------------------------------------------------------

    def vetoed(self, category: str, motif: str) -> tuple[bool, float]:
        """True when the cell is a high-confidence, repeatedly-failing motif."""
        cell = self.cells.get((category, motif))
        if cell is None or cell.n < self.min_observations:
            return False, 0.0
        if cell.fail_rate >= self.veto_threshold:
            return True, cell.fail_rate
        return False, 0.5 * cell.fail_rate

    def vetoed_motifs(self, category: str) -> set[str]:
        """All motifs vetoed for a category (used to skip plans cheaply)."""
        return {m for (c, m) in self.cells if c == category and self.vetoed(c, m)[0]}

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "n_conf": self.n_conf,
            "veto_threshold": self.veto_threshold,
            "min_observations": self.min_observations,
            "cells": {f"{c}::{m}": asdict(cell) for (c, m), cell in self.cells.items()},
            "parent_bucket_baselines": {k: v for k, v in self.parent_bucket_baselines.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResidualMemory":
        mem = cls(
            n_conf=int(data.get("n_conf", 12)),
            veto_threshold=float(data.get("veto_threshold", 0.80)),
            min_observations=int(data.get("min_observations", 5)),
        )
        for key, blob in (data.get("cells") or {}).items():
            category, motif = key.split("::", 1)
            cell = ResidualCell(
                residuals=list(blob.get("residuals", [])),
                successes=int(blob.get("successes", 0)),
                failures=int(blob.get("failures", 0)),
            )
            mem.cells[(category, motif)] = cell
        for k, v in (data.get("parent_bucket_baselines") or {}).items():
            mem.parent_bucket_baselines[k] = list(v)
        return mem

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ResidualMemory":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def size(self) -> int:
        return len(self.cells)
