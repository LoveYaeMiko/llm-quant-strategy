"""Signal calculator — the compiled, deterministic factor engine (Phase 5).

In the TiMi split the offline layer may call LLMs freely, but *this* layer must
produce identical scores on identical inputs with no randomness. A factor is
"compiled" once (parse + structural validation), then evaluated through the
closed operator library over a ``(date, symbol)`` panel.

:class:`CompiledFactor` is the on-the-wire artifact: it serialises to JSON so a
factory can deploy the factor without shipping any code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

from ..factors.code_generator import (
    FormulaError,
    Node,
    NodeCall,
    NodeVar,
    extract_lookbacks,
    parse_expression,
    validate,
)


@dataclass
class CompiledFactor:
    """A formula that is validated once and evaluated many times."""

    name: str
    formula: str
    node_json: str = ""
    lookbacks: list[int] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    operators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, blob: str) -> "CompiledFactor":
        data = json.loads(blob)
        return cls(**data)


def compile_factor(formula: str, *, name: Optional[str] = None) -> CompiledFactor:
    """Parse + validate a formula into a :class:`CompiledFactor`.

    Raises :class:`~src.factors.code_generator.FormulaError` for unknown
    operators or arity violations — the online layer never ships a broken
    factor.
    """
    node: Node = parse_expression(formula)
    validate(node)
    nodes = list(_walk(node))
    fields = sorted({n.name for n in nodes if isinstance(n, NodeVar)})
    ops = [n.name for n in nodes if isinstance(n, NodeCall)]
    return CompiledFactor(
        name=name or formula,
        formula=formula,
        node_json=_node_to_json(node),
        lookbacks=sorted({lb for lb in extract_lookbacks(node) if lb > 0}),
        fields=fields,
        operators=ops,
    )


def _walk(node):
    yield node
    if hasattr(node, "args"):
        for arg in node.args:
            yield from _walk(arg)


def _node_to_json(node: Node) -> str:
    import json as _json

    def rec(n):
        if hasattr(n, "value"):
            return {"t": "lit", "v": n.value}
        if hasattr(n, "name") and not hasattr(n, "args"):
            return {"t": "var", "n": n.name}
        return {"t": "call", "n": n.name, "a": [rec(x) for x in n.args]}

    return _json.dumps(rec(node))


def compute_signal(
    compiled: CompiledFactor,
    data: pd.DataFrame,
    *,
    fill: float = 0.0,
) -> pd.Series:
    """Evaluate a compiled factor against raw ``(date, symbol)`` fields.

    ``data`` is the point-in-time panel; rows where any referenced field is NaN
    yield NaN and are filled to ``fill`` only when ``fill`` is not NaN.
    """
    from ..factors.code_generator import FactorContext, evaluate, parse_expression

    ctx = FactorContext(data)
    node = parse_expression(compiled.formula)
    scores = evaluate(node, ctx)
    if fill is not None and not pd.isna(fill):
        scores = scores.fillna(fill)
    return scores.astype(float)
