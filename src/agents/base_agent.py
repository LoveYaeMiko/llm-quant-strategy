"""Base agent + shared context/result types for the pipeline.

Agents share a common ``AgentContext`` (the point-in-time horizon, the universe,
and the runtime config) and return a typed ``AgentResult`` — the structured
contract the blueprint's "type-safe JSON" pipeline expects. The LLM is optional:
every agent falls back to a deterministic implementation so the whole system
runs offline, and the FinCAD wrapper from ``bias_control`` is the only gate on
the way to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from ..bias_control.context_decoder import FinCADWrapper, LLMBackend, MockLLMBackend
from ..config import Config


@dataclass
class AgentContext:
    """Everything an agent is allowed to know at a point in time."""

    as_of: pd.Timestamp
    symbols: list[str]
    config: Optional[Config] = None
    data: Any = None            # PIT panel or prepared features
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.as_of = pd.Timestamp(self.as_of)


@dataclass
class AgentResult:
    agent: str
    summary: str
    artifacts: dict = field(default_factory=dict)
    context: Optional[AgentContext] = None

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "summary": self.summary,
            "artifacts": self.artifacts,
        }


class BaseAgent:
    """Common plumbing for every pipeline agent."""

    name = "base"

    def __init__(
        self,
        llm: Optional[LLMBackend] = None,
        config: Optional[Config] = None,
        fincad: Optional[FinCADWrapper] = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.fincad = fincad
        if fincad is None and llm is not None:
            # default: wrap every call in look-ahead control at its context horizon
            self.fincad = FinCADWrapper(llm)

    # -- model access with cost + bias bookkeeping ---------------------------

    def _complete(
        self,
        prompt: str,
        context: AgentContext,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Route a completion through the FinCAD wrapper when one is present."""
        if self.fincad is not None:
            result = self.fincad.complete(
                prompt, as_of=context.as_of, temperature=temperature, max_tokens=max_tokens
            )
            return result.text
        if self.llm is not None:
            return self.llm.complete(prompt, temperature=temperature, max_tokens=max_tokens)
        raise RuntimeError(f"{self.name} has no LLM backend configured")

    def run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError


__all__ = [
    "AgentContext",
    "AgentResult",
    "BaseAgent",
    "FinCADWrapper",
    "LLMBackend",
    "MockLLMBackend",
]
