"""FinCAD inference-time adaptation layer (blueprint Phase 1 step 3).

Two cooperating mechanisms:

1. **Prompt sanitisation** — any date mention in the *prompt* after the current
   timestamp T is redacted, and a system instruction pins the model to the T
   information horizon. This is cheap and handles the common case where the
   caller's own context leaks.

2. **Output gating** — after generation, the reply is scanned for future-date
   mentions (the recall vector for memorised outcomes). If any are found the
   wrapper re-asks the backend once with an explicit penalty instruction, then
   redacts any remaining mentions from the returned text and records them in the
   audit trail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

import pandas as pd

from .look_ahead_detector import future_mentions

_DATELIKE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-/]?(\d{1,2})?[-/]?(\d{1,2})?(?!\d)|"
                       r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:19|20)\d{2}\b",
                       re.I)


class LLMBackend(Protocol):
    """Any chat-style backend the framework can call."""

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str: ...


class MockLLMBackend:
    """Deterministic backend for tests / offline demos.

    Returns the canned text; if the prompt contains ``-> echo:`` it echoes the
    rest verbatim (used to simulate a leaky model).
    """

    def __init__(self, responses: Optional[dict[str, str]] = None) -> None:
        self.responses: dict[str, str] = responses or {}
        self.call_count = 0

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        self.call_count += 1
        for key, value in self.responses.items():
            if key in prompt:
                return value
        marker = "-> echo:"
        if marker in prompt:
            return prompt.split(marker, 1)[1].strip()
        return "I have no opinion."


@dataclass
class FinCADResult:
    text: str
    as_of: pd.Timestamp
    redacted_mentions: list[str] = field(default_factory=list)
    suppressed_count: int = 0
    retried: bool = False

    @property
    def leak_free(self) -> bool:
        return not future_mentions(self.text, self.as_of)


def sanitize_prompt(text: str, as_of: str | pd.Timestamp) -> tuple[str, list[str]]:
    """Redact future-date mentions from ``text``, return ``(clean, redacted)``."""
    t = pd.Timestamp(as_of)
    redacted: list[str] = []
    def _repl(match: re.Match) -> str:
        mention = match.group(0)
        redacted.append(mention)
        return "[DATE_REDACTED]"
    clean = _DATELIKE.sub(_repl, text)
    return clean, redacted


def contextual_system_prompt(as_of: str | pd.Timestamp, extra: str = "") -> str:
    """System prompt that pins the model to the information horizon at ``as_of``."""
    t = pd.Timestamp(as_of)
    return (
        "You are a quantitative researcher. It is now "
        f"{t.date().isoformat()}. You MUST NOT refer to, imply, or reason about "
        "any event, price, or announcement dated after this timestamp. Treat "
        "anything after this date as unknown. "
        f"{extra}".strip()
    )


class FinCADWrapper:
    """Wrap a backend so every completion is look-ahead-controlled."""

    def __init__(
        self,
        backend: LLMBackend,
        as_of: str | pd.Timestamp | None = None,
        penalty_scale: float = 2.0,
        retry_on_leak: bool = True,
    ) -> None:
        self.backend = backend
        self.as_of = pd.Timestamp(as_of) if as_of is not None else None
        self.penalty_scale = penalty_scale
        self.retry_on_leak = retry_on_leak
        self._suppressed_count = 0

    @property
    def total_suppressions(self) -> int:
        return self._suppressed_count

    def complete(
        self,
        prompt: str,
        as_of: str | pd.Timestamp | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> FinCADResult:
        t = pd.Timestamp(as_of) if as_of is not None else self.as_of
        if t is None:
            raise ValueError("as_of must be provided (constructor or call)")

        clean_prompt, redacted = sanitize_prompt(prompt, t)
        system = contextual_system_prompt(t)
        full_prompt = f"{system}\n\n{clean_prompt}"
        text = self.backend.complete(full_prompt, temperature=temperature, max_tokens=max_tokens)

        leaks = future_mentions(text, t)
        retried = False
        if leaks and self.retry_on_leak:
            retried = True
            penalty_instruction = (
                f"\n\nCorrection: your previous reply referred to dates after {t.date().isoformat()} "
                "which you cannot know. Remove every such reference and rewrite using only prior information."
            )
            text = self.backend.complete(full_prompt + penalty_instruction, temperature=temperature, max_tokens=max_tokens)

        final_leaks = future_mentions(text, t)
        # Redact anything that still slipped through so the caller never sees it.
        final_text, residual = sanitize_prompt(text, t)
        self._suppressed_count += len(residual)
        return FinCADResult(
            text=final_text,
            as_of=t,
            redacted_mentions=[m for m, _ in final_leaks],
            suppressed_count=len(residual),
            retried=retried,
        )
