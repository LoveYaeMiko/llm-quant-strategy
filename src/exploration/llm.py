"""Controlled LLM exploration — the re-opened hypothesis path.

The production firewall (``factor_mining.force_combination_templates``) stays ON
and is untouched; this module is a SEPARATE, opt-in LLM path used only by the
exploration track. "Controlled" means the same three guardrails the production
path applies — the closed operator library, the structural blacklist (the crash-
continuation reversal family), and a lookback floor — but with a WIDER lookback
floor (20 vs 60 days) so higher-frequency high-risk/high-reward routes are not
ruled out a priori.
"""

from __future__ import annotations

import json
import re
from typing import Optional

# Structural blacklist only — the crash-continuation reversal families. The
# sub-60-day ``TS_Return`` pattern is intentionally OMITTED here (exploration
# allows shorter lookbacks); the lookback floor is enforced by bump_lookbacks.
STRUCTURAL_FORBIDDEN: list[str] = [
    r"TS_Rank\s*\(\s*TS_Return",
    r"Neg\s*\(\s*TS_ZScore",
    r"Inv\s*\(\s*TS_",
]

_JSON_ARR = re.compile(r"\[[\s\S]*\]")


def _operator_list() -> str:
    from ..factors.code_generator import OPERATOR_LIBRARY

    return " ".join(sorted(OPERATOR_LIBRARY))


def propose_formulas(
    backend,
    n: int,
    *,
    temperature: float = 0.8,
    min_lookback: int = 20,
    max_lookback: int = 240,
) -> list[str]:
    """Ask the LLM to propose ``n`` raw formula strings, then validate each
    against the closed library. Returns only parseable, structurally-clean
    formulas (falling back to an empty list when the model is unavailable or
    uncooperative — the deterministic sweep carries the run either way)."""
    ops = _operator_list()
    # The prompt is deliberately TIGHT. deepseek-v4-pro is a *reasoning* model:
    # open-ended "explore directions you have not seen before" phrasing sends its
    # chain-of-thought into runaway generation that eats the whole output budget
    # and returns empty content (finish_reason="length"). A short, directive
    # contract makes it finish in a bounded number of reasoning tokens.
    prompt = (
        f"Propose exactly {n} DISTINCT single-factor formulas for Chinese A-share "
        f"stock selection (HS300/HS500).\n"
        f"Use ONLY operators: {ops}\n"
        f"Use ONLY fields: Close Open High Low Volume.\n"
        f"Every TS_* lookback between {min_lookback} and {max_lookback}.\n"
        f"Prefer Rank()/cs_zscore() normalisation; compose 2+ operators for novel signals.\n"
        f"Avoid: TS_Rank(TS_Return(...)), Neg(TS_ZScore(...)), Inv(TS_*(...)).\n"
        f"Return a JSON array of strings only."
    )
    try:
        # Reasoning models burn most of max_tokens on chain-of-thought, and their
        # generation is slow (n=8 took ~2min). Budget max_tokens for the reasoning
        # (~n×800) + the answer, and give the client a generous timeout. With
        # temperature>0 the model is stochastic and occasionally runs away into
        # reasoning (empty content), so retry a couple of times before giving up.
        for _attempt in range(3):
            text = backend.complete(
                prompt, temperature=temperature,
                max_tokens=max(4096, n * 1024), timeout=300,
            )
            formulas = _parse_formulas(text, min_lookback=min_lookback)
            if formulas:
                return formulas
        return []
    except Exception:  # noqa: BLE001 — a flaky model must not kill the run
        return []


def _parse_formulas(text: str, *, min_lookback: int) -> list[str]:
    from ..factors.code_generator import FormulaError, bump_lookbacks, parse_expression, validate

    m = _JSON_ARR.search(text or "")
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        formula = item.strip()
        if not formula or formula in seen:
            continue
        # guardrail 1: closed library (parse + validate)
        try:
            node = parse_expression(formula)
            validate(node)
        except FormulaError:
            continue
        # guardrail 2: structural blacklist (crash-continuation reversal)
        if any(re.search(p, formula, re.IGNORECASE) for p in STRUCTURAL_FORBIDDEN):
            continue
        # guardrail 3: lookback floor (bump sub-floor windows up to the floor)
        try:
            formula = bump_lookbacks(formula, min_lookback)
        except FormulaError:
            continue
        seen.add(formula)
        out.append(formula)
    return out
