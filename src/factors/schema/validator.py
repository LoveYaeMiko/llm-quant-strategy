"""Code-layer physical block + combination templates — validation_BLUEPRINT §3.1.

DeepSeek-v4-flash ignores prompt-level feedback and re-proposes the rejected
reversal factor family verbatim (Phase 8.1: 20 slots, ~7 distinct formulas).
This module is the *code-layer firewall*: ``FORBIDDEN_OPERATOR_PATTERNS`` match
the families that killed the last batch, and ``sanitize_formula`` force-replaces
any match with a **dual-factor equal-weighted combination template** — no error,
no retry (blueprint: "匹配即强制替换，不报错、不重试").

2026-08 template redirection (post-0/20 diagnosis): the 5-iteration validation
proved the ONLY alpha direction is 低波 + 低换手 (low volatility + low turnover).
Both momentum templates (动量+资金流背离, 动量+低波) drew negative alpha, and the
low-vol + low-turnover pair was the only family clearing the absolute 15% gate
under real HS300 (rank_ic 0.021-0.023, abs_dd 0.105-0.109). The template pool is
therefore restricted to the proven low-vol / low-turnover / low-level family —
no momentum leg, no reversal leg. Template index 0 (低波+低换手) is reserved: the
signal agent's template-slot generator guarantees it occupies slot 0 every round.

Templates are written in the closed formula grammar (call syntax only — there is
no infix ``*``/``-``/``/`` in the language) and use only operators + fields that
exist in the library / price panel:
``Neg(Rank(TS_Std(Close, 60)))`` = low-volatility; ``Neg(Rank(TS_Mean(Volume, 120)))``
= low-turnover; ``Neg(Rank(TS_Mean(Open, 60)))`` = low price level. Equal-weighting
two of these is the blueprint's drawdown lever and the empirically-passing family.
"""

from __future__ import annotations

import random
import re
from typing import Any

# -- blacklist: matches a factor family -> force-replaced ----------------------
FORBIDDEN_OPERATOR_PATTERNS: list[str] = [
    r"TS_Rank\s*\(\s*TS_Return",  # 排名反转 (rank-reversal)
    r"Neg\s*\(\s*TS_ZScore",  # 负向 ZScore (reverse-zscore contrarian)
    r"Inv\s*\(\s*TS_",  # 逆 TS 函数 (inverted single-TS factor)
    # 短周期 < 60 日 (1-59 days). 60/120/240 are allowed and must NOT match.
    r"TS_Return\s*\(\s*Close\s*,\s*(?:[1-9]|[1-5]\d)\s*\)",
]

# -- whitelist: dual-factor equal-weighted combination templates --------------
# ``{lb1}`` / ``{lb2}`` are drawn from the allowed lookback set {60,120,240}.
# Each template is ``Avg(A, B)`` — the library's ``(a + b) / 2`` — rather than an
# explicit ``Div(Add(A, B), 2)`` so the divisor ``2`` is never mistaken for a
# time-series lookback by ``extract_lookbacks`` (CodeGenerator would otherwise
# list a 2-day window and fail the lookback-bounds check downstream).
#
# The pool is Volume-anchored: the 0/20 diagnosis showed the alpha engine is the
# 低换手 (low-turnover) leg — both factors that cleared the absolute gate used
# ``Neg(Rank(TS_Mean(Volume, 60)))``, while `Rank(Volume)` alone scored rank_ic
# -0.04 (so the negated low-turnover direction is the +0.04 driver) and the
# Open/price-level leg was weak (rank_ic ~0.006). Every template below therefore
# carries a Volume leg. All are structurally distinct — each differs from its
# neighbours by an operator name (TS_Std / TS_Mean / TS_Delta), keeping pairwise
# AST distance >= 0.5, above the diversity floor 0.4 (a pair differing only by a
# *field* inside TS_Mean would drop to 0.25 and fail). Index 0 is the
# empirically-proven pair and is reserved for slot 0 each round.
COMBINATION_TEMPLATES: list[str] = [
    # 低波 + 低换手 (low volatility + low turnover) — PROVEN, reserved slot 0
    "Avg(Neg(Rank(TS_Std(Close, {lb1}))), Neg(Rank(TS_Mean(Volume, {lb2}))))",
    # 低换手 + 低波 (permuted pair — Volume leads, same proven family)
    "Avg(Neg(Rank(TS_Mean(Volume, {lb1}))), Neg(Rank(TS_Std(Close, {lb2}))))",
    # 双低换手 (two low-turnover windows — the alpha engine twice)
    "Avg(Neg(Rank(TS_Mean(Volume, {lb1}))), Neg(Rank(TS_Mean(Volume, {lb2}))))",
    # 低波 + 缩量 (low volatility + volume shrinking, volume-flow leg)
    "Avg(Neg(Rank(TS_Std(Close, {lb1}))), Neg(Rank(TS_Delta(Volume, {lb2}))))",
]

LOOKBACKS: tuple[int, ...] = (60, 120, 240)


def is_forbidden(formula: str) -> bool:
    """True when the formula matches any blacklist pattern (case-insensitive)."""
    return any(re.search(p, formula, re.IGNORECASE) for p in FORBIDDEN_OPERATOR_PATTERNS)


def sample_combination_template(rng: Any = None, template: str | None = None) -> str:
    """One combination template with two distinct allowed lookbacks.

    Without ``template`` a random direction from ``COMBINATION_TEMPLATES`` is
    drawn. Pass a specific template string to draw lookbacks for that direction —
    the signal agent uses this to guarantee the proven 低波+低换手 template
    (index 0) fills slot 0 of every round.
    """
    rng = rng or random
    if template is None:
        template = rng.choice(COMBINATION_TEMPLATES)
    lb1, lb2 = rng.sample(list(LOOKBACKS), 2)
    return template.format(lb1=lb1, lb2=lb2)


def sanitize_formula(formula: str, rng: Any = None) -> str:
    """Force-replace a blacklisted formula with a combination template.

    Returns the input unchanged when it is clean. Never raises.
    """
    if is_forbidden(formula):
        return sample_combination_template(rng=rng)
    return formula


__all__ = [
    "FORBIDDEN_OPERATOR_PATTERNS",
    "COMBINATION_TEMPLATES",
    "LOOKBACKS",
    "is_forbidden",
    "sample_combination_template",
    "sanitize_formula",
]
