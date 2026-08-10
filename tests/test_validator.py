"""validation_BLUEPRINT §3.1 — code-layer physical block + combination templates.

The validator is the firewall that fixes Phase 8.1's root cause B (deepseek-
v4-flash ignores prompt feedback and re-proposes the banned reversal family):
``FORBIDDEN_OPERATOR_PATTERNS`` match the families that killed the 0/20 batch and
``sanitize_formula`` force-replaces them with a dual-factor equal-weight
combination template — no error, no retry. These tests pin the blacklist, the
template grammar (must parse AND evaluate on a real panel — no FormulaError) and
the lookback policy (60/120/240 only; <60 is blacklisted, >=60 is allowed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.code_generator import FormulaError, eval_expression, parse_expression, validate
from src.factors.schema.validator import (
    COMBINATION_TEMPLATES,
    FORBIDDEN_OPERATOR_PATTERNS,
    LOOKBACKS,
    is_forbidden,
    sample_combination_template,
    sanitize_formula,
)

BLACKLISTED = [
    "Neg(TS_ZScore(Close, 240))",          # negative-zscore contrarian (根因B主犯)
    "Neg(TS_ZScore(Close, 120))",
    "Inv(TS_Std(Close, 240))",             # inverted single-TS factor
    "Inv(TS_Mean(Close, 120))",
    "TS_Rank(TS_Return(Close, 240), 240)",  # rank-reversal
    "TS_Rank(TS_Return(Close, 60), 60)",
    "TS_Return(Close, 5)",                  # short-cycle <60 (LIMIT_DOWN 方案 D)
    "TS_Return(Close, 59)",
]

ALLOWED = [
    "TS_Return(Close, 60)",                  # exactly the floor — allowed
    "TS_Return(Close, 240)",                 # long window — allowed
    "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 240)))",  # momentum
    "Rank(Close)",                            # value
    "Rank(Volume)",                           # liquidity
    "Div(Add(Rank(TS_Return(Close, 60)), Neg(Rank(TS_Delta(Volume, 120)))), 2)",  # a template
]


@pytest.mark.parametrize("formula", BLACKLISTED)
def test_blacklist_hits_every_rejected_family(formula):
    assert is_forbidden(formula), f"{formula!r} must be blocked"


@pytest.mark.parametrize("formula", ALLOWED)
def test_blacklist_spares_legit_factor_families(formula):
    assert not is_forbidden(formula), f"{formula!r} must be allowed"


def test_sanitize_force_replaces_without_error_or_retry():
    for bad in BLACKLISTED:
        out = sanitize_formula(bad)
        assert out != bad, f"blacklisted formula was returned unchanged: {bad!r}"
        assert not is_forbidden(out), f"replacement is still blacklisted: {out!r}"
        node = parse_expression(out)
        validate(node)  # must be well-formed call-syntax
    # a clean formula passes through untouched
    clean = "Rank(Close)"
    assert sanitize_formula(clean) == clean


def test_all_templates_parse_and_evaluate_on_a_real_panel():
    dates = pd.bdate_range("2020-01-01", periods=300)
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D", "E"]], names=["date", "symbol"])
    rng = np.random.default_rng(7)
    data = pd.DataFrame(
        {
            "open": np.abs(rng.normal(100, 3, len(idx))),
            "high": np.abs(rng.normal(102, 3, len(idx))),
            "low": np.abs(rng.normal(98, 3, len(idx))),
            "close": np.abs(rng.normal(100, 3, len(idx))),
            "volume": np.abs(rng.normal(1e6, 2e5, len(idx))),
        },
        index=idx,
    )
    from src.factors.code_generator import FactorContext

    ctx = FactorContext(data)
    for template in COMBINATION_TEMPLATES:
        formula = template.format(lb1=60, lb2=120)
        node = parse_expression(formula)
        validate(node)  # operators exist + arity correct
        series = eval_expression(formula, ctx)
        assert series.notna().sum() > 0, f"template produced no finite values: {formula}"
        # equal-weight dual-factor: values stay bounded (no blow-up from /2)


def test_combination_templates_are_dual_factor_equal_weight():
    assert len(COMBINATION_TEMPLATES) >= 4, "blueprint requires >= 4 template directions"
    for template in COMBINATION_TEMPLATES:
        # each template averages a 2-way combination (Avg = (a+b)/2) and
        # references two distinct lookbacks — the equal-weight dual-factor
        # construction. Avg (not Div(...,2)) keeps the divisor out of the
        # lookback-extraction path.
        assert "{lb1}" in template and "{lb2}" in template
        assert template.startswith("Avg("), f"template must be equal-weighted: {template}"


def test_sample_combination_template_uses_allowed_lookbacks():
    seen = set()
    for _ in range(30):
        formula = sample_combination_template()
        seen.add(formula)
        node = parse_expression(formula)
        validate(node)
    assert len(seen) > 1, "sampling should produce variety across templates"
    # exactly two numeric literals — the two sampled lookbacks — and both allowed
    for formula in seen:
        literals = _numeric_literals(formula)
        assert len(literals) == 2, f"expected exactly two lookbacks in {formula}, got {literals}"
        for token in literals:
            assert int(token) in LOOKBACKS, f"illegal lookback {token} in {formula}"


def _numeric_literals(formula: str) -> list[str]:
    return [t for t in formula.replace("(", " ").replace(")", " ").replace(",", " ").split() if t.replace(".", "", 1).isdigit()]


def test_sanitized_formula_determinism_not_required_but_always_valid():
    # two calls may differ (random template), but both must be valid + clean
    for bad in ("Neg(TS_ZScore(Close, 240))", "Inv(TS_Std(Close, 60))", "TS_Return(Close, 5)"):
        for _ in range(5):
            out = sanitize_formula(bad)
            try:
                validate(parse_expression(out))
            except FormulaError as exc:  # pragma: no cover - failure diagnostic
                raise AssertionError(f"sanitize output invalid for {bad!r}: {out!r} ({exc})") from exc
            assert not is_forbidden(out)
