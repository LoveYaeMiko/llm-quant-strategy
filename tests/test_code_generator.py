"""Hardcoded operator library + safe evaluator tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.code_generator import (
    CodeGenerator,
    FactorContext,
    FormulaError,
    ast_distance,
    ast_subtrees,
    canonical,
    default_formula_for,
    eval_expression,
    extract_lookbacks,
    node_to_python,
    operator_count,
    parse_expression,
)
from src.factors.semantic_space import SchemaPlan


def test_operator_count_meets_blueprint():
    assert operator_count() >= 66


def test_evaluate_ts_return(fctx):
    scores = eval_expression("TS_Return(Close, 5)", fctx)
    assert isinstance(scores, pd.Series)
    assert len(scores) == len(fctx.data)
    # no NaN after warmup windows
    assert scores.dropna().shape[0] > len(fctx.data) * 0.8


def test_evaluate_rank_expression(fctx):
    scores = eval_expression("Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))", fctx)
    assert scores.index.names == ["date", "symbol"]
    assert np.isfinite(scores.dropna()).all()


def test_unknown_operator_rejected():
    gen = CodeGenerator()
    with pytest.raises(FormulaError):
        gen.parse("magic_op(Close, 5)")


def test_wrong_arity_rejected():
    gen = CodeGenerator()
    with pytest.raises(FormulaError):
        gen.parse("ts_mean(Close)")  # needs a window


def test_no_python_eval_path():
    # formula language cannot call arbitrary python — quotes break the grammar
    with pytest.raises(FormulaError):
        parse_expression("eval('__import__(\"os\")')")


def test_field_case_insensitive():
    ctx = FactorContext(pd.DataFrame({"close": [1.0, 2.0]}, index=pd.MultiIndex.from_product([[pd.Timestamp("2024-01-01")], ["A", "B"]])))
    s = ctx.field("CLOSE")
    assert list(s) == [1.0, 2.0]


def test_ast_distance_identical_is_zero():
    a = parse_expression("Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))")
    b = parse_expression("Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))")
    assert ast_distance(a, b) == 0.0


def test_ast_distance_different_is_one():
    a = parse_expression("Rank(Close)")
    b = parse_expression("Neg(Close)")
    assert ast_distance(a, b) == 1.0


def test_canonical_and_subtrees():
    node = parse_expression("Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))")
    assert canonical(node).startswith("call:Rank_Mul(")
    subs = ast_subtrees(node)
    assert len(subs) > 3
    assert "var:Close" in subs
    assert any(s.startswith("call:TS_Return(") for s in subs)


def test_extract_lookbacks():
    node = parse_expression("TS_Return(Close, 10)")
    assert extract_lookbacks(node) == [10]


def test_node_to_python():
    node = parse_expression("Rank(Close)")
    assert "Rank(Field('Close'))" in node_to_python(node)


def test_code_generator_generate():
    gen = CodeGenerator()
    gf = gen.generate("TS_Return(Close, 10)", name="mom", meaning="10d momentum", category="Momentum")
    assert gf.formula == "TS_Return(Close, 10)"
    assert gf.data_fields_used == ["Close"]
    assert gf.lookback_periods == [10]
    assert gf.python_code.startswith("import numpy")


def test_within_lookback_bounds():
    gen = CodeGenerator()
    assert gen.within_lookback_bounds("TS_Return(Close, 20)", 60, 5)
    assert not gen.within_lookback_bounds("TS_Return(Close, 100)", 60, 5)


def test_default_formula_for_deterministic():
    plan = SchemaPlan(event="Earnings Surprise", context="Bull Market", qualities=("Momentum",), direction="long", output="score")
    assert default_formula_for(plan) == default_formula_for(plan)
