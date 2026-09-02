"""Tests for the factor-zoo translator (src/exploration/translate.py)."""

from __future__ import annotations

import pytest

from src.exploration.translate import translate


def test_gtja_001_corr_rank_delta_log():
    expr = "(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))"
    t = translate(expr)
    assert t.status == "ok", t.reasons
    assert t.fqa == (
        "Mul(Neg(1), Ts_Corr(Rank(Ts_Delta(Log(Volume), 1)), "
        "Rank(Div(Sub(Close, Open), Open)), 6))"
    )


def test_gtja_sma_and_vwap():
    t = translate("SMA(CLOSE, 10, 2)")
    assert t.status == "ok" and t.fqa == "Ts_Sma(Close, 10, 2)"
    t = translate("VWAP(CLOSE, VOLUME, 10)")
    assert t.status == "ok" and t.fqa == "Ts_Vwap(Close, Volume, 10)"


def test_worldquant_ternary_and_signedpower():
    # Alpha001 original: rank(Ts_ArgMax(SignedPower(((returns<0)?stddev(returns,20):close), 2), 5)) - 0.5
    expr = "rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2), 5)) - 0.5"
    t = translate(expr)
    assert t.status == "ok", t.reasons
    # SignedPower expands to Mul(Sign(x), Power(Abs(x), 2)); ternary → Cond
    assert "Cond(Less(TS_Return(Close, 1), 0)" in t.fqa
    assert "Power(Abs(TS_Return" in t.fqa or "Abs(Cond" in t.fqa


def test_aqml_if_style():
    expr = "Rank(Ts_ArgMax(If(returns < 0, Ts_Std(returns, 20), close), 5)) - 0.5"
    t = translate(expr)
    assert t.status == "ok", t.reasons
    assert t.fqa == (
        "Sub(Rank(Ts_Argmax(Cond(Less(TS_Return(Close, 1), 0), "
        "Ts_Std(TS_Return(Close, 1), 20), Close), 5)), 0.5)"
    )


def test_benchmark_field_defers():
    t = translate("CORR(RANK(CLOSE), RANK(BENCHMARKINDEXCLOSE), 10)")
    assert t.status == "deferred"
    assert any("benchmark" in r.lower() for r in t.reasons)


def test_unknown_operator_defers():
    t = translate("FOOBAR(CLOSE, 3)")
    assert t.status == "deferred"
    assert any("unknown operator" in r for r in t.reasons)


def test_arity_mismatch_defers():
    t = translate("MEAN(CLOSE)")
    assert t.status == "deferred"
    assert any("arity" in r for r in t.reasons)


def test_parse_error_defers():
    t = translate("CORR((CLOSE, 5")
    assert t.status == "deferred"
    assert any("parse" in r for r in t.reasons)


def test_qlib_dialect_kbar_and_rank():
    # qlib Greater(x, y) is element-wise max; Rank(x, d) is time-series
    t = translate("($high-Greater($open, $close))/$open")
    assert t.status == "ok", t.reasons
    assert t.fqa == "Div(Sub(High, Max2(Open, Close)), Open)"
    t = translate("Rank($close, 5)")
    assert t.status == "ok" and t.fqa == "Ts_Rank(Close, 5)"


def test_qlib_scientific_notation_renders_decimal():
    # FQA rejects "1e-12" (embedded minus) — must expand to a plain decimal
    t = translate("($close-$open)/($high-$low+1e-12)")
    assert t.status == "ok", t.reasons
    assert "1e" not in t.fqa
    assert t.fqa == "Div(Sub(Close, Open), Add(Sub(High, Low), 0.000000000001))"


def test_qlib_resi_and_quantile():
    t = translate("Resi($close, 20)/$close")
    assert t.status == "ok", t.reasons
    assert t.fqa == "Div(Ts_Resi(Close, 20), Close)"
    t = translate("Quantile($close, 30, 0.8)/$close")
    assert t.status == "ok", t.reasons
    assert t.fqa == "Div(Ts_Quantile(Close, 30, 0.8), Close)"


def test_simple_arith_and_neg():
    t = translate("-(CLOSE - OPEN)")
    assert t.status == "ok" and t.fqa == "Neg(Sub(Close, Open))"
    t = translate("CLOSE ^ 2")
    assert t.status == "ok" and t.fqa == "Power(Close, 2)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
