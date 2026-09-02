"""Tests for the ML promote gate (src/ml/promote.py) — pure, offline."""

from __future__ import annotations

import pytest

from src.ml.promote import DEFAULT_GATES, evaluate_promotion


def _model(**kw):
    base = {"sharpe": 1.5, "max_dd": 0.10, "rank_ic": 0.04}
    base.update(kw)
    return base


def _incumbent(**kw):
    base = {"sharpe": 1.0, "max_dd": 0.15}
    base.update(kw)
    return base


def test_promotes_when_beating_incumbent_on_both_axes():
    d = evaluate_promotion(_model(), _incumbent())
    assert d.promote and not d.reasons


def test_blocks_on_sharpe_floor():
    d = evaluate_promotion(_model(sharpe=0.8), None)
    assert not d.promote
    assert any("Sharpe" in r for r in d.reasons)


def test_blocks_on_drawdown_ceiling():
    d = evaluate_promotion(_model(max_dd=0.20), None)
    assert not d.promote
    assert any("maxDD" in r for r in d.reasons)


def test_blocks_on_ic_floor():
    d = evaluate_promotion(_model(rank_ic=0.005), None)
    assert not d.promote
    assert any("rank_ic" in r for r in d.reasons)


def test_blocks_when_incumbent_beats_on_one_axis():
    # model Sharpe higher but drawdown worse — must NOT promote
    d = evaluate_promotion(_model(sharpe=1.8, max_dd=0.18), _incumbent(sharpe=1.0, max_dd=0.10))
    assert not d.promote
    assert any("both axes" in r for r in d.reasons)


def test_first_deployment_still_needs_absolute_floors():
    d = evaluate_promotion(_model(sharpe=0.5, max_dd=0.30, rank_ic=0.01), None)
    assert not d.promote
    assert len(d.reasons) == 3


def test_custom_gates_override():
    d = evaluate_promotion(_model(sharpe=0.9), None, gates={"min_sharpe": 0.8})
    assert d.promote


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
