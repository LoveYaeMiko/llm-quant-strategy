"""Red-line applicability per alpha source (D-only panel usefulness).

A long-only pullback book cannot have a short-leg imbalance and does not consume
the PEAD tilt, so those two lines are omitted for it instead of permanently
showing a meaningless "warning" that drowns the real signals.
"""
from __future__ import annotations

from src.paper.shadow import _applicable_red_lines

ALL_LINES = [
    {"name": "cost_deviation", "level": "ok"},
    {"name": "short_leg_deviation", "level": "ok"},
    {"name": "regime_switch", "level": "ok"},
    {"name": "pead_anomaly", "level": "warning"},
]


def test_pullback_book_keeps_only_relevant_lines():
    names = [rl["name"] for rl in _applicable_red_lines(ALL_LINES, "pullback")]
    assert names == ["cost_deviation", "regime_switch"]


def test_ml_book_keeps_every_line():
    out = _applicable_red_lines(ALL_LINES, "ml")
    # compare CONTENT, not the same list object (the old assertion was tautological)
    assert [rl["name"] for rl in out] == [rl["name"] for rl in ALL_LINES]
    assert len(out) == len(ALL_LINES)


def test_missing_alpha_source_keeps_every_line():
    out = _applicable_red_lines(ALL_LINES, None)
    assert [rl["name"] for rl in out] == [rl["name"] for rl in ALL_LINES]


def test_filter_does_not_mutate_the_input():
    before = [dict(rl) for rl in ALL_LINES]
    _applicable_red_lines(ALL_LINES, "pullback")
    assert ALL_LINES == before
