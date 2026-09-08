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
    assert _applicable_red_lines(ALL_LINES, "ml") == ALL_LINES


def test_missing_alpha_source_keeps_every_line():
    assert _applicable_red_lines(ALL_LINES, None) == ALL_LINES
