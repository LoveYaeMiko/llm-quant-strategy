"""Phase 9.2 — PEAD factor tests (SUE math, PIT non-leakage, expiry, ranks)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.financials import symbol_to_baostock
from src.factors.pead import PEADFactor


def _panel(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """Build a profit panel: (symbol, statDate, pubDate, eps_cum)."""
    return pd.DataFrame(
        [
            {"symbol": s, "statDate": pd.Timestamp(stat), "pubDate": pd.Timestamp(pub), "eps_cum": eps}
            for s, stat, pub, eps in rows
        ]
    )


# ---------------------------------------------------------------------------
# baostock symbol conversion
# ---------------------------------------------------------------------------


def test_symbol_to_baostock():
    assert symbol_to_baostock("600519.SH") == "sh.600519"
    assert symbol_to_baostock("000001.SZ") == "sz.000001"


# ---------------------------------------------------------------------------
# SUE math — seasonal same-quarter-a-year-ago comparison
# ---------------------------------------------------------------------------


def test_sue_seasonal_baseline():
    # A: Q1-2022 beats Q1-2021 (0.5 vs 0.4) -> +25%; Q2-2022 beats Q2-2021 (1.1 vs 1.0)
    panel = _panel(
        [
            ("A", "2021-03-31", "2021-04-30", 0.4),
            ("A", "2021-06-30", "2021-08-31", 1.0),
            ("A", "2022-03-31", "2022-04-30", 0.5),
            ("A", "2022-06-30", "2022-08-31", 1.1),
        ]
    )
    f = PEADFactor(panel, min_eps_history=1)
    sue_q1, pub = f.sue("A", "2022-05-01")
    assert pub == pd.Timestamp("2022-04-30")
    assert abs(sue_q1 - 0.25) < 1e-9  # (0.5-0.4)/0.4
    sue_q2, _ = f.sue("A", "2022-09-01")
    assert abs(sue_q2 - 0.10) < 1e-9  # (1.1-1.0)/1.0


def test_sue_negative_surprise():
    panel = _panel(
        [
            ("B", "2021-03-31", "2021-04-30", 0.5),
            ("B", "2022-03-31", "2022-04-30", 0.3),
        ]
    )
    f = PEADFactor(panel, min_eps_history=1)
    sue, _ = f.sue("B", "2022-05-01")
    assert abs(sue - (-0.4)) < 1e-9  # (0.3-0.5)/0.5


# ---------------------------------------------------------------------------
# PIT non-leakage
# ---------------------------------------------------------------------------


def test_pit_excludes_future_reports():
    # report for 2022Q1 announced 2022-05-01; ask on 2022-04-15 -> not yet public
    panel = _panel(
        [
            ("A", "2021-03-31", "2021-04-30", 0.4),
            ("A", "2022-03-31", "2022-05-01", 0.9),  # not public before 05-01
        ]
    )
    f = PEADFactor(panel, min_eps_history=1)
    sue, _ = f.sue("A", "2022-04-15")
    # latest public report = 2021Q1, which is a baseline with no prior-year -> no signal
    assert pd.isna(sue)
    sue_after, _ = f.sue("A", "2022-05-02")
    assert abs(sue_after - 1.25) < 1e-9  # (0.9-0.4)/0.4


def test_expiry_drops_old_signals():
    panel = _panel(
        [
            ("A", "2021-03-31", "2021-04-30", 0.4),
            ("A", "2022-03-31", "2022-04-30", 0.5),
        ]
    )
    f = PEADFactor(panel, signal_expiry_days=60, min_eps_history=1)
    assert not pd.isna(f.sue("A", "2022-05-01")[0])      # 1 day after -> valid
    assert pd.isna(f.sue("A", "2022-07-30")[0])          # 91 days after -> expired


def test_missing_baseline_or_zero_eps_is_no_signal():
    panel = _panel(
        [
            ("A", "2022-03-31", "2022-04-30", 0.5),      # no prior-year baseline
            ("B", "2021-03-31", "2021-04-30", 0.0),      # zero baseline
            ("B", "2022-03-31", "2022-04-30", 0.2),
        ]
    )
    f = PEADFactor(panel, min_eps_history=1)
    assert pd.isna(f.sue("A", "2022-05-01")[0])
    assert pd.isna(f.sue("B", "2022-05-01")[0])


def test_min_history_filters_young_symbols():
    panel = _panel(
        [
            ("A", "2021-03-31", "2021-04-30", 0.4),
            ("A", "2022-03-31", "2022-04-30", 0.5),
        ]
    )
    f = PEADFactor(panel, min_eps_history=8)
    assert pd.isna(f.sue("A", "2022-05-01")[0])          # only 2 reports


# ---------------------------------------------------------------------------
# score_panel — cross-sectional percentile ranks
# ---------------------------------------------------------------------------


def test_score_panel_cross_sectional_rank():
    panel = _panel(
        [
            ("A", "2021-03-31", "2021-04-30", 0.4),
            ("B", "2021-03-31", "2021-04-30", 0.6),
            ("C", "2021-03-31", "2021-04-30", 0.8),
            ("A", "2022-03-31", "2022-04-30", 0.5),   # A +25%
            ("B", "2022-03-31", "2022-04-30", 0.6),   # B 0%
            ("C", "2022-03-31", "2022-04-30", 0.7),   # C -12.5%
        ]
    )
    f = PEADFactor(panel, min_eps_history=1)
    scores = f.score_panel(["2022-05-01"], ["A", "B", "C"])
    assert len(scores) == 3
    by_sym = {s: float(scores.xs(s, level="symbol").iloc[0]) for s in ("A", "B", "C")}
    # A has the highest SUE -> highest percentile rank (1.0); C the lowest
    assert by_sym["A"] == pytest.approx(1.0)
    assert by_sym["C"] == pytest.approx(1 / 3)
    # unknown symbol -> NaN (excluded from the book)
    scores_with_d = f.score_panel(["2022-05-01"], ["A", "B", "C", "D"])
    assert bool(scores_with_d.xs("D", level="symbol").isna().iloc[0])
