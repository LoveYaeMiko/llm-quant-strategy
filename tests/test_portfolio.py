"""Phase 10 — three-layer portfolio unit tests.

Covers the blueprint §2.5 contract: alpha-book construction (top-decile
selection / cap / NaN drop), PEAD tilt trigger + amplitude + min-weight guard,
sentiment risk-cut trigger + 5-day freeze + history guard, and the layer
integration priority (risk > tilt > alpha) + normalisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.alpha_core import AlphaCore, long_book_weights
from src.portfolio.layer_integration import ThreeLayerPortfolio
from src.portfolio.risk_overlay import SentimentRiskOverlay
from src.portfolio.seasonal_tilt import PEADSeasonalTilt


# ---------------------------------------------------------------------------
# alpha core — long book construction
# ---------------------------------------------------------------------------


def test_long_book_top_decile_equal_weight():
    scores = pd.Series({"A": 1.0, "B": 0.9, "C": 0.8, "D": 0.1, "E": 0.0, "F": -1.0,
                        "G": -0.5, "H": -0.3, "I": 0.2, "J": 0.15})
    w = long_book_weights(scores, long_pct=0.20, max_position_pct=1.0)
    # top 20% of 10 = 2 names (A, B), equal weight 0.5 each
    assert set(w) == {"A", "B"}
    assert w["A"] == pytest.approx(0.5)
    assert w["B"] == pytest.approx(0.5)


def test_long_book_drops_nan():
    scores = pd.Series({"A": 1.0, "B": np.nan, "C": 0.5})
    w = long_book_weights(scores, long_pct=1.0, max_position_pct=1.0)
    assert set(w) == {"A", "C"}


def test_long_book_empty_when_all_nan():
    assert long_book_weights(pd.Series({"A": np.nan})) == {}


def test_long_book_cap_after_normalisation():
    # small universe: top decile = 1 name, weight 1.0 -> capped at 0.05
    scores = pd.Series({"A": 1.0, "B": 0.5, "C": 0.4, "D": 0.3})
    w = long_book_weights(scores, long_pct=0.25, max_position_pct=0.05)
    assert w["A"] == pytest.approx(0.05)


def test_long_book_weights_drops_groupby_date_level():
    # AlphaCore feeds long_book_weights a groupby(level=0) group, whose index is
    # still the full (date, symbol) MultiIndex — the keys must be plain symbols.
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2023-04-15"), s) for s in "ABCDE"], names=["date", "symbol"]
    )
    day = pd.Series([1.0, 0.9, 0.8, 0.1, 0.0], index=idx)
    w = long_book_weights(day, long_pct=0.4, max_position_pct=1.0)
    assert set(w) == {"A", "B"}
    assert all(isinstance(k, str) for k in w)


def test_alpha_core_composite_equal_weight():
    # legacy long-only mode (short_pct=0): gross 1, all positive
    from src.factors.code_generator import FactorContext
    from src.data.synthetic import make_synthetic_market

    market = make_synthetic_market(symbols=10, days=120, seed=7)
    fctx = FactorContext(market.long)
    core = AlphaCore(fctx, ["Rank(Close)", "Rank(Close)"], long_pct=0.20,
                     short_pct=0.0, max_position_pct=1.0, neutralize=False)
    # two identical factors -> composite = the single factor z-score
    day = sorted(core.dates)[0]
    w = core.weights_on(day)
    assert w  # non-empty book
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)


def test_alpha_core_market_neutral_net_zero():
    # Phase 10 default: long top decile + short bottom decile, net ~ 0, gross 1
    from src.factors.code_generator import FactorContext
    from src.data.synthetic import make_synthetic_market

    market = make_synthetic_market(symbols=10, days=120, seed=7)
    fctx = FactorContext(market.long)
    core = AlphaCore(fctx, ["Rank(Close)", "Rank(Close)"], long_pct=0.20,
                     short_pct=0.20, max_position_pct=1.0, neutralize=False)
    day = sorted(core.dates)[0]
    w = core.weights_on(day)
    assert w
    assert sum(w.values()) == pytest.approx(0.0, abs=1e-9)          # dollar-neutral
    assert sum(abs(v) for v in w.values()) == pytest.approx(1.0, abs=1e-9)  # gross 1
    assert any(v > 0 for v in w.values()) and any(v < 0 for v in w.values())


# ---------------------------------------------------------------------------
# momentum neutralization + regime-adaptive short control (paper-backed)
# ---------------------------------------------------------------------------


def _big_market(days=400, seed=7, drift=0.0):
    from src.data.synthetic import make_synthetic_market
    from src.factors.code_generator import FactorContext

    market = make_synthetic_market(symbols=40, days=days, seed=seed, market_drift=drift)
    return market, FactorContext(market.long)


def test_neutralize_removes_momentum_exposure():
    # composite = momentum + independent noise. Neutralization must strip the
    # momentum component, leaving ~the noise -> residual ~0 correlation with
    # momentum. (A composite that IS pure momentum would give a numerically-zero
    # residual whose "correlation" is meaningless floating-point noise.)
    from src.portfolio.alpha_core import _neutralize_composite, _momentum_panel, _zscore

    market, _ = _big_market()
    close = market.long["close"].unstack()
    mom = _momentum_panel(close, (20, 60, 120, 252))
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.normal(0.0, 1.0, len(mom["mom120"].dropna())),
                      index=mom["mom120"].dropna().index)
    z = _zscore(mom["mom120"]) + 0.5 * _zscore(noise)
    resid = _neutralize_composite(z, close, (20, 60, 120, 252))
    assert not resid.empty
    joint = pd.concat([resid.rename("resid"), mom["mom120"]], axis=1).dropna()
    corrs = []
    for _, day in joint.groupby(level=0):
        if len(day) >= 30:
            corrs.append(np.corrcoef(day["resid"].rank(), day["mom120"].rank())[0, 1])
    assert np.mean(np.abs(corrs)) < 0.2  # momentum component removed


def test_neutralize_removes_beta_exposure():
    # A low-vol composite is structurally negative-beta (low-vol = low beta).
    # Momentum-only neutralization does NOT remove this (beta != momentum), so a
    # dollar-neutral long/short book leaks market beta. beta_neutralize adds the
    # market-beta regressor and must strip the beta component from the residual.
    from src.portfolio.alpha_core import _beta_panel, _neutralize_composite, _zscore

    market, _ = _big_market(days=500, seed=3)
    close = market.long["close"].unstack()
    beta = _beta_panel(close, 252)
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.normal(0.0, 1.0, len(beta.dropna())),
                      index=beta.dropna().index)
    z = -_zscore(beta) + 0.5 * _zscore(noise)   # score negatively correlated with beta
    resid = _neutralize_composite(z, close, (20, 60, 120, 252), beta_neutralize=True)
    assert not resid.empty
    joint = pd.concat([resid.rename("resid"), beta.rename("beta")], axis=1).dropna()
    corrs = []
    for _, day in joint.groupby(level=0):
        if len(day) >= 30:
            corrs.append(np.corrcoef(day["resid"].rank(), day["beta"].rank())[0, 1])
    assert np.mean(np.abs(corrs)) < 0.3  # beta component removed


def test_alpha_core_neutralize_produces_book():
    # end-to-end: 40-symbol market with neutralize=True yields a valid book
    market, fctx = _big_market()
    core = AlphaCore(fctx, ["Rank(Close)", "Rank(Close)"], long_pct=0.20,
                     short_pct=0.20, max_position_pct=1.0, neutralize=True)
    day = sorted(core.dates)[len(core.dates) // 2]
    w = core.weights_on(day)
    assert w
    assert sum(abs(v) for v in w.values()) == pytest.approx(1.0, abs=1e-9)


def test_alpha_core_regime_short_trims_short_leg_in_uptrend():
    # strong uptrend (drift) -> 60d market trend > gate -> short leg scaled to
    # short_scale before gross normalisation; flat market leaves it at ~0.5.
    from src.portfolio.alpha_core import _market_trend

    market, fctx = _big_market(days=300, seed=11, drift=0.002)
    close = market.long["close"].unstack()
    trend = _market_trend(close, 60)
    core = AlphaCore(fctx, ["Rank(Close)", "Rank(Close)"], long_pct=0.20,
                     short_pct=0.20, max_position_pct=1.0, neutralize=False,
                     regime_short=True, trend_gate=0.01, short_scale=0.4)
    late = sorted(core.dates)[-20]
    assert trend[late] > 0.01  # regime active
    w = core.weights_on(late)
    short_total = -sum(v for v in w.values() if v < 0)
    # gross-normalised book: long 1.0, short 0.4 -> short total 0.4/1.4 ≈ 0.286
    assert short_total == pytest.approx(0.4 / 1.4, abs=0.03)

    # flat market: regime inactive, short leg ~0.5
    market2, fctx2 = _big_market(days=300, seed=13, drift=0.0)
    core2 = AlphaCore(fctx2, ["Rank(Close)", "Rank(Close)"], long_pct=0.20,
                      short_pct=0.20, max_position_pct=1.0, neutralize=False,
                      regime_short=True, trend_gate=0.01, short_scale=0.4)
    day2 = sorted(core2.dates)[-20]
    w2 = core2.weights_on(day2)
    short2 = -sum(v for v in w2.values() if v < 0)
    assert short2 == pytest.approx(0.5, abs=0.03)


# ---------------------------------------------------------------------------
# seasonal tilt
# ---------------------------------------------------------------------------


class _FakePEAD:
    def __init__(self, sues: dict):
        self.sues = sues

    def sue_snapshot(self, symbols, date):
        return pd.Series({s: self.sues[s] for s in symbols if s in self.sues})


def _tilt_monthly(date_str: str):
    # 10 symbols so the top/bottom SUE names land outside the 0.8/0.2 cutoffs
    sues = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0,
            "F": 0.0, "G": -1.0, "H": -2.0, "I": -3.0, "J": -5.0}
    # percentile is measured over the full universe (the blueprint's 全市场分位)
    return PEADSeasonalTilt(_FakePEAD(sues), universe=list(sues.keys()),
                            amplitude=0.20, min_weight=0.03)


def test_tilt_noop_outside_earnings_season():
    tilt = _tilt_monthly("2023-07-01")
    w = {"A": 0.10, "B": 0.10}
    assert tilt.apply(w, "2023-07-01") == w  # July not in (1,2,4,8,10)


def test_tilt_high_sue_down_low_sue_up():
    tilt = _tilt_monthly("2023-04-15")  # April earnings season
    w = {"A": 0.10, "B": 0.10, "C": 0.10, "D": 0.10, "J": 0.10}
    out = tilt.apply(w, "2023-04-15")
    assert out["A"] == pytest.approx(0.10 * 0.80)  # top SUE (pct=1.0) -> -20%
    assert out["J"] == pytest.approx(0.10 * 1.20)  # bottom SUE (pct=0.1) -> +20%
    assert out["D"] == pytest.approx(0.10)  # mid SUE -> unchanged


def test_tilt_min_weight_guard():
    tilt = _tilt_monthly("2023-04-15")
    w = {"A": 0.01}  # below min_weight -> untouched
    out = tilt.apply(w, "2023-04-15")
    assert out["A"] == pytest.approx(0.01)


def test_tilt_short_leg_direction_aware():
    # sign-aware amplitude: short high-SUE (reverts down) adds, short low-SUE cuts
    tilt = _tilt_monthly("2023-04-15")
    w = {"A": -0.10, "J": -0.10}
    out = tilt.apply(w, "2023-04-15")
    assert out["A"] == pytest.approx(-0.10 * 1.20)  # short high-SUE -> more short
    assert out["J"] == pytest.approx(-0.10 * 0.80)  # short low-SUE -> less short


def test_tilt_ignores_symbols_without_sue():
    tilt = _tilt_monthly("2023-04-15")
    w = {"A": 0.10, "ZZZ": 0.10}
    out = tilt.apply(w, "2023-04-15")
    assert out["ZZZ"] == pytest.approx(0.10)  # no SUE -> unchanged


# ---------------------------------------------------------------------------
# risk overlay — sentiment circuit breaker
# ---------------------------------------------------------------------------


def _sentiment_panel(crash_on="AAA"):
    dates = pd.bdate_range("2022-01-01", periods=320)
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "symbol"])
    vals = np.full(len(idx), 0.5, dtype=float)
    # AAA: stable 0.5 history, collapse to 0.05 in the last 5 days
    for i, (d, s) in enumerate(zip(idx.get_level_values(0), idx.get_level_values(1))):
        if s == "AAA":
            day = int(np.searchsorted(dates, d))
            if day >= len(dates) - 5:
                vals[i] = 0.05
    return pd.Series(vals, index=idx)


def test_risk_trigger_and_fifty_percent_cut():
    panel = _sentiment_panel()
    overlay = SentimentRiskOverlay(panel, zscore_threshold=-2.5, position_cut=0.50,
                                   freeze_days=5, min_trigger_samples=20)
    last_day = panel.index.get_level_values(0).max()
    w = {"AAA": 0.10, "BBB": 0.10}
    out = overlay.apply(w, last_day)
    assert out["AAA"] == pytest.approx(0.05)  # halved
    assert out["BBB"] == pytest.approx(0.10)  # untouched
    assert len(overlay.trigger_log()) == 1
    assert overlay.trigger_log()[0]["symbol"] == "AAA"


def test_risk_freeze_last_five_days():
    panel = _sentiment_panel()
    overlay = SentimentRiskOverlay(panel, zscore_threshold=-2.5, position_cut=0.50,
                                   freeze_days=5, min_trigger_samples=20)
    dates = list(panel.index.get_level_values(0).unique())
    last = dates[-1]
    overlay.apply({"AAA": 0.10}, last)
    # next trading day still frozen (halved), no new trigger logged
    n_triggers = len(overlay.trigger_log())
    out = overlay.apply({"AAA": 0.10}, dates[-2])
    assert out["AAA"] == pytest.approx(0.05)
    assert len(overlay.trigger_log()) == n_triggers


def test_risk_no_trigger_without_history():
    # only a few days of data -> below min_trigger_samples -> no trigger
    dates = pd.bdate_range("2022-01-01", periods=10)
    idx = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "symbol"])
    panel = pd.Series(0.05, index=idx)
    overlay = SentimentRiskOverlay(panel, zscore_threshold=-2.5, position_cut=0.50,
                                   freeze_days=5, min_trigger_samples=20)
    out = overlay.apply({"AAA": 0.10}, dates[-1])
    assert out["AAA"] == pytest.approx(0.10)  # untouched
    assert overlay.trigger_log() == []


def test_risk_no_trigger_when_recent_not_extreme():
    panel = _sentiment_panel()
    # replace the crash so the recent sentiment stays ~0.5
    panel = panel.copy()
    dates = panel.index.get_level_values(0).unique()
    for i, (d, s) in enumerate(zip(panel.index.get_level_values(0), panel.index.get_level_values(1))):
        if s == "AAA" and d >= dates[-5]:
            panel.iat[i] = 0.5
    overlay = SentimentRiskOverlay(panel, zscore_threshold=-2.5, position_cut=0.50,
                                   freeze_days=5, min_trigger_samples=20)
    out = overlay.apply({"AAA": 0.10}, dates[-1])
    assert out["AAA"] == pytest.approx(0.10)
    assert overlay.trigger_log() == []


# ---------------------------------------------------------------------------
# layer integration — priority + normalisation
# ---------------------------------------------------------------------------


class _FakeAlpha:
    def weights_on(self, date, symbols=None):
        return {"A": 0.40, "B": 0.35, "C": 0.25}


class _FakeTilt:
    def apply(self, weights, date):
        return {s: v * 1.2 for s, v in weights.items()}


class _FakeRisk:
    def apply(self, weights, date):
        out = dict(weights)
        out["A"] *= 0.5  # cut A (highest priority: applied last overrides tilt)
        return out


def test_three_layer_order_risk_overrides_tilt():
    pf = ThreeLayerPortfolio(_FakeAlpha(), tilt=_FakeTilt(), risk=_FakeRisk())
    w = pf.compute_weights(["A", "B", "C"], "2023-04-15")
    # tilt: A=0.48, B=0.42, C=0.30; risk cuts A: A=0.24, B=0.42, C=0.30
    # normalise by the POST-cut total (0.96), so A = 0.24/0.96 = 0.25
    total = 0.24 + 0.42 + 0.30
    assert w["A"] == pytest.approx(0.24 / total)
    assert w["B"] == pytest.approx(0.42 / total)
    assert sum(w.values()) == pytest.approx(1.0)  # normalised


def test_weights_frame_index_is_chronological():
    # from_dict(orient="index") does not guarantee row order — a scrambled index
    # corrupts cumprod/cummax (maxDD). The frame must come back time-sorted.
    pf = ThreeLayerPortfolio(_FakeAlpha())
    dates = [pd.Timestamp("2023-04-10"), pd.Timestamp("2023-04-11"), pd.Timestamp("2023-04-12")]
    wdf = pf.weights_frame(["A", "B", "C"], dates)
    assert wdf.index.is_monotonic_increasing
    assert list(wdf.index) == sorted(dates)


def test_three_layer_no_overlays_is_alpha():
    pf = ThreeLayerPortfolio(_FakeAlpha(), tilt=None, risk=None)
    w = pf.compute_weights(["A", "B", "C"], "2023-04-15")
    assert w["A"] == pytest.approx(0.40)
    assert sum(w.values()) == pytest.approx(1.0)
