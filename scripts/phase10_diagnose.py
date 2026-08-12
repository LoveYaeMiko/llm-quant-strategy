"""Phase 10 sub-period diagnosis — risk-drawdown-reduction & earnings-season tilt.

Blueprint §4.2/§4.3 checks on the four scenarios:

* **risk overlay (4.2)**: does the sentiment circuit-breaker reduce max drawdown
  in the crisis windows where it fires? Sentiment coverage is 2022-2025, so the
  effect is measured there (2015/2018 crises pre-date the overlay by construction).
* **seasonal tilt (4.3)**: does the PEAD reversal tilt beat the baseline during
  earnings-season months (1,2,4,8,10) in the PEAD coverage window (2020-2025)?
* **drawdown structure**: where is the headline 34.5% maxDD (baseline) located,
  and how much of it is market beta vs idiosyncratic?

Prints a console report; no files written.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CRISIS = {
    "2015股灾": ("2015-05-01", "2015-09-30"),
    "2018熊市": ("2018-01-01", "2018-12-31"),
    "2024微盘股": ("2024-01-01", "2024-02-29"),
    "2022熊市": ("2022-01-01", "2022-04-30"),
}
EARN_MONTHS = (1, 2, 4, 8, 10)
TILT_WINDOW = ("2020-01-01", "2025-12-31")
RISK_WINDOW = ("2022-01-01", "2025-12-31")


def _dd(port: pd.Series) -> pd.Series:
    cum = (1 + port).cumprod()
    return cum / cum.cummax() - 1.0


def _maxdd(port: pd.Series) -> float:
    return float(-_dd(port).min())


def main() -> int:
    from src.cli import _cached_universe_json
    from src.config import load_config
    from src.data.financials import ensure_profit_panel
    from src.factors.code_generator import FactorContext
    from src.factors.pead import PEADFactor
    from src.portfolio.alpha_core import AlphaCore
    from src.portfolio.backtest_runner import load_hs300_market
    from src.portfolio.layer_integration import ThreeLayerPortfolio
    from src.portfolio.risk_overlay import SentimentRiskOverlay
    from src.portfolio.seasonal_tilt import PEADSeasonalTilt

    cfg = load_config()
    symbols = _cached_universe_json("hs300", cfg)
    t0 = time.time()
    market = load_hs300_market(cfg, symbols, start="2009-01-06", end="2025-12-31")
    formulas = [x.get("factor", {}).get("formula", x.get("formula"))
                for x in json.loads((ROOT / "outputs" / "factors.json").read_text(encoding="utf-8"))]
    _fwd_wide = market.forward_returns_tradable.unstack(fill_value=0.0)
    _bench = _fwd_wide.reindex(columns=symbols).mean(axis=1)
    _trend = (1 + _bench).shift(1).rolling(60).apply(lambda x: x.prod() - 1, raw=True)
    alpha = AlphaCore(FactorContext(market.long), formulas, long_pct=0.10, max_position_pct=0.05,
                      trend_series=_trend)

    pead_cfg = cfg.get("pead") or {}
    panel = ensure_profit_panel(symbols, list(range(2020, 2026)),
                                cache_dir=str(pead_cfg.get("cache_dir", "data/financials")))
    pead = PEADFactor(panel, signal_expiry_days=int(pead_cfg.get("signal_expiry_days", 60)))

    from src.sentiment.ingestion import ReportIngestor
    from src.sentiment.triagent import build_report_signal
    sent_cfg = cfg.get("sentiment") or {}
    have = set(symbols) & ReportIngestor(str(sent_cfg.get("report_dir", "data/reports"))).cached_symbols()
    reports = ReportIngestor(str(sent_cfg.get("report_dir", "data/reports"))).load(symbols=sorted(have))
    scores = pd.read_parquet(str(sent_cfg.get("score_cache", "data/reports/report_sentiment.parquet")))
    trading = sorted(market.forward_returns.index.get_level_values(0).unique())
    sig = build_report_signal(reports, scores, trading, sorted(have),
                              decay_days=int(sent_cfg.get("decay_days", 10)))

    tilt = PEADSeasonalTilt(pead, universe=symbols)

    def _fresh_risk():
        return SentimentRiskOverlay(sig, zscore_threshold=-2.5, position_cut=0.50,
                                    freeze_days=5, min_trigger_samples=20)

    dates = [d for d in sorted(market.forward_returns.index.get_level_values(0).unique())
             if pd.Timestamp("2010-01-01") <= d <= pd.Timestamp("2025-12-31")]
    fwd_wide = market.forward_returns_tradable.unstack(fill_value=0.0)
    aligned = fwd_wide.reindex(columns=sorted(alpha.composite.index.get_level_values(1).unique())).fillna(0.0)

    scenarios = {
        "baseline": (None, None),
        "alpha_risk": (None, _fresh_risk()),
        "alpha_tilt": (tilt, None),
        "three_layer": (tilt, _fresh_risk()),
    }
    ports: dict[str, pd.Series] = {}
    for name, (t, r) in scenarios.items():
        pf = ThreeLayerPortfolio(alpha, tilt=t, risk=r)
        wdf = pf.weights_frame(symbols, dates)
        # align the fwd panel to the *actual* book dates (weights.index starts at
        # the first non-empty book). Without the index slice, `wdf * aligned`
        # aligns on the union and the pre-book dates (2009 + warm-up 2010) come
        # out as zero-return rows, diluting Sharpe/ann in the headline.
        port = (wdf.reindex(columns=aligned.columns, fill_value=0.0)
                * aligned.reindex(index=wdf.index)).sum(axis=1).sort_index()
        ports[name] = port.dropna()
    print(f"diagnose: built {len(ports)} scenarios in {time.time() - t0:.1f}s\n")
    for name in ("alpha_risk", "three_layer"):
        r = scenarios[name][1]
        if r is not None:
            print(f"  risk triggers ({name}): {len(r.trigger_log())}")

    # ------------------------------------------------------- headline
    print("=== HEADLINE (2010-2025) ===")
    for name, p in ports.items():
        print(f"  {name:<12} sharpe={p.mean()/p.std()*np.sqrt(252):.3f} "
              f"ann={(1+p).prod()**(252/len(p))-1:.1%} maxDD={_maxdd(p):.1%}")

    # ------------------------------------------------------- 4.2 risk reduction
    print("\n=== 4.2 RISK OVERLAY — maxDD reduction in crisis windows (2022-2025 only) ===")
    risk_r = scenarios["alpha_risk"][1]
    print(f"  risk triggers (2022-2025, alpha_risk scenario): {len(risk_r.trigger_log())}")
    for cname, (cs, ce) in CRISIS.items():
        for name in ("baseline", "alpha_risk"):
            p = ports[name]
            sub = p[(p.index >= cs) & (p.index <= ce)]
            if len(sub) < 10:
                continue
            print(f"  {cname}: {name:<12} maxDD={_maxdd(sub):.2%} days={len(sub)}")
        print(f"  {'':22}---")
    # whole 2022-2025 window
    for name in ("baseline", "alpha_risk"):
        p = ports[name]
        sub = p[(p.index >= RISK_WINDOW[0]) & (p.index <= RISK_WINDOW[1])]
        print(f"  {RISK_WINDOW[0]}..{RISK_WINDOW[1]}: {name:<12} maxDD={_maxdd(sub):.2%} "
              f"ann={(1+sub).prod()**(252/len(sub))-1:.2%}")

    # ------------------------------------------------------- 4.3 earnings-season tilt
    print("\n=== 4.3 SEASONAL TILT — earnings-season months 1/2/4/8/10 (2020-2025) ===")
    for name in ("baseline", "alpha_tilt"):
        p = ports[name]
        sub = p[(p.index >= TILT_WINDOW[0]) & (p.index <= TILT_WINDOW[1])]
        season = sub[sub.index.month.isin(EARN_MONTHS)]
        nonseason = sub[~sub.index.month.isin(EARN_MONTHS)]
        print(f"  {name:<12} season ann={(1+season).prod()**(252/max(len(season),1))-1:.3%} "
              f"({len(season)}d)  non-season ann={(1+nonseason).prod()**(252/max(len(nonseason),1))-1:.3%} "
              f"({len(nonseason)}d)  maxDD={_maxdd(sub):.2%}")

    # ------------------------------------------------------- drawdown structure
    print("\n=== DRAWDDOWN STRUCTURE (baseline) ===")
    for name in ("baseline",):
        p = ports[name]
        dd = _dd(p)
        cum = (1 + p).cumprod()
        for i, (dt, v) in enumerate(dd.sort_values().head(4).items(), 1):
            peak = cum[:dt].idxmax()  # max cum level before the trough = the peak
            print(f"  DD#{i}: {v:.1%} trough={dt.date()} peak={peak.date()} "
                  f"cum@peak={cum.loc[peak]:.2f} cum@trough={cum.loc[dt]:.2f}")
        # yearly
        y = (1 + p).groupby(p.index.year).prod() - 1
        print("  yearly:", " ".join(f"{k}:{v:+.0%}" for k, v in y.items()))

    # market beta of the book
    bench = fwd_wide.reindex(index=ports["baseline"].index, columns=aligned.columns).mean(axis=1)
    print(f"\n  corr(baseline, equal-weight market): {ports['baseline'].corr(bench):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
