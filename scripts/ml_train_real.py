"""First real-data ML run — pre-registered price/volume feature set.

Walk-forward LightGBM on the real PIT store (hs300_500 research universe):
train 2010-2019 / val 2020-2021 / test 2022-2025, horizon 10d, purged CV +
embargo. The feature list is FIXED before looking at any result (no
test-peeking); the artifact lands in outputs/models/.

Usage:  python scripts/ml_train_real.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cli import _market_data  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ml import walk_forward_fit  # noqa: E402

# Pre-registered feature set — diverse price/volume families, fixed in advance.
FORMULAS = [
    # momentum
    "TS_Return(Close, 5)",
    "TS_Return(Close, 20)",
    "TS_Return(Close, 60)",
    "TS_Return(Close, 120)",
    "TS_Return(Close, 252)",
    # short-term reversal
    "Neg(TS_Return(Close, 5))",
    "Neg(TS_Return(Close, 20))",
    # volatility (the surviving FQA family)
    "TS_Std(Close, 20)",
    "TS_Std(Close, 60)",
    "TS_Std(Close, 120)",
    "TS_Std(Close, 240)",
    # turnover / volume
    "TS_Mean(Volume, 20)",
    "TS_Mean(Volume, 60)",
    "TS_Mean(Volume, 120)",
    "TS_Rel_Volume(Volume, 20)",
    "TS_Rel_Volume(Volume, 60)",
    # price-volume interaction
    "TS_Corr(Close, Volume, 60)",
    "TS_Corr(Close, Volume, 120)",
    # volatility structure
    "TS_Semi_Std(Close, 60)",
    "TS_Semi_Std(Close, 120)",
    "TS_Max_Drawdown(Close, 60)",
    "TS_Max_Drawdown(Close, 120)",
    "TS_Autocorr(Close, 60)",
    "TS_Skew(Close, 120)",
    "TS_Sharpe(TS_Return(Close, 1), 60)",
    "TS_Price_Position(Close, 60)",
    "TS_Price_Position(Close, 120)",
    "TS_ZScore(Close, 60)",
    # OHLC volatility structure
    "TS_Parkinson(High, Low, 20)",
    "TS_Parkinson(High, Low, 60)",
    "TS_Garman_Klass(Open, High, Low, Close, 60)",
    # illiquidity
    "TS_Illiquidity(Close, Volume, 20)",
    # the surviving pool composites as single features
    "Avg(Neg(Rank(TS_Std(Close, 120))), Neg(Rank(TS_Mean(Volume, 60))))",
    "Avg(Neg(Rank(TS_Std(Close, 240))), Neg(Rank(TS_Mean(Volume, 120))))",
]


def main() -> int:
    cfg = load_config()
    market = _market_data(cfg, seed=1)
    print(f"market: {market.n_symbols} symbols, {market.n_days} days "
          f"({market.price_panel.index.min().date()} .. {market.price_panel.index.max().date()})",
          flush=True)
    result = walk_forward_fit(
        market,
        FORMULAS,
        horizon=10,
        train_window=("2010-01-01", "2019-12-31"),
        val_window=("2020-01-01", "2021-12-31"),
        test_window=("2022-01-01", "2025-12-31"),
        n_estimators=300,
        early_stopping=30,
        n_folds=5,
        embargo_frac=0.01,
        cost_bps=5.0,
        out_dir=str(ROOT / "outputs" / "models"),
    )
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
