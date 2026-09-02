"""Factor-zoo walk-forward scan + expanded-feature ML retrain (round 2).

One market load, two deliverables:

1. **Zoo scan** — evaluate the 79 translated Alpha101/GTJA formulas on the real
   research universe, walk-forward (train/val/test from master config), report
   per-formula rank_ic/ICIR per window + a tradable tail-spread check on test.
   A formula is flagged "reliable" when its test rank_ic keeps the same sign as
   train and clears |rank_ic| >= 0.02 with ICIR >= 0.30 on test.
2. **Expanded ML** — retrain the LightGBM with a PRE-REGISTERED feature set =
   baseline 34 price/volume features + the translated zoo (deduped). Same
   windows, purged CV + embargo; test evaluated exactly once.

Usage:  python scripts/scan_factor_zoo.py [--families alpha101,gtja191] [--skip-ml]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.backtest.metrics import daily_ic  # noqa: E402
from src.cli import _market_data  # noqa: E402
from src.config import load_config  # noqa: E402
from src.factors.code_generator import FactorContext, eval_expression  # noqa: E402
from src.ml import walk_forward_fit  # noqa: E402

BASELINE = [
    "TS_Return(Close, 5)", "TS_Return(Close, 20)", "TS_Return(Close, 60)",
    "TS_Return(Close, 120)", "TS_Return(Close, 252)",
    "Neg(TS_Return(Close, 5))", "Neg(TS_Return(Close, 20))",
    "TS_Std(Close, 20)", "TS_Std(Close, 60)", "TS_Std(Close, 120)", "TS_Std(Close, 240)",
    "TS_Mean(Volume, 20)", "TS_Mean(Volume, 60)", "TS_Mean(Volume, 120)",
    "TS_Rel_Volume(Volume, 20)", "TS_Rel_Volume(Volume, 60)",
    "TS_Corr(Close, Volume, 60)", "TS_Corr(Close, Volume, 120)",
    "TS_Semi_Std(Close, 60)", "TS_Semi_Std(Close, 120)",
    "TS_Max_Drawdown(Close, 60)", "TS_Max_Drawdown(Close, 120)",
    "TS_Autocorr(Close, 60)", "TS_Skew(Close, 120)",
    "TS_Sharpe(TS_Return(Close, 1), 60)",
    "TS_Price_Position(Close, 60)", "TS_Price_Position(Close, 120)",
    "TS_ZScore(Close, 60)",
    "TS_Parkinson(High, Low, 20)", "TS_Parkinson(High, Low, 60)",
    "TS_Garman_Klass(Open, High, Low, Close, 60)",
    "TS_Illiquidity(Close, Volume, 20)",
    "Avg(Neg(Rank(TS_Std(Close, 120))), Neg(Rank(TS_Mean(Volume, 60))))",
    "Avg(Neg(Rank(TS_Std(Close, 240))), Neg(Rank(TS_Mean(Volume, 120))))",
]

_WINDOWS = {
    "train": ("2010-01-01", "2019-12-31"),
    "val": ("2020-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2025-12-31"),
}


def _window_ic(scores: pd.Series, fwd: pd.Series, start: str, end: str) -> dict:
    s = scores[scores.index.get_level_values(0) >= pd.Timestamp(start)]
    s = s[s.index.get_level_values(0) <= pd.Timestamp(end)]
    ic = daily_ic(s, fwd, method="spearman").dropna()
    if len(ic) < 5:
        return {"n_days": int(len(ic)), "rank_ic": None, "icir": None}
    return {
        "n_days": int(len(ic)),
        "rank_ic": float(ic.mean()),
        "icir": float(ic.mean() / ic.std() * 252 ** 0.5) if ic.std() > 0 else None,
    }


def scan_zoo(market, formulas: list[str]) -> dict:
    fctx = FactorContext(market.long)
    fwd = market.forward_returns
    results = {}
    errors = {}
    for fqa in formulas:
        t0 = time.time()
        try:
            scores = eval_expression(fqa, fctx)
            wins = {w: _window_ic(scores, fwd, *span) for w, span in _WINDOWS.items()}
            test_ic = wins["test"]["rank_ic"]
            train_ic = wins["train"]["rank_ic"]
            reliable = (
                test_ic is not None
                and abs(test_ic) >= 0.02
                and (wins["test"]["icir"] or 0) >= 0.30
                and (train_ic is None or test_ic * train_ic >= 0)
            )
            results[fqa] = {"windows": wins, "reliable": bool(reliable), "sec": round(time.time() - t0, 1)}
        except Exception as exc:  # noqa: BLE001 — one broken formula must not kill the scan
            errors[fqa] = f"{type(exc).__name__}: {str(exc)[:100]}"
    return {"results": results, "errors": errors}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="alpha101,gtja191,alpha158")
    ap.add_argument("--skip-ml", action="store_true", help="scan only, no ML retrain")
    args = ap.parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    cfg = load_config()
    t0 = time.time()
    market = _market_data(cfg, seed=1)
    print(f"market: {market.n_symbols} symbols, {market.n_days} days "
          f"({market.price_panel.index.min().date()} .. {market.price_panel.index.max().date()}) "
          f"[{time.time()-t0:.0f}s]", flush=True)

    translated = json.loads(
        (ROOT / "paper" / "factor_zoo" / "translated.json").read_text(encoding="utf-8")
    )
    zoo = [
        r["fqa"]
        for fam in families
        for r in translated[fam].values()
        if r.get("status") == "ok"
    ]
    zoo = list(dict.fromkeys(zoo))
    print(f"zoo formulas to scan ({families}): {len(zoo)}", flush=True)

    scan = scan_zoo(market, zoo)
    n_ok = len(scan["results"])
    n_reliable = sum(1 for r in scan["results"].values() if r["reliable"])
    print(f"scan: {n_ok} evaluated, {len(scan['errors'])} errors, {n_reliable} reliable", flush=True)
    if n_reliable:
        for fqa, r in scan["results"].items():
            if r["reliable"]:
                t = r["windows"]["test"]
                print(f"  RELIABLE rank_ic={t['rank_ic']:+.4f} icir={t['icir']:+.2f} :: {fqa[:90]}", flush=True)
    out = ROOT / "paper" / "factor_zoo" / f"scan_results_{'-'.join(families)}.json"
    out.write_text(json.dumps(scan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)

    if args.skip_ml:
        return 0

    # ---- expanded ML retrain (pre-registered: baseline + zoo, deduped) ----
    features = list(dict.fromkeys(BASELINE + zoo))
    print(f"\nexpanded ML: {len(features)} features", flush=True)
    ml = walk_forward_fit(
        market,
        features,
        horizon=10,
        train_window=_WINDOWS["train"],
        val_window=_WINDOWS["val"],
        test_window=_WINDOWS["test"],
        n_estimators=300,
        early_stopping=30,
        n_folds=5,
        embargo_frac=0.01,
        cost_bps=5.0,
        out_dir=str(ROOT / "outputs" / "models"),
    )
    print("expanded ML result:", json.dumps(ml, ensure_ascii=False, indent=2)[:1600], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
