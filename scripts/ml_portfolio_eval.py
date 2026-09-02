"""Three-way showdown v3 — dual capital standards + 2026 live segment.

Books: MLP v1 (108 features), MLP v2 (345), XGBoost-GPU (345), LightGBM (345),
their per-date rank ensemble, and the incumbent 5-factor pool. Each book runs
through PaperRunner with the REAL A-share cost model under BOTH starting
capitals (2,000,000 and 100,000) on BOTH segments: the test window
(2022-2025) and the 2026 live segment (2026-01 .. data end).

The 100k column is deliberately shown: at that scale the 5-yuan minimum
commission on a 60-name book is structurally dominant (the live shadow's
-5.24% is exactly this) — the honest deployment picture.

Usage:  python scripts/ml_portfolio_eval.py [--capitals 2000000,100000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.cli import _market_data  # noqa: E402
from src.config import load_config  # noqa: E402
from src.factors.code_generator import FactorContext  # noqa: E402
from src.ml import build_feature_matrix, load_artifact, score_artifact  # noqa: E402
from src.ml.ensemble import rank_ensemble  # noqa: E402
from src.ml.promote import evaluate_promotion  # noqa: E402
from src.ml.torch_model import load_torch_artifact  # noqa: E402
from src.paper.ledger import PaperLedger  # noqa: E402
from src.paper.runner import PaperRunner  # noqa: E402
from src.portfolio.alpha_core import (  # noqa: E402
    AlphaCore,
    _market_trend,
    long_book_weights,
)

SEGMENTS = {
    "test_2022_2025": ("2022-01-01", "2025-12-31"),
    "live_2026": ("2026-01-01", "2026-08-28"),
}


class _BooksPortfolio:
    def __init__(self, books: dict[pd.Timestamp, dict[str, float]]):
        self._books = books

    def compute_weights(self, symbols, date):
        return self._books.get(pd.Timestamp(date), {})


class _AlphaCorePortfolio:
    def __init__(self, alpha: AlphaCore):
        self.alpha = alpha

    def compute_weights(self, symbols, date):
        return self.alpha.weights_on(date, symbols=symbols)


def _run_book(market, portfolio, symbols, cost: dict, rebalance_days: int,
              cash: float, start: str, end: str) -> dict:
    ledger_path = ROOT / "outputs" / f"_evalledger_{rebalance_days}_{cash:.0f}.sqlite"
    if ledger_path.exists():
        ledger_path.unlink()
    ledger = PaperLedger(str(ledger_path))
    runner = PaperRunner(
        portfolio, market, ledger, symbols=symbols, cash=cash,
        slippage_bps=cost["slippage"], commission_bps=cost["commission"],
        min_commission=cost["min"], stamp_tax_sell_bps=cost["stamp"],
        transfer_fee_bps=cost["transfer"], rebalance_days=rebalance_days,
        pit_strict=True, seed=7,
    )
    out = runner.run(start=start, end=end)
    ledger.close()
    ledger_path.unlink(missing_ok=True)
    m = out.get("metrics", {})
    return {
        "ann_return": m.get("annualized_return", 0.0),
        "sharpe": m.get("sharpe", 0.0),
        "max_dd": m.get("max_drawdown", 0.0),
        "n_fills": m.get("n_fills", 0),
        "cost": m.get("total_commission", 0.0),
    }


def _ml_books(market, scores: pd.Series, start: str, end: str) -> dict:
    comp = scores[
        (scores.index.get_level_values(0) >= pd.Timestamp(start))
        & (scores.index.get_level_values(0) <= pd.Timestamp(end))
    ]
    trend = _market_trend(market.price_panel, 60)
    books: dict[pd.Timestamp, dict[str, float]] = {}
    for d, day in comp.groupby(level=0):
        ss = 1.0
        if pd.Timestamp(d) in trend.index:
            t = trend[pd.Timestamp(d)]
            if np.isfinite(t) and t > 0.03:
                ss = 0.5
        books[pd.Timestamp(d)] = long_book_weights(
            day, long_pct=0.10, short_pct=0.10, max_position_pct=0.05, short_scale=ss
        )
    return books


def _score_artifact(model_path: Path, meta_path: Path, market, features=None, cfg=None):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if features is None:
        features = build_feature_matrix(market.long, meta["feature_formulas"], n_jobs=None)
        extras = meta.get("metadata", {}).get("extra_features") or []
        if extras:
            from scripts.ml_common import load_margin_extras

            margin = load_margin_extras(cfg)
            assert set(extras) <= set(margin), f"missing extras: {set(extras) - set(margin)}"
            features = features.join(
                pd.concat([margin[k].rename(k) for k in extras], axis=1), how="left"
            )
    assert list(features.columns) == meta["features"], "artifact feature columns out of sync"
    if meta.get("kind") == "ml_mlp_torch":
        adapter = load_torch_artifact(model_path, meta_path)
        scores = pd.Series(adapter.predict(features), index=features.index)
    elif meta.get("kind") == "ml_xgb":
        from src.ml.xgb_model import XGBAdapter, load_xgb_artifact

        adapter = XGBAdapter(load_xgb_artifact(model_path))
        scores = pd.Series(adapter.predict(features), index=features.index)
    else:
        booster = load_artifact(model_path)
        scores = score_artifact(booster, features)
    return scores, meta


def _artifacts(out_dir: Path, per_kind: int = 2) -> list[tuple[Path, Path]]:
    metas = sorted(out_dir.glob("ml_*.json")) + sorted(out_dir.glob("mlp_*.json")) + sorted(out_dir.glob("xgb_*.meta.json"))
    newest: dict[str, list[Path]] = {}
    for p in metas:
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if meta.get("feature_formulas"):
            newest.setdefault(meta.get("kind", "ml_lgbm"), []).append(p)
    out: list[tuple[Path, Path]] = []
    for kind_paths in newest.values():
        for meta_path in kind_paths[-per_kind:]:
            kind = json.loads(meta_path.read_text(encoding="utf-8")).get("kind", "ml_lgbm")
            if kind == "ml_mlp_torch":
                model_path = meta_path.with_suffix(".pt")
            elif kind == "ml_xgb":
                model_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".json"))
            else:
                model_path = meta_path.with_suffix(".txt")
            out.append((meta_path, model_path))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capitals", default="2000000,100000")
    args = ap.parse_args()
    capitals = [float(c) for c in args.capitals.split(",")]

    cfg = load_config()
    market = _market_data(cfg, seed=1)
    symbols = list(market.price_panel.columns)
    print(f"market: {len(symbols)} symbols", flush=True)

    cost = {"slippage": 2.0, "commission": 2.5, "min": 5.0, "stamp": 5.0, "transfer": 0.1}

    # ---- score every artifact once ----
    named_scores: dict[str, pd.Series] = {}
    feature_cache: dict[tuple, pd.DataFrame] = {}
    for meta_path, model_path in _artifacts(ROOT / "outputs" / "models"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        key = (tuple(meta["feature_formulas"]), tuple(meta.get("metadata", {}).get("extra_features") or []))
        if key not in feature_cache:
            feature_cache[key] = build_feature_matrix(market.long, meta["feature_formulas"], n_jobs=None)
            extras = meta.get("metadata", {}).get("extra_features") or []
            if extras:
                from scripts.ml_common import load_margin_extras

                margin = load_margin_extras(cfg)
                feature_cache[key] = feature_cache[key].join(
                    pd.concat([margin[k].rename(k) for k in extras], axis=1), how="left"
                )
        scores, meta = _score_artifact(model_path, meta_path, market, feature_cache[key])
        named_scores[meta_path.name] = scores
        print(f"artifact scored: {meta_path.name} kind={meta.get('kind')}", flush=True)

    # ensemble over all available model scores
    ensemble = rank_ensemble(list(named_scores.values()))
    named_scores["RANK_ENSEMBLE"] = ensemble
    print(f"ensemble over {len(named_scores) - 1} models", flush=True)

    # ---- incumbent ----
    pool = json.loads((ROOT / "outputs" / "factors.json").read_text(encoding="utf-8"))
    formulas = [e["factor"]["formula"] for e in pool]
    fctx = FactorContext(market.long)
    alpha = AlphaCore(
        fctx, formulas, long_pct=0.10, short_pct=0.10, max_position_pct=0.05,
        neutralize=True, momentum_lookbacks=(20, 60, 120, 252),
        beta_neutralize=True, beta_lookback=252, regime_short=True,
        trend_days=60, trend_gate=0.03, short_scale=0.5,
    )
    inc_port = _AlphaCorePortfolio(alpha)

    # ---- run every book on every segment at every capital ----
    report: dict[str, dict] = {}
    for seg, (s, e) in SEGMENTS.items():
        for cash in capitals:
            label = f"{seg} @ {cash:,.0f}"
            report[label] = {}
            for name, scores in named_scores.items():
                books = _ml_books(market, scores, s, e)
                report[label][name] = _run_book(
                    market, _BooksPortfolio(books), symbols, cost, 10, cash, s, e
                )
            report[label]["incumbent"] = _run_book(
                market, inc_port, symbols, cost, 10, cash, s, e
            )
            rows = sorted(report[label].items(), key=lambda kv: -kv[1]["sharpe"])
            print(f"\n== {label} ==", flush=True)
            for name, m in rows:
                print(f"{name:26s} ann={m['ann_return']:+.2%} sharpe={m['sharpe']:+.2f} "
                      f"maxDD={m['max_dd']:.2%} fills={m['n_fills']} cost={m['cost']:,.0f}", flush=True)

    out = ROOT / "outputs" / "showdown_v3.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)

    # ---- promote verdict per capital on the test segment ----
    for cash in capitals:
        label = f"test_2022_2025 @ {cash:,.0f}"
        models = {k: v for k, v in report[label].items() if k != "incumbent"}
        best_name, best = max(models.items(), key=lambda kv: kv[1]["sharpe"])
        decision = evaluate_promotion(
            {"sharpe": best["sharpe"], "max_dd": best["max_dd"], "rank_ic": 0.05},
            {"sharpe": report[label]["incumbent"]["sharpe"],
             "max_dd": report[label]["incumbent"]["max_dd"]},
        )
        print(f"\npromote @ {cash:,.0f}: best={best_name} "
              f"{'PASS' if decision.promote else 'BLOCKED'} — {decision.reasons}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
