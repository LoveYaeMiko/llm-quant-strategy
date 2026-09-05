"""D-track model-cycle feasibility evidence (walk-forward, NO future data).

Stages (resumable — each stage skips work whose artifacts exist):

0. extend the intraday minute caches back to 2025-09-01 so the tail-volume gate
   can run for Q4-2025 evaluation windows;
1. build the shared feature matrix (formula zoo + margin extras) ONCE;
2. refit three challenger artifacts at historical cutoffs (rolling windows with
   a 10-day label embargo) into ``outputs/models_challenger/`` — NOT into
   ``outputs/models/``, so production artifact resolution is untouched;
3. replay the D track (identical params/execution/intraday semantics) over each
   challenger's OUT-OF-SAMPLE window and compare with the incumbent / the real
   D ledger — point-in-time, same board-lot/PIT discipline as production;
4. write ``outputs/d_model_cycle_evidence.json``.

Usage: python scripts/d_model_cycle_eval.py [--skip-stage0] [--skip-refits]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HORIZON = 10
EMBARGO_TD = 10  # trading days — labels never cross the deployment boundary
CHALLENGER_DIR = ROOT / "outputs" / "models_challenger"
MATRIX_CACHE = ROOT / "outputs" / "_dcycle_matrix.parquet"

# (name, cutoff, eval_start, eval_end, compare_mode)
# compare_mode: "incumbent_replay" (no real history) | "actual_ledger"
CYCLES = [
    ("CH_A", "2025-09-16", "2025-10-01", "2025-12-31", "incumbent_replay"),
    ("CH_B", "2025-12-17", "2026-01-01", "2026-09-04", "actual_ledger"),
    ("CH_C", "2026-06-16", "2026-07-01", "2026-09-04", "actual_ledger"),
]


def _cfg_and_symbols():
    from src.cli import load_config
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    acc = next(a for a in cfg.get("shadow.accounts") if a["name"] == "D_5W")
    symbols = resolve_shadow_universe(cfg, acc.get("universe"))
    return cfg, acc, symbols


def _d_params(acc):
    from src.paper.pullback_book import PullbackParams

    return PullbackParams(
        k=int(acc.get("pb_k", 6)), rank_source="ml", rank_min=float(acc.get("pb_rank_min", 0.8)),
        mom_window=int(acc.get("pb_mom_window", 63)),
        mom_long_rank_min=float(acc.get("pb_mom_long_rank_min", 0.0)),
        bounce_confirm=bool(acc.get("pb_bounce_confirm", False)),
        ema_fast=int(acc.get("pb_ema_fast", 21)), ema_zone=int(acc.get("pb_ema_zone", 21)),
        zone_band=float(acc.get("pb_zone_band", 0.02)), pullback_min=float(acc.get("pb_pullback_min", 0.03)),
        vol_shrink=bool(acc.get("pb_vol_shrink", True)), atr_mult=float(acc.get("pb_atr_mult", 1.5)),
        stop_lo=float(acc.get("pb_stop_lo", 0.025)), stop_hi=float(acc.get("pb_stop_hi", 0.04)),
        breakeven_r=float(acc.get("pb_breakeven_r", 1.0)), trail_r=float(acc.get("pb_trail_r", 1.5)),
        exit_into_strength_r=float(acc.get("pb_exit_into_strength_r", 0.0)),
        max_hold=int(acc.get("pb_max_hold", 40)), entry_gate=float(acc.get("pb_entry_gate", 0.0)),
        exit_gate=float(acc.get("pb_exit_gate", -0.03)), trend_days=int(acc.get("pb_trend_days", 60)),
        vwap_filter=float(acc.get("pb_vwap_filter", 0.0)), stop_rv=bool(acc.get("pb_stop_rv", False)),
        tail_vol_max=float(acc.get("pb_tail_vol_max", 0.0)), open30_max=float(acc.get("pb_open30_max", 0.0)),
        range_max=float(acc.get("pb_range_max", 0.0)), full_invest=bool(acc.get("pb_full_invest", False)),
        stop_trigger=str(acc.get("pb_stop_trigger", "close")),
        stop_buffer=float(acc.get("pb_stop_buffer", 0.0)),
        stop_open_minutes=int(acc.get("pb_stop_open_minutes", 0)),
    )


def stage0_extend_intraday(cfg, symbols) -> None:
    from src.data.intraday import refresh_intraday

    rollup = ROOT / "data" / "intraday" / "daily_features.parquet"
    if rollup.is_file():
        store = pd.read_parquet(rollup)
        idx = store.index
        if len(idx) and idx.min() <= pd.Timestamp("2025-09-10"):
            print("stage0: intraday already covers 2025-09 — skip", flush=True)
            return
    print("stage0: extending minute caches back to 2025-09-01 ...", flush=True)
    for start, end in (("2025-09-01", "2025-10-25"), ("2025-10-20", "2025-12-15")):
        s = refresh_intraday(cfg, symbols, start, end)
        print(f"  [{start}..{end}] calls={s['calls']} updated={s['symbols_updated']} "
              f"rollup {s['first_day']}..{s['last_day']}", flush=True)


def _sanitize_inf_chunked(X: pd.DataFrame, step: int = 32) -> pd.DataFrame:
    """Replace ±inf with NaN column-block by column-block.

    A whole-frame ``replace`` copies the entire matrix (~6 GB at 2.3M rows ×
    337 cols) and OOMs on 32 GB machines alongside LightGBM's own copies;
    block-wise replacement caps the extra memory at one block.
    """
    import gc

    for i in range(0, X.shape[1], step):
        X.iloc[:, i : i + step] = X.iloc[:, i : i + step].replace([np.inf, -np.inf], np.nan)
    gc.collect()
    return X


def stage1_matrix(cfg, market, formulas, extras):
    if MATRIX_CACHE.is_file():
        try:
            df = pd.read_parquet(MATRIX_CACHE)
            print("stage1: matrix cache hit — skip", flush=True)
            return df
        except Exception:  # noqa: BLE001 — a truncated write must rebuild, not crash
            print("stage1: corrupt matrix cache — rebuilding", flush=True)
            MATRIX_CACHE.unlink(missing_ok=True)
    import os

    from src.ml.train import align_features_labels, build_feature_matrix, forward_return_labels, standardize_per_date

    t0 = time.time()
    # 6 workers max: with Windows spawn each worker pickles its own copy of the
    # full market panel — 12 workers duplicated GBs of RAM (OOM observed).
    # dtype="float32" keeps the 2.3M×337 matrix at ~3 GB with no float64
    # intermediate (the original pipeline OOM'd the machine and ballooned the
    # pagefile to ~39 GB on 2026-09-04).
    X = build_feature_matrix(market.long, formulas, n_jobs=min(6, (os.cpu_count() or 4) - 2), dtype="float32")
    if extras:
        X = X.join(pd.concat([s.rename(k) for k, s in extras.items()], axis=1), how="left")
    # float32 end-to-end: the 2.3M×337 matrix is ~3 GB instead of ~6 GB, and
    # LightGBM converts internally anyway — rank-based books are insensitive
    # to the precision (OOM-driven decision, documented).
    X = _sanitize_inf_chunked(X.astype(np.float32))
    labels = standardize_per_date(forward_return_labels(market.price_panel, (HORIZON,)))
    tradable = getattr(market, "forward_returns_tradable", None)
    Xm, y = align_features_labels(X, labels, f"fwd_{HORIZON}", tradable)
    Xm = Xm.copy()
    Xm["__y__"] = y.astype(np.float32)
    # atomic write: a reboot/power-loss mid-write corrupts the cache (observed
    # 2026-09-05 — the machine rebooted during the write), so write to a temp
    # name and rename; the resume logic then never sees a half-written file.
    import os as _os
    import shutil as _shutil

    free_gb = _shutil.disk_usage(str(ROOT)).free / 1e9
    if free_gb < 4.0:
        raise RuntimeError(f"low disk space ({free_gb:.1f} GB free) — refusing to write the matrix cache")
    tmp_path = MATRIX_CACHE.with_name(MATRIX_CACHE.name + ".tmp")
    Xm.to_parquet(tmp_path)
    _os.replace(tmp_path, MATRIX_CACHE)
    print(f"stage1: matrix {Xm.shape} built in {time.time() - t0:.0f}s — cached", flush=True)
    return Xm


def stage2_refit(cfg, market, formulas, extras, name, cutoff):
    """Refit one challenger: train 2010→(cutoff-2mo), val (cutoff-2mo)→cutoff."""
    out_paths_file = CHALLENGER_DIR / f"{name}_paths.json"
    if out_paths_file.is_file():
        print(f"stage2: {name} already refit — skip", flush=True)
        return json.loads(out_paths_file.read_text(encoding="utf-8"))

    import lightgbm as lgb  # noqa: F401

    from src.ml.train import (
        MLArtifact, _purged_best_iterations, _slice, _window_metrics,
    )
    from src.ml.labels import forward_return_labels, standardize_per_date

    X = stage1_matrix(cfg, market, formulas, extras)
    y = X.pop("__y__")
    cutoff_ts = pd.Timestamp(cutoff)
    val_start = cutoff_ts - pd.DateOffset(months=2)
    train_start = pd.Timestamp("2010-01-01")

    X_tr = _slice(X, str(train_start.date()), str((val_start - pd.Timedelta(days=1)).date()))
    X_tv = _slice(X, str(train_start.date()), str(cutoff_ts.date()))
    y_tr = y[X_tr.index]
    y_tv = y[X_tv.index]

    params = {
        "objective": "regression", "learning_rate": 0.03, "num_leaves": 31,
        "min_child_samples": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "reg_lambda": 1.0,
        "num_threads": max(4, (__import__("os").cpu_count() or 4) - 2),
        "verbose": -1, "seed": 7,
    }
    t0 = time.time()
    best_iter = _purged_best_iterations(
        X_tr, y_tr, params, HORIZON, n_folds=5, embargo_frac=0.01,
        n_estimators=600, early_stopping=40,
    )
    fit_params = dict(params)
    fit_params.pop("seed", None)
    final = lgb.LGBMRegressor(n_estimators=int(best_iter), **fit_params)
    final.fit(X_tv, y_tv)

    labels_raw = forward_return_labels(market.price_panel, (HORIZON,))
    y_raw = labels_raw[f"fwd_{HORIZON}"].rename("fwd_raw").reindex(X.index)
    X_te = _slice(X, str((cutoff_ts + pd.Timedelta(days=1)).date()), "2026-09-04")
    test_metrics = _window_metrics(final, X_te, y_raw[X_te.index], HORIZON, cost_bps=5.0) if len(X_te) else {}

    artifact = MLArtifact(
        model_text=final.booster_.model_to_string(),
        features=list(X.columns),
        feature_formulas=list(formulas),
        horizon=HORIZON,
        fit_window=(str(train_start.date()), str(cutoff_ts.date())),
        params=fit_params,
        best_iteration=int(best_iter),
        metadata={
            "train_window": [str(train_start.date()), str((val_start - pd.Timedelta(days=1)).date())],
            "val_window": [str(val_start.date()), str(cutoff_ts.date())],
            "test_window": [str((cutoff_ts + pd.Timedelta(days=1)).date()), "2026-09-04"],
            "n_train": int(len(X_tr)), "n_val": int(len(X_tv) - len(X_tr)),
            "n_test": int(len(X_te)),
            "extra_features": list(extras.keys()) if extras else [],
            "cycle": name,
        },
    )
    CHALLENGER_DIR.mkdir(parents=True, exist_ok=True)
    paths = artifact.save(CHALLENGER_DIR)
    out_paths_file.write_text(json.dumps(paths), encoding="utf-8")
    print(f"stage2: {name} refit in {time.time() - t0:.0f}s best_iter={best_iter} "
          f"test_ic={test_metrics.get('rank_ic')} -> {paths}", flush=True)
    return paths


def _replay(cfg, acc, symbols, artifact_paths, eval_start, eval_end, label, ledger_path, live_gate):
    """D-track replay over an OOS window with a given artifact (production semantics)."""
    from src.cli import _build_market_for_paper
    from src.data.intraday import load_intraday_frames, make_minute_provider
    from src.ml.train import load_artifact, score_artifact
    from src.paper.ledger import PaperLedger
    from src.paper.ml_book import _feature_frame
    from src.paper.pullback_book import PullbackPortfolio
    from src.paper.runner import PaperRunner

    meta_path = Path(artifact_paths["meta"])
    model_path = Path(artifact_paths["model"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    t0 = time.time()
    market = _build_market_for_paper(cfg, symbols, eval_start, eval_end, seed=1)
    frame = _feature_frame(market, meta, cfg, n_jobs=min(6, (__import__("os").cpu_count() or 4) - 2))
    assert list(frame.columns) == meta["features"], "artifact columns out of sync"
    scores = score_artifact(load_artifact(model_path), frame)

    intraday = load_intraday_frames(cfg, symbols)
    book = PullbackPortfolio(
        market, _d_params(acc), symbols=symbols, scores=scores,
        intraday=intraday, minute_provider=make_minute_provider(cfg),
    )
    if live_gate:
        book.live_intraday_from = str(live_gate)

    ledger = PaperLedger(str(ledger_path))
    runner = PaperRunner(
        book, market, ledger, symbols=symbols, cash=float(acc.get("cash", 50_000)),
        slippage_bps=2.0, commission_bps=2.5, min_commission=5.0,
        stamp_tax_sell_bps=5.0, transfer_fee_bps=0.1,
        rebalance_days=int(acc.get("rebalance_days", 1)),
        notional_floor=float(acc.get("notional_floor", 2000.0)),
        band_frac=float(acc.get("band_frac", 0.0)),
        max_position_pct=float(acc.get("max_position_pct", 0.40)),
        pit_strict=True, seed=1,
    )
    result = runner.run(start=eval_start, end=eval_end)
    m = result["metrics"]
    out = {
        "label": label,
        "window": [eval_start, eval_end],
        "total_return": float(m.get("total_return", 0.0)),
        "annualized_return": float(m.get("annualized_return", 0.0)),
        "sharpe": float(m.get("sharpe", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "n_fills": int(m.get("n_fills", 0)),
        "total_commission": float(m.get("total_commission", 0.0)),
        "final_equity": float(m.get("final_equity", 0.0)),
    }
    from scripts.audit_tracks import legality_audit

    aud = legality_audit(label, str(ledger_path), market)
    out["violations"] = {
        "t_plus_1": int(aud.get("same_day_flips", 0)),
        "odd_lot": int(aud.get("odd_lot_fills", 0)),
        "star_lot": int(aud.get("star_lot_violations", 0)),
        "off_tick": int(aud.get("off_tick_fills", 0)),
        "limit_locked": int(aud.get("limit_locked_fills", 0)),
    }
    ledger.close()
    print(f"  replay[{label}] {out['total_return']:+.2%} sharpe={out['sharpe']:.2f} "
          f"fills={out['n_fills']} viol={sum(out['violations'].values())} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


def _actual_ledger_window(eval_start, eval_end):
    from src.paper.ledger import PaperLedger

    led = PaperLedger(str(ROOT / "outputs" / "shadow_ledger_D_5W.sqlite"))
    eq = led.equity_curve()
    led.close()
    w = eq[(eq.index >= eval_start) & (eq.index <= eval_end)]
    if len(w) < 2:
        return None
    ret = float(w.iloc[-1] / w.iloc[0] - 1.0)
    daily = w.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 2 and daily.std() > 0 else 0.0
    dd = float((w / w.cummax() - 1.0).min())
    return {"label": "actual_ledger", "window": [eval_start, eval_end],
            "total_return": ret, "sharpe": sharpe, "max_drawdown": dd,
            "note": "D_5W 真实账本同窗"}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-stage0", action="store_true")
    ap.add_argument("--skip-refits", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="run only this cycle (CH_A/CH_B/CH_C)")
    args = ap.parse_args()

    cfg, acc, symbols = _cfg_and_symbols()
    from scripts.ml_common import load_margin_extras, load_zoo_formulas
    from scripts.scan_factor_zoo import BASELINE

    formulas = list(dict.fromkeys(BASELINE + load_zoo_formulas()))
    extras = load_margin_extras(cfg)
    print(f"universe={len(symbols)} formulas={len(formulas)} extras={len(extras)}", flush=True)

    if not args.skip_stage0:
        stage0_extend_intraday(cfg, symbols)

    from src.cli import _market_data

    market = _market_data(cfg, seed=1)

    evidence: dict[str, dict] = {}
    for name, cutoff, eval_start, eval_end, mode in CYCLES:
        if args.only and name != args.only:
            continue
        print(f"\n===== {name} (cutoff {cutoff}) =====", flush=True)
        # intraday guard: the tail-volume gate blocks ALL entries when the rollup
        # lacks the eval window (minute data older than ~2025-12 is API-limited);
        # skip such cycles instead of replaying a zero-trade book.
        rollup_path = ROOT / "data" / "intraday" / "daily_features.parquet"
        if rollup_path.is_file():
            store = pd.read_parquet(rollup_path)
            if len(store.index) and store.index.min() > pd.Timestamp(eval_start):
                evidence[name] = {
                    "skipped": True,
                    "note": f"intraday rollup starts {store.index.min().date()} > eval_start {eval_start} — tail gate would block all entries",
                }
                print(f"  {name}: SKIPPED (intraday rollup starts {store.index.min().date()})", flush=True)
                continue
        paths = None
        if not args.skip_refits:
            paths = stage2_refit(cfg, market, formulas, extras, name, cutoff)
        if paths is None:
            pf = CHALLENGER_DIR / f"{name}_paths.json"
            if not pf.is_file():
                print(f"  {name}: no artifact paths and --skip-refits — skip", flush=True)
                continue
            paths = json.loads(pf.read_text(encoding="utf-8"))

        ledger_path = ROOT / "outputs" / f"_dcycle_{name}.sqlite"
        if ledger_path.is_file():
            ledger_path.unlink()
        live_gate = "2026-09-04" if eval_end >= "2026-09-04" else None
        challenger = _replay(cfg, acc, symbols, paths, eval_start, eval_end, name,
                             ledger_path, live_gate)
        challenger["cutoff"] = cutoff

        if mode == "incumbent_replay":
            inc_paths = {
                "meta": str(ROOT / "outputs" / "models" / "ml_20260901_004119.json"),
                "model": str(ROOT / "outputs" / "models" / "ml_20260901_004119.txt"),
            }
            inc_ledger = ROOT / "outputs" / f"_dcycle_{name}_inc.sqlite"
            if inc_ledger.is_file():
                inc_ledger.unlink()
            incumbent = _replay(cfg, acc, symbols, inc_paths, eval_start, eval_end,
                                f"{name}_incumbent", inc_ledger, live_gate)
        else:
            incumbent = _actual_ledger_window(eval_start, eval_end)
            if incumbent is None:
                print(f"  {name}: actual ledger window empty — fallback to incumbent replay", flush=True)
                inc_paths = {
                    "meta": str(ROOT / "outputs" / "models" / "ml_20260901_004119.json"),
                    "model": str(ROOT / "outputs" / "models" / "ml_20260901_004119.txt"),
                }
                inc_ledger = ROOT / "outputs" / f"_dcycle_{name}_inc.sqlite"
                if inc_ledger.is_file():
                    inc_ledger.unlink()
                incumbent = _replay(cfg, acc, symbols, inc_paths, eval_start, eval_end,
                                    f"{name}_incumbent", inc_ledger, live_gate)

        margin = 0.003
        challenger_ret = challenger["total_return"]
        incumbent_ret = incumbent["total_return"]
        challenger_wins = challenger_ret > incumbent_ret + margin
        clean = sum(challenger["violations"].values()) == 0
        evidence[name] = {
            "challenger": challenger,
            "baseline": incumbent,
            "delta_pp": round((challenger_ret - incumbent_ret) * 100.0, 2),
            "margin_pp": margin * 100.0,
            "challenger_wins": bool(challenger_wins),
            "legality_clean": bool(clean),
            "verdict": ("正面（建议晋升）" if challenger_wins and clean
                        else "负面/不足（建议留任现役）"),
        }

    out_path = ROOT / "outputs" / "d_model_cycle_evidence.json"
    out_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n===== EVIDENCE SUMMARY =====", flush=True)
    for name, e in evidence.items():
        c, b = e["challenger"], e["baseline"]
        print(f"{name}: challenger {c['total_return']:+.2%} (fills {c['n_fills']}) vs "
              f"baseline {b['total_return']:+.2%} -> delta {e['delta_pp']:+.2f}pp | "
              f"{e['verdict']}", flush=True)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
