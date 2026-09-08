"""D-track model self-optimization loop — monthly rolling refit + parallel
shadow challenger + forward promotion gate (replaces Saturday calibration and
Sunday weekly).

Industry design ("train often, deploy selectively", arXiv 2607.28577;
champion-challenger, MDPI Applied Sciences 16(17):8406):

* ``refit``   — monthly (first Sunday): rolling refit with a 10-day label
  embargo (cutoff = last month end − 10 trading days; train 2010→cutoff−2mo,
  val cutoff−2mo→cutoff, final artifact = train+val). Written to
  ``outputs/models_challenger/`` — NEVER to ``outputs/models/``, so production
  artifact resolution is untouched until a promotion.
* ``challenger`` — daily (17:45): advance the challenger ledger with the SAME
  D-track book/executor/intraday semantics, only the ML scanner artifact
  differs. Point-in-time like the main D track.
* ``decide``   — monthly (first Sunday, after refit): compare the challenger
  against the REAL D ledger over the trailing OOS window; promote only if the
  challenger wins by the configured margin with a clean legality audit and a
  minimum fill count. Promotion = move the artifact into ``outputs/models/``.
* ``audit-cost`` — replaces the old §7 calibrate: the real A-share cost
  structure is regulatory-fixed; verify the configured cost model matches it
  and alert on any drift (no parameter tuning).
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

CHALLENGER_DIR = ROOT / "outputs" / "models_challenger"
STATE_PATH = ROOT / "outputs" / "d_model_cycle.json"

HORIZON = 10
EMBARGO_TD = 10  # trading days — labels never cross the deployment boundary


def _cycle_cfg(cfg) -> dict[str, Any]:
    return dict((cfg.get("d_model_cycle") or {}) or {})


def _challenger_ledger_path(cfg) -> Path:
    rel = str(_cycle_cfg(cfg).get("challenger_ledger", "outputs/shadow_ledger_D_5W_CH.sqlite"))
    return Path(rel) if Path(rel).is_absolute() else ROOT / rel


def _incumbent_paths() -> dict[str, str]:
    from .paper.ml_book import _resolve_artifact

    meta_path, model_path = _resolve_artifact("lgbm", "")
    return {"meta": str(meta_path), "model": str(model_path)}


def _last_month_cutoff(market, today: pd.Timestamp | None = None) -> str:
    """End of the previous month − 10 trading days (label embargo).

    ``today`` is injectable for tests.
    """
    bars = pd.Series(market.price_panel.index).sort_values()
    today = (today or pd.Timestamp.today()).normalize()
    month_end = (today.replace(day=1) - pd.Timedelta(days=1)).normalize()
    prior = bars[bars <= month_end]
    if len(prior) < EMBARGO_TD + 1:
        return str(prior.iloc[-1].date())
    return str(prior.iloc[-1 - EMBARGO_TD].date())


def _deploy_from(market, month_end: pd.Timestamp) -> str:
    """First trading day after ``month_end`` (labels of bars ≤ cutoff expire)."""
    bars = pd.Series(market.price_panel.index).sort_values()
    after = bars[bars > month_end]
    if len(after) == 0:
        return str((month_end + pd.Timedelta(days=1)).date())
    return str(after.iloc[0].date())


def _sanitize_inf_chunked(X: pd.DataFrame, step: int = 32) -> pd.DataFrame:
    """Replace ±inf with NaN column-block by column-block (OOM-safe).

    A whole-frame ``replace`` copies the entire matrix (~6 GB) and OOMs on
    32 GB machines alongside LightGBM's own copies.
    """
    import gc

    for i in range(0, X.shape[1], step):
        X.iloc[:, i : i + step] = X.iloc[:, i : i + step].replace([float("inf"), float("-inf")], float("nan"))
    gc.collect()
    return X


def refit_challenger(cfg, name: str = "CH") -> dict[str, Any]:
    """Monthly rolling refit → challenger artifact (isolated directory)."""
    import lightgbm as lgb

    from scripts.ml_common import load_margin_extras, load_zoo_formulas
    from scripts.scan_factor_zoo import BASELINE
    from .cli import _market_data
    from .ml.train import (
        MLArtifact, _purged_best_iterations, _slice, align_features_labels,
        build_feature_matrix, forward_return_labels, standardize_per_date,
    )

    market = _market_data(cfg, seed=1)
    formulas = list(dict.fromkeys(BASELINE + load_zoo_formulas()))
    extras = load_margin_extras(cfg)
    import os

    # 6 workers max: with Windows spawn each worker pickles its own copy of the
    # full market panel — more workers duplicated GBs of RAM (OOM observed).
    # dtype="float32" keeps the 2.3M×337 matrix at ~3 GB with no float64
    # intermediate.
    n_jobs = min(6, (os.cpu_count() or 4) - 2)
    X = build_feature_matrix(market.long, formulas, n_jobs=n_jobs, dtype="float32")
    if extras:
        X = X.join(pd.concat([s.rename(k) for k, s in extras.items()], axis=1), how="left")
    # float32 end-to-end: the 2.3M×337 matrix is ~3 GB instead of ~6 GB, and
    # LightGBM converts internally anyway — rank-based books are insensitive
    # to the precision (OOM-driven decision, documented).
    X = _sanitize_inf_chunked(X.astype("float32"))
    labels = standardize_per_date(forward_return_labels(market.price_panel, (HORIZON,)))
    tradable = getattr(market, "forward_returns_tradable", None)
    Xm, y = align_features_labels(X, labels, f"fwd_{HORIZON}", tradable)
    y = y.astype("float32")

    today = pd.Timestamp.today().normalize()
    month_end = (today.replace(day=1) - pd.Timedelta(days=1)).normalize()
    cutoff = pd.Timestamp(_last_month_cutoff(market, today=today))
    deploy_from = _deploy_from(market, month_end)
    val_start = cutoff - pd.DateOffset(months=2)
    train_start = pd.Timestamp("2010-01-01")

    X_tr = _slice(Xm, str(train_start.date()), str((val_start - pd.Timedelta(days=1)).date()))
    X_tv = _slice(Xm, str(train_start.date()), str(cutoff.date()))
    y_tr = y[X_tr.index]
    y_tv = y[X_tv.index]

    params = {
        "objective": "regression", "learning_rate": 0.03, "num_leaves": 31,
        "min_child_samples": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "reg_lambda": 1.0, "num_threads": n_jobs,
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

    artifact = MLArtifact(
        model_text=final.booster_.model_to_string(),
        features=list(Xm.columns),
        feature_formulas=list(formulas),
        horizon=HORIZON,
        fit_window=(str(train_start.date()), str(cutoff.date())),
        params=fit_params,
        best_iteration=int(best_iter),
        metadata={
            "train_window": [str(train_start.date()), str((val_start - pd.Timedelta(days=1)).date())],
            "val_window": [str(val_start.date()), str(cutoff.date())],
            "test_window": [deploy_from, None],
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_tv) - len(X_tr)),
            "n_test": 0,
            "extra_features": list(extras.keys()) if extras else [],
            "cycle": name,
            # deployment starts on the first trading day after the previous
            # month end: the 10-day label embargo expires by then, so the
            # challenger's whole forward window is out-of-sample.
            "deploy_from": deploy_from,
        },
    )
    CHALLENGER_DIR.mkdir(parents=True, exist_ok=True)
    # one active challenger at a time: retire previous artifacts AND its ledger
    # (the new challenger re-replays from its own deploy_from)
    for old in CHALLENGER_DIR.glob("ml_*.json"):
        old.with_suffix(".txt").unlink(missing_ok=True)
        old.unlink(missing_ok=True)
    _challenger_ledger_path(cfg).unlink(missing_ok=True)
    paths = artifact.save(CHALLENGER_DIR)
    (CHALLENGER_DIR / "active.json").write_text(json.dumps(paths), encoding="utf-8")
    return {
        "ok": True,
        "cutoff": str(cutoff.date()),
        "deploy_from": deploy_from,
        "fit_window": [str(train_start.date()), str(cutoff.date())],
        "best_iteration": int(best_iter),
        "paths": paths,
        "train_sec": round(time.time() - t0, 1),
    }


def run_challenger(cfg) -> dict[str, Any]:
    """Advance the challenger ledger through the latest bar (daily 17:45).

    Replays from the artifact's ``deploy_from`` (first trading day after the
    previous month end) — every bar it trades is out-of-sample for its model.
    """
    from .cli import _build_market_for_paper
    from .data.intraday import ensure_intraday_current, load_intraday_frames, make_minute_provider
    from .ml.train import load_artifact, score_artifact
    from .paper.ledger import PaperLedger
    from .paper.ml_book import _feature_frame
    from .paper.pullback_book import PullbackPortfolio
    from .paper.runner import PaperRunner
    from .paper.shadow import resolve_shadow_universe

    active_file = CHALLENGER_DIR / "active.json"
    if not active_file.is_file():
        return {"ok": False, "error": "no active challenger artifact — run refit first"}
    paths = json.loads(active_file.read_text(encoding="utf-8"))
    meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))

    acc = next(a for a in cfg.get("shadow.accounts") if a["name"] == "D_5W")
    symbols = resolve_shadow_universe(cfg, acc.get("universe"))
    start = str(meta.get("metadata", {}).get("deploy_from") or cfg.section("shadow").get("start_date", "2026-01-01"))

    market = _build_market_for_paper(cfg, symbols, start, None, seed=1)
    latest = pd.Timestamp(market.price_panel.index.max()).date().isoformat()
    if pd.Timestamp(latest) < pd.Timestamp(start):
        return {"ok": True, "skipped": f"no bars after deploy_from {start} yet"}
    ensure_intraday_current(cfg, symbols, latest)

    import os

    frame = _feature_frame(market, meta, cfg, n_jobs=max(2, (os.cpu_count() or 4) - 2))
    assert list(frame.columns) == meta["features"], "artifact columns out of sync"
    scores = score_artifact(load_artifact(paths["model"]), frame)

    ledger = PaperLedger(str(_challenger_ledger_path(cfg)))
    intraday = load_intraday_frames(cfg, symbols)
    params = _pullback_params(acc)
    book = PullbackPortfolio(
        market, params, symbols=symbols, scores=scores,
        intraday=intraday, minute_provider=make_minute_provider(cfg),
        ledger=ledger,
    )
    book.live_intraday_from = str(acc.get("pb_live_intraday_from", "") or "") or None

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
    result = runner.run(start=start, end=latest)
    m = result["metrics"]
    ledger.close()
    return {
        "ok": True,
        "as_of": latest,
        "total_return": float(m.get("total_return", 0.0)),
        "sharpe": float(m.get("sharpe", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "n_fills": int(m.get("n_fills", 0)),
        "final_equity": float(m.get("final_equity", 0.0)),
    }


def _pullback_params(acc):
    from .paper.pullback_book import PullbackParams

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
        # Diagnostic A/B switch (default True = correct): see
        # PullbackParams.intraday_basis_adjust. Production never sets it False.
        intraday_basis_adjust=bool(acc.get("pb_intraday_basis_adjust", True)),
    )


def decide_promotion(cfg, market=None, today: pd.Timestamp | None = None) -> dict[str, Any]:
    """Forward promotion gate: challenger vs REAL D ledger over the trailing window.

    ``market``/``today`` are injectable for tests (production passes None).
    """
    from .paper.ledger import PaperLedger

    cyc = _cycle_cfg(cfg)
    margin = float(cyc.get("margin_pp", 0.3)) / 100.0
    min_fills = int(cyc.get("min_fills", 5))
    window_days = int(cyc.get("window_days", 30))

    challenger_path = _challenger_ledger_path(cfg)
    if not challenger_path.is_file():
        return {"ok": False, "error": "challenger ledger missing"}
    main_path = ROOT / "outputs" / "shadow_ledger_D_5W.sqlite"
    if not main_path.is_file():
        return {"ok": False, "error": "main D ledger missing"}

    ch = PaperLedger(str(challenger_path))
    ch_eq = ch.equity_curve()
    ch.close()
    main = PaperLedger(str(main_path))
    main_eq = main.equity_curve()
    main.close()
    ch_eq.index = pd.to_datetime(ch_eq.index)
    main_eq.index = pd.to_datetime(main_eq.index)

    today = (today or pd.Timestamp.today()).normalize()
    start = today - pd.Timedelta(days=window_days)
    ch_w = ch_eq[(ch_eq.index >= start) & (ch_eq.index <= today)]
    m_w = main_eq[(main_eq.index >= start) & (main_eq.index <= today)]
    if len(ch_w) < 2 or len(m_w) < 2:
        return {"ok": False, "error": "evaluation window too short"}

    ch_ret = float(ch_w.iloc[-1] / ch_w.iloc[0] - 1.0)
    main_ret = float(m_w.iloc[-1] / m_w.iloc[0] - 1.0)

    from .paper.ledger import PaperLedger as _PL

    ch2 = _PL(str(challenger_path))
    n_fills = ch2.n_fills()
    ch2.close()

    from scripts.audit_tracks import legality_audit  # noqa: PLC0415

    if market is None:
        from .cli import _market_data

        market = _market_data(cfg, seed=1)
    aud = legality_audit("challenger", str(challenger_path), market)
    violations = {
        "t_plus_1": int(aud.get("same_day_flips", 0)),
        "odd_lot": int(aud.get("odd_lot_fills", 0)),
        "star_lot": int(aud.get("star_lot_violations", 0)),
        "off_tick": int(aud.get("off_tick_fills", 0)),
        "limit_locked": int(aud.get("limit_locked_fills", 0)),
    }
    clean = sum(violations.values()) == 0
    enough = n_fills >= min_fills
    wins = ch_ret > main_ret + margin
    promote = wins and clean and enough

    decision = {
        "date": date.today().isoformat(),
        "window": [str(start.date()), str(today.date())],
        "challenger_return": round(ch_ret, 6),
        "incumbent_return": round(main_ret, 6),
        "delta_pp": round((ch_ret - main_ret) * 100.0, 2),
        "margin_pp": margin * 100.0,
        "challenger_fills": n_fills,
        "violations": violations,
        "promote": bool(promote),
        "reasons": [] if promote else [
            *( [] if wins else ["收益未超现役+边际"]),
            *( [] if clean else ["合法性审计存在违规"]),
            *( [] if enough else [f"成交笔数不足 {min_fills}"]),
        ],
    }

    if promote:
        active_file = CHALLENGER_DIR / "active.json"
        paths = json.loads(active_file.read_text(encoding="utf-8")) if active_file.is_file() else None
        if paths:
            from .paper.ml_book import _ARTIFACT_DIR

            dest_meta = _ARTIFACT_DIR / Path(paths["meta"]).name
            dest_model = _ARTIFACT_DIR / Path(paths["model"]).name
            shutil.copy2(paths["meta"], dest_meta)
            shutil.copy2(paths["model"], dest_model)
            decision["promoted_artifact"] = dest_meta.name
            # the incumbent is kept on disk (artifact history); the newest file wins.
    else:
        for old in CHALLENGER_DIR.glob("ml_*"):
            old.unlink(missing_ok=True)
        (CHALLENGER_DIR / "active.json").unlink(missing_ok=True)
        decision["challenger_dropped"] = True

    _update_state(decision)
    return {"ok": True, "decision": decision}


def audit_cost_consistency(cfg) -> dict[str, Any]:
    """Replaces §7 calibrate: the real cost structure is regulatory-fixed.

    Verify the configured cost model equals the real A-share structure and the
    accumulated ledger commission deviates by zero; alert on any drift. No
    parameters are tuned.
    """
    from .paper.ledger import PaperLedger
    from .paper.shadow import compute_cost_deviation, real_cost_model

    real = real_cost_model(cfg)
    pcfg = cfg.section("paper")
    configured = {
        "commission_bps": float(pcfg.get("commission_bps", 5.0)),
        "min_commission": float(pcfg.get("min_commission", 1.0)),
        "stamp_tax_sell_bps": float(pcfg.get("stamp_tax_sell_bps", 0.0)),
        "transfer_fee_bps": float(pcfg.get("transfer_fee_bps", 0.0)),
        "slippage_bps": float(pcfg.get("slippage_bps", 2.0)),
    }
    drift = {k: (configured.get(k), real.get(k)) for k in real if configured.get(k) != real.get(k)}
    fills = pd.DataFrame()
    for p in sorted((ROOT / "outputs").glob("shadow_ledger*.sqlite")):
        try:
            led = PaperLedger(str(p))
            f = led.fills()
            led.close()
        except Exception:  # noqa: BLE001
            continue
        if len(f):
            fills = pd.concat([fills, f], ignore_index=True)
    dev = compute_cost_deviation(fills, real) if len(fills) else {"deviation_pct": 0.0}
    return {
        "ok": not drift and abs(float(dev["deviation_pct"])) <= 1e-9,
        "drift": drift,
        "accumulated_deviation_pct": float(dev["deviation_pct"]),
        "note": "真实 A 股成本结构为监管固定值——仅校验不调参",
    }


def _update_state(decision: dict[str, Any]) -> None:
    history: list[dict[str, Any]] = []
    if STATE_PATH.is_file():
        try:
            old = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            history = old.get("history", []) if isinstance(old, dict) else []
        except Exception:  # noqa: BLE001
            history = []
    history.append(decision)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"history": history[-60:]}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


__all__ = [
    "audit_cost_consistency",
    "decide_promotion",
    "refit_challenger",
    "run_challenger",
]
