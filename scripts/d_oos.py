"""D-track out-of-sample (OOS) validation harness — 11 hard assertions.

Why this exists (2026-09-08 independent audit): the D track's evidence was
almost entirely IN-SAMPLE (the 2026 segment the parameters were tuned on), the
single OOS datapoint was negative, and several defects (grid/production
isomorphism, mixed-basis ATR, replay-vs-live intraday stops) meant even the
in-sample numbers were not measured on the deployed assembly. This script makes
the validation reproducible and self-checking:

* it runs the PRODUCTION assembly (``src.cli._shadow_cycle`` → the same
  ``_build_account_portfolio`` the 15:10 job uses) into a FRESH ledger, never the
  production ledger, and writes no status/report artifacts;
* it asserts, not assumes, that the run is what it claims to be (fresh ledger,
  no calendar gap, intraday feature pack + minute provider wired, param
  fingerprint identical to production, no future-dated or post-15:00 intraday
  fills, T+1 respected, limit-locked bars never filled, live-gated dates never
  replayed, costs consistent with the ledger, and the Sharpe reported with its
  standard error).

Usage::

    python scripts/d_oos.py 2025-09-01 2025-12-31 --label oos_2025q4
    python scripts/d_oos.py 2026-01-01 2026-08-28 --label is_2026
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

#: A calendar gap wider than this (in calendar days) means the window contains a
#: hole in the data — results over it cannot be read as a continuous curve.
#: 15 days tolerates the Spring-Festival break (a 10-11 day market closure).
MAX_CALENDAR_GAP_DAYS = 15

#: Minimum share of the universe that must carry a minute feature on EVERY day
#: of the window. The 2025-10-27→12-12 hole had ~274/800 (0.34) while the
#: day-level probe reported 100% — this threshold is what makes that visible.
MIN_SYMBOL_COVERAGE = 0.90

#: Minimum share of a SINGLE symbol's own tradable days (days with a daily price
#: bar inside the window) that must carry a minute feature. The day-level probe
#: above cannot see one name missing a month; this one can. The denominator is
#: deliberately the symbol's tradable days, not the window's trading days: a
#: suspension (信达证券 601059.SH, no bar 2025-11-20→12-17) and a pre-listing
#: period (601112.SH, first bar 2026-01-29) are NOT data holes — a missing
#: feature on a day the name actually traded is.
MIN_SYMBOL_WINDOW_COVERAGE = 0.90

#: Intraday decisions must stop at the close; the auction layer owns 15:00+.
INTRADAY_CUTOFF = "15:00"


def _calendar_gap_days(dates: list[pd.Timestamp]) -> int:
    if len(dates) < 2:
        return 0
    diffs = pd.Series(dates).diff().dropna()
    return int(diffs.max().days) if len(diffs) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="D-track OOS validation (11 assertions)")
    ap.add_argument("start", help="window start (YYYY-MM-DD)")
    ap.add_argument("end", help="window end (YYYY-MM-DD)")
    ap.add_argument("--label", default="oos", help="output label")
    ap.add_argument("--account", default="D_5W")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--refresh", action="store_true", help="refresh market data first")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one account parameter for a candidate run (repeatable), "
             "e.g. --set pb_atr_mult=1.0 --set pb_stop_hi=0.035",
    )
    args = ap.parse_args()

    from src.cli import _shadow_cycle
    from src.config import load_config
    from src.data.intraday import load_intraday_frames
    from src.paper.ledger import PaperLedger
    from src.paper.shadow import resolve_shadow_universe

    cfg = load_config()
    account = next(
        (a for a in (cfg.get("shadow.accounts") or []) if str(a.get("name")) == args.account), None
    )
    if account is None:
        print(f"ERROR: account {args.account!r} not in shadow.accounts", file=sys.stderr)
        return 2
    account = dict(account)
    overrides: dict[str, object] = {}
    for item in args.set:
        if "=" not in item:
            print(f"ERROR: --set expects KEY=VALUE, got {item!r}", file=sys.stderr)
            return 2
        key, raw = item.split("=", 1)
        try:
            value: object = float(raw) if "." in raw or "e" in raw.lower() else int(raw)
        except ValueError:
            value = raw
        account[key.strip()] = value
        overrides[key.strip()] = value

    symbols = resolve_shadow_universe(cfg, account.get("universe"))

    ledger_path = ROOT / "outputs" / f"_doos_{args.label}.sqlite"
    ledger_existed_before = ledger_path.exists()
    ledger_path.unlink(missing_ok=True)

    print(f"[oos] window=[{args.start}, {args.end}] account={args.account} "
          f"universe={len(symbols)} ledger={ledger_path.name}", flush=True)
    # Build the market with the PRODUCTION slice shape (end=None) so the feature
    # matrix cache hits; the runner still stops at ``end``. A fresh slice would
    # force a multi-GB rebuild and can OOM the worker pool.
    from src.cli import _build_market_for_paper

    market = _build_market_for_paper(cfg, symbols, args.start, None, seed=args.seed)
    symbols = [s for s in symbols if s in market.price_panel.columns]
    # Same kill-switch state as the production 15:10 run (a research run without
    # the gate is NOT isomorphic once the gate leaves `normal`).
    from src.autopilot.state import ControlState

    state_path = str(
        ROOT / str(cfg.get("autopilot.state_file", "outputs/autopilot_state.json"))
    ).replace(".json", f"_{args.account}.json")
    control = ControlState.load(state_path)
    print(f"[oos] kill-switch: mode={control.mode} gross={control.gross_scale:g}", flush=True)

    probe: dict = {}
    status, ledger_out = _shadow_cycle(
        cfg, symbols, args.start, args.end, args.seed, skip_refresh=not args.refresh,
        control_scale=control.gross_scale, account=account,
        ledger_override=str(ledger_path),
        write_artifacts=False, probe=probe, market_override=market,
    )

    ledger = PaperLedger(str(ledger_path))
    fills = ledger.fills()
    equity = ledger.curve() if hasattr(ledger, "curve") else ledger.equity_curve()
    ledger_cost = float(ledger.total_commission())
    ledger.close()

    # ---- production assembly, built from the CONFIG account (no overrides) ---
    # This is the real production fingerprint. Comparing the run against a probe
    # built from the SAME override set (the earlier behaviour) is a self-
    # comparison that can never detect "this is not the production config".
    prod_probe: dict = {}
    from src.cli import _build_account_portfolio, _book_fingerprint

    prod_account = next(
        (a for a in (cfg.get("shadow.accounts") or []) if str(a.get("name")) == args.account),
        account,
    )
    prod_ledger = PaperLedger(str(ROOT / "outputs" / "_doos_prod_probe.sqlite"))
    try:
        prod_book, _ = _build_account_portfolio(
            cfg, market, symbols, prod_account, control.gross_scale, ledger=prod_ledger
        )
        from src.paper.shadow import paper_runner_kwargs

        prod_kwargs = paper_runner_kwargs(cfg)
        prod_kwargs.update({
            "cash": float(prod_account.get("cash", prod_kwargs["cash"])),
            "notional_floor": float(prod_account.get("notional_floor", 0.0)),
            "band_frac": float(prod_account.get("band_frac", 0.0)),
            "rebalance_days": int(prod_account.get("rebalance_days", 1)),
            "max_position_pct": float(prod_account.get("max_position_pct", 0.05)),
        })
        prod_probe = _book_fingerprint(prod_book, runner_kwargs=prod_kwargs, universe=symbols)
    finally:
        prod_ledger.close()
        (ROOT / "outputs" / "_doos_prod_probe.sqlite").unlink(missing_ok=True)

    # ---- window data sanity -------------------------------------------------
    equity.index = pd.to_datetime(equity.index)  # ledger dates are TEXT
    window = equity.index[(equity.index >= pd.Timestamp(args.start)) & (equity.index <= pd.Timestamp(args.end))]
    dates = list(pd.to_datetime(window))
    gap_days = _calendar_gap_days(dates)
    frames = load_intraday_frames(cfg, symbols)
    frame_days = int(len(frames.get("tail_vol", pd.DataFrame()).index)) if frames else 0

    # ---- fills-derived checks ----------------------------------------------
    intraday = fills[fills["time"].fillna("").astype(str) != ""] if len(fills) else fills
    close_fills = fills[fills["time"].fillna("").astype(str) == ""] if len(fills) else fills
    live_from = account.get("pb_live_intraday_from")
    live_dates = (
        pd.to_datetime(intraday["date"]) >= pd.Timestamp(live_from)
        if (len(intraday) and live_from) else pd.Series(dtype=bool)
    )

    # T+1: a sell may not share a date with the buy that opened the lot.
    t1_ok = True
    if len(fills):
        opens: dict[str, list[str]] = {}
        for _, f in fills.sort_values(["date", "seq"] if "seq" in fills.columns else ["date"]).iterrows():
            sym = str(f["symbol"])
            if float(f["shares"]) > 0:
                opens.setdefault(sym, []).append(str(f["date"]))
            elif float(f["shares"]) < 0 and opens.get(sym):
                if opens[sym][-1] == str(f["date"]):
                    t1_ok = False
                opens[sym].pop()

    # Limit-lock legality, direction-aware — exactly the executor's rule: buying
    # into a limit-UP close and selling into a limit-DOWN close are impossible;
    # selling into a limit-up (or buying a limit-down) is perfectly legal and
    # must NOT be flagged. A direction-blind mask would flag every limit-up exit.
    from src.backtest.limit_locked import board_limit

    rets_panel = market.price_panel.pct_change(fill_method=None)
    locked_fills = 0
    locked_examples: list[dict] = []
    if len(fills):
        for _, f in fills.iterrows():
            day = pd.Timestamp(f["date"])
            sym = str(f["symbol"])
            try:
                lv = float(rets_panel.loc[day, sym])
            except (KeyError, TypeError):
                continue
            if not np.isfinite(lv):
                continue
            lim = board_limit(sym, day, True) - 0.005
            is_buy = float(f["shares"]) > 0
            illegal = (is_buy and lv >= lim) or ((not is_buy) and lv <= -lim)
            if illegal:
                locked_fills += 1
                if len(locked_examples) < 5:
                    locked_examples.append(
                        {"date": str(f["date"]), "symbol": sym, "side": f["side"],
                         "day_return": round(lv, 4)}
                    )

    # minute-feature coverage of the window. TWO levels are needed:
    #  * day level — does the rollup have a row for each trading day?
    #  * SYMBOL level — on each day, how much of the universe actually carries a
    #    feature? The 2025-10-27→12-12 hole had rows every day but only ~274/800
    #    symbols (every SH name missing), and the tail-volume gate turns a missing
    #    symbol into "no entry". A day-level 100% therefore said nothing.
    tail = frames.get("tail_vol") if frames else None
    symbol_cov: pd.Series = pd.Series(dtype=float)
    low_cov_days: list[str] = []
    if tail is not None and len(tail):
        win = tail.index[(tail.index >= pd.Timestamp(args.start)) & (tail.index <= pd.Timestamp(args.end))]
        sub = tail.loc[win]
        if len(sub):
            symbol_cov = sub.notna().sum(axis=1) / max(1, sub.shape[1])
        coverage = round(len(set(pd.to_datetime(win))) / max(1, len(dates)), 4)
        low_cov_days = [str(pd.Timestamp(d).date()) for d, v in symbol_cov.items() if v < MIN_SYMBOL_COVERAGE]
    else:
        coverage = 0.0

    # fills that happened on a low-coverage day (they are inside a data hole)
    fills_in_low_cov = 0
    if len(fills) and low_cov_days:
        low = set(low_cov_days)
        fills_in_low_cov = int(pd.to_datetime(fills["date"]).dt.date.astype(str).isin(low).sum())

    # SYMBOL-WINDOW coverage — the day-level probe above answers "how much of the
    # universe carries a feature today"; it cannot see a SINGLE name missing a
    # month. Both defects it must catch (a per-name data hole vs a legitimate
    # suspension/IPO) are separated by using the symbol's OWN tradable days (days
    # with a daily price bar in the window) as the denominator.
    sym_cov = pd.Series(dtype=float)
    sym_cov_raw = pd.Series(dtype=float)
    below_rows: list[dict] = []
    if tail is not None and len(tail):
        panel = market.price_panel
        pwin = panel.loc[(panel.index >= pd.Timestamp(args.start)) & (panel.index <= pd.Timestamp(args.end))]
        for sym in tail.columns:
            if sym not in pwin.columns:
                continue
            tradable = pwin[sym].notna()
            n_tr = int(tradable.sum())
            if n_tr == 0:  # never traded in the window (pre-listing / halted) → no denominator
                continue
            have = tail[sym].reindex(pwin.index).notna()
            n_have = int((have & tradable).sum())
            tr_cov = n_have / n_tr
            win_cov = float(have.mean())
            sym_cov[sym] = tr_cov
            sym_cov_raw[sym] = win_cov
            if tr_cov < MIN_SYMBOL_WINDOW_COVERAGE:
                missing = pwin.index[tradable & ~have]
                below_rows.append(
                    {
                        "symbol": sym,
                        "tradable_days": n_tr,
                        "feature_days": n_have,
                        "coverage_of_tradable_days": round(tr_cov, 4),
                        "coverage_of_window_days": round(win_cov, 4),
                        "n_missing_tradable_days": int(len(missing)),
                        "first_missing": str(pd.Timestamp(missing.min()).date()) if len(missing) else None,
                        "last_missing": str(pd.Timestamp(missing.max()).date()) if len(missing) else None,
                    }
                )
    below_rows.sort(key=lambda r: (r["coverage_of_tradable_days"], r["symbol"]))
    min_sym_cov = round(float(sym_cov.min()), 4) if len(sym_cov) else 0.0

    # cost consistency: ledger commission == sum of per-fill commissions
    if len(fills):
        cost_sum = float(fills["commission"].sum())
    else:
        cost_sum = 0.0
    cost_from_metrics = float(status.get("equity", {}).get("total_commission", 0.0))

    n_days = int(len(dates))
    sharpe = float(status.get("equity", {}).get("sharpe", 0.0) or 0.0)
    sharpe_se = math.sqrt(252.0 / n_days) if n_days > 0 else float("inf")

    n_live_dates = int(live_dates.sum()) if len(live_dates) else 0
    params_match = bool(
        probe.get("params_hash") and probe.get("params_hash") == prod_probe.get("params_hash")
    )
    assembly_match = bool(
        probe.get("book_class") == prod_probe.get("book_class")
        and probe.get("has_intraday_frames") == prod_probe.get("has_intraday_frames")
        and probe.get("has_minute_provider") == prod_probe.get("has_minute_provider")
        and probe.get("gross_scale_wired") == prod_probe.get("gross_scale_wired")
        and probe.get("execution") == prod_probe.get("execution")
    )
    min_symbol_cov = round(float(symbol_cov.min()), 4) if len(symbol_cov) else 0.0

    checks = {
        # the ledger is unlinked just above, so assert THAT instead of a result
        # field that can only ever be False (the old check was a tautology).
        "ledger_was_fresh": bool(not ledger_existed_before),
        "calendar_gap_within_tolerance": gap_days <= MAX_CALENDAR_GAP_DAYS,
        "minute_symbol_coverage_ok": bool(len(symbol_cov) and min_symbol_cov >= MIN_SYMBOL_COVERAGE),
        # per-NAME hole detector: a symbol missing features on days it actually
        # traded (suspensions/IPOs are excluded by construction)
        "minute_symbol_window_coverage_ok": bool(
            len(sym_cov) and min_sym_cov >= MIN_SYMBOL_WINDOW_COVERAGE
        ),
        "no_fills_inside_data_hole": bool(fills_in_low_cov == 0),
        "intraday_frames_loaded": bool(
            probe.get("has_intraday_frames") and "tail_vol" in (frames or {})
        ),
        "minute_provider_loaded": bool(probe.get("has_minute_provider")),
        "assembly_is_production": assembly_match,
        "no_fill_after_window_end": bool(
            not len(fills) or pd.to_datetime(fills["date"]).max() <= pd.Timestamp(args.end)
        ),
        "intraday_fills_before_1500": bool(
            not len(intraday)
            or (intraday["time"].astype(str).str.slice(0, 5) < INTRADAY_CUTOFF).all()
        ),
        # only meaningful when the window actually contains live-execution dates
        "live_dates_not_replayed": bool(not len(intraday) or n_live_dates == 0),
        "t_plus_1_respected": bool(t1_ok),
        "no_fill_on_limit_locked_bar": bool(locked_fills == 0),
        "cost_model_consistent": bool(
            abs(cost_sum - cost_from_metrics) < 0.01 and abs(cost_sum - ledger_cost) < 0.01
        ),
    }
    # `params_match_production` is reported separately: a CANDIDATE run (--set) is
    # expected to differ, so it must not be folded into all_passed — but it also
    # must not be silently dropped, or a candidate artifact could be mistaken for
    # the deployed configuration.
    is_candidate_run = bool(overrides)
    citable = bool(all(checks.values()) and params_match and not is_candidate_run)
    # Data/assembly soundness alone — a CANDIDATE run can be compared on sound
    # data even though it is not the deployed configuration.
    data_checks = (
        "ledger_was_fresh", "calendar_gap_within_tolerance",
        "minute_symbol_coverage_ok", "minute_symbol_window_coverage_ok",
        "no_fills_inside_data_hole",
        "intraday_frames_loaded", "minute_provider_loaded",
        "assembly_is_production", "t_plus_1_respected",
        "no_fill_on_limit_locked_bar", "cost_model_consistent",
    )
    data_ok = bool(all(checks.get(k) for k in data_checks))

    result = {
        "label": args.label,
        "window": {"start": args.start, "end": args.end},
        "account": args.account,
        #: candidate parameter overrides (empty for the deployed configuration)
        "overrides": overrides,
        "is_candidate_run": is_candidate_run,
        "params_match_production": params_match,
        #: TRUE only for a fresh-ledger, gap-free, production-config run — the
        #: only kind of artifact whose numbers may be quoted as evidence.
        "citable": citable,
        #: data/assembly soundness only (a candidate run can be compared on
        #: sound data even though it is not the deployed configuration)
        "data_ok": data_ok,
        "ledger": str(ledger_path),
        "ledger_existed_before": bool(ledger_existed_before),
        "n_days": n_days,
        "n_fills": int(len(fills)),
        "n_intraday_fills": int(len(intraday)),
        "n_close_fills": int(len(close_fills)),
        "n_live_dates_in_window": n_live_dates,
        "metrics": status.get("equity", {}),
        "sharpe_standard_error": round(sharpe_se, 3),
        "sharpe_t_stat": round(sharpe / sharpe_se, 3) if sharpe_se > 0 and sharpe_se != float("inf") else None,
        "max_calendar_gap_days": gap_days,
        "minute_frame_days": frame_days,
        "minute_window_coverage": coverage,
        "minute_min_symbol_coverage": min_symbol_cov,
        "minute_min_symbol_tradable_coverage": min_sym_cov,
        "symbols_below_90pct": [r["symbol"] for r in below_rows],
        "symbol_coverage_below_90pct": below_rows,
        "n_symbols_below_90pct": len(below_rows),
        "low_coverage_days": low_cov_days,
        "n_low_coverage_days": len(low_cov_days),
        "fills_in_low_coverage_days": fills_in_low_cov,
        "kill_switch": {"mode": control.mode, "gross_scale": control.gross_scale},
        "limit_locked_fills": locked_fills,
        "limit_locked_examples": locked_examples,
        "fingerprint": probe,
        "production_fingerprint": prod_probe,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }
    out_path = ROOT / "outputs" / f"d_oos_{args.label}.json"
    # Provenance (audit P-6): an OOS number is only quotable together with the
    # slice, the convention and the code that produced it. ``data_as_of`` is the
    # last bar actually present in the panel — a recent window on stale data
    # (the exact failure mode that made the pre-backfill OOS artifact misleading)
    # is now visible in the artifact itself.
    from src.provenance import stamp_artifact

    data_as_of = str(pd.Timestamp(market.price_panel.index.max()).date())
    convention = (
        "adjusted-close price basis; daily close rebalance at the panel close; "
        f"intraday stop trigger={account.get('pb_stop_trigger')} "
        f"open_minutes={account.get('pb_stop_open_minutes')}; "
        f"live dates (>={account.get('pb_live_intraday_from')}) fill at the 15:00 "
        "auction from outputs/preclose_orders_<acct>.json (no orders → no close trades); "
        "T+1; no leverage; commission+stamp duty per the configured cost model"
    )
    result = stamp_artifact(
        result, window={"start": args.start, "end": args.end},
        convention=convention, data_as_of=data_as_of,
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"[oos] days={n_days} fills={result['n_fills']} "
          f"(intraday {result['n_intraday_fills']}) "
          f"minute_coverage={coverage:.0%} (min symbol {min_symbol_cov:.0%}, "
          f"low-cov days {len(low_cov_days)}, fills in hole {fills_in_low_cov}) "
          f"sym-window cov min={min_sym_cov:.1%} below-90%={len(below_rows)} "
          f"{[r['symbol'] for r in below_rows[:6]]} "
          f"cum={status.get('equity', {}).get('total_return', 0):+.2%} "
          f"ann={status.get('equity', {}).get('annualized_return', 0):+.2%} "
          f"sharpe={sharpe:+.2f} (SE {sharpe_se:.2f}, t={result['sharpe_t_stat']}) "
          f"maxDD={status.get('equity', {}).get('max_drawdown', 0):.2%}", flush=True)
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    print(f"[oos] params_match_production={params_match} candidate={is_candidate_run} "
          f"citable={citable}", flush=True)
    print(f"[oos] provenance: data_as_of={data_as_of} commit={result['provenance']['code_commit'][:12]} "
          f"sha256={result['provenance']['artifact_sha256'][:16]}…", flush=True)
    print(f"[oos] all_passed={result['all_passed']} → {out_path}", flush=True)
    return 0 if citable else 1


if __name__ == "__main__":
    raise SystemExit(main())
