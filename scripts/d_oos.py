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

#: A calendar gap wider than this (in calendar days) means the window contains a
#: hole in the data — results over it cannot be read as a continuous curve.
MAX_CALENDAR_GAP_DAYS = 10

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
    args = ap.parse_args()

    from src.cli import _market_data, _shadow_cycle
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

    symbols = resolve_shadow_universe(cfg, account.get("universe"))
    market = _market_data(cfg, seed=args.seed)
    symbols = [s for s in symbols if s in market.price_panel.columns]

    ledger_path = ROOT / "outputs" / f"_doos_{args.label}.sqlite"
    ledger_path.unlink(missing_ok=True)

    print(f"[oos] window=[{args.start}, {args.end}] account={args.account} "
          f"universe={len(symbols)} ledger={ledger_path.name}", flush=True)
    probe: dict = {}
    status, ledger_out = _shadow_cycle(
        cfg, symbols, args.start, args.end, args.seed, skip_refresh=not args.refresh,
        control_scale=None, account=account, ledger_override=str(ledger_path),
        write_artifacts=False, probe=probe,
    )

    ledger = PaperLedger(str(ledger_path))
    fills = ledger.fills()
    equity = ledger.equity_curve()
    ledger_cost = float(ledger.total_commission())
    ledger.close()

    # ---- independent production assembly (fingerprint comparison) -----------
    prod_probe: dict = {}
    from src.cli import _build_account_portfolio

    prod_ledger = PaperLedger(str(ROOT / "outputs" / "_doos_prod_probe.sqlite"))
    try:
        prod_book, _ = _build_account_portfolio(
            cfg, market, symbols, account, None, ledger=prod_ledger
        )
        from src.cli import _book_fingerprint

        prod_probe = _book_fingerprint(prod_book)
    finally:
        prod_ledger.close()
        (ROOT / "outputs" / "_doos_prod_probe.sqlite").unlink(missing_ok=True)

    # ---- window data sanity -------------------------------------------------
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

    # limit-lock: no fill may sit on a locked bar
    from src.backtest.limit_locked import limit_lock_mask

    locked = limit_lock_mask(market.long)
    locked_set = set(locked[locked].index)
    locked_fills = 0
    if len(fills):
        for _, f in fills.iterrows():
            key = (pd.Timestamp(f["date"]), str(f["symbol"]))
            if key in locked_set:
                locked_fills += 1

    # minute-feature coverage of the window: the tail-volume entry gate treats a
    # missing day as FAIL, so a gap silently suppresses entries (and the OOS
    # number must be read with that in mind).
    tail = frames.get("tail_vol") if frames else None
    if tail is not None and len(tail):
        covered = tail.index[(tail.index >= pd.Timestamp(args.start)) & (tail.index <= pd.Timestamp(args.end))]
        coverage = round(len(set(pd.to_datetime(covered))) / max(1, len(dates)), 4)
    else:
        coverage = 0.0

    # cost consistency: ledger commission == sum of per-fill commissions
    if len(fills):
        cost_sum = float(fills["commission"].sum())
    else:
        cost_sum = 0.0
    cost_from_metrics = float(status.get("equity", {}).get("total_commission", 0.0))

    n_days = int(len(dates))
    sharpe = float(status.get("equity", {}).get("sharpe", 0.0) or 0.0)
    sharpe_se = math.sqrt(252.0 / n_days) if n_days > 0 else float("inf")

    checks = {
        "fresh_ledger_not_resumed": bool(probe.get("resumed") is False),
        "contiguous_window_no_gap": gap_days <= MAX_CALENDAR_GAP_DAYS,
        "intraday_frames_loaded": bool(probe.get("has_intraday_frames")),
        "minute_provider_loaded": bool(probe.get("has_minute_provider")),
        "strategy_fingerprint_is_production": bool(
            probe.get("params_hash") and probe.get("params_hash") == prod_probe.get("params_hash")
            and probe.get("book_class") == prod_probe.get("book_class")
        ),
        "no_fill_after_window_end": bool(
            not len(fills) or pd.to_datetime(fills["date"]).max() <= pd.Timestamp(args.end)
        ),
        "intraday_fills_before_1500": bool(
            not len(intraday)
            or (intraday["time"].astype(str).str.slice(0, 5) < INTRADAY_CUTOFF).all()
        ),
        "live_dates_not_replayed": bool(not len(intraday) or live_dates.sum() == 0),
        "t_plus_1_respected": bool(t1_ok),
        "no_fill_on_limit_locked_bar": bool(locked_fills == 0),
        "cost_model_consistent": bool(
            abs(cost_sum - cost_from_metrics) < 0.01 and abs(cost_sum - ledger_cost) < 0.01
        ),
    }

    result = {
        "label": args.label,
        "window": {"start": args.start, "end": args.end},
        "account": args.account,
        "ledger": str(ledger_path),
        "n_days": n_days,
        "n_fills": int(len(fills)),
        "n_intraday_fills": int(len(intraday)),
        "n_close_fills": int(len(close_fills)),
        "metrics": status.get("equity", {}),
        "sharpe_standard_error": round(sharpe_se, 3),
        "sharpe_t_stat": round(sharpe / sharpe_se, 3) if sharpe_se > 0 and sharpe_se != float("inf") else None,
        "max_calendar_gap_days": gap_days,
        "minute_frame_days": frame_days,
        "minute_window_coverage": coverage,
        "fingerprint": probe,
        "production_fingerprint": prod_probe,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
    }
    out_path = ROOT / "outputs" / f"d_oos_{args.label}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"[oos] days={n_days} fills={result['n_fills']} "
          f"(intraday {result['n_intraday_fills']}) "
          f"minute_coverage={coverage:.0%} "
          f"cum={status.get('equity', {}).get('total_return', 0):+.2%} "
          f"ann={status.get('equity', {}).get('annualized_return', 0):+.2%} "
          f"sharpe={sharpe:+.2f} (SE {sharpe_se:.2f}, t={result['sharpe_t_stat']}) "
          f"maxDD={status.get('equity', {}).get('max_drawdown', 0):.2%}", flush=True)
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    print(f"[oos] all_passed={result['all_passed']} → {out_path}", flush=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
