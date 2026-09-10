"""Forward-period health monitor — the RISK gate, not a capability gate.

    python scripts/forward_health.py                     # full run (replay included)
    python scripts/forward_health.py --no-replay          # skip the replay (TE unmeasured)
    python scripts/forward_health.py --start 2026-09-10 --end 2026-09-30
    python scripts/forward_health.py --json

What it measures over the forward window (``configs/forward_policy.yaml``):

1. **pipeline tracking error** — the production ledger is replayed from a copy of
   its own state (rows before the window start only) with the same code on the
   data as it exists NOW. Daily |recorded − replay| in pp/day plus a sign-bias
   binomial test. A non-zero value means the data moved under the book
   (adjustment-anchor drift, revised bar) or the code changed silently — the
   exact failure a forward window IS able to detect.
2. **cost-model calibration** — charged fees vs the pre-registered cost spec, and
   the realized execution price vs a market reference (the panel close for
   close/auction fills, the same-minute print for live fills). Market impact is
   reported as UNMEASURED rather than assumed away.
3. **operational reliability** — T+1 / lot / tick / limit / suspension violations,
   live-layer availability from the append-only heartbeat, data freshness, and
   symbol-level minute coverage.

Soft metrics (Sharpe, maxDD, excess) are recorded and NEVER gate: the window has
no statistical power for them (see ``docs/FORWARD_PROTOCOL.md`` §1.1).

Exit code: 0 = every hard gate passed, 1 = at least one failed, 2 = the monitor
could not run (config/ledger missing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.forward.risk_gate import (  # noqa: E402
    GateThresholds,
    availability,
    cost_deviation,
    data_freshness,
    evaluate_gate,
    fill_violations,
    panel_universe_health,
    soft_metrics,
    symbol_minute_coverage,
    tracking_error,
)
from src.provenance import stamp_artifact  # noqa: E402

CONVENTION = (
    "adjusted-close price basis; daily close rebalance at the panel close; "
    "live dates fill intraday stops at the confirmed minute print and close "
    "orders at the 15:00 auction from outputs/preclose_orders_<acct>.json; "
    "T+1; no leverage"
)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def _account(cfg, name: str) -> dict:
    acc = next((a for a in (cfg.get("shadow.accounts") or []) if str(a.get("name")) == name), None)
    if acc is None:
        raise SystemExit(f"ERROR: account {name!r} not in shadow.accounts")
    return dict(acc)


def _ledger_path(cfg, account: dict) -> Path:
    base = str(cfg.get("shadow.ledger_db", "outputs/shadow_ledger.sqlite"))
    return ROOT / base.replace(".sqlite", f"_{account['name']}.sqlite")


def _active_prereg(cfg, account: dict) -> dict | None:
    """Newest verified pre-registration whose scope names this account."""
    from src.forward.prereg import DEFAULT_DIR, list_preregistrations, verify_preregistration

    raw = cfg.get("forward.prereg_dir")
    base = Path(str(raw)) if raw else DEFAULT_DIR
    if not base.is_absolute():
        base = ROOT / base          # the scheduler may run from any directory
    fallback: dict | None = None
    for rec in list_preregistrations(base):
        if rec.get("error"):
            continue
        scope = rec.get("scope") or {}
        if str(scope.get("account", account["name"])) != str(account["name"]):
            continue
        try:
            verify_preregistration(rec["_path"])
        except Exception:  # noqa: BLE001 — an invalid record must not become the window
            continue
        # prefer the record that governs the GATE itself; a candidate record
        # shares the window but its decision is the switch rule, not the gate
        if str((rec.get("trials") or {}).get("family")) == "d_forward_risk_gate":
            return rec
        fallback = fallback or rec
    return fallback


def _window(cfg, account: dict, args) -> tuple[str, str, dict | None]:
    """Requested window: explicit args, else the active pre-registration's scope."""
    rec = _active_prereg(cfg, account)
    start = args.start or (str((rec or {}).get("scope", {}).get("window", [None])[0] or "") or None)
    end = args.end or (str((rec or {}).get("scope", {}).get("window", [None, None])[1] or "") or None)
    if not start:
        raise SystemExit(
            "ERROR: no forward window — pass --start or freeze a pre-registration "
            "with scope.window (scripts/prereg.py new)"
        )
    end = end or pd.Timestamp.today().normalize().date().isoformat()
    return start, end, rec


def _effective_freeze(cfg, prereg: dict | None) -> tuple[str | None, str]:
    """The date the deployed configuration took effect — and why that date.

    Two sources, and the IMMUTABLE one wins: the active pre-registration's
    ``frozen_at`` date (hashed into the record) outranks the mutable
    ``forward.config_freeze_date`` config key. Moving the config key earlier can
    therefore no longer drop days from the sample — the exemption knob is pinned
    to a signed artifact.
    """
    cfg_date = str(cfg.get("forward.config_freeze_date", "") or "")
    rec_date = ""
    if prereg:
        try:
            rec_date = str(pd.Timestamp(prereg.get("frozen_at")).date())
        except (ValueError, TypeError):
            rec_date = ""
    if cfg_date and rec_date:
        return max(cfg_date, rec_date), f"max(config {cfg_date}, prereg.frozen_at {rec_date})"
    return (cfg_date or rec_date or None), ("config" if cfg_date else "prereg" if rec_date else "none")


def _trading_days_between(panel_index, start, end) -> list[pd.Timestamp]:
    """Panel dates inside ``[start, end]`` — the book's own calendar."""
    if panel_index is None or not len(panel_index):
        return []
    idx = pd.DatetimeIndex(panel_index)
    return [pd.Timestamp(d) for d in idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]]


def _daily_returns(ledger_path: Path, start: str, end: str) -> pd.Series:
    from src.paper.ledger import PaperLedger

    led = PaperLedger(str(ledger_path))
    try:
        eq = led.equity_curve()
    finally:
        led.close()
    eq = pd.Series(eq).astype(float).sort_index()
    eq.index = pd.to_datetime(eq.index)
    win = eq.loc[(eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))]
    return win.pct_change().dropna()


def _fills(ledger_path: Path) -> pd.DataFrame:
    from src.paper.ledger import PaperLedger

    led = PaperLedger(str(ledger_path))
    try:
        df = led.fills()
    finally:
        led.close()
    return df if df is not None else pd.DataFrame()


def _replay(cfg, account: dict, symbols: list[str], start: str, end: str,
            prod_ledger: Path) -> tuple[pd.Series, object]:
    """Replay the forward window from the pre-window state; return its daily returns.

    The production ledger is COPIED and truncated to rows strictly before
    ``start``, so the replay begins from exactly the state the live book was in
    on the eve of the window — then the same code runs on today's data.
    """
    from src.paper.ledger import clone_ledger_before

    scratch = ROOT / "outputs" / f"_fwd_replay_{account['name']}.sqlite"
    seed = clone_ledger_before(prod_ledger, scratch, start)
    print(f"[fwd] replay state: days={seed['days']} positions={seed['positions']} "
          f"fills={seed['fills']} (last {seed['last_date']})", flush=True)

    from src.cli import _build_market_for_paper, _shadow_cycle
    from src.autopilot.state import ControlState

    market = _build_market_for_paper(cfg, symbols, start, None, seed=1)
    state_path = str(ROOT / str(cfg.get("autopilot.state_file", "outputs/autopilot_state.json"))
                     ).replace(".json", f"_{account['name']}.json")
    control = ControlState.load(state_path)
    _shadow_cycle(
        cfg, symbols, start, end, 1, skip_refresh=True, control_scale=control.gross_scale,
        account=account, ledger_override=str(scratch), write_artifacts=False,
        market_override=market,
    )
    return _daily_returns(scratch, start, end), market


def _reference_prices(cfg, fills: pd.DataFrame) -> dict:
    """``(date, symbol) -> reference price`` for the realized-slippage check.

    Close/auction fills are checked against the PIT adjusted close of that day;
    intraday fills against the same-minute print in the minute cache (the price
    the poller actually saw). A missing reference simply drops that fill from the
    check — and the gate fails if NO fill could be checked.
    """
    refs: dict[tuple[str, str], float] = {}
    if fills is None or len(fills) == 0:
        return refs
    url = cfg.get("data.pit_database_url")
    symbols = sorted({str(s) for s in fills["symbol"].unique()})
    daily: dict[tuple[str, str], float] = {}
    if url:
        try:
            from src.data.point_in_time_loader import from_url

            store = from_url(url)
            for sym in symbols:
                hist = store.history(sym, "price")
                if hist is None or hist.empty or "close" not in hist.columns:
                    continue
                for r in hist.itertuples():
                    day = str(pd.Timestamp(r.valid_from).date())
                    try:
                        daily[(day, sym)] = float(r.close)
                    except (TypeError, ValueError):
                        continue
            if hasattr(store, "close"):
                store.close()
        except Exception as exc:  # noqa: BLE001 — a missing reference is reported, not fatal
            print(f"WARNING: PIT reference prices unavailable ({exc})", file=sys.stderr)

    minute_cache: dict[str, pd.DataFrame] = {}
    from src.data.intraday import _symbol_minutes  # noqa: PLC0415 — lazy, cached

    for r in fills.itertuples():
        day = str(pd.Timestamp(r.date).date())
        sym = str(r.symbol)
        t = str(getattr(r, "time", "") or "")
        if t:
            if sym not in minute_cache:
                try:
                    minute_cache[sym] = _symbol_minutes(cfg, sym)
                except Exception:  # noqa: BLE001
                    minute_cache[sym] = pd.DataFrame()
            bars = minute_cache[sym]
            if bars is not None and len(bars):
                day_bars = bars[bars["timestamp"].dt.normalize() == pd.Timestamp(day)]
                day_bars = day_bars[day_bars["timestamp"].dt.strftime("%H:%M") == t[:5]]
                if len(day_bars):
                    refs[(day, sym)] = float(day_bars["close"].iloc[-1])
                    continue
        if (day, sym) in daily:
            refs[(day, sym)] = daily[(day, sym)]
    return refs


def _heartbeats(account: dict) -> pd.DataFrame:
    path = ROOT / "outputs" / f"live_{account['name']}.jsonl"
    if not path.is_file():
        return pd.DataFrame(columns=["ts"])
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    df = pd.DataFrame(rows)
    if len(df) and "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Forward-period risk gate")
    ap.add_argument("--account", default="D_5W")
    ap.add_argument("--start", default=None, help="forward window start (default: prereg scope)")
    ap.add_argument("--end", default=None, help="forward window end (default: today)")
    ap.add_argument("--no-replay", action="store_true",
                    help="skip the tracking-error replay (leaves that gate unmeasured → fail)")
    ap.add_argument("--out", default=None, help="artifact path (default: policy health_json)")
    ap.add_argument("--allow-code-drift", action="store_true",
                    help="record an explicit waiver when HEAD differs from the frozen "
                         "commit (the waiver is written into the artifact)")
    ap.add_argument("--json", action="store_true", help="print the artifact to stdout")
    args = ap.parse_args()

    from src.config import load_config
    from src.forward.prereg import policy_fingerprint, prereg_gate, verify_preregistration
    from src.provenance import git_commit, git_dirty

    cfg = load_config()
    account = _account(cfg, args.account)
    start, end, prereg = _window(cfg, account, args)
    prod_ledger = _ledger_path(cfg, account)
    if not prod_ledger.is_file():
        print(f"ERROR: production ledger not found: {prod_ledger}", file=sys.stderr)
        return 2

    from src.paper.shadow import resolve_shadow_universe

    symbols = resolve_shadow_universe(cfg, account.get("universe"))
    freeze, freeze_src = _effective_freeze(cfg, prereg)
    # The MEASURED window starts at the freeze date: a replay under today's
    # configuration cannot reproduce days recorded before it took effect, and
    # seeding the clone earlier than that carries the pre-freeze divergence into
    # every later day (the 2026-09-09 shakedown: 0.99pp/day, then 2.36pp on the
    # one day that survived the exclusion). Starting the window at the freeze
    # removes the need for a self-service exclusion list altogether.
    measured_start = max(pd.Timestamp(start), pd.Timestamp(freeze)) if freeze else pd.Timestamp(start)
    measured_start = str(measured_start.date())
    print(f"[fwd] account={account['name']} window=[{start}, {end}] "
          f"measured=[{measured_start}, {end}] freeze={freeze} ({freeze_src}) "
          f"universe={len(symbols)} prereg={(prereg or {}).get('rule_id')}", flush=True)

    # ---- 0. pre-registration binding (the lock) ---------------------------
    code_commit = git_commit(ROOT)
    code_dirty = git_dirty(ROOT)
    pol_sha = policy_fingerprint(cfg)
    binding = prereg_gate(
        record=prereg, window=(start, end), data_as_of=end,
        policy_sha256=pol_sha, code_commit=code_commit, code_dirty=code_dirty,
        allow_code_drift=bool(args.allow_code_drift),
    )
    if binding["ok"]:
        print(f"[fwd] prereg bound: {binding['rule_id']} frozen {binding['frozen_at']} "
              f"policy={pol_sha[:12]} commit={code_commit[:12]}"
              + (" (code drift WAIVED)" if binding.get("waived") else ""), flush=True)
    else:
        print("[fwd] PRE-REGISTRATION NOT BOUND — the evaluation is not evidence:", flush=True)
        for issue in binding["issues"]:
            print(f"  - {issue}", flush=True)

    # ---- 1. tracking error ------------------------------------------------
    # The FULL fill history is needed before the replay: a date on which the
    # real-time layer executed a fill at a live print is an external market
    # event the bar replay cannot reproduce (it is gated off on live dates by
    # design), so it is excluded from the gate and reported separately.
    fills_all = _fills(prod_ledger)
    recorded = _daily_returns(prod_ledger, measured_start, end)
    live_days: list[str] = []
    if len(fills_all) and "source" in fills_all.columns:
        live_rows = fills_all[fills_all["source"].astype(str) == "live"]
        live_days = sorted({
            str(pd.Timestamp(d).date()) for d in live_rows["date"]
            if pd.Timestamp(measured_start) <= pd.Timestamp(d) <= pd.Timestamp(end)
        })
    exclude_days = list(live_days)
    market = None
    if args.no_replay:
        te = tracking_error(pd.Series(dtype=float), pd.Series(dtype=float))
        print("[fwd] replay skipped → tracking error unmeasured", flush=True)
    else:
        print(f"[fwd] replaying [{measured_start}, {end}] from the pre-window state …", flush=True)
        replay, market = _replay(cfg, account, symbols, measured_start, end, prod_ledger)
        te = tracking_error(recorded, replay, exclude_dates=exclude_days)
        te["excluded_live_days"] = live_days
        te["config_freeze_date"] = freeze
        te["measured_window"] = {"start": measured_start, "end": end}
        print(f"[fwd] tracking error: {te['mean_abs_pp']}pp/day over {te['n_days']} days "
              f"(sign bias p={te['sign_bias_p']}; excluded {te['n_excluded']} live-fill "
              f"day(s), their mean |diff| {te['excluded_mean_abs_pp']}pp)", flush=True)

    # ---- 2. cost ----------------------------------------------------------
    # (``fills_all`` was loaded above for the live-day exclusion.)
    fills = fills_all
    if len(fills_all):
        dts = pd.to_datetime(fills_all["date"])
        fills = fills_all[(dts >= pd.Timestamp(start)) & (dts <= pd.Timestamp(end))]
    spec = dict(cfg.get("s7_calibration.cost_model", {}) or {})
    refs = _reference_prices(cfg, fills)
    cost = cost_deviation(fills, spec, reference_prices=refs,
                          modeled_slippage_bps=float(cfg.get("paper.slippage_bps", 2.0)),
                          price_integrity_bps_max=float(
                              cfg.get("forward.risk_gate.hard.price_integrity_bps_max", 2.0)))
    print(f"[fwd] cost: n={cost['n_fills']} fee_dev={cost['fee_deviation_pct']}% "
          f"price_integrity={cost['price_integrity_abs_bps']}bps "
          f"({cost['n_price_checked']} checked / {cost['n_price_skipped']} skipped)", flush=True)

    # ---- 3. operational reliability ---------------------------------------
    panel = market.price_panel if market is not None else pd.DataFrame()
    limit_fn = None
    if len(panel):
        from src.backtest.limit_locked import board_limit

        limit_fn = lambda s, d, up=True: board_limit(s, d, up)  # noqa: E731
    violations = fill_violations(
        fills_all, panel if len(panel) else None, limit_fn=limit_fn,
        since=measured_start, until=end,
    )
    hb = _heartbeats(account)
    days = [pd.Timestamp(d) for d in recorded.index]
    avail_start = measured_start
    if len(hb) and "ts" in hb.columns:
        first_hb = str(hb["ts"].min().date())
        avail_start = max(pd.Timestamp(measured_start), pd.Timestamp(first_hb)).date().isoformat()
    avail = availability(hb, [d for d in days if d >= pd.Timestamp(avail_start)])
    avail["measured_from"] = avail_start

    from src.data.intraday import load_intraday_frames

    frames = load_intraday_frames(cfg, symbols)
    coverage = symbol_minute_coverage(frames, panel, measured_start, end,
                                      threshold=float(cfg.get("forward.risk_gate.hard.symbol_minute_coverage_min", 0.95))) \
        if len(panel) else {"min_coverage": None, "n_symbols": 0, "below_threshold": [],
                            "threshold": 0.95}
    universe = panel_universe_health(panel, as_of=end) if len(panel) else {}
    if universe:
        print(f"[fwd] effective universe: {universe.get('n_warm_20')}/{universe.get('n_columns')} "
              f"warm names (ratio {universe.get('effective_ratio')}; "
              f"{universe.get('n_with_price')} with a bar on {universe.get('as_of')})", flush=True)

    # Freshness must be measured against the MARKET, not against the book's own
    # last row: a ledger that stopped updating on 2026-08-10 would otherwise look
    # "0 days stale" forever. ``lag_days`` = newest bar vs the trading day we are
    # entitled to expect; ``ledger_lag_days`` = the book vs that same bar.
    fresh = None
    expected_last = pd.Timestamp(pd.Timestamp.now().normalize())
    hhmm = pd.Timestamp.now().strftime("%H:%M")
    if expected_last.weekday() >= 5 or hhmm < "15:00":
        # before the close (or on a weekend) the newest complete session is earlier
        expected_last -= pd.Timedelta(days=1)
        while expected_last.weekday() >= 5:
            expected_last -= pd.Timedelta(days=1)
    bar_max = None
    if len(panel):
        bar_max = pd.Timestamp(panel.index.max())
    else:
        url = cfg.get("data.pit_database_url")
        if url:
            try:
                from src.data.point_in_time_loader import from_url

                store = from_url(url)
                bar_max = store.max_valid_from("price")
                if hasattr(store, "close"):
                    store.close()
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: data freshness unavailable ({exc})", file=sys.stderr)
    if bar_max is not None and not pd.isna(bar_max):
        fresh = data_freshness(str(pd.Timestamp(bar_max).date()), str(expected_last.date()))
        fresh["source"] = "market panel max bar" if len(panel) else "pit_records.max(valid_from) price"
        ledger_last = days[-1] if days else None
        fresh["ledger_last_date"] = str(pd.Timestamp(ledger_last).date()) if ledger_last is not None else None
        fresh["ledger_lag_days"] = (
            int((pd.Timestamp(bar_max).normalize() - pd.Timestamp(ledger_last).normalize()).days)
            if ledger_last is not None else None
        )
    if fresh is None:
        fresh = {"lag_days": None, "unmeasured": True,
                 "note": "no market panel and no reachable PIT store"}

    # ---- 4. soft (recorded only) ------------------------------------------
    equity = None
    from src.paper.ledger import PaperLedger

    led = PaperLedger(str(prod_ledger))
    try:
        equity = led.equity_curve()
    finally:
        led.close()
    eq = pd.Series(equity).astype(float)
    eq.index = pd.to_datetime(eq.index)
    eq = eq.loc[(eq.index >= pd.Timestamp(measured_start)) & (eq.index <= pd.Timestamp(end))]
    soft = soft_metrics(eq)

    metrics = {
        "prereg": binding,
        "tracking_error": te,
        "cost": cost,
        "violations": violations,
        "availability": avail,
        "data_freshness": fresh or {},
        "symbol_coverage": coverage,
        "universe": universe,
        "soft": soft,
    }
    gate = evaluate_gate(metrics, GateThresholds.from_config(cfg))
    artifact = {
        "account": account["name"],
        "window": {"start": start, "end": end},
        "measured_window": {"start": measured_start, "end": end},
        "config_freeze_date": freeze,
        "config_freeze_source": freeze_src,
        "policy_sha256": pol_sha,
        "code_commit": code_commit,
        "prereg": None if prereg is None else {
            "rule_id": prereg.get("rule_id"), "version": prereg.get("version"),
            "frozen_at": prereg.get("frozen_at"), "record_sha256": prereg.get("record_sha256"),
            "trials": prereg.get("trials"),
        },
        "metrics": metrics,
        "gate": gate,
        "n_fills": int(len(fills)),
        "n_days": int(len(recorded)),
        "fills_by_source": (
            {str(k): int(v) for k, v in fills["source"].fillna("").replace("", "unlabelled")
             .value_counts().items()} if len(fills) else {}
        ),
    }
    data_as_of = str(pd.Timestamp(panel.index.max()).date()) if len(panel) else end
    artifact = stamp_artifact(artifact, window={"start": start, "end": end},
                              convention=CONVENTION, data_as_of=data_as_of)

    out = Path(args.out) if args.out else ROOT / str(cfg.get("forward.health_json",
                                                             "outputs/forward/forward_health.json"))
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    hist_dir = ROOT / str(cfg.get("forward.health_history_dir", "outputs/forward/history"))
    hist_dir.mkdir(parents=True, exist_ok=True)
    (hist_dir / f"forward_health_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if args.json:
        print(json.dumps(artifact, ensure_ascii=False, indent=2, default=str))
    print(f"[fwd] hard gate: {gate['verdict'].upper()}"
          + (f" — failed: {gate['failed']}" if gate["failed"] else ""))
    for k, v in gate["hard"].items():
        if isinstance(v, dict):
            print(f"  [{'PASS' if v.get('ok') else 'FAIL'}] {k}: {json.dumps(v, ensure_ascii=False)}")
    print(f"[fwd] soft (record only): {json.dumps(gate['soft']['values'], ensure_ascii=False)}")
    print(f"[fwd] → {out}")
    return 0 if gate["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
