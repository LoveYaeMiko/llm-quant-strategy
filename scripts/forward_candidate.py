"""Forward candidate comparison — ``atr_1p0_25_40`` vs the deployed flat 3.5% stop.

    python scripts/forward_candidate.py run                # advance BOTH arms one day
    python scripts/forward_candidate.py run --date 2026-09-10
    python scripts/forward_candidate.py report             # paired comparison
    python scripts/forward_candidate.py daily              # run + report (PAICC job)
    python scripts/forward_candidate.py report --json

Design (audit item 2, ``docs/FORWARD_PROTOCOL.md`` §2) — and the 2026-09-10 fix:

* **Two arms, one difference.** ``incumbent`` (= the deployed flat 3.5% stop) and
  ``candidate`` (= ATR 1.0 clipped to [2.5%, 4.0%]) run the SAME code, the same
  data slice, the same universe, the same control state and the same seed state;
  only the stop width differs. Each arm owns its ledger
  (``outputs/forward/<arm>/ledger.sqlite``) and can never touch the production
  ledger.
* **Both arms are REPLAYS, not the live book.** The first version of this script
  had the candidate inherit the production 14:50 order list
  (``pb_preclose_account``). On live dates the runner executes that list instead
  of computing its own targets AND the intraday sweep is gated off — so the
  candidate's stop width drove NOTHING and the two ledgers came out byte-identical
  (verified 2026-09-10: same equity, same 253 fills). The experiment could not
  answer its own question. Now both arms are counterfactual replays of the frozen
  rule set from minute bars (``pb_live_intraday_from`` cleared for the arm
  accounts), so a stop-width difference actually shows up on every day.
  The production ledger remains the OPERATIONAL record and is not the comparison
  arm — it mixes live execution with close execution and cannot be paired with a
  replay.
* **Record only.** Nothing is auto-promoted. The switch rule is pre-registered:
  over the forward window, a paired daily-difference mean > 0 AND paired t > 1.5
  ⇒ switch (which COUNTS AS A NEW TRIAL and must be re-pre-registered); anything
  else ⇒ hold. The two books correlate ≈ 0.88, so the paired difference — not the
  Sharpe difference — is the test statistic.
* **"May never separate" is an accepted outcome.** The candidate is not given a
  deadline and the incumbent is not destabilised to force a decision.
* **Catch-up.** ``run`` advances every trading day missing from BOTH arms, not
  just today: the runner walks ``[first_missing, newest_bar]`` in one cycle, so a
  missed 15:20 job (or a holiday) leaves no hole in the paired series.

Exit code: 0 on success, 1 when the report cannot be produced (missing ledgers).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.forward.risk_gate import paired_comparison  # noqa: E402
from src.provenance import stamp_artifact  # noqa: E402

DEFAULT_RULE = "atr_1p0_25_35"
CONVENTION = (
    "adjusted-close basis; two REPLAY arms of the frozen D-track rule set on the "
    "same data slice and universe; the only difference is the stop width "
    "(flat 3.5% vs ATR 1.0 clipped to [2.5%, 4.0%]); intraday stops evaluated on "
    "minute bars, close rebalance at the panel close; T+1; no leverage"
)


def _cfg():
    from src.config import load_config

    return load_config()


def _spec(cfg, rule_id: str) -> dict:
    spec = dict((cfg.get("forward.candidates", {}) or {}).get(rule_id, {}) or {})
    if not spec:
        raise SystemExit(f"ERROR: forward.candidates.{rule_id} not in configs/forward_policy.yaml")
    return spec


def _production_account(cfg, name: str) -> dict:
    acc = next((a for a in (cfg.get("shadow.accounts") or []) if str(a.get("name")) == name), None)
    if acc is None:
        raise SystemExit(f"ERROR: account {name!r} not in shadow.accounts")
    return dict(acc)


def _replay_account(cfg, spec: dict, arm: str) -> dict:
    """The account for one arm: production shape + that arm's stop width.

    ``pb_live_intraday_from`` is CLEARED deliberately — that gate exists so the
    live record never rewrites a past session, but a counterfactual replay must
    run the whole rule set (intraday stops + its own close targets) or the two
    arms cannot differ.
    """
    acc = _production_account(cfg, "D_5W")
    if arm == "candidate":
        acc.update({k: v for k, v in dict(spec.get("params", {})).items()})
    acc["name"] = f"D_5W_{arm.upper()}"
    acc["pb_live_intraday_from"] = ""
    acc.pop("pb_preclose_account", None)
    return acc


def _arm_paths(cfg, spec: dict) -> dict[str, Path]:
    """Ledger + meta path per arm (candidate keeps the path from the config)."""
    cand = ROOT / str(spec.get("ledger",
                               f"outputs/forward/candidate_{DEFAULT_RULE}/ledger.sqlite"))
    inc = ROOT / "outputs" / "forward" / "incumbent_replay" / "ledger.sqlite"
    return {"candidate": cand, "incumbent": inc}


def _pit_max_bar(cfg) -> str | None:
    """Newest price bar in the PIT store (cheap aggregate; no full snapshot)."""
    url = cfg.get("data.pit_database_url")
    if not url:
        return None
    try:
        from src.data.point_in_time_loader import from_url

        store = from_url(url)
        try:
            bar = store.max_valid_from("price")
        finally:
            if hasattr(store, "close"):
                store.close()
        return None if bar is None or pd.isna(bar) else str(pd.Timestamp(bar).date())
    except Exception as exc:  # noqa: BLE001 — the caller treats None as "cannot advance"
        print(f"WARNING: PIT max bar unavailable ({exc})", file=sys.stderr)
        return None


def _last_advanced(ledger: Path) -> str | None:
    from src.paper.ledger import PaperLedger

    if not ledger.is_file():
        return None
    led = PaperLedger(str(ledger))
    try:
        return led.last_date()
    finally:
        led.close()


def cmd_run(args) -> int:
    cfg = _cfg()
    spec = _spec(cfg, args.rule)
    prod_ledger = _ledger_paths(cfg, spec)["production"]
    if not prod_ledger.is_file():
        print(f"ERROR: production ledger missing: {prod_ledger}", file=sys.stderr)
        return 1

    # ---- cheap guards: never build a market just to learn there is nothing ----
    today = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    if today.weekday() >= 5 and not args.date:
        print(f"[fwd-cand] {today.date()} is a weekend — nothing to advance", flush=True)
        return 0
    target_end = _pit_max_bar(cfg)
    if not target_end:
        print("[fwd-cand] cannot read the newest PIT bar — skipping (no status written)",
              flush=True)
        return 0
    if pd.Timestamp(target_end) > today:
        target_end = str(today.date())
    if not args.fresh and not args.start:
        lags = {arm: _last_advanced(p) for arm, p in _arm_paths(cfg, spec).items()}
        if all(v is not None and pd.Timestamp(v) >= pd.Timestamp(target_end)
               for v in lags.values()):
            # a holiday (or an already-advanced day) lands here: no bar beyond what
            # the arms already have, so there is nothing to do and NOTHING is written
            print(f"[fwd-cand] both arms already advanced through {target_end} "
                  f"(newest bar) — no-op", flush=True)
            return 0

    from src.autopilot.state import ControlState
    from src.cli import _build_market_for_paper, _shadow_cycle
    from src.paper.ledger import clone_ledger_before
    from src.paper.shadow import resolve_shadow_universe

    accounts = {arm: _replay_account(cfg, spec, arm) for arm in ("incumbent", "candidate")}
    symbols = resolve_shadow_universe(cfg, accounts["incumbent"].get("universe"))
    paths = _arm_paths(cfg, spec)
    state_path = str(ROOT / str(cfg.get("autopilot.state_file", "outputs/autopilot_state.json"))
                     ).replace(".json", "_D_5W.json")
    control = ControlState.load(state_path)

    # window start: explicit, else the pre-registration window, else the seed day
    start = args.start
    if not start:
        try:
            from src.forward.prereg import list_preregistrations

            recs = [r for r in list_preregistrations(ROOT / "outputs/forward/prereg")
                    if str((r.get("trials") or {}).get("family")) == "d_stop_width"]
            win = (recs[0].get("scope") or {}).get("window") if recs else None
            start = str(win[0]) if win else target_end
        except Exception:  # noqa: BLE001
            start = target_end
    start = str(pd.Timestamp(start).date())

    # ---- seed both arms from the SAME production state, once -----------------
    for arm, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = path.parent / "arm_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        if args.fresh or not path.is_file():
            seed = clone_ledger_before(prod_ledger, path, start)
            # ``seed_cutoff`` is the last day whose history is SHARED between the
            # arms (a copy of the production ledger), not the exclusive boundary:
            # the paired window starts the day after it. Storing the boundary here
            # would push the window one day forward and silently skip a session.
            meta = {"arm": arm, "rule_id": args.rule, "seeded_from": str(prod_ledger),
                    "seed_cutoff": seed["last_date"], "seed_boundary": seed["cutoff"],
                    "seed_days": seed["days"],
                    "params": dict(accounts[arm]), "seeded_at":
                        pd.Timestamp.now().isoformat(timespec="seconds")}
            print(f"[fwd-cand] arm={arm} seeded from production (state < {start}): "
                  f"days={seed['days']} positions={seed['positions']} fills={seed['fills']}",
                  flush=True)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                                 encoding="utf-8")
        elif not meta.get("seed_cutoff"):
            # a ledger that predates arm_meta.json (the first, degenerate version):
            # everything it holds up to its last date is shared history — both arms
            # must agree on that boundary or the paired window would include days
            # that cannot differ
            prev = _last_advanced(path)
            meta = {"arm": arm, "rule_id": args.rule, "seeded_from": "existing ledger",
                    "seed_cutoff": prev, "inferred": True,
                    "params": dict(accounts[arm]), "seeded_at":
                        pd.Timestamp.now().isoformat(timespec="seconds")}
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                                 encoding="utf-8")
            print(f"[fwd-cand] arm={arm}: inferred seed_cutoff={prev} from the existing ledger",
                  flush=True)

    # ---- cheap re-check AFTER seeding --------------------------------------
    # The guard above runs before the arms exist, so it cannot see the freshly
    # seeded state: on 2026-09-10 18:28 the new candidate arm was seeded from
    # production (state < 2026-09-11) and every arm already stood at the newest
    # bar (2026-09-10 = ``target_end``), yet the run still built the full ~10 GB
    # market slice just to print 「nothing to advance」. Since
    # ``end = min(target_end, panel.max()) <= target_end``, an arm whose last
    # advanced day is already >= ``target_end`` can be proven to have nothing to
    # do WITHOUT the panel — so prove it here and skip the build entirely.
    # ``None`` (no ledger yet) means the proof does not hold → fall through.
    lags = {arm: _last_advanced(p) for arm, p in paths.items()}
    if all(v is not None and pd.Timestamp(v) >= pd.Timestamp(target_end)
           for v in lags.values()):
        print(f"[fwd-cand] all arms already advanced through {target_end} after seeding "
              f"(window starts {start}) — no market build, nothing written", flush=True)
        return 0

    # ---- one market build, shared by both arms ------------------------------
    print(f"[fwd-cand] building the market slice once for both arms …", flush=True)
    market = _build_market_for_paper(cfg, symbols, start, None, seed=1)
    panel = market.price_panel.index
    end = min(pd.Timestamp(target_end), pd.Timestamp(panel.max()))
    first_missing = min(
        (pd.Timestamp(_last_advanced(p) or "1900-01-01") + pd.Timedelta(days=1))
        for p in paths.values()
    )
    run_start = max(pd.Timestamp(start), first_missing)
    if run_start > end:
        print(f"[fwd-cand] nothing to advance (next {run_start.date()} > newest bar {end.date()})",
              flush=True)
        return 0

    statuses: dict[str, dict] = {}
    for arm, path in paths.items():
        print(f"[fwd-cand] arm={arm} {run_start.date()}..{end.date()} ledger={path}", flush=True)
        status, _ = _shadow_cycle(
            cfg, symbols, str(run_start.date()), str(end.date()), 1, skip_refresh=True,
            control_scale=control.gross_scale, account=accounts[arm],
            ledger_override=str(path), write_artifacts=False, market_override=market,
        )
        eq = status.get("equity", {})
        last = str(status.get("last_trading_date") or "")
        if not last or eq.get("latest") in (None, 0):
            print(f"[fwd-cand] arm={arm} produced no usable day — status NOT written",
                  file=sys.stderr, flush=True)
            return 1
        statuses[arm] = status
        status_path = (ROOT / str(spec.get("status_json",
                                           f"outputs/forward/candidate_{args.rule}/status.json"))
                       if arm == "candidate"
                       else path.parent / "status.json")
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str),
                               encoding="utf-8")
        meta_path = path.parent / "arm_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        meta.update({"last_run_date": last,
                     "last_run_at": pd.Timestamp.now().isoformat(timespec="seconds")})
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                             encoding="utf-8")
        print(f"[fwd-cand] arm={arm} equity={eq.get('latest')} cum={eq.get('total_return')} "
              f"sharpe={eq.get('sharpe')} fills={eq.get('n_fills')} → {status_path}", flush=True)
    return 0


def _ledger_paths(cfg, spec: dict) -> dict[str, Path]:
    prod = ROOT / str(cfg.get("shadow.ledger_db", "outputs/shadow_ledger.sqlite")).replace(
        ".sqlite", "_D_5W.sqlite")
    out = _arm_paths(cfg, spec)
    out["production"] = prod
    return out


def _returns(path: Path) -> pd.Series:
    from src.paper.ledger import PaperLedger

    led = PaperLedger(str(path))
    try:
        eq = pd.Series(led.equity_curve()).astype(float).sort_index()
    finally:
        led.close()
    eq.index = pd.to_datetime(eq.index)
    return eq.pct_change().dropna()


def _arm_meta(path: Path) -> dict:
    meta_path = path.parent / "arm_meta.json"
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def cmd_report(args) -> int:
    cfg = _cfg()
    spec = _spec(cfg, args.rule)
    rule = dict(spec.get("switch_rule", {}) or {})
    paths = _arm_paths(cfg, spec)
    for arm, p in paths.items():
        if not p.is_file():
            print(f"ERROR: {arm} ledger missing: {p} "
                  "(run `python scripts/forward_candidate.py run` first)", file=sys.stderr)
            return 1

    inc = _returns(paths["incumbent"])
    cand = _returns(paths["candidate"])
    # only days both arms advanced INDEPENDENTLY count: the seeded prefix is a
    # byte-copy of production history in both arms and cannot differ
    seed_cutoffs = [_arm_meta(p).get("seed_cutoff") for p in paths.values()]
    seed_cutoffs = [c for c in seed_cutoffs if c]
    independent_from = (
        str((pd.Timestamp(max(seed_cutoffs)) + pd.Timedelta(days=1)).date())
        if seed_cutoffs else None
    )
    cut = None
    if args.start:
        cut = args.start
    if independent_from:
        cut = max([c for c in (cut, independent_from) if c])
    if cut:
        inc = inc[inc.index >= pd.Timestamp(cut)]
        cand = cand[cand.index >= pd.Timestamp(cut)]

    out = paired_comparison(
        inc, cand,
        window_days=int(args.window_days or rule.get("window_days", 120)),
        t_min=float(rule.get("t_min", 1.5)),
        diff_gt=float(rule.get("paired_diff_gt", 0.0)),
    )
    out.update({
        "rule_id": args.rule,
        "arms": {
            "incumbent": {"label": "flat 3.5% (pb_stop_lo = pb_stop_hi = 0.035)",
                          "ledger": str(paths["incumbent"])},
            "candidate": {"label": "ATR 1.0 clipped to [2.5%, 4.0%]",
                          "params": dict(spec.get("params", {})),
                          "ledger": str(paths["candidate"])},
        },
        "seed_cutoffs": seed_cutoffs or None,
        "paired_from": cut,
        "paired_note": (
            "paired days are those BOTH arms advanced independently; the seeded "
            "prefix is a copy of production history in both arms and cannot differ"
        ),
        "switch_rule": rule,
        "counts_as_new_trial": bool(rule.get("counts_as_new_trial", True)),
        "may_never_separate": bool(rule.get("may_never_separate", True)),
        "counts_as_new_trial_note": (
            "a switch would be a NEW trial: re-run scripts/prereg.py with "
            "trials.this_trial+1 before adopting the candidate"
        ),
    })
    win_start = str(inc.index.min().date()) if len(inc) else ""
    win_end = str(inc.index.max().date()) if len(inc) else ""
    if not win_end:
        # too few paired days to form a single return yet (day 1 of the window):
        # fall back to the arm ledger's own last date so the artifact is still
        # stampable — an empty window is reported, not crashed on
        win_end = str(pd.Timestamp(_last_advanced(paths["candidate"]) or pd.Timestamp.today()
                                   ).date())
        win_start = win_start or win_end
    artifact = stamp_artifact(
        {"paired": out},
        window={"start": win_start, "end": win_end},
        convention=CONVENTION,
        data_as_of=win_end,
    )
    out_path = ROOT / "outputs" / "forward" / f"paired_{args.rule}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    if args.json:
        print(json.dumps(artifact, ensure_ascii=False, indent=2, default=str))
    p = artifact["paired"]
    print(f"[fwd-cand] {args.rule} vs flat 3.5%: n={p['n_days']} ready={p['ready']} "
          f"corr={p['corr']} mean_diff={p['mean_diff_pp']}pp/day "
          f"cum_diff={p['cum_diff_pp']}pp t={p['t_stat']} → {p['verdict'].upper()}")
    if not p.get("ready"):
        print(f"[fwd-cand] {p.get('reason', 'not enough paired days yet')} — HOLD "
              f"({p['n_days']}/{p['window_days']} paired days)")
    if p.get("days_needed_for_t"):
        print(f"[fwd-cand] at the observed effect size, t>{p['t_min']} needs ~"
              f"{p['days_needed_for_t']} paired days")
    print(f"[fwd-cand] → {out_path}")
    return 0


def cmd_daily(args) -> int:
    rc = cmd_run(args)
    if rc != 0:
        return rc
    return cmd_report(args)


def main() -> int:
    ap = argparse.ArgumentParser(description="Forward candidate arms + paired report")
    ap.add_argument("--rule", default=DEFAULT_RULE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="advance BOTH arms by every missing trading day")
    p.add_argument("--date", default=None, help="do not look past this date (default: today)")
    p.add_argument("--start", default=None,
                   help="window start / seed day (default: the pre-registered window)")
    p.add_argument("--fresh", action="store_true",
                   help="re-seed BOTH arms from the production state before --start "
                        "(discards the days they had advanced independently)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="paired comparison between the two arms")
    p.add_argument("--window-days", type=int, default=None)
    p.add_argument("--start", default=None, help="ignore days before this date")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("daily", help="run + report (the daily scheduler entry point)")
    p.add_argument("--date", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--window-days", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_daily)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
