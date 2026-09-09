"""Forward candidate shadow — ``atr_1p0_25_40`` vs the deployed flat 3.5% stop.

    python scripts/forward_candidate.py run                # advance one day
    python scripts/forward_candidate.py run --date 2026-09-10
    python scripts/forward_candidate.py report             # paired comparison
    python scripts/forward_candidate.py daily              # run + report (PAICC job)
    python scripts/forward_candidate.py report --json

Design (audit item 2, ``docs/FORWARD_PROTOCOL.md`` §2):

* **Isolated state.** The candidate keeps its OWN ledger and status file
  (``outputs/forward/candidate_atr_1p0_25_40/``) — it can never touch the
  production ledger, and the production book can never read its fills.
* **Same regime, one difference.** Data slice, universe, code path, entry rules,
  14:50 order list and the 15:00 auction execution are IDENTICAL to production;
  only the stop width differs (``pb_atr_mult=1.0, pb_stop_lo=0.025,
  pb_stop_hi=0.040``). Any second difference would make the comparison
  uninterpretable.
* **Record only.** Nothing is auto-promoted. The switch rule is pre-registered:
  over the forward window, a paired daily-difference mean > 0 AND paired t > 1.5
  ⇒ switch (which COUNTS AS A NEW TRIAL and must be re-pre-registered); anything
  else ⇒ hold. The two books correlate ≈ 0.84, so the paired difference — not the
  Sharpe difference — is the test statistic.
* **"May never separate" is an accepted outcome.** The candidate is not given a
  deadline and the incumbent is not destabilised to force a decision.

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

DEFAULT_RULE = "atr_1p0_25_40"
CONVENTION = (
    "adjusted-close basis; identical universe/data/entry/exit/execution to the "
    "D_5W production book; only the stop width differs; close orders inherited "
    "from the production 14:50 order list; T+1; no leverage"
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


def _candidate_account(cfg, spec: dict) -> dict:
    """Production account + the single pre-registered difference."""
    acc = _production_account(cfg, "D_5W")
    base_name = acc["name"]
    acc.update({k: v for k, v in dict(spec.get("params", {})).items()})
    acc["name"] = f"{base_name}{spec.get('account_suffix', '_FWD')}"
    # read the PRODUCTION order list so both books execute the same 15:00 auction
    acc["pb_preclose_account"] = base_name
    return acc


def _ledger_paths(cfg, spec: dict) -> tuple[Path, Path]:
    prod = ROOT / str(cfg.get("shadow.ledger_db", "outputs/shadow_ledger.sqlite")).replace(
        ".sqlite", "_D_5W.sqlite")
    cand = ROOT / str(spec.get("ledger", f"outputs/forward/candidate_{DEFAULT_RULE}/ledger.sqlite"))
    return prod, cand


def cmd_run(args) -> int:
    cfg = _cfg()
    spec = _spec(cfg, args.rule)
    account = _candidate_account(cfg, spec)
    prod_ledger, cand_ledger = _ledger_paths(cfg, spec)
    if not prod_ledger.is_file():
        print(f"ERROR: production ledger missing: {prod_ledger}", file=sys.stderr)
        return 1

    from src.autopilot.state import ControlState
    from src.cli import _shadow_cycle
    from src.paper.shadow import resolve_shadow_universe

    symbols = resolve_shadow_universe(cfg, account.get("universe"))
    end = args.date or pd.Timestamp.today().normalize().date().isoformat()
    start = args.start or end
    state_path = str(ROOT / str(cfg.get("autopilot.state_file", "outputs/autopilot_state.json"))
                     ).replace(".json", "_D_5W.json")
    control = ControlState.load(state_path)
    cand_ledger.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fwd-cand] {args.rule} {start}..{end} ledger={cand_ledger}", flush=True)
    status, ledger_out = _shadow_cycle(
        cfg, symbols, start, end, 1, skip_refresh=True, control_scale=control.gross_scale,
        account=account, ledger_override=str(cand_ledger), write_artifacts=False,
    )
    status_path = ROOT / str(spec.get("status_json",
                                      f"outputs/forward/candidate_{args.rule}/status.json"))
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str),
                           encoding="utf-8")
    eq = status.get("equity", {})
    print(f"[fwd-cand] equity={eq.get('equity')} cum={eq.get('total_return')} "
          f"sharpe={eq.get('sharpe')} fills={eq.get('n_fills')} → {status_path}", flush=True)
    return 0


def _returns(path: Path) -> pd.Series:
    from src.paper.ledger import PaperLedger

    led = PaperLedger(str(path))
    try:
        eq = pd.Series(led.equity_curve()).astype(float).sort_index()
    finally:
        led.close()
    eq.index = pd.to_datetime(eq.index)
    return eq.pct_change().dropna()


def cmd_report(args) -> int:
    cfg = _cfg()
    spec = _spec(cfg, args.rule)
    rule = dict(spec.get("switch_rule", {}) or {})
    prod_ledger, cand_ledger = _ledger_paths(cfg, spec)
    for p in (prod_ledger, cand_ledger):
        if not p.is_file():
            print(f"ERROR: ledger missing: {p} "
                  "(run `python scripts/forward_candidate.py run` first)", file=sys.stderr)
            return 1

    prod = _returns(prod_ledger)
    cand = _returns(cand_ledger)
    if args.start:
        prod = prod[prod.index >= pd.Timestamp(args.start)]
        cand = cand[cand.index >= pd.Timestamp(args.start)]
    out = paired_comparison(
        prod, cand,
        window_days=int(args.window_days or rule.get("window_days", 120)),
        t_min=float(rule.get("t_min", 1.5)),
        diff_gt=float(rule.get("paired_diff_gt", 0.0)),
    )
    out.update({
        "rule_id": args.rule,
        "candidate_params": dict(spec.get("params", {})),
        "incumbent": "flat 3.5% (pb_stop_lo = pb_stop_hi = 0.035)",
        "incumbent_ledger": str(prod_ledger),
        "candidate_ledger": str(cand_ledger),
        "switch_rule": rule,
        "counts_as_new_trial": bool(rule.get("counts_as_new_trial", True)),
        "may_never_separate": bool(rule.get("may_never_separate", True)),
        "counts_as_new_trial_note": (
            "a switch would be a NEW trial: re-run scripts/prereg.py with "
            "trials.this_trial+1 before adopting the candidate"
        ),
    })
    win_start = str(prod.index.min().date()) if len(prod) else ""
    win_end = str(prod.index.max().date()) if len(prod) else ""
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
    ap = argparse.ArgumentParser(description="Forward candidate shadow + paired report")
    ap.add_argument("--rule", default=DEFAULT_RULE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="advance the candidate ledger by one day")
    p.add_argument("--date", default=None, help="date to advance (default: today)")
    p.add_argument("--start", default=None, help="window start (default: same as --date)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="paired comparison vs the incumbent")
    p.add_argument("--window-days", type=int, default=None)
    p.add_argument("--start", default=None, help="ignore days before this date")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("daily", help="run + report (the daily scheduler entry point)")
    p.add_argument("--date", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--window-days", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_daily)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
