"""Pre-registration CLI — freeze a rule before its forward window opens.

    python scripts/prereg.py template --rule-id d_forward_2026h2 > /tmp/rule.json
    # ... fill scope / decision / stopping / trials ...
    python scripts/prereg.py new --file /tmp/rule.json
    python scripts/prereg.py list
    python scripts/prereg.py verify                 # every record in the dir
    python scripts/prereg.py verify --require-commit

The record is stamped with ``frozen_at`` (now), ``code_commit`` (HEAD) and
``config_sha256`` (the merged config, so a threshold change is visible), then
written append-only to ``outputs/forward/prereg/``. Changing a rule requires a
new version naming the record it ``supersedes`` — see ``docs/FORWARD_PROTOCOL.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.forward.prereg import (  # noqa: E402
    DEFAULT_DIR,
    PREREG_FIELDS,
    PreregError,
    list_preregistrations,
    new_record,
    record_sha256,
    verify_preregistration,
    write_preregistration,
)
from src.provenance import code_fingerprint, git_commit, sha256_of  # noqa: E402

import pandas as pd  # noqa: E402


def _template(rule_id: str) -> dict:
    return {
        "rule_id": rule_id,
        "version": 1,
        "scope": {
            "account": "D_5W",
            "universe": "hs300_500",
            "window": ["YYYY-MM-DD", "YYYY-MM-DD"],
            "params": {},
            "what_changes": "describe the single rule under test",
        },
        "decision": {
            "hard": [
                "tracking_error_daily_pp <= 0.2 and sign_bias_p >= 0.05",
                "|cost fee deviation| <= 20% and |slippage deviation| <= 20%",
                "violations (T+1 / lot / tick / limit / suspension) == 0",
                "availability >= 99%, data freshness <= 1 day, symbol minute coverage >= 95%",
            ],
            "soft": ["sharpe", "max_drawdown", "excess_return"],
            "verdict": "PASS only when every hard gate passes; soft metrics are recorded, never gate",
        },
        "stopping": {
            "window_days": 120,
            "kill": "any hard gate failing on 3 consecutive days → stop and investigate the pipeline",
            "may_never_separate": True,
        },
        "trials": {"family": "d_forward", "prior_trials": 0, "this_trial": 1,
                   "notes": "count every parameter/rule change as a new trial"},
        "notes": "",
    }


def _config_sha() -> str:
    """Fingerprint of the POLICY surface the freeze binds to (see prereg.policy_payload)."""
    try:
        from src.forward.prereg import policy_fingerprint

        return policy_fingerprint(load_config())
    except Exception as exc:  # noqa: BLE001 — provenance must not break the CLI
        print(f"WARNING: cannot fingerprint the policy ({exc})", file=sys.stderr)
        return "unknown"


def cmd_template(args) -> int:
    print(json.dumps(_template(args.rule_id), ensure_ascii=False, indent=2))
    return 0


def cmd_new(args) -> int:
    try:
        raw = json.loads(Path(args.file).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read {args.file}: {exc}", file=sys.stderr)
        return 2
    # ``frozen_at`` is stamped by the CLI (now) unless explicitly supplied — a
    # record can never be created with a future timestamp, and back-dating one
    # requires saying so out loud (--frozen-at) because that is only legitimate
    # when reconstructing a record that was frozen earlier elsewhere.
    missing = [f for f in PREREG_FIELDS if f != "frozen_at" and f not in raw]
    if missing:
        print(f"ERROR: template is missing {missing}", file=sys.stderr)
        return 2
    frozen_at = args.frozen_at or raw.get("frozen_at")
    if frozen_at and pd.Timestamp(frozen_at) > pd.Timestamp.now() + pd.Timedelta(minutes=5):
        print(f"ERROR: frozen_at {frozen_at} is in the future", file=sys.stderr)
        return 2
    rec = new_record(
        rule_id=str(raw["rule_id"]),
        scope=raw["scope"],
        decision=raw["decision"],
        stopping=raw["stopping"],
        trials=raw["trials"],
        version=int(raw.get("version", 1)),
        supersedes=raw.get("supersedes"),
        notes=str(raw.get("notes", "")),
        frozen_at=frozen_at,
        code_commit=args.code_commit or git_commit(ROOT),
        policy_sha256=_config_sha(),
        code_fingerprint=code_fingerprint(ROOT),
    )
    try:
        path = write_preregistration(rec, dir=args.dir or DEFAULT_DIR, force=args.force)
    except PreregError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[prereg] frozen {rec['rule_id']} v{rec['version']} at {rec['frozen_at']}")
    print(f"[prereg] commit={rec['code_commit'][:12]} "
          f"code={str(rec.get('code_fingerprint'))[:12]}… "
          f"policy={str(rec.get('policy_sha256'))[:12]}…")
    print(f"[prereg] record_sha256={rec['record_sha256']}")
    print(f"[prereg] → {path}")
    return 0


def cmd_verify(args) -> int:
    paths = [Path(p) for p in args.paths] or sorted((args.dir or DEFAULT_DIR).glob("prereg_*.json"))
    if not paths:
        # An empty record directory is NOT success: the whole forward protocol is
        # built on a frozen record, so "nothing to verify" means the gate has no
        # lock to bind to (a fresh clone used to exit 0 here, which reads as OK).
        print("[prereg] NO RECORDS FOUND — nothing is frozen; freeze one with "
              "`scripts/prereg.py new` before evaluating a forward window", file=sys.stderr)
        return 1
    bad = 0
    for p in paths:
        try:
            rec = verify_preregistration(
                p, require_commit=args.require_commit, repo_root=ROOT,
            )
        except PreregError as exc:
            bad += 1
            print(f"  [FAIL] {p.name}: {exc}")
            continue
        trials = rec.get("trials", {})
        print(f"  [PASS] {p.name}  frozen={rec['frozen_at']}  "
              f"family={trials.get('family')} trial={trials.get('this_trial')}  "
              f"sha256={str(rec['record_sha256'])[:12]}…")
    print(f"[prereg] {len(paths) - bad}/{len(paths)} verified")
    return 1 if bad else 0


def cmd_list(args) -> int:
    recs = list_preregistrations(args.dir or DEFAULT_DIR)
    if not recs:
        print("[prereg] no records", file=sys.stderr)
        return 0
    for rec in recs:
        if rec.get("error"):
            print(f"  [UNREADABLE] {rec['_path']}")
            continue
        trials = rec.get("trials") or {}
        scope = rec.get("scope") or {}
        win = scope.get("window")
        print(f"  {rec.get('rule_id')} v{rec.get('version', 1)}  frozen={rec.get('frozen_at')}  "
              f"window={win}  family={trials.get('family')} trial={trials.get('this_trial')}")
    if args.json:
        print(json.dumps(recs, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_check(args) -> int:
    """Re-hash a record file and print its digest (for cross-checking a copy)."""
    p = Path(args.path)
    rec = json.loads(p.read_text(encoding="utf-8"))
    print(f"stored : {rec.get('record_sha256')}")
    print(f"computed: {record_sha256(rec)}")
    return 0 if rec.get("record_sha256") == record_sha256(rec) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-registration records (append-only)")
    ap.add_argument("--dir", default=None, help=f"record directory (default {DEFAULT_DIR})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("template", help="print a fillable template")
    p.add_argument("--rule-id", required=True)
    p.set_defaults(func=cmd_template)

    p = sub.add_parser("new", help="freeze a record from a JSON file")
    p.add_argument("--file", required=True)
    p.add_argument("--frozen-at", default=None,
                   help="override the freeze timestamp (only to reconstruct a record "
                        "frozen earlier; a future value is refused)")
    p.add_argument("--code-commit", default=None, help="override HEAD (default: git HEAD)")
    p.add_argument("--force", action="store_true", help="repair an identical rewrite only")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("verify", help="verify records (hash, fields, frozen_at, commit)")
    p.add_argument("paths", nargs="*")
    p.add_argument("--require-commit", action="store_true",
                   help="also require code_commit == current HEAD")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("list", help="list records")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("hash", help="recompute a record file's sha256")
    p.add_argument("path")
    p.set_defaults(func=cmd_check)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
