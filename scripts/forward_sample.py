"""Append one line per forward day to the forward sample (follow-up ③).

    python scripts/forward_sample.py                  # append today (idempotent)
    python scripts/forward_sample.py --date 2026-09-11
    python scripts/forward_sample.py show             # print the accumulated sample
    python scripts/forward_sample.py check            # exit 1 if any day is dirty

Why a separate append-only file: the forward period is the only clean sample this
project will ever have, and "clean" has a precise meaning here —

* the day was produced **under the frozen convention** (policy fingerprint and code
  fingerprint still match the pre-registration), and
* the day's inputs were complete (data fresh, live layer available, minute
  coverage present).

The gate in ``scripts/forward_health.py`` answers "is the pipeline trustworthy
right now" by replaying; this file answers "what did the pipeline actually do,
day by day, under which convention" — and it is the raw material any later
analysis must use instead of the mixed-convention production curve
(``scripts/shadow_series.py`` shows why that curve cannot be quoted).

Each line carries the day's ledger row, the fills by source, the heartbeat
coverage, the panel width, and the convention fingerprints. A day whose
convention does NOT match the frozen one is appended with ``convention_ok:
false`` — recorded, not hidden, and excluded from the clean count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


def sample_path(label: str = "") -> Path:
    name = f"forward_samples{('_' + label) if label else ''}.jsonl"
    return ROOT / "outputs" / "forward" / name


def load_samples(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def heartbeat_coverage(account: str, day: str) -> dict:
    """Decision-window heartbeat coverage for one day (cheap, from the jsonl).

    Uses the gate's own :func:`~src.forward.risk_gate.availability` so the daily
    sample and the weekly gate can never disagree — including the lunch break,
    which is NOT downtime (a single 09:30-15:00 interval reported ~62% for a
    perfectly healthy session until 2026-09-10).
    """
    from src.forward.risk_gate import availability

    path = ROOT / "outputs" / f"live_{account}.jsonl"
    if not path.is_file():
        return {"measured": False, "note": "no heartbeat file"}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ts = json.loads(line).get("ts")
        except ValueError:
            continue
        if ts and str(ts)[:10] == day:
            rows.append({"ts": ts})
    if not rows:
        return {"measured": False, "note": "no heartbeat on this day"}
    out = availability(pd.DataFrame(rows), [pd.Timestamp(day)])
    out.update({"measured": True, "n_ticks": len(rows),
                "first": min(r["ts"] for r in rows), "last": max(r["ts"] for r in rows)})
    return out


def build_sample(cfg, account_name: str, day: str, prereg: dict | None,
                 policy_sha: str, code_fp: str) -> dict:
    """One day's sample row (pure assembly from artifacts; no market build)."""
    from src.data.intraday import load_intraday_frames
    from src.paper.ledger import PaperLedger
    from src.paper.shadow import resolve_shadow_universe

    ledger_path = (ROOT / str(cfg.get("shadow.ledger_db", "outputs/shadow_ledger.sqlite"))
                   ).with_name(f"shadow_ledger_{account_name}.sqlite")
    led = PaperLedger(str(ledger_path))
    try:
        states = led.daily_states().copy()
        fills = led.fills()
    finally:
        led.close()
    states["date"] = pd.to_datetime(states["date"])
    day_ts = pd.Timestamp(day)
    row = states[states["date"] == day_ts]
    prev = states[states["date"] < day_ts].tail(1)
    if row.empty:
        return {"date": day, "recorded": False,
                "note": "the daily loop has not recorded this day yet"}
    r = row.iloc[0]
    daily_return = None
    if not prev.empty and float(prev.iloc[0]["equity"]) > 0:
        daily_return = round(float(r["equity"]) / float(prev.iloc[0]["equity"]) - 1.0, 6)
    day_fills = fills[pd.to_datetime(fills["date"]) == day_ts] if len(fills) else fills
    by_source = (day_fills["source"].fillna("").replace("", "unlabelled").value_counts().to_dict()
                 if len(day_fills) else {})

    status_path = ROOT / "outputs" / f"shadow_status_{account_name}.json"
    skipped_orders: list[dict] = []
    if status_path.is_file():
        try:
            st = json.loads(status_path.read_text(encoding="utf-8"))
            skipped_orders = list((st.get("equity") or {}).get("skipped_orders") or [])
        except ValueError:
            skipped_orders = []

    symbols = resolve_shadow_universe(cfg, cfg.get("shadow.accounts")[0].get("universe") if
                                      cfg.get("shadow.accounts") else None)
    frames = load_intraday_frames(cfg, symbols)
    tail = frames.get("tail_vol")
    panel_symbols = int(tail.shape[1]) if tail is not None and len(tail) else None

    frozen_policy = str((prereg or {}).get("policy_sha256") or "")
    frozen_code = str((prereg or {}).get("code_fingerprint") or "")
    convention_ok = bool(
        prereg and frozen_policy == policy_sha and frozen_code and frozen_code == code_fp
    )
    return {
        "date": day,
        "recorded": True,
        "account": account_name,
        "equity": round(float(r["equity"]), 2),
        "cash": round(float(r["cash"]), 2),
        "gross_exposure": round(float(r.get("gross_exposure", 0.0)), 2),
        "n_positions": int(r.get("n_positions", 0)),
        "n_fills": int(r.get("n_fills", 0)),
        "commission": round(float(r.get("commission", 0.0)), 2),
        "daily_return": daily_return,
        "fills_by_source": {str(k): int(v) for k, v in by_source.items()},
        "skipped_orders": skipped_orders,
        "panel_symbols": panel_symbols,
        "live_heartbeat": heartbeat_coverage(account_name, day),
        "convention": {
            "rule_id": (prereg or {}).get("rule_id"),
            "record_sha256": (prereg or {}).get("record_sha256"),
            "frozen_at": (prereg or {}).get("frozen_at"),
            "policy_sha256": policy_sha,
            "code_fingerprint": code_fp,
            "policy_match": bool(prereg) and frozen_policy == policy_sha,
            "code_match": bool(prereg) and bool(frozen_code) and frozen_code == code_fp,
        },
        #: True only when the day was produced under the FROZEN convention — the
        #: defining property of a clean forward sample.
        "convention_ok": convention_ok,
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }


def _window_start(cfg, account: str) -> str | None:
    """The pre-registered forward window start (rows before it are context, not sample)."""
    try:
        prereg = _active_prereg(ROOT / str(cfg.get("forward.prereg_dir",
                                                    "outputs/forward/prereg")), account)
        win = ((prereg or {}).get("scope") or {}).get("window") or []
        return str(win[0])[:10] if win else None
    except Exception:  # noqa: BLE001 — a missing window must not break the append
        return None


def _in_window(rows: list[dict], start: str | None) -> list[dict]:
    """Forward-sample rows only (a day before the window is context, never evidence)."""
    if not start:
        return list(rows)
    return [r for r in rows if str(r.get("date") or "") >= start]


def cmd_append(args) -> int:
    from src.config import load_config
    from src.forward.prereg import policy_fingerprint, verify_preregistration
    from src.provenance import code_fingerprint

    cfg = load_config()
    day = str(pd.Timestamp(args.date).date()) if args.date else str(pd.Timestamp.today().date())
    account = args.account
    prereg_dir = ROOT / str(cfg.get("forward.prereg_dir", "outputs/forward/prereg"))
    prereg = _active_prereg(prereg_dir, account)
    policy_sha = policy_fingerprint(cfg)
    code_fp = code_fingerprint(ROOT)
    sample = build_sample(cfg, account, day, prereg, policy_sha, code_fp)
    path = sample_path(args.label)
    rows = load_samples(path)
    if any(r.get("date") == day for r in rows):
        print(f"[sample] {day} already present in {path.name} — append-only, nothing written")
        return 0
    if not sample.get("recorded"):
        print(f"[sample] {day}: {sample.get('note')} — nothing appended", flush=True)
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
    start = _window_start(cfg, account)
    in_win = _in_window(rows + [sample], start)
    clean = sum(1 for r in in_win if r.get("convention_ok"))
    print(f"[sample] appended {day}: equity={sample['equity']} "
          f"ret={sample['daily_return']} fills={sample['n_fills']} "
          f"panel={sample['panel_symbols']} convention_ok={sample['convention_ok']} "
          f"→ clean forward day(s) {clean}/{len(in_win)} (window from {start})", flush=True)
    return 0


def _active_prereg(prereg_dir: Path, account: str) -> dict | None:
    from src.forward.prereg import list_preregistrations, verify_preregistration

    fallback = None
    for rec in list_preregistrations(prereg_dir):
        if rec.get("error"):
            continue
        if str((rec.get("scope") or {}).get("account", account)) != account:
            continue
        try:
            verify_preregistration(rec["_path"])
        except Exception:  # noqa: BLE001
            continue
        if str((rec.get("trials") or {}).get("family")) == "d_forward_risk_gate":
            return rec
        fallback = fallback or rec
    return fallback


def cmd_show(args) -> int:
    from src.config import load_config

    cfg = load_config()
    start = _window_start(cfg, args.account)
    path = sample_path(args.label)
    rows = load_samples(path)
    if not rows:
        print(f"[sample] {path} is empty (the forward window has not produced a day yet)")
        return 0
    print(f"{'date':12s} {'equity':>10s} {'ret':>8s} {'fills':>6s} {'panel':>6s} "
          f"{'hb':>6s} {'conv':>5s} {'win':>4s}")
    for r in rows:
        hb = (r.get("live_heartbeat") or {})
        in_win = (not start) or str(r.get("date") or "") >= start
        print(f"{r.get('date'):12s} {r.get('equity', 0):10.2f} "
              f"{(r.get('daily_return') or 0) * 100:7.2f}% {r.get('n_fills', 0):6d} "
              f"{r.get('panel_symbols') or 0:6d} "
              f"{((hb.get('availability') if hb.get('measured') else 0) or 0) * 100:5.0f}% "
              f"{'yes' if r.get('convention_ok') else 'NO':>5s} "
              f"{'yes' if in_win else 'ctx':>4s}")
    in_win = _in_window(rows, start)
    clean = sum(1 for r in in_win if r.get("convention_ok"))
    print(f"[sample] {len(in_win)} forward day(s) (window from {start}), {clean} clean, "
          f"{len(in_win) - clean} dirty · {len(rows) - len(in_win)} pre-window row(s) kept "
          f"as context → {path}")
    return 0


def cmd_check(args) -> int:
    from src.config import load_config

    cfg = load_config()
    start = _window_start(cfg, args.account)
    rows = _in_window(load_samples(sample_path(args.label)), start)
    dirty = [r for r in rows if not r.get("convention_ok")]
    if dirty:
        print(f"[sample] {len(dirty)}/{len(rows)} forward day(s) were NOT produced under the "
              f"frozen convention: {[r.get('date') for r in dirty]}")
        return 1
    today = str(pd.Timestamp.today().date())
    if start and today >= start and not rows:
        # the window is open and no day has been archived: "nothing recorded" is not
        # a pass — the sample is the only clean forward evidence there will ever be
        print(f"[sample] the forward window opened on {start} but NO day has been archived "
              "yet — run scripts/forward_sample.py from the daily job")
        return 1
    print(f"[sample] all {len(rows)} forward day(s) are convention-clean (window from {start})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Append-only forward sample")
    ap.add_argument("--date", default=None)
    ap.add_argument("--account", default="D_5W")
    ap.add_argument("--label", default="")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("show", help="print the accumulated sample")
    p.set_defaults(func=cmd_show)
    p = sub.add_parser("check", help="exit 1 when any day is convention-dirty")
    p.set_defaults(func=cmd_check)
    args = ap.parse_args()
    if getattr(args, "func", None) is not None:
        return int(args.func(args))
    return cmd_append(args)


if __name__ == "__main__":
    raise SystemExit(main())
