"""Annotate the shadow book's history with its conventions, and re-issue a comparable series.

    python scripts/shadow_series.py --label 2026h1          # annotate + replay + compare
    python scripts/shadow_series.py --label x --no-replay   # annotation only (fast)

Why (2026-09-10 follow-up ①): the production ledger is a single continuous equity
curve (2026-01-05 → today, +24.58%) that was NOT produced under one rule. Across
that span the deployed configuration changed at least four times — the stop width
(2.5% → 3.5%), the true-range basis fix, the intraday/live execution regime
(2026-09-04) and the cash guard — and, for the whole of 2026 until 2026-09-08, the
book ran on a **301-name cross-section** while claiming 800 (the defect found and
backfilled on 2026-09-10). Quoting "+24.58%" therefore mixes incompatible
conventions, and the mixed number is not comparable with anything.

This script does two things:

1. **annotates** every recorded day with the convention in force at the time —
   read from the REPOSITORY (the newest commit touching code before that day, and
   the D-track parameters inside `configs/master_config.yaml` at that commit), so
   the labels are auditable rather than remembered — and splits the curve into
   contiguous segments of identical convention with per-segment metrics;
2. **re-issues** the series by re-running the same window under the CURRENT
   convention on the CURRENT (corrected, 800-name) data from a flat start, and
   reports recorded vs replay overall and per segment, marking explicitly which
   comparisons are meaningful and which are not.

Artifact: ``outputs/shadow_series_<label>.json`` (stamped with provenance).
Read-only with respect to the production ledger: the replay goes into a scratch
ledger that is deleted afterwards (``write_artifacts=False``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.provenance import git_commit, stamp_artifact  # noqa: E402

CONVENTION = (
    "adjusted-close basis; daily close rebalance at the panel close; intraday "
    "stops on minute bars below 15:00; T+1; no leverage; recorded days annotated "
    "from the repository state in force at the time, replay days from the current "
    "configuration"
)

#: Cross-section size in force per era. This is a MEASURED external fact, not a
#: derivation: on 2026-09-10 the 2026 panel was found to carry 301 symbols/day
#: (`select count(distinct symbol) … between '2026-01-01' and '2026-09-07'`),
#: while 2025 windows carried ~5,150 and the daily ingest widened to the declared
#: 800 names on 2026-09-08. The 499 missing names were backfilled the same
#: evening, so today's DB can no longer reproduce the historical width — hence
#: the table. See docs/D_TRACK_EVIDENCE.md §六 and scripts/backfill_price_gap.py.
PANEL_ERAS: tuple[tuple[str, str | None, int, str], ...] = (
    ("2026-01-01", "2026-09-07", 301,
     "daily incremental ingest covered ~300 names (measured 2026-09-10)"),
    ("2026-09-08", None, 800, "ingest widened to the declared hs300_500 universe"),
)

#: Documented change points for the era the repository does NOT cover: the ledger
#: starts 2026-01-05 while the first commit is 2026-08-08, so seven months of the
#: record cannot be annotated from git. Each entry states what it asserts and
#: WHERE that claim comes from; a day's convention is the latest entry ≤ that day,
#: overlaid by the repo-derived values when a commit exists.
DOCUMENTED_ERAS: tuple[dict, ...] = (
    {
        "from": "2026-01-05",
        "stop": "ATR x1.5 band [2.5%, 4.0%] — effectively flat 2.5%",
        "tr_basis": "collapsed (per-date true range, defect D-8b) so every stop sat on the floor",
        "cash_guard": "none (negative cash possible; audit V-1 found 12 such days)",
        "live_regime": False,
        "source": "docs/D_TRACK_EVIDENCE.md §三 (D-8b/D-8c) + §十 (V-1)",
    },
    {
        "from": "2026-09-04",
        "stop": "ATR x1.5 band [2.5%, 4.0%] — effectively flat 2.5%",
        "tr_basis": "collapsed (defect D-8b)",
        "cash_guard": "none",
        "live_regime": True,
        "source": "configs/master_config.yaml pb_live_intraday_from + docs/D_TRACK_EVIDENCE.md §四",
    },
    {
        "from": "2026-09-08",
        "stop": "ATR x1.5 band [2.5%, 4.0%] (true range now 2-D and correct)",
        "tr_basis": "fixed (per-symbol true range)",
        "cash_guard": "sells-first + affordable-share clip",
        "live_regime": True,
        "source": "docs/D_TRACK_EVIDENCE.md §三/§十 (2026-09-08 audit fixes, defect D-8/V-1)",
    },
    {
        "from": "2026-09-09",
        "stop": "flat 3.5% (D-8c selection; stop_lo == stop_hi)",
        "tr_basis": "fixed",
        "cash_guard": "sells-first + affordable-share clip",
        "live_regime": True,
        "source": "configs/master_config.yaml pb_stop_lo/hi + docs/D_TRACK_EVIDENCE.md §六",
    },
)


def documented_era(day: str | pd.Timestamp) -> dict:
    """Latest documented change point at or before ``day`` (empty dict if none)."""
    d = pd.Timestamp(day)
    applicable = [e for e in DOCUMENTED_ERAS if d >= pd.Timestamp(e["from"])]
    return dict(applicable[-1]) if applicable else {}


def panel_width(day: str | pd.Timestamp) -> int | None:
    """Cross-section size that was IN FORCE on ``day`` (documented, see PANEL_ERAS)."""
    d = pd.Timestamp(day)
    for start, end, n, _note in PANEL_ERAS:
        if d >= pd.Timestamp(start) and (end is None or d <= pd.Timestamp(end)):
            return n
    return None


def _git(*args: str) -> str:
    # encoding pinned: the historical configs carry Chinese comments and the
    # locale default (GBK on this host) cannot decode them — a silent failure here
    # would erase the whole annotation.
    out = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    return (out.stdout or "").strip() if out.returncode == 0 else ""


def commit_in_force(day: str | pd.Timestamp) -> str:
    """Newest commit touching the code paths STRICTLY before ``day`` (annotation)."""
    stamp = f"{pd.Timestamp(day).date()}T00:00:00"
    sha = _git("rev-list", "-1", f"--before={stamp}", "HEAD", "--",
               "src", "scripts", "configs")
    return sha or "unknown"


def _config_at(commit: str) -> dict:
    """The D-track parameters + live regime recorded in ``configs/master_config.yaml``."""
    if not commit or commit == "unknown":
        return {}
    text = _git("show", f"{commit}:configs/master_config.yaml")
    if not text:
        return {}
    try:
        import yaml

        cfg = yaml.safe_load(text) or {}
    except Exception:  # noqa: BLE001 — an unparseable historical config is reported as unknown
        return {}
    accounts = ((cfg.get("shadow") or {}).get("accounts") or [])
    acc = next((a for a in accounts if str(a.get("name")) == "D_5W"), {}) or {}
    live = cfg.get("live") or {}
    return {
        "stop_lo": acc.get("pb_stop_lo"),
        "stop_hi": acc.get("pb_stop_hi"),
        "atr_mult": acc.get("pb_atr_mult"),
        "stop_trigger": acc.get("pb_stop_trigger"),
        "stop_open_minutes": acc.get("pb_stop_open_minutes"),
        "tail_vol_max": acc.get("pb_tail_vol_max"),
        "full_invest": acc.get("pb_full_invest"),
        "k": acc.get("pb_k"),
        "universe": acc.get("universe"),
        "live_from": acc.get("pb_live_intraday_from"),
        "live_enabled": live.get("enabled"),
    }


def convention_for(day: str | pd.Timestamp, *, cache: dict | None = None) -> dict:
    """The convention in force on ``day`` — documented change log + repo state.

    The ledger predates the repository (2026-01-05 vs the first commit on
    2026-08-08), so neither source alone covers it: the documented eras carry the
    pre-repo facts, the repo supplies the exact parameters whenever a commit
    exists, and ``sources`` records which was used so a reader can audit it.
    """
    cache = cache if cache is not None else {}
    day_s = str(pd.Timestamp(day).date())
    if day_s not in cache:
        era = documented_era(day_s)
        sha = commit_in_force(day_s)
        params = _config_at(sha)
        live_from = str(params.get("live_from") or "")
        repo_live = bool(live_from and day_s >= live_from)
        stop = "unknown"
        if params.get("stop_lo") is not None:
            if params.get("stop_lo") == params.get("stop_hi"):
                stop = f"flat {float(params['stop_lo']):.1%}"
            else:
                stop = f"ATR x{params.get('atr_mult')} [{params.get('stop_lo')}, {params.get('stop_hi')}]"
        sources = []
        if era:
            sources.append(f"documented:{era.get('from')} ({era.get('source')})")
        if sha and sha != "unknown":
            sources.append(f"repo:{sha[:12]}")
        cache[day_s] = {
            "date": day_s,
            "code_commit": sha[:12] if sha else "unknown",
            "panel_symbols": panel_width(day_s),
            "stop": stop if stop != "unknown" else era.get("stop", "unknown"),
            "tr_basis": era.get("tr_basis", "fixed" if sha and sha != "unknown" else "unknown"),
            "cash_guard": era.get("cash_guard", "sells-first + affordable-share clip"
                                  if sha and sha != "unknown" else "unknown"),
            "params": params,
            "live_regime": repo_live or bool(era.get("live_regime")),
            "sources": sources,
        }
    return cache[day_s]


def _signature(conv: dict) -> str:
    """Segment key: everything about the convention that makes two days comparable."""
    p = conv.get("params") or {}
    parts = [
        str(p.get("stop_lo")), str(p.get("stop_hi")), str(p.get("atr_mult")),
        str(p.get("stop_trigger")), str(p.get("stop_open_minutes")),
        str(p.get("tail_vol_max")), str(p.get("full_invest")), str(p.get("k")),
        str(p.get("universe")),
        "live" if conv.get("live_regime") else "replay",
        str(conv.get("panel_symbols")),
        str(conv.get("stop")), str(conv.get("tr_basis")), str(conv.get("cash_guard")),
    ]
    return "|".join(parts)


def segment_days(rows: list[dict]) -> list[dict]:
    """Split annotated day rows into contiguous segments of identical convention.

    Pure function (no I/O) so the segmentation is unit-testable: consecutive rows
    with the same :func:`_signature` form one segment; a change starts a new one.
    """
    segments: list[dict] = []
    for row in rows:
        sig = _signature(row)
        if segments and segments[-1]["signature"] == sig:
            segments[-1]["days"].append(row["date"])
            segments[-1]["rows"].append(row)
        else:
            segments.append({"signature": sig, "days": [row["date"]], "rows": [row],
                             "convention": {k: v for k, v in row.items()
                                            if k not in ("equity",)}})
    return segments


def _segment_metrics(eq: pd.Series, days: list[str], fills: pd.DataFrame) -> dict:
    """Return / max drawdown / fills over one segment."""
    idx = [d for d in days if pd.Timestamp(d) in eq.index]
    if not idx:
        return {"n_days": 0}
    sub = eq.loc[[pd.Timestamp(d) for d in idx]]
    ret = float(sub.iloc[-1] / sub.iloc[0] - 1.0) if len(sub) > 1 else 0.0
    dd = float((sub / sub.cummax() - 1.0).min()) if len(sub) else 0.0
    n_fills = 0
    if len(fills):
        f_days = pd.to_datetime(fills["date"]).dt.date.astype(str)
        n_fills = int(f_days.isin(set(idx)).sum())
    return {"n_days": len(sub), "return": round(ret, 4),
            "max_drawdown": round(dd, 4), "n_fills": n_fills,
            "first_equity": round(float(sub.iloc[0]), 2),
            "last_equity": round(float(sub.iloc[-1]), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Annotate and re-issue the shadow series")
    ap.add_argument("--label", default="latest")
    ap.add_argument("--account", default="D_5W")
    ap.add_argument("--no-replay", action="store_true",
                    help="skip the current-convention replay (annotation only)")
    ap.add_argument("--replay-start", default=None,
                    help="replay window start (default: the ledger's first day); use it to "
                         "reproduce another harness's window, e.g. 2026-01-01")
    ap.add_argument("--replay-end", default=None, help="replay window end (default: the last day)")
    ap.add_argument("--keep-ledger", action="store_true",
                    help="keep the scratch replay ledger for inspection (default: delete)")
    ap.add_argument("--replay-label", default="",
                    help="suffix for the scratch ledger name (compare two runs side by side)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from src.config import load_config
    from src.paper.ledger import PaperLedger

    cfg = load_config()
    ledger_path = (ROOT / str(cfg.get("shadow.ledger_db", "outputs/shadow_ledger.sqlite"))
                   ).with_name(f"shadow_ledger_{args.account}.sqlite")
    if not ledger_path.is_file():
        print(f"ERROR: production ledger missing: {ledger_path}", file=sys.stderr)
        return 2

    led = PaperLedger(str(ledger_path))
    try:
        states = led.daily_states()
        fills = led.fills()
        eq = pd.Series(led.equity_curve()).astype(float)
    finally:
        led.close()
    eq.index = pd.to_datetime(eq.index)
    states = states.copy()
    states["date"] = pd.to_datetime(states["date"])

    cache: dict = {}
    rows = [
        {**convention_for(d, cache=cache), "equity": float(e)}
        for d, e in zip(states["date"], states["equity"], strict=False)
    ]
    segments = segment_days(rows)
    for seg in segments:
        seg["metrics"] = _segment_metrics(eq, seg["days"], fills)
        seg.pop("rows")
    start = str(states["date"].min().date())
    end = str(states["date"].max().date())
    print(f"[series] recorded: {len(rows)} days [{start}, {end}] "
          f"{len(segments)} convention segment(s)", flush=True)
    for i, seg in enumerate(segments, 1):
        c, m = seg["convention"], seg["metrics"]
        print(f"  #{i} {seg['days'][0]}→{seg['days'][-1]} ({m['n_days']}d) "
              f"stop={c['stop']} panel={c['panel_symbols']} "
              f"regime={'live' if c['live_regime'] else 'replay'} "
              f"ret={m['return']:+.2%} fills={m['n_fills']} commit={c['code_commit']}",
              flush=True)

    overall_recorded = None
    if len(eq) > 1:
        overall_recorded = {
            "return": round(float(eq.iloc[-1] / eq.iloc[0] - 1.0), 4),
            "max_drawdown": round(float((eq / eq.cummax() - 1.0).min()), 4),
            "n_days": int(len(eq)), "n_fills": int(len(fills)),
            "note": ("NOT a single-convention number: it spans "
                     f"{len(segments)} convention segment(s) "
                     f"(see `segments`); quote a segment, or the replay below"),
        }

    replay_block: dict = {"performed": False}
    if not args.no_replay:
        from src.cli import _build_market_for_paper, _shadow_cycle
        from src.paper.shadow import resolve_shadow_universe

        account = next((a for a in (cfg.get("shadow.accounts") or [])
                        if str(a.get("name")) == args.account), None)
        if account is None:
            print(f"ERROR: account {args.account!r} not in shadow.accounts", file=sys.stderr)
            return 2
        # A COUNTERFACTUAL under one convention: clear the live gate and never read
        # production's 14:50 order list. Otherwise, on live dates the runner
        # executes production's submitted list instead of computing its own targets
        # AND the intraday sweep is gated off — the replay would then inherit the
        # live book's positions and simply mark them to market, which is not "what
        # the current rule would have done" (the same degeneracy the candidate arms
        # hit on 2026-09-10, where both ledgers came out byte-identical).
        account = dict(account)
        account["pb_live_intraday_from"] = ""
        account.pop("pb_preclose_account", None)
        symbols = resolve_shadow_universe(cfg, account.get("universe"))
        scratch = ROOT / "outputs" / (
            f"_series_replay_{args.account}{('_' + args.replay_label) if args.replay_label else ''}"
            ".sqlite")
        scratch.unlink(missing_ok=True)
        r_start = str(pd.Timestamp(args.replay_start).date()) if args.replay_start else start
        r_end = str(pd.Timestamp(args.replay_end).date()) if args.replay_end else end
        print(f"[series] replaying [{r_start}, {r_end}] under the CURRENT convention "
              f"(800-name pool) …", flush=True)
        market = _build_market_for_paper(cfg, symbols, r_start, None, seed=1)
        _shadow_cycle(cfg, symbols, r_start, r_end, 1, skip_refresh=True, control_scale=None,
                      account=account, ledger_override=str(scratch),
                      write_artifacts=False, market_override=market)
        rled = PaperLedger(str(scratch))
        try:
            req = pd.Series(rled.equity_curve()).astype(float)
            rfills = rled.fills()
        finally:
            rled.close()
        if not args.keep_ledger:
            scratch.unlink(missing_ok=True)
        else:
            print(f"[series] kept the replay ledger at {scratch}", flush=True)
        req.index = pd.to_datetime(req.index)
        replay_block = {
            "performed": True,
            "window": {"start": r_start, "end": r_end},
            # NOT publishable as "the corrected series" yet (2026-09-10): replaying
            # [2026-01-01, 2026-08-28] reproduces the IS evidence exactly (+5.72%),
            # but replaying [2026-01-05, 2026-09-10] — the ledger's own range —
            # ends at −6.33% and sits in cash from late July. The two runs differ
            # ONLY in the market slice's warmup start (540 calendar days before
            # `start`), i.e. the outcome is slice-start sensitive. Until that is
            # explained, the replay is reported for diagnosis, not for quoting.
            "publishable": False,
            "known_issue": (
                "slice-start sensitivity: start=2026-01-01 → +5.72% at 08-28 "
                "(= outputs/d_oos_is_2026_v5.json), start=2026-01-05 → −3.59% at 08-28. "
                "Same code, same convention, same panel; only the warmup slice moves."
            ),
            "convention": "current configuration, corrected 800-name panel, flat start",
            "metrics": {
                "return": round(float(req.iloc[-1] / req.iloc[0] - 1.0), 4) if len(req) > 1 else None,
                "max_drawdown": round(float((req / req.cummax() - 1.0).min()), 4) if len(req) else None,
                "n_days": int(len(req)), "n_fills": int(len(rfills)),
                "first_equity": round(float(req.iloc[0]), 2) if len(req) else None,
                "last_equity": round(float(req.iloc[-1]), 2) if len(req) else None,
            },
            "equity_curve": [{"date": str(pd.Timestamp(d).date()), "equity": round(float(v), 2)}
                             for d, v in req.items()],
            "comparison_note": (
                "recorded vs replay is an AGGREGATE comparison only: the recorded "
                "curve spans several conventions on a 301-name cross-section, while "
                "the replay applies one convention to the corrected 800-name pool. "
                "They share a start date and a flat start, nothing else."
            ),
        }
        if overall_recorded and replay_block["metrics"]["return"] is not None:
            replay_block["delta_vs_recorded"] = round(
                replay_block["metrics"]["return"] - overall_recorded["return"], 4)
        print(f"[series] replay under the current convention: "
              f"{replay_block['metrics']['return']:+.2%} "
              f"(recorded, mixed conventions: {overall_recorded['return']:+.2%})", flush=True)

    artifact = {
        "label": args.label,
        "account": args.account,
        "window": {"start": start, "end": end},
        "panel_eras": [{"start": s, "end": e, "symbols": n, "note": note}
                       for s, e, n, note in PANEL_ERAS],
        "recorded": {
            "metrics": overall_recorded,
            "equity_curve": [{"date": str(pd.Timestamp(d).date()), "equity": round(float(v), 2)}
                             for d, v in eq.items()],
        },
        "segments": segments,
        "replay_current_convention": replay_block,
        "reading": [
            "A segment is the smallest unit that may be quoted: within a segment the "
            "code commit, the D-track parameters, the execution regime and the "
            "cross-section width are all constant.",
            "The recorded curve's headline return mixes segments and is therefore not "
            "comparable with any single-convention number.",
            "The replay is the corrected series: one convention, corrected panel, flat "
            "start. Compare aggregates, never day-by-day.",
            "The replay is marked publishable=false while the slice-start sensitivity "
            "recorded in `replay_current_convention.known_issue` is unexplained.",
        ],
    }
    artifact = stamp_artifact(artifact, window={"start": start, "end": end},
                              convention=CONVENTION, data_as_of=end)
    out = ROOT / "outputs" / f"shadow_series_{args.label}.json"
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    if args.json:
        print(json.dumps(artifact, ensure_ascii=False, indent=2, default=str))
    print(f"[series] → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
