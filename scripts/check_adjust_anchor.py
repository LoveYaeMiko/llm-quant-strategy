"""Freeze, capture and drift-check the price-adjustment anchor (defect C3).

The PIT price series is backward-adjusted against an **implicit, unversioned**
anchor: the newest bar of the ingest batch always carries
``adjust_factor == 1.0`` and every re-ingest that sees a new corporate-action
event silently re-bases the entire history. This CLI makes that anchor explicit
and drift-detectable (see ``src/data/adjust_anchor.py`` for the mechanism).

Usage::

    # freeze the reference anchor (refuses to overwrite without --force)
    python scripts/check_adjust_anchor.py --baseline --force

    # snapshot the current anchor + a timestamped copy in the history dir
    python scripts/check_adjust_anchor.py --capture --json

    # is today's store still on the baseline's price basis?
    python scripts/check_adjust_anchor.py --compare            # exit 1 on drift

    # compare any two anchor files
    python scripts/check_adjust_anchor.py --compare --prev a.json --curr b.json

Exit codes: ``0`` = stable / write succeeded, ``1`` = drift detected or the
requested write was refused, ``2`` = usage / IO error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.adjust_anchor import (  # noqa: E402
    DEFAULT_TOL,
    capture_anchor,
    compare_anchors,
    format_drift_report,
    load_anchor,
    write_anchor,
)

BASELINE_NAME = "adjust_anchor_baseline.json"
CURRENT_NAME = "adjust_anchor.json"
HISTORY_NAME = "adjust_anchor_history"


def _outputs_root() -> Path:
    """``outputs/`` root, overridable with ``LLM_QUANT_OUTPUTS`` (repo convention)."""
    return Path(os.environ.get("LLM_QUANT_OUTPUTS", ROOT / "outputs"))


def _data_dir() -> Path:
    return _outputs_root() / "data"


def _baseline_path() -> Path:
    return _data_dir() / BASELINE_NAME


def _current_path() -> Path:
    return _data_dir() / CURRENT_NAME


def _history_dir() -> Path:
    return _data_dir() / HISTORY_NAME


def _log(message: str, *, json_mode: bool) -> None:
    """Progress goes to stderr in ``--json`` mode so stdout stays pure JSON."""
    print(message, file=sys.stderr if json_mode else sys.stdout, flush=True)


def _anchor_summary(anchor: dict, *, top: int = 5) -> str:
    stats = anchor.get("factor_stats") or {}
    lines = [
        f"captured_at:      {anchor.get('captured_at')}",
        f"data_as_of:       {anchor.get('data_as_of')}",
        f"anchor_policy:    {anchor.get('anchor_policy')}",
        f"symbols / rows:   {anchor.get('n_symbols')} / {anchor.get('n_rows')}",
        f"factor != 1:      {stats.get('n_symbols_factor_ne_1')} "
        f"(<1: {stats.get('n_symbols_factor_lt_1')}, >1: {stats.get('n_symbols_factor_gt_1')})",
        f"factor min/max:   {stats.get('min')} / {stats.get('max')}",
        f"factors_sha256:   {anchor.get('factors_sha256')}",
        f"top {top} factors (largest |factor-1|):",
    ]
    for sym, factor in (anchor.get("top_factors") or [])[:top]:
        lines.append(f"  {sym:<12} {factor!r}")
    return "\n".join(lines)


def _capture(args, *, json_mode: bool) -> dict:
    cfg = load_config()
    url = cfg.get("data.pit_database_url")
    if not url:
        raise SystemExit("ERROR: data.pit_database_url is empty — set PIT_DATABASE_URL (see .env)")
    tail = str(url).split("@")[-1]
    _log(f"capturing anchor from {tail} (one read pass) ...", json_mode=json_mode)
    symbols = args.symbols or None
    anchor = capture_anchor(cfg, symbols)
    _log(f"captured {anchor['n_symbols']} symbols / {anchor['n_rows']} price rows", json_mode=json_mode)
    return anchor


def _do_baseline(args, *, json_mode: bool) -> tuple[dict, Path]:
    path = _baseline_path()
    if path.exists() and not args.force:
        # Refuse before spending the ~1-minute store read.
        raise FileExistsError(
            f"{path} already exists — pass --force to re-freeze the reference anchor "
            "(and record why in docs/ADJUST_ANCHOR.md)"
        )
    anchor = _capture(args, json_mode=json_mode)
    path = write_anchor(path, anchor, force=args.force, history_dir=_history_dir())
    _log(f"baseline frozen at {path}", json_mode=json_mode)
    _log(_anchor_summary(anchor), json_mode=json_mode)
    return anchor, path


def _do_capture(args, *, json_mode: bool) -> tuple[dict, Path]:
    anchor = _capture(args, json_mode=json_mode)
    # The current-anchor file is a snapshot, not a reference: every capture
    # overwrites it, and every capture is preserved in the history directory.
    path = write_anchor(_current_path(), anchor, force=True, history_dir=_history_dir())
    _log(f"wrote {path} (+ copy under {_history_dir()})", json_mode=json_mode)
    _log(_anchor_summary(anchor), json_mode=json_mode)
    return anchor, path


def _do_compare(args, *, json_mode: bool) -> dict:
    prev_path = Path(args.prev) if args.prev else _baseline_path()
    curr_path = Path(args.curr) if args.curr else _current_path()
    for label, path in (("prev", prev_path), ("curr", curr_path)):
        if not path.is_file():
            raise SystemExit(
                f"ERROR: {label} anchor {path} not found — run "
                f"`python scripts/check_adjust_anchor.py "
                f"{'--baseline' if label == 'prev' else '--capture'}` first"
            )
    prev = load_anchor(prev_path)
    curr = load_anchor(curr_path)
    report = compare_anchors(prev, curr, tol=args.tol)
    _log(f"prev: {prev_path}", json_mode=json_mode)
    _log(f"curr: {curr_path}", json_mode=json_mode)
    _log(format_drift_report(report), json_mode=json_mode)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_adjust_anchor.py",
        description=(
            "Freeze / capture / compare the PIT price-adjustment anchor. "
            "A drifted anchor means the stored backward-adjusted series was "
            "re-based by a re-ingest, so metrics from before and after are on "
            "different price bases and must not be compared."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=f"capture the current anchor and freeze it at outputs/data/{BASELINE_NAME}",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help=f"capture and write outputs/data/{CURRENT_NAME} + a timestamped history copy",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare baseline vs current (or --prev/--curr); exit 1 when drifted",
    )
    parser.add_argument("--prev", default=None, help="explicit previous anchor file for --compare")
    parser.add_argument("--curr", default=None, help="explicit current anchor file for --compare")
    parser.add_argument(
        "--force", action="store_true", help="allow --baseline to overwrite an existing baseline"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_TOL,
        help=f"relative factor-change tolerance for drift (default {DEFAULT_TOL:g})",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="comma-separated subset to capture (default: every stored price symbol)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_mode",
        help="print the machine-readable result to stdout (progress goes to stderr)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.symbols:
        args.symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not (args.baseline or args.capture or args.compare):
        parser.error("nothing to do — pass at least one of --baseline / --capture / --compare")

    json_mode = bool(args.json_mode)
    result: dict[str, dict] = {}
    exit_code = 0
    try:
        if args.baseline:
            anchor, path = _do_baseline(args, json_mode=json_mode)
            result["baseline"] = {**anchor, "path": str(path)}
        if args.capture:
            anchor, path = _do_capture(args, json_mode=json_mode)
            result["capture"] = {**anchor, "path": str(path)}
        if args.compare:
            report = _do_compare(args, json_mode=json_mode)
            result["compare"] = report
            exit_code = 1 if report["drifted"] else 0
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2

    if json_mode:
        payload = next(iter(result.values())) if len(result) == 1 else result
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
