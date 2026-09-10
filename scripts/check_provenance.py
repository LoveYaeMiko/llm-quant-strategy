"""Fail when an external-facing artifact is missing provenance (audit item P-6).

The contract lives in :mod:`src.provenance`: every quotable JSON must carry
``window / convention / data_as_of / artifact_sha256 / code_commit``. This script
is the enforcement point — run it in CI, before publishing evidence, and at the
end of any research session.

Default sweep (``--preset evidence``): the artifacts that back published claims.

    python scripts/check_provenance.py                    # evidence preset
    python scripts/check_provenance.py outputs/forward/*.json
    python scripts/check_provenance.py --glob "outputs/d_oos_*.json"
    python scripts/check_provenance.py --all              # everything in outputs/
    python scripts/check_provenance.py --all --no-verify-hash

Exit code: 0 when every checked file satisfies the contract, 1 otherwise. A
legacy artifact that predates the contract fails on purpose — re-run the script
that produced it (the harnesses stamp on write) rather than hand-editing the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.provenance import check_artifacts  # noqa: E402

#: Artifacts whose numbers are CURRENTLY quotable — every one of them must exist
#: and carry a complete, hash-verified provenance block. Update this list when a
#: new run supersedes an old one (and move the old file to the legacy set below).
#:
#: Deliberate exceptions (they carry their own, purpose-built contract instead of
#: the five fields): ``outputs/forward/prereg/*.json`` (six pre-registration
#: fields + ``record_sha256``, verified by ``scripts/prereg.py verify``) and
#: ``outputs/data/adjust_anchor*.json`` (``factors_sha256`` over the per-symbol
#: factor map + ``record_sha256`` + an append-only history directory, verified by
#: ``scripts/check_adjust_anchor.py --compare``).
EVIDENCE_REQUIRED: tuple[str, ...] = (
    "outputs/d_oos_is_2026_v5.json",
    "outputs/d_oos_oos_2025h2_v5.json",
    "outputs/shadow_status_D_5W.json",
    "outputs/forward/forward_health*.json",
    "outputs/forward/paired_*.json",
    "outputs/bias_stress_*.json",
    "docs/evidence/d_oos_is_2026_v5.json",
    "docs/evidence/d_oos_oos_2025h2_v5.json",
    "docs/evidence/bias_stress_oos_2025h2_v1.json",
)

#: Artifacts kept for the historical record. They PREDATE the contract, so they
#: are reported as legacy (not as failures) unless ``--strict`` is passed.
#: ``outputs/forward/prereg/*.json`` is deliberately absent: pre-registration
#: records have their own contract (``scripts/prereg.py verify``).
EVIDENCE_LEGACY: tuple[str, ...] = (
    "outputs/d_oos_*.json",
    "outputs/d_stop_grid.json",
    "docs/evidence/*.json",
)


def _resolve(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        hits = sorted(ROOT.glob(pat))
        out.extend(hits)
    # de-dup, keep order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description="Provenance completeness check")
    ap.add_argument("paths", nargs="*", help="explicit JSON files to check")
    ap.add_argument("--glob", action="append", default=[], help="glob pattern (repeatable)")
    ap.add_argument("--all", action="store_true", help="check every JSON under outputs/ and docs/evidence/")
    ap.add_argument("--preset", default="evidence", choices=["evidence", "none"])
    ap.add_argument("--strict", action="store_true",
                    help="treat legacy (pre-contract) artifacts as failures too")
    ap.add_argument("--no-verify-hash", action="store_true",
                    help="only check that the fields exist and look well-formed")
    ap.add_argument("--json", action="store_true", help="emit the machine-readable report")
    ap.add_argument("--quiet", action="store_true", help="only print failures")
    args = ap.parse_args()

    patterns: list[str] = list(args.glob)
    if args.all:
        patterns += ["outputs/**/*.json", "docs/evidence/*.json"]

    paths = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.paths]
    legacy: list[Path] = []
    if args.preset == "evidence" and not args.paths and not patterns:
        # required set must exist; legacy set is only reported
        required = _resolve(list(EVIDENCE_REQUIRED))
        missing_required = [p for p in EVIDENCE_REQUIRED if not _resolve([p])]
        have = {str(p.resolve()).lower() for p in required}
        legacy = [p for p in _resolve(list(EVIDENCE_LEGACY))
                  if str(p.resolve()).lower() not in have]
        paths += required
        if missing_required:
            print("[provenance] REQUIRED artifact(s) absent: "
                  + ", ".join(missing_required), file=sys.stderr)
    else:
        paths += _resolve(patterns)

    if not paths and not legacy:
        print("[provenance] nothing to check (no matching artifacts)", file=sys.stderr)
        return 0

    report = check_artifacts(paths, verify_hash=not args.no_verify_hash)
    report["patterns"] = patterns
    report["n_legacy"] = len(legacy)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if not args.quiet:
        print(f"[provenance] checked {report['n_checked']} required artifact(s), "
              f"{report['n_failed']} failing; {len(legacy)} legacy file(s) skipped"
              f"{' (strict: counted)' if args.strict else ''}", flush=True)
    for rep in report["reports"]:
        rel = Path(rep["path"]).relative_to(ROOT) if str(rep["path"]).startswith(str(ROOT)) else rep["path"]
        if rep["ok"]:
            if not args.quiet:
                v = rep["values"]
                dirty = " DIRTY-TREE" if rep.get("code_dirty") else ""
                print(f"  [PASS] {rel}  data_as_of={v['data_as_of']} "
                      f"commit={str(v['code_commit'])[:12]}{dirty} "
                      f"sha256={str(v['artifact_sha256'])[:12]}…")
        else:
            detail = (f"missing {rep['missing']}" if rep["missing"] else "; ".join(rep["problems"]))
            print(f"  [FAIL] {rel}: {detail}")
    if legacy and not args.quiet:
        print(f"  [LEGACY] {len(legacy)} pre-contract artifact(s) not checked "
              "(see docs/evidence/README.md; use --strict to enforce)")
    if args.strict:
        return 0 if report["ok"] and not legacy else 1
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
