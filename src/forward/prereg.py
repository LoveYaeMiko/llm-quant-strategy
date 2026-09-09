"""Pre-registration records — freeze a rule BEFORE its forward window opens.

Why (2026-09-09 audit, item 1.4): the D track's stop width was re-evaluated after
the fact on the same window the parameters were chosen on. That is how a research
program drifts into reporting the maximum of many trials as if it were the
estimate of one. The precedent in this repository is the Kronos/AurumQ track's
"Pre-registered kill criteria" (``paper/repos/aurumq-rl/scripts/p3/master_lib.py``):
the criteria, the cost spec and the trial count are written down before the run,
and changing any number after looking at the results voids the verdict.

A record has exactly six required fields (the audit's template)::

    rule_id      stable identifier of the rule under test (never reused)
    frozen_at    ISO-8601 timestamp; must be in the past when verified
    scope        what the rule applies to (account, universe, window, params)
    decision     the pass/fail logic, in enough detail to be mechanical
    stopping     when the evaluation stops (window length, kill criteria)
    trials       multiplicity accounting (how many attempts this family has had)

plus provenance (``code_commit``, ``config_sha256``, ``artifact_sha256``) and a
self-hash ``record_sha256`` over the canonical payload. Records are APPEND-ONLY:
:func:`write_preregistration` refuses to overwrite a file, and changing a rule
requires a new version that names the record it ``supersedes``.

Usage::

    from src.forward.prereg import new_record, write_preregistration, verify_preregistration

    rec = new_record(
        rule_id="d_forward_2026h2",
        scope={"account": "D_5W", "window": ["2026-09-10", "2027-03-09"]},
        decision={"hard": [...], "verdict": "all hard gates must pass"},
        stopping={"window_days": 120, "kill": "any hard gate failure for 3 days"},
        trials={"family": "d_forward", "prior_trials": 2, "this_trial": 3},
    )
    path = write_preregistration(rec)          # outputs/forward/prereg/...
    verify_preregistration(path)               # raises on tamper / missing field
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from ..provenance import git_commit, sha256_of

ROOT = Path(__file__).resolve().parents[2]

#: The audit's pre-registration template — every field is required and checked.
PREREG_FIELDS: tuple[str, ...] = (
    "rule_id",
    "frozen_at",
    "scope",
    "decision",
    "stopping",
    "trials",
)

#: Default location of the append-only record directory.
DEFAULT_DIR = ROOT / "outputs" / "forward" / "prereg"


class PreregError(ValueError):
    """A pre-registration record is invalid, tampered with, or being overwritten."""


def _now() -> str:
    return pd.Timestamp.now().isoformat(timespec="seconds")


def new_record(
    *,
    rule_id: str,
    scope: Mapping[str, Any],
    decision: Mapping[str, Any],
    stopping: Mapping[str, Any],
    trials: Mapping[str, Any],
    frozen_at: Optional[str] = None,
    version: int = 1,
    supersedes: Optional[str] = None,
    notes: str = "",
    code_commit: Optional[str] = None,
    repo_root: str | Path | None = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build a complete record (validated by :func:`validate_record`).

    ``frozen_at`` defaults to now; pass an explicit timestamp to reconstruct a
    record that was frozen earlier (the value is what the record CLAIMS — the
    verifier only checks that it is in the past and that the hash holds).
    """
    record: dict[str, Any] = {
        "rule_id": str(rule_id),
        "version": int(version),
        "frozen_at": str(frozen_at or _now()),
        "scope": dict(scope),
        "decision": dict(decision),
        "stopping": dict(stopping),
        "trials": dict(trials),
        "notes": notes,
        "supersedes": supersedes,
        "code_commit": (code_commit or git_commit(repo_root)).strip(),
    }
    if extra:
        record.update(dict(extra))
    record["record_sha256"] = record_sha256(record)
    validate_record(record)
    return record


def record_sha256(record: Mapping[str, Any]) -> str:
    """Self-hash over the record WITHOUT ``record_sha256`` (canonical JSON)."""
    payload = {k: v for k, v in record.items() if k != "record_sha256"}
    return sha256_of(payload)


def validate_record(record: Mapping[str, Any]) -> None:
    """Raise :class:`PreregError` when a record cannot be trusted as a pre-registration."""
    missing = [f for f in PREREG_FIELDS if f not in record or record[f] in (None, "", {}, [])]
    if missing:
        raise PreregError(f"pre-registration missing required field(s): {missing}")
    for f in ("scope", "decision", "stopping", "trials"):
        if not isinstance(record[f], Mapping) or not record[f]:
            raise PreregError(f"{f} must be a non-empty mapping")
    try:
        frozen = pd.Timestamp(record["frozen_at"])
    except (ValueError, TypeError) as exc:
        raise PreregError(f"frozen_at is not a timestamp: {record['frozen_at']!r}") from exc
    if pd.isna(frozen):
        raise PreregError(f"frozen_at is not a timestamp: {record['frozen_at']!r}")
    stored = str(record.get("record_sha256", ""))
    expect = record_sha256(record)
    if stored != expect:
        raise PreregError(
            f"record_sha256 mismatch for {record.get('rule_id')!r}: stored {stored!r}, "
            f"recomputed {expect!r} — the record was edited after freezing"
        )


def record_path(rule_id: str, version: int = 1, dir: str | Path | None = None) -> Path:
    """``<dir>/prereg_<rule_id>_v<version>.json`` (stable, collision-free naming)."""
    base = Path(dir) if dir is not None else DEFAULT_DIR
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(rule_id))
    return base / f"prereg_{safe}_v{int(version)}.json"


def write_preregistration(
    record: Mapping[str, Any],
    *,
    dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Write a record to the append-only directory and return its path.

    Refuses to overwrite an existing file (``force=True`` only for repairing a
    truncated write of byte-identical content). Use a new ``version`` +
    ``supersedes`` to change a rule.
    """
    validate_record(record)
    path = record_path(str(record["rule_id"]), int(record.get("version", 1)), dir)
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("record_sha256") != record.get("record_sha256"):
            raise PreregError(
                f"{path.name} already exists with a different record — pre-registration is "
                "append-only; bump version and set supersedes (a changed rule is a new trial)"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def load_preregistration(path: str | Path) -> dict:
    """Read a record file (no validation — call :func:`verify_preregistration`)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_preregistration(
    path: str | Path,
    *,
    require_past: bool = True,
    require_commit: bool = False,
    repo_root: str | Path | None = None,
) -> dict:
    """Verify a record file end to end; raises :class:`PreregError` on any problem.

    ``require_commit`` additionally demands that the record's ``code_commit``
    equals the current ``HEAD`` (the rule must be frozen against the code that
    will run it).
    """
    p = Path(path)
    try:
        record = load_preregistration(p)
    except (OSError, ValueError) as exc:
        raise PreregError(f"{p}: unreadable pre-registration: {exc}") from exc
    validate_record(record)
    if require_past:
        frozen = pd.Timestamp(record["frozen_at"])
        if frozen > pd.Timestamp.now() + pd.Timedelta(minutes=5):
            raise PreregError(f"{p.name}: frozen_at {record['frozen_at']} is in the future")
    if require_commit:
        head = git_commit(repo_root)
        if str(record.get("code_commit")) != head:
            raise PreregError(
                f"{p.name}: frozen against {record.get('code_commit')} but HEAD is {head} — "
                "re-freeze (new version) before evaluating"
            )
    return record


def list_preregistrations(dir: str | Path | None = None) -> list[dict]:
    """All records in the directory, newest ``frozen_at`` first (never raises)."""
    base = Path(dir) if dir is not None else DEFAULT_DIR
    out: list[dict] = []
    for p in sorted(base.glob("prereg_*.json")):
        try:
            rec = load_preregistration(p)
            rec["_path"] = str(p)
            out.append(rec)
        except (OSError, ValueError):
            out.append({"_path": str(p), "rule_id": p.stem, "error": "unreadable"})
    out.sort(key=lambda r: str(r.get("frozen_at", "")), reverse=True)
    return out


def trial_count(dir: str | Path | None, family: str) -> int:
    """How many records of ``family`` already exist (multiplicity accounting)."""
    n = 0
    for rec in list_preregistrations(dir):
        trials = rec.get("trials") or {}
        if isinstance(trials, Mapping) and str(trials.get("family")) == str(family):
            n += 1
    return n


__all__ = [
    "DEFAULT_DIR",
    "PREREG_FIELDS",
    "PreregError",
    "list_preregistrations",
    "load_preregistration",
    "new_record",
    "record_path",
    "record_sha256",
    "trial_count",
    "validate_record",
    "verify_preregistration",
    "write_preregistration",
]
