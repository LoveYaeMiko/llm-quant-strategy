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

from ..provenance import code_fingerprint as code_fingerprint_of
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


#: Config subtrees that decide what the forward gate MEASURES and how it judges.
#: Hashing the whole merged config would be over-strict (it carries API keys and
#: unrelated research knobs whose rotation must not invalidate a freeze), while
#: hashing none of it — the pre-2026-09-10 behaviour — made the freeze a
#: decoration: thresholds could be edited after seeing the results without any
#: failure. These five subtrees are the policy surface.
POLICY_PATHS: tuple[str, ...] = (
    "forward",          # gate thresholds, candidate params, switch rule, power
    "deployment",       # observe vs live, the real-money gate
    "red_lines",        # the shadow red-line thresholds
    "paper",            # cost model actually charged by the executor
    "s7_calibration.cost_model",
)


def policy_payload(cfg) -> dict:
    """The policy surface of a config, as a plain dict (see :data:`POLICY_PATHS`).

    The deployed account is included under ``account`` because its parameters
    (stop width, gates, universe) ARE the rule under test.
    """
    payload: dict = {}
    for path in POLICY_PATHS:
        node = cfg.get(path, None)
        if node is not None:
            payload[path] = node
    accounts = cfg.get("shadow.accounts", None)
    if accounts:
        payload["account"] = next(
            (dict(a) for a in accounts if str(a.get("name")) == "D_5W"), None
        )
    return payload


def policy_fingerprint(cfg) -> str:
    """SHA-256 over :func:`policy_payload` — the value a freeze is bound to."""
    return sha256_of(policy_payload(cfg))


def prereg_gate(
    *,
    record: Optional[Mapping[str, Any]],
    window: Mapping[str, Any] | tuple[str, str],
    data_as_of: str,
    policy_sha256: str,
    code_commit: str,
    code_fingerprint: Optional[str] = None,
    code_dirty: Optional[bool] = None,
    allow_code_drift: bool = False,
) -> dict:
    """Hard gate: the evaluation must be BOUND to a frozen pre-registration.

    An unbound evaluation is not evidence — it is a number produced by whatever
    the thresholds happened to be at the moment it ran. Five checks, all of which
    must pass:

    1. **a verified record exists** for this account/window;
    2. **frozen before the window opened** (``frozen_at`` strictly earlier than the
       window start) and before the data cut-off;
    3. **the policy has not moved since the freeze** — the live policy fingerprint
       must equal the record's ``policy_sha256`` (this is the lock: editing a
       threshold after seeing the results fails until a NEW version is frozen);
    4. **the evaluated window is the frozen window** (not a hand-picked sub-range);
    5. **the code matches the frozen commit** — unless the caller explicitly passes
       ``allow_code_drift``, which is recorded as a waiver in the artifact rather
       than silently accepted.

    Returns a gate-shaped mapping (``ok`` + the evidence), never raises.
    """
    start, end = (window.get("start"), window.get("end")) if isinstance(window, Mapping) \
        else (window[0], window[1])
    out: dict = {
        "ok": False, "rule_id": None, "frozen_at": None,
        "window_match": False, "frozen_before_window": False,
        "policy_sha256_match": False, "code_commit_match": False,
        "waived": False, "issues": [],
    }
    if not record:
        out["issues"].append("no verified pre-registration covers this window — "
                             "freeze one with `python scripts/prereg.py new`")
        return out
    out["rule_id"] = record.get("rule_id")
    out["frozen_at"] = record.get("frozen_at")

    frozen = pd.Timestamp(record.get("frozen_at"))
    win_start, win_end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(frozen):
        out["issues"].append("the record has no usable frozen_at")
    else:
        if frozen.date() >= win_start.date():
            out["issues"].append(
                f"frozen_at {frozen.date()} is not before the window start {win_start.date()}"
                " — a rule frozen inside its own window is not pre-registered"
            )
        else:
            out["frozen_before_window"] = True
        if data_as_of and frozen.date() > pd.Timestamp(data_as_of).date():
            out["issues"].append(f"frozen_at {frozen.date()} is after data_as_of {data_as_of}")

    scope = dict(record.get("scope") or {})
    rec_win = scope.get("window") or []
    if len(rec_win) == 2:
        out["window_match"] = (str(rec_win[0])[:10] == str(start)[:10]
                               and str(rec_win[1])[:10] == str(end)[:10])
        if not out["window_match"]:
            out["issues"].append(
                f"evaluated window [{start}, {end}] is not the frozen window "
                f"[{rec_win[0]}, {rec_win[1]}] — a sub-range chosen after the fact is a new trial"
            )
    else:
        out["issues"].append("the record has no scope.window to bind against")

    stored_policy = str(record.get("policy_sha256") or record.get("config_sha256") or "")
    out["policy_sha256_match"] = bool(stored_policy) and stored_policy == policy_sha256
    if not out["policy_sha256_match"]:
        out["issues"].append(
            "the live policy fingerprint does not match the frozen record — thresholds "
            "or deployed parameters changed after the freeze; freeze a new version "
            "(`scripts/prereg.py new` with version+1 and supersedes) before evaluating"
        )
    out["policy_sha256_stored"] = stored_policy[:16] or None
    out["policy_sha256_live"] = str(policy_sha256)[:16]

    stored_commit = str(record.get("code_commit") or "")
    stored_fp = str(record.get("code_fingerprint") or "")
    out["code_commit"] = stored_commit or None
    out["code_fingerprint_stored"] = stored_fp[:16] or None
    out["code_fingerprint_live"] = str(code_fingerprint or "")[:16] or None
    if code_fingerprint and stored_fp:
        # Preferred binding: the CONTENT of the behaviour-deciding code. Immune to
        # docs-only commits, sensitive to any real edit (committed or not).
        out["code_commit_match"] = stored_fp == code_fingerprint
        drift = ("" if out["code_commit_match"] else
                 "the behaviour-deciding code changed since the freeze "
                 f"(record {stored_fp[:12]} vs live {str(code_fingerprint)[:12]})")
    else:
        # Fallback for records frozen before the fingerprint existed: commit sha.
        out["code_commit_match"] = bool(stored_commit) and stored_commit == code_commit
        drift = ("" if out["code_commit_match"] else
                 f"code drifted since the freeze: record {stored_commit[:12] or '—'} "
                 f"vs HEAD {str(code_commit)[:12]}")
    if not drift and code_dirty and not code_fingerprint:
        # the fingerprint already covers uncommitted edits; this only bites when
        # the older commit-based binding is in use
        drift = "the working tree was DIRTY at evaluation time"
    if drift:
        if allow_code_drift:
            out["waived"] = True
            out["waiver_reason"] = drift + " (explicitly waived for this run)"
        else:
            out["issues"].append(
                drift + " — re-freeze, or pass --allow-code-drift to record the "
                "waiver (a behaviour change invalidates the frozen rule)"
            )
    out["code_dirty"] = code_dirty

    out["ok"] = not out["issues"]
    return out


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
    policy_sha256: Optional[str] = None,
    code_fingerprint: Optional[str] = None,
    repo_root: str | Path | None = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build a complete record (validated by :func:`validate_record`).

    ``frozen_at`` defaults to now; pass an explicit timestamp to reconstruct a
    record that was frozen earlier (the value is what the record CLAIMS — the
    verifier only checks that it is in the past and that the hash holds).

    ``policy_sha256`` binds the record to the policy surface (:func:`policy_payload`:
    forward thresholds, deployment gate, red lines, cost model and the deployed
    account). The gate refuses to evaluate when the live fingerprint differs, so
    a threshold edited after the freeze cannot be used to judge the window.
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
        "code_fingerprint": (code_fingerprint or code_fingerprint_of(repo_root)),
        "policy_sha256": policy_sha256,
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
    "POLICY_PATHS",
    "PREREG_FIELDS",
    "PreregError",
    "list_preregistrations",
    "load_preregistration",
    "new_record",
    "policy_fingerprint",
    "policy_payload",
    "prereg_gate",
    "record_path",
    "record_sha256",
    "trial_count",
    "validate_record",
    "verify_preregistration",
    "write_preregistration",
]
