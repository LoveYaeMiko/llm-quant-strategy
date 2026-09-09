"""Artifact provenance — the five fields every external-facing JSON must carry.

Why this exists (2026-09-09 audit, item P-6): the evidence artifacts answered
"what happened" but not "under which slice of data, computed by which code,
under which convention". A number without those four answers cannot be compared
with another number, cannot be reproduced, and cannot be quoted to a third party
— which is exactly how a stale ``d_oos_*`` run came to be cited after the
configuration had moved on.

The contract is deliberately tiny. Every external-facing artifact must carry:

``window``
    The evaluation slice: ``{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}`` (a
    ``[start, end]`` list is accepted for legacy artifacts).
``convention``
    One string naming the accounting/execution convention the numbers are on —
    e.g. fills at the 15:00 auction close, adjusted-close price basis, T+1. Two
    artifacts with different conventions are not comparable.
``data_as_of``
    The last bar date actually present in the input data (``YYYY-MM-DD``). This
    is what makes a stale run visible: a ``window`` can be recent while the data
    behind it is not.
``artifact_sha256``
    SHA-256 of the artifact's canonical JSON *without* the provenance block, so
    the file is tamper-evident and a truncated/edited copy is detectable.
``code_commit``
    The git commit that produced the numbers (``git rev-parse HEAD``).

:func:`stamp_artifact` attaches a complete, self-consistent block;
:func:`check_provenance` verifies one (including recomputing the hash);
:func:`require_provenance` raises. ``scripts/check_provenance.py`` sweeps the
artifact directory and fails the run when anything is missing or unverifiable.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

#: The five fields the contract requires (kept in one place so the checker, the
#: stamper and the docs can never drift apart).
PROVENANCE_REQUIRED: tuple[str, ...] = (
    "window",
    "convention",
    "data_as_of",
    "artifact_sha256",
    "code_commit",
)

#: Where a caller may put the block. ``provenance`` (preferred) or top level
#: (legacy artifacts such as the early ``d_oos_*`` runs put ``window`` there).
PROVENANCE_KEYS: tuple[str, ...] = ("provenance", "")

_HEX = re.compile(r"^[0-9a-f]{7,64}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


class ProvenanceError(ValueError):
    """An artifact violates the provenance contract."""


# --------------------------------------------------------------------------- #
# canonical form / hashing
# --------------------------------------------------------------------------- #
def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, UTF-8 kept as-is."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_of(obj: Any) -> str:
    """SHA-256 of an object's canonical JSON form."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit(repo_root: str | Path | None = None) -> str:
    """Current ``HEAD`` sha (``"unknown"`` when git is unavailable/dirty-repo-less).

    Never raises: a research artifact produced outside a git checkout must still
    be stampable, but the field will read ``unknown`` and
    :func:`check_provenance` flags it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = (out.stdout or "").strip()
        return sha if out.returncode == 0 and _HEX.match(sha) else "unknown"
    except Exception:  # noqa: BLE001 — provenance must never break the caller
        return "unknown"


def git_dirty(repo_root: str | Path | None = None) -> Optional[bool]:
    """True when the working tree has uncommitted changes (``None`` if unknown).

    ``code_commit`` alone is not enough: an artifact produced from a dirty tree is
    stamped with HEAD but was NOT produced by that commit. Recording the flag
    keeps the artifact honest without blocking the run.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        return bool((out.stdout or "").strip())
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# stamp / check
# --------------------------------------------------------------------------- #
def _window_block(window: Any) -> dict:
    """Normalize a window to ``{"start": .., "end": ..}`` (raises on nonsense)."""
    if isinstance(window, Mapping):
        start, end = window.get("start"), window.get("end")
    elif isinstance(window, (list, tuple)) and len(window) == 2:
        start, end = window[0], window[1]
    else:
        raise ProvenanceError(f"window must be a mapping or a 2-item sequence, got {window!r}")
    if not start or not end:
        raise ProvenanceError(f"window needs both start and end, got {window!r}")
    return {"start": str(start)[:10], "end": str(end)[:10]}


def stamp_artifact(
    artifact: Mapping[str, Any],
    *,
    window: Any,
    convention: str,
    data_as_of: Any,
    code_commit: Optional[str] = None,
    repo_root: str | Path | None = None,
    key: str = "provenance",
) -> dict:
    """Return a copy of ``artifact`` carrying a complete provenance block.

    The hash covers the artifact WITHOUT the block, so stamping is idempotent
    (re-stamping the same payload yields the same ``artifact_sha256``) and the
    payload can be verified independently of the provenance metadata itself.

    ``convention`` and ``data_as_of`` are validated here: an artifact that cannot
    state its convention or its data cut-off is refused rather than stamped with
    a placeholder.
    """
    if not isinstance(convention, str) or not convention.strip():
        raise ProvenanceError("convention must be a non-empty string")
    das = str(data_as_of)[:10]
    if not _DATE.match(das):
        raise ProvenanceError(f"data_as_of must start with YYYY-MM-DD, got {data_as_of!r}")
    out = {k: v for k, v in artifact.items() if k != key}
    block = {
        "window": _window_block(window),
        "convention": convention.strip(),
        "data_as_of": das,
        "artifact_sha256": sha256_of(out),
        "code_commit": (code_commit or git_commit(repo_root)).strip(),
        # informational: HEAD was dirty when the numbers were produced, so the
        # artifact is reproducible only together with the working tree
        "code_dirty": git_dirty(repo_root),
    }
    out[key] = block
    return out


def _locate(artifact: Mapping[str, Any]) -> tuple[dict, str]:
    """Find the provenance block (preferred key first, then top level).

    Returns ``(block, where)`` where ``where`` is ``"provenance"`` or ``"top"``;
    an empty block means "nothing found" and is reported by the caller.
    """
    for k in PROVENANCE_KEYS:
        if k and isinstance(artifact.get(k), Mapping):
            block = dict(artifact[k])  # type: ignore[index]
            # accept a PARTIAL block too: reporting exactly which field is absent
            # is the point of the checker (an all-or-nothing locate would blame
            # the caller for a missing file that is actually just incomplete)
            if any(f in block for f in PROVENANCE_REQUIRED):
                return block, "provenance"
    top = {f: artifact[f] for f in PROVENANCE_REQUIRED if f in artifact}
    return top, "top"


def check_provenance(
    artifact: Mapping[str, Any], *, verify_hash: bool = True
) -> dict:
    """Verify the contract. Returns a JSON-serializable report (never raises).

    ``ok`` is True only when every required field is present AND well-formed AND
    (unless ``verify_hash=False``) the ``artifact_sha256`` recomputes.
    """
    block, where = _locate(artifact)
    missing = [f for f in PROVENANCE_REQUIRED if f not in block or block[f] in (None, "")]
    problems: list[str] = []
    if not missing:
        try:
            _window_block(block["window"])
        except ProvenanceError as exc:
            problems.append(str(exc))
        if not isinstance(block["convention"], str) or not block["convention"].strip():
            problems.append("convention must be a non-empty string")
        if not _DATE.match(str(block["data_as_of"])):
            problems.append(f"data_as_of must start with YYYY-MM-DD, got {block['data_as_of']!r}")
        if not _HEX.match(str(block["artifact_sha256"])):
            problems.append(f"artifact_sha256 must be 7-64 lowercase hex, got {block['artifact_sha256']!r}")
        if not _HEX.match(str(block["code_commit"])):
            problems.append(
                f"code_commit must be a git sha (got {block['code_commit']!r}; "
                "'unknown' means the artifact was produced outside a git checkout)"
            )
        if verify_hash and not problems:
            payload = (
                {k: v for k, v in artifact.items() if k != "provenance"}
                if where == "provenance"
                else {k: v for k, v in artifact.items() if k not in PROVENANCE_REQUIRED}
            )
            expect = sha256_of(payload)
            if expect != str(block["artifact_sha256"]):
                problems.append(
                    f"artifact_sha256 mismatch: stored {block['artifact_sha256']}, "
                    f"recomputed {expect} (the payload changed after stamping)"
                )
    return {
        "ok": not missing and not problems,
        "where": where,
        "missing": missing,
        "problems": problems,
        "values": {f: block.get(f) for f in PROVENANCE_REQUIRED},
        "code_dirty": block.get("code_dirty"),
    }


def require_provenance(artifact: Mapping[str, Any], *, label: str = "artifact") -> dict:
    """Like :func:`check_provenance` but raises :class:`ProvenanceError` on failure."""
    report = check_provenance(artifact)
    if not report["ok"]:
        detail = report["missing"] and f"missing {report['missing']}" or "; ".join(report["problems"])
        raise ProvenanceError(f"{label} violates the provenance contract: {detail}")
    return report


def check_artifact_file(path: str | Path, *, verify_hash: bool = True) -> dict:
    """Check one JSON file on disk; a non-JSON/absent file is a failure, not a crash."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"path": str(p), "ok": False, "missing": list(PROVENANCE_REQUIRED),
                "problems": [f"unreadable JSON: {exc}"], "values": {}}
    report = check_provenance(data, verify_hash=verify_hash)
    report["path"] = str(p)
    return report


def check_artifacts(paths: Iterable[str | Path], *, verify_hash: bool = True) -> dict:
    """Check many files → ``{ok, n_checked, n_failed, failures: [...], reports: [...]}``."""
    reports = [check_artifact_file(p, verify_hash=verify_hash) for p in paths]
    failures = [r for r in reports if not r["ok"]]
    return {
        "ok": not failures,
        "n_checked": len(reports),
        "n_failed": len(failures),
        "failures": failures,
        "reports": reports,
    }


__all__ = [
    "PROVENANCE_REQUIRED",
    "ProvenanceError",
    "canonical_json",
    "check_artifact_file",
    "check_artifacts",
    "check_provenance",
    "git_commit",
    "require_provenance",
    "sha256_file",
    "sha256_of",
    "stamp_artifact",
]
