"""Reproducibility / execution-realism audit (Beyond Agent Architecture §2.4).

That paper audits 30 LLM-trading studies and finds evaluation assumptions are
reported worse than architectures — the failure that makes results impossible to
compare or reproduce. This module is the counter-measure: every mining run gets
an :class:`AuditRecord` capturing, in one machine-readable blob:

* **config hash** — a digest of the YAML gates actually used;
* **data horizon** — the PIT window and universe construction;
* **model versions** — which models/routing were in effect;
* **evaluation assumptions** — IC method, thresholds, annualisation, trials;
* **per-factor metrics + verdicts**, and the blueprint verification checklist.

Writing the audit file is part of ``cli.py mine`` / ``backtest`` so every result
carries its own reproducibility envelope.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import Config, ROOT


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_sha(repo_root: Path = ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


@dataclass
class AuditRecord:
    run_id: str
    name: str
    created_at: str
    git_commit: str = ""
    config_hash: str = ""
    pit_window: dict = field(default_factory=dict)
    model_versions: dict = field(default_factory=dict)
    evaluation_assumptions: dict = field(default_factory=dict)
    factors: list[dict] = field(default_factory=list)
    checklist: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)

    def add_factor(self, factor: dict, metrics: dict, verdict: str) -> None:
        self.factors.append({"factor": factor, "metrics": metrics, "verdict": verdict})

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2, default=str)
        return path


class ExperimentAuditor:
    """Builds :class:`AuditRecord` snapshots of the environment."""

    def __init__(self, repo_root: Path = ROOT) -> None:
        self.repo_root = Path(repo_root)

    def begin(self, name: str) -> AuditRecord:
        return AuditRecord(
            run_id=uuid.uuid4().hex[:12],
            name=name,
            created_at=_utcnow(),
            git_commit=_git_sha(self.repo_root),
        )

    def snapshot_config(self, record: AuditRecord, config: Config) -> None:
        payload = json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False, default=str)
        record.config_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        record.evaluation_assumptions = {
            "ic_threshold": config.get("factor_mining.ic_threshold"),
            "rank_ic_threshold": config.get("factor_mining.rank_ic_threshold"),
            "max_lookback": config.get("factor_mining.max_lookback"),
            "min_lookback": config.get("factor_mining.min_lookback"),
            "max_drawdown": config.get("risk_management.max_sharpe_drawdown"),
            "annualization": 252,
            "ic_method": "spearman",
        }

    def snapshot_routing(self, record: AuditRecord, config: Config) -> None:
        record.model_versions = {
            "generator": config.get("routing.generator.model"),
            "code": config.get("routing.code.model"),
            "critic": config.get("routing.critic.model"),
            "sentiment": config.get("routing.sentiment.model"),
        }

    def set_pit_window(self, record: AuditRecord, start: str, end: str, universe: int) -> None:
        record.pit_window = {
            "start_date": start,
            "end_date": end,
            "universe_size": int(universe),
        }

    def set_checklist(self, record: AuditRecord, **items: Any) -> None:
        record.checklist.update(items)
