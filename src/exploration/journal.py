"""Exploration research journal — persistence + markdown report.

All exploration artifacts are written under ``outputs/exploration/`` — NEVER the
production ``outputs/factors.json`` — so the daily production autopilot loop is
completely unaffected by whatever this track discovers.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_journal(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def render_report(results: list[dict], survivors: list[dict], promoted: list[dict]) -> str:
    """Human-readable markdown summary of one exploration run."""
    lines: list[str] = []
    lines.append("# 探索轨迹报告 — high-risk/high-reward")
    lines.append("")
    lines.append(f"生成时间：{_now()}")
    lines.append("")
    lines.append(f"- 候选总数：{len(results)}")
    lines.append(f"- 可靠候选（train+test 双窗过门且 IC 同号）：{len(survivors)}")
    lines.append(f"- 满足**生产硬门**（可 promote）：{len(promoted)}")
    lines.append("")

    if promoted:
        lines.append("## 可 promote 候选（已过生产硬门）")
        lines.append("")
        lines.append("| 公式 | rank_ic(train) | rank_ic(test) | 回撤 | Sharpe |")
        lines.append("|---|---|---|---|---|")
        for p in promoted:
            t = p.get("windows", {}).get("train", {})
            te = p.get("windows", {}).get("test", {})
            lines.append(
                f"| `{p.get('formula','')}` | {t.get('rank_ic',0):.4f} | "
                f"{te.get('rank_ic',0):.4f} | {te.get('max_drawdown',0):.2%} | "
                f"{te.get('sharpe',0):.2f} |"
            )
        lines.append("")

    if survivors:
        lines.append("## 可靠候选（探索门通过，但未过生产硬门）")
        lines.append("")
        lines.append("| 公式 | 算子 | rank_ic(train) | rank_ic(test) | 回撤 |")
        lines.append("|---|---|---|---|---|")
        for s in survivors[:50]:
            t = s.get("windows", {}).get("train", {})
            te = s.get("windows", {}).get("test", {})
            lines.append(
                f"| `{s.get('formula','')}` | {s.get('operator','')} | "
                f"{t.get('rank_ic',0):.4f} | {te.get('rank_ic',0):.4f} | "
                f"{te.get('max_drawdown',0):.2%} |"
            )
        lines.append("")

    lines.append("## 全量结果摘要")
    lines.append("")
    lines.append("| 公式 | 算子 | 来源 | train | val | test |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        w = r.get("windows", {})
        lines.append(
            f"| `{r.get('formula','')}` | {r.get('operator','')} | {r.get('source','')} | "
            f"{w.get('train',{}).get('exploration','-')} | {w.get('val',{}).get('exploration','-')} | "
            f"{w.get('test',{}).get('exploration','-')} |"
        )
    lines.append("")
    return "\n".join(lines)
