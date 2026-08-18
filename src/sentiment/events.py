"""TimelineEvent + single-file HTML replay for the TriAgent sentiment pipeline.

C1 (studio ``core/events.py``): the TriAgent's three tiers (lexicon → BERT →
critic) currently surface only the final score and tier; the intermediate
decisions — escalation, dispersion, the critic's fusion weights — are discarded.
Recording them as a timeline lets a run export "the full decision path for
symbol X on day T" as one self-contained HTML file: no server, no external
assets, openable straight from disk.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from typing import Iterable

_KINDS = {"note", "score", "escalate", "decision"}


@dataclass
class TimelineEvent:
    """One step in a pipeline's decision path.

    ``ts``     — timestamp / date label (free text, e.g. ``"2024-03-15"``)
    ``phase``  — pipeline stage (``word`` / ``bert`` / ``critic`` / ``decision``)
    ``agent``  — which agent emitted it (``lexicon`` / ``bert`` / ``critic`` /
                 ``triagent``)
    ``content``— human-readable description of what happened
    ``kind``   — ``note`` | ``score`` | ``escalate`` | ``decision``
    ``meta``   — structured numbers (scores, thresholds, weights) for the replay
    """

    ts: str
    phase: str
    agent: str
    content: str
    kind: str = "note"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "phase": self.phase,
            "agent": self.agent,
            "content": self.content,
            "kind": self.kind,
            "meta": self.meta,
        }


def _meta_block(meta: dict) -> str:
    if not meta:
        return ""
    items = "".join(
        f"<span class=\"k\">{html.escape(str(k))}</span>"
        f"<span class=\"v\">{html.escape(str(v))}</span>"
        for k, v in meta.items()
    )
    return f"<div class=\"meta\">{items}</div>"


def render_html(
    events: Iterable[TimelineEvent],
    title: str = "TriAgent 决策回放",
    subtitle: str = "",
) -> str:
    """Render events as a single self-contained HTML document (inline CSS only)."""
    rows = []
    for e in events:
        kind = e.kind if e.kind in _KINDS else "note"
        rows.append(
            f"""
        <li class="ev phase-{html.escape(e.phase)}">
          <div class="bar">
            <span class="ts">{html.escape(e.ts)}</span>
            <span class="phase">{html.escape(e.phase)}</span>
            <span class="agent">{html.escape(e.agent)}</span>
            <span class="kind kind-{kind}">{kind}</span>
          </div>
          <div class="body">{html.escape(e.content)}</div>{_meta_block(e.meta)}
        </li>"""
        )
    body = "\n".join(rows) or '<li class="empty">无事件</li>'
    return f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         max-width: 860px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.4rem; }} .sub {{ color: #888; font-size: .9rem; }}
  ul {{ list-style: none; padding: 0; }}
  li.ev {{ border-left: 3px solid #ccc; margin: .6rem 0; padding: .35rem .8rem;
          background: rgba(127,127,127,.06); border-radius: 4px; }}
  li.phase-word   {{ border-left-color: #4caf50; }}
  li.phase-bert   {{ border-left-color: #2196f3; }}
  li.phase-critic {{ border-left-color: #ff9800; }}
  li.phase-decision {{ border-left-color: #e91e63; }}
  .bar {{ display: flex; gap: .6rem; align-items: baseline; font-size: .85rem; }}
  .ts {{ color: #666; font-variant-numeric: tabular-nums; }}
  .phase {{ font-weight: 600; text-transform: uppercase; letter-spacing: .03em; }}
  .agent {{ color: #888; }}
  .kind {{ font-size: .72rem; border-radius: 3px; padding: 0 .4rem; background: #eee; }}
  .kind-decision {{ background: #fce4ec; }} .kind-escalate {{ background: #fff3e0; }}
  .body {{ margin: .25rem 0 0; white-space: pre-wrap; }}
  .meta {{ display: flex; flex-wrap: wrap; gap: .3rem .9rem; margin-top: .3rem;
          font-size: .78rem; color: #666; }}
  .meta .k {{ font-weight: 600; margin-right: .3rem; }}
  .empty {{ color: #999; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
{subtitle and f'<div class="sub">{html.escape(subtitle)}</div>' or ""}
<ul>{body}</ul>
<script>
  // no behaviour needed — the file is a static audit trail
</script>
</body>
</html>
"""


def events_to_json(events: Iterable[TimelineEvent]) -> str:
    """Serialize events for machine consumption (the HTML replay's JSON cousin)."""
    return json.dumps([e.to_dict() for e in events], ensure_ascii=False, indent=2)
