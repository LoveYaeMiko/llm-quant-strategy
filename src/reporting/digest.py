"""C2 — structured daily digest + webhook notifier.

Two concerns, one file:

* :func:`build_digest` turns a run's raw state (accepted factors, checklist
  verdicts, cost snapshot) into a **deterministic** four-section markdown digest
  so a researcher can skim "what survived the gates today" without reading logs.
* :class:`WebhookNotifier` pushes text to a 飞书 / DingTalk-compatible webhook.
  It is a **no-op** when no URL is configured (returns ``False`` instead of
  raising), so the loop can always "send" the digest and only actually transmit
  when a human has wired ``notify.webhook_url``.

The webhook URL carries a token, so it must live in the gitignored ``.env`` and
be referenced as ``${NOTIFY_WEBHOOK_URL}`` — the same pattern every other secret
in this project uses.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Iterable, Optional

# 飞书 / DingTalk both accept this minimal "text" message shape. A bare "text"
# body also works for a generic Slack-style incoming webhook.
_TEXT_PAYLOAD = {"msg_type": "text", "content": {"text": ""}}

_SECTION_HEADERS = ("一、今日产出", "二、门控与风控", "三、成本", "四、下一步")


def _fmt(v: object, nd: int = 4) -> str:
    """Render a numeric value compactly, tolerating None/NaN-ish inputs."""
    if v is None:
        return "-"
    try:
        f = float(v)  # type: ignore[arg-type]
        if f != f:  # NaN
            return "-"
        return f"{f:.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _factor_line(f: dict) -> str:
    # ``pool``/``cli`` accepted entries are nested ``{factor: {name, formula},
    # metrics: {...}}``; a flat ``{name, formula, rank_ic, ...}`` dict also works.
    inner = f.get("factor") if isinstance(f.get("factor"), dict) else f
    metrics = f.get("metrics") if isinstance(f.get("metrics"), dict) else f
    name = inner.get("name") or f.get("name") or "?"
    formula = inner.get("formula") or f.get("formula") or ""
    ic = _fmt(metrics.get("rank_ic", metrics.get("ic")))
    ts = _fmt(metrics.get("tail_spread"))
    dsr = _fmt(metrics.get("deflated_sharpe"))
    line = f"- **{name}** — RankIC `{ic}`"
    if ts != "-":
        line += f" · 尾部价差 `{ts}`"
    if dsr != "-":
        line += f" · DeflatedSharpe `{dsr}`"
    if formula:
        line += f"\n  - `{formula}`"
    return line


def build_digest(
    run_id: str,
    accepted: Optional[Iterable[dict]] = None,
    checklist: Optional[Iterable[object]] = None,
    cost_snapshot: Optional[dict] = None,
    notes: Optional[str] = None,
) -> str:
    """Render one run's state as a four-section Chinese markdown digest.

    ``accepted``      — the factors that passed today's gates (dicts from
                        ``pool.evaluate_pool`` output, read defensively).
    ``checklist``     — ``CheckResult`` objects (or dicts) from
                        ``checklist.run_all``.
    ``cost_snapshot`` — ``CostTracker.snapshot()`` output.
    ``notes``         — free-text "next steps"; auto-filled when omitted.
    """
    accepted = list(accepted or [])
    checklist = list(checklist or [])
    lines: list[str] = [f"# 因子挖掘日报 — {run_id}", ""]

    # -- 一、今日产出 --------------------------------------------------------
    lines.append(f"## {_SECTION_HEADERS[0]}")
    if accepted:
        lines.append(f"本周期通过门控的因子 **{len(accepted)}** 个：")
        lines.extend(_factor_line(f) for f in accepted)
    else:
        lines.append("本周期无因子通过门控。")
    lines.append("")

    # -- 二、门控与风控 ------------------------------------------------------
    lines.append(f"## {_SECTION_HEADERS[1]}")
    if checklist:
        for c in checklist:
            if isinstance(c, dict):
                name, passed, detail = c.get("name", "?"), bool(c.get("passed")), c.get("detail", "")
            else:
                name = getattr(c, "name", "?")
                passed = bool(getattr(c, "passed", False))
                detail = getattr(c, "detail", "")
            mark = "✅" if passed else "❌"
            lines.append(f"- {mark} `{name}` — {detail}")
    else:
        lines.append("- 无门控记录。")
    lines.append("")

    # -- 三、成本 ------------------------------------------------------------
    lines.append(f"## {_SECTION_HEADERS[2]}")
    if cost_snapshot:
        total = _fmt(cost_snapshot.get("total_cost_usd"), 4)
        budget = _fmt(cost_snapshot.get("monthly_budget_usd"), 2)
        n_calls = cost_snapshot.get("n_calls", 0)
        ok = "✅ 预算内" if cost_snapshot.get("under_budget") else "⚠️ 超预算"
        lines.append(f"- 累计成本 `${total}` / 月预算 `${budget}`（{ok}，`{n_calls}` 次调用）")
        by_model = cost_snapshot.get("by_model") or {}
        if by_model:
            lines.append(f"- 按模型拆分：{', '.join(f'{m} `${_fmt(c, 4)}`' for m, c in sorted(by_model.items()))}")
    else:
        lines.append("- 无成本快照。")
    lines.append("")

    # -- 四、下一步 ----------------------------------------------------------
    lines.append(f"## {_SECTION_HEADERS[3]}")
    lines.append(notes or "继续下一轮演化/回测循环。")
    return "\n".join(lines)


class WebhookNotifier:
    """Push a text message to a 飞书 / DingTalk-compatible incoming webhook.

    ``url=None`` (the default) makes every call a silent no-op returning
    ``False`` — the loop can unconditionally call ``send`` without knowing
    whether a human wired a webhook. ``timeout`` guards against a hung endpoint
    stalling the research loop.
    """

    def __init__(self, url: Optional[str] = None, timeout: float = 10.0) -> None:
        self.url = (url or "").strip()
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.url)

    def send(self, text: str) -> bool:
        """POST ``text``; return whether it was actually transmitted."""
        if not self.configured:
            return False
        payload = dict(_TEXT_PAYLOAD)
        payload["content"] = {"text": text}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return 200 <= resp.status < 300

    def send_digest(self, digest: str) -> bool:
        return self.send(digest)


def build_notifier(config) -> WebhookNotifier:
    """Read ``notify.webhook_url`` (``${NOTIFY_WEBHOOK_URL}``) off the config.

    ``config`` is a :class:`src.config.Config`; any object exposing ``.get`` with
    a default also works, so the helper stays testable without the full loader.
    """
    url = config.get("notify.webhook_url", "") if config else ""
    return WebhookNotifier(url)
