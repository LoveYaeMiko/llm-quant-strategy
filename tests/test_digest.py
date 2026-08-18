"""C2 — structured daily digest + webhook notifier."""

from __future__ import annotations

import json
import urllib.request

from src.config import Config
from src.reporting import WebhookNotifier, build_digest, build_notifier


class _Check:
    def __init__(self, name, passed, detail):
        self.name, self.passed, self.detail = name, passed, detail


def test_build_digest_four_sections():
    digest = build_digest(
        run_id="run-001",
        accepted=[
            {"name": "f1", "formula": "Rank(Close)", "rank_ic": 0.05,
             "tail_spread": 0.12, "deflated_sharpe": 1.1},
        ],
        checklist=[_Check("pit", True, "clean"), _Check("cost", False, "over")],
        cost_snapshot={
            "total_cost_usd": 1.2345, "monthly_budget_usd": 500.0,
            "under_budget": True, "n_calls": 42,
            "by_model": {"deepseek-v4-flash": 1.2345},
        },
        notes="明天跑全样本回测。",
    )
    assert "因子挖掘日报 — run-001" in digest
    for hdr in ("一、今日产出", "二、门控与风控", "三、成本", "四、下一步"):
        assert hdr in digest
    # accepted factor line carries the metrics
    assert "f1" in digest and "0.0500" in digest and "DeflatedSharpe" in digest
    # checklist verdicts render pass/fail
    assert "✅ `pit`" in digest and "❌ `cost`" in digest
    # cost + notes
    assert "1.2345" in digest and "明天跑全样本回测。" in digest


def test_build_digest_empty_degrades_gracefully():
    digest = build_digest(run_id="empty")
    assert "本周期无因子通过门控。" in digest
    assert "无门控记录。" in digest
    assert "无成本快照。" in digest
    assert "继续下一轮演化/回测循环。" in digest


def test_build_digest_handles_dict_checklist_and_none_metrics():
    digest = build_digest(
        run_id="r",
        accepted=[{"name": "f", "rank_ic": None, "tail_spread": None}],
        checklist=[{"name": "x", "passed": True, "detail": "ok"}],
    )
    assert "✅ `x` — ok" in digest
    assert "-" in digest  # None metrics render as a dash


def test_webhook_notifier_noop_without_url():
    n = WebhookNotifier()  # url=None
    assert not n.configured
    assert n.send("hello") is False


def test_webhook_notifier_posts_feishu_payload(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["timeout"] = timeout
        return _FakeResp(200)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    n = WebhookNotifier("https://open.feishu.cn/hook/abc")
    assert n.configured
    assert n.send_digest("# 日报\n内容") is True
    body = json.loads(captured["data"].decode("utf-8"))
    assert body["msg_type"] == "text"
    assert body["content"]["text"] == "# 日报\n内容"
    assert captured["timeout"] == 10.0


class _FakeResp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_build_notifier_reads_config_or_defaults():
    assert not build_notifier(Config({})).configured
    assert not build_notifier(Config({"notify": {"webhook_url": ""}})).configured
    n = build_notifier(Config({"notify": {"webhook_url": "https://x/hook"}}))
    assert n.url == "https://x/hook"
