"""Phase 9.1 — TriAgent tests (layer triggering, score range, PIT signal panel)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.sentiment.report_factor import ensure_report_scores
from src.sentiment.triagent import TriAgentSentiment, build_report_signal


# ---------------------------------------------------------------------------
# layer triggering (BERT mocked — heavy, not needed for routing logic)
# ---------------------------------------------------------------------------


class _FakeIngestor:
    def __init__(self, items):
        self._items = items

    def get_news(self, symbol, date):
        return self._items.get((symbol, date), [])


class _FakeBert:
    """BERT with genuine cross-article dispersion so the critic tier can fire."""

    def predict_batch(self, texts):
        return [0.75 if ("拉升" in t or "异动" in t) else -0.7 for t in texts]


class _FakeCritic:
    available = True

    def __init__(self):
        self.calls = 0

    def analyze(self, symbol, articles, bert_scores):
        self.calls += 1
        return 0.6


class _FakeLexicon:
    def score(self, text):
        if "利好" in text or "增长" in text:
            return 0.8 if "非常" not in text else 0.9
        if "下跌" in text or "亏损" in text:
            return -0.8
        return 0.0


def _make_news_item(text):
    from src.sentiment.ingestion import NewsItem

    return NewsItem(symbol="600519.SH", title=text, content="",
                    publish_time="2026-08-11 09:00", source="em", url="http://x/1")


def test_compute_emotion_none_when_no_news():
    agent = TriAgentSentiment(
        ingestor=_FakeIngestor({}),
        lexicon=_FakeLexicon(), bert=_FakeBert(), critic=_FakeCritic(),
    )
    s, tier = agent.compute_emotion("600519.SH", "2026-08-11")
    assert s == 0.5 and tier == "none"


def test_word_tier_decides_strong_sentiment():
    critic = _FakeCritic()
    agent = TriAgentSentiment(
        ingestor=_FakeIngestor({("600519.SH", "2026-08-11"): [_make_news_item("业绩增长超预期")]}),
        lexicon=_FakeLexicon(), bert=_FakeBert(), critic=critic,
    )
    s, tier = agent.compute_emotion("600519.SH", "2026-08-11")
    assert tier == "word"
    assert 0.5 < s <= 1.0
    assert critic.calls == 0


def test_bert_tier_handles_ambiguous_high_dispersion():
    critic = _FakeCritic()
    # lexicon 0.0 -> word tier won't decide; bert scores disperse & >=3 items -> critic
    items = [_make_news_item(t) for t in
             ("今日窄幅震荡", "盘中大幅异动", "尾盘快速拉升", "市场情绪谨慎")]
    agent = TriAgentSentiment(
        ingestor=_FakeIngestor({("600519.SH", "2026-08-11"): items}),
        lexicon=_FakeLexicon(), bert=_FakeBert(), critic=critic,
    )
    s, tier = agent.compute_emotion("600519.SH", "2026-08-11")
    assert tier == "critic"
    assert 0.0 <= s <= 1.0
    assert critic.calls == 1


def test_score_range_and_tier_on_report_titles():
    critic = _FakeCritic()
    agent = TriAgentSentiment(
        ingestor=_FakeIngestor({}), lexicon=_FakeLexicon(), bert=_FakeBert(), critic=critic,
    )
    out = agent.score_titles(["业绩增长超预期", "今日窄幅震荡"])
    assert list(out["tier"]) == ["word", "bert"]
    assert out["final"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# PIT report-signal panel
# ---------------------------------------------------------------------------


def test_build_report_signal_pit_and_decay():
    reports = pd.DataFrame(
        {
            "symbol": ["600519.SH", "600519.SH"],
            "title": ["t1", "t2"],
            "date": ["2026-08-01", "2026-08-20"],
        }
    )
    scores = pd.DataFrame(
        {"title": ["t1", "t2"], "final": [0.9, 0.2]}
    )
    trading = pd.date_range("2026-07-28", "2026-08-25", freq="B")
    sig = build_report_signal(reports, scores, trading, ["600519.SH"], decay_days=10)

    # no look-ahead: before the first report the signal must be NaN
    assert pd.isna(sig.loc[(pd.Timestamp("2026-07-28"), "600519.SH")])
    # report 08-01 active through 08-11 (decay 10d), then the 08-20 report takes over
    assert sig.loc[(pd.Timestamp("2026-08-03"), "600519.SH")] == pytest.approx(0.9)
    assert pd.isna(sig.loc[(pd.Timestamp("2026-08-13"), "600519.SH")])
    assert sig.loc[(pd.Timestamp("2026-08-24"), "600519.SH")] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# report score caching
# ---------------------------------------------------------------------------


def test_ensure_report_scores_cache_dedupes(tmp_path):
    class _FakeAgent:
        def __init__(self):
            self.calls = 0

        def score_titles(self, titles):
            self.calls += 1
            import pandas as pd

            return pd.DataFrame(
                {"title": titles, "final": [0.6 + i * 0.05 for i in range(len(titles))],
                 "lexicon": [0.5] * len(titles), "bert": [None] * len(titles),
                 "tier": ["word"] * len(titles)}
            )

    agent = _FakeAgent()
    reports = pd.DataFrame(
        {"symbol": ["600519.SH", "600519.SH", "000001.SZ"],
         "title": ["t1", "t2", "t1"], "date": ["2026-08-01"] * 3}
    )
    cache = str(tmp_path / "scores.parquet")
    s1 = ensure_report_scores(reports, agent, cache, tier="triagent", workers=1)
    assert len(s1) == 2 and set(s1["title"]) == {"t1", "t2"}
    assert agent.calls == 1  # one scoring pass

    # re-run: all titles cached -> no scoring call
    s2 = ensure_report_scores(reports, agent, cache, tier="triagent", workers=1)
    assert agent.calls == 1
    assert s2["final"].between(0.0, 1.0).all()
