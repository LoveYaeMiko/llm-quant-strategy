"""Text feeds — point-in-time news / transcript ingestion + TriAgent sentiment.

Beyond the blueprint's bare ``text_feeds.py`` this module implements **TriAgent**
(review.md §2.3): a sentiment committee layered by contextual granularity —

1. **word-level**     : a VADER-style financial lexicon (dependency-free);
2. **sentence-level** : an optional FinBERT-class transformer;
3. **cross-sentence** : an optional LLM critic that reasons across sentences.

The router promotes cheap tiers first and only escalates to the LLM critic for
ambiguous items. That escalation-avoidance is where the 9.3M USD/yr saving at
10M users (TriAgent) comes from — replicated here as a cost-conscious default.
All retrieval is point-in-time: ``as_of(t)`` only returns items timestamped
<= t, so text can never leak into a backtest before it existed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class TextItem:
    timestamp: pd.Timestamp
    symbol: str
    source: str
    title: str
    body: str
    lang: str = "en"

    def to_text(self) -> str:
        return f"{self.title}. {self.body}"

    def __post_init__(self) -> None:
        self.timestamp = pd.Timestamp(self.timestamp)


class TextFeed:
    """In-memory store of news/transcripts with PIT retrieval."""

    def __init__(self, items: Optional[Iterable[TextItem]] = None) -> None:
        self._items: list[TextItem] = list(items or [])

    def add(self, item: TextItem | Iterable[TextItem]) -> None:
        if isinstance(item, TextItem):
            self._items.append(item)
        else:
            self._items.extend(item)

    def __len__(self) -> int:
        return len(self._items)

    def as_of(self, timestamp: str | pd.Timestamp, symbols: Optional[Iterable[str]] = None) -> list[TextItem]:
        """All items timestamped <= t (PIT guarantee for text)."""
        t = pd.Timestamp(timestamp)
        out = [i for i in self._items if i.timestamp <= t]
        if symbols is not None:
            allowed = set(symbols)
            out = [i for i in out if i.symbol in allowed]
        return sorted(out, key=lambda i: i.timestamp)


# ---------------------------------------------------------------------------
# Sentiment committee (TriAgent)
# ---------------------------------------------------------------------------


class SentimentScorer(Protocol):
    """Anything that maps text to a [-1, +1] sentiment score."""

    def score(self, text: str) -> float: ...


# A compact financial sentiment lexicon (VADER-style unigram weights).
_FIN_LEXICON: dict[str, float] = {
    # positive
    "beat": 0.5, "beats": 0.5, "beating": 0.4, "outperform": 0.5, "outperforms": 0.5,
    "growth": 0.4, "grew": 0.4, "record": 0.35, "strong": 0.4, "stronger": 0.35,
    "upgrade": 0.5, "upgraded": 0.5, "raised": 0.35, "raise": 0.3, "increase": 0.25,
    "surge": 0.4, "surged": 0.4, "rally": 0.35, "gain": 0.25, "gains": 0.25,
    "profit": 0.35, "profitable": 0.45, "positive": 0.35, "buy": 0.3, "good": 0.3,
    "exceed": 0.4, "exceeded": 0.45, "boost": 0.3, "confidence": 0.2, "bullish": 0.5,
    "expansion": 0.3, "momentum": 0.3, "award": 0.35, "win": 0.3, "success": 0.4,
    # negative
    "miss": -0.5, "missed": -0.5, "underperform": -0.5, "downgrade": -0.5,
    "downgraded": -0.5, "cut": -0.3, "cuts": -0.3, "decline": -0.35, "declined": -0.35,
    "drop": -0.35, "dropped": -0.35, "fall": -0.3, "fell": -0.3, "loss": -0.4,
    "losses": -0.45, "negative": -0.35, "sell": -0.3, "weak": -0.35, "weaker": -0.35,
    "warning": -0.4, "warn": -0.3, "risk": -0.25, "risks": -0.25, "lawsuit": -0.4,
    "litigation": -0.35, "fraud": -0.6, "investigation": -0.4, "bearish": -0.5,
    "bankrupt": -0.6, "bankruptcy": -0.6, "layoff": -0.4, "layoffs": -0.4,
    "recall": -0.4, "delist": -0.5, "concern": -0.25, "struggle": -0.35,
}

_WORD_RE = re.compile(r"[a-z]+")


class LexiconSentiment(SentimentScorer):
    """Word-level scorer — VADER-style but with a financial lexicon."""

    def __init__(self, lexicon: Optional[dict[str, float]] = None) -> None:
        self.lexicon = dict(lexicon or _FIN_LEXICON)

    def score(self, text: str) -> float:
        words = _WORD_RE.findall(text.lower())
        if not words:
            return 0.0
        pos = neg = 0.0
        for w in words:
            if w in self.lexicon:
                v = self.lexicon[w]
                if v > 0:
                    pos += v
                else:
                    neg += v
        total = pos + abs(neg)
        if total == 0:
            return 0.0
        # normalized to [-1, 1] with the net/absolute ratio
        return (pos + neg) / (pos + abs(neg)) if total > 0 else 0.0


class FinBertSentiment(SentimentScorer):
    """Sentence-level transformer (TriAgent mid tier). Optional import."""

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        self.model_name = model_name
        self._pipeline = None
        try:  # pragma: no cover - environment dependent
            from transformers import pipeline

            self._pipeline = pipeline("sentiment-analysis", model=model_name)
        except Exception:
            self._pipeline = None

    @property
    def available(self) -> bool:
        return self._pipeline is not None

    def score(self, text: str) -> float:
        if self._pipeline is None:
            return LexiconSentiment().score(text)
        out = self._pipeline(text[:512])[0]
        label, conf = out["label"], out["score"]
        return conf if label.lower() in ("positive", "bullish") else -conf


class LLMCriticSentiment(SentimentScorer):
    """Cross-sentence reasoning critic (TriAgent top tier). LLM backend injected.

    The critic reads the full item and returns a structured JSON sentiment. If no
    backend is configured it degrades to the lexicon, keeping the pipeline
    offline-runnable.
    """

    def __init__(self, backend=None) -> None:
        self.backend = backend

    def score(self, text: str) -> float:
        if self.backend is None:
            return LexiconSentiment().score(text)
        prompt = (
            "Rate the sentiment of this financial text on [-1, +1] considering "
            "the *combined* meaning across sentences, not isolated keywords.\n\n"
            f"TEXT: {text[:1500]}\n\nReturn only a JSON object "
            '{"sentiment": <float between -1 and 1>}.'
        )
        try:
            raw = self.backend.complete(prompt, temperature=0.0)
            import json

            parsed = json.loads(raw)
            val = float(parsed["sentiment"])
            return max(-1.0, min(1.0, val))
        except Exception:
            return LexiconSentiment().score(text)


class TriAgentSentiment:
    """Layered committee with cheap-first routing (review.md §2.3).

    ``score`` returns ``(value, tier)`` where tier is one of ``word``, ``sentence``,
    ``critic``. The router only escalates when the cheap tier's confidence
    (distance from 0) is below ``escalate_threshold`` — this is the cost lever.
    """

    def __init__(
        self,
        lexicon: Optional[SentimentScorer] = None,
        sentence: Optional[SentimentScorer] = None,
        critic: Optional[SentimentScorer] = None,
        escalate_threshold: float = 0.25,
    ) -> None:
        self.lexicon = lexicon or LexiconSentiment()
        self.sentence = sentence or FinBertSentiment()
        self.critic = critic or LLMCriticSentiment()
        self.escalate_threshold = escalate_threshold

    def score(self, text: str) -> tuple[float, str]:
        word = self.lexicon.score(text)
        if abs(word) >= self.escalate_threshold:
            return word, "word"
        sent = self.sentence.score(text)
        if abs(sent) >= self.escalate_threshold:
            return sent, "sentence"
        return self.critic.score(text), "critic"

    def aggregate_daily_signal(
        self,
        feed: TextFeed,
        timestamp: str | pd.Timestamp,
        symbols: Optional[Iterable[str]] = None,
    ) -> pd.Series:
        """Mean sentiment per symbol as-of ``timestamp`` (PIT text signal)."""
        items = feed.as_of(timestamp, symbols=symbols)
        if not items:
            return pd.Series(dtype=float)
        rows: dict[str, list[float]] = {}
        for item in items:
            val, _ = self.score(item.to_text())
            rows.setdefault(item.symbol, []).append(val)
        return pd.Series({s: float(np.mean(v)) for s, v in rows.items()})
