"""TriAgentSentiment — layered Chinese sentiment committee (Phase 9.1).

Escalation path (blueprint §3.5, with the scale mixing cleaned up):

1. **word tier**     — :class:`ChineseFinancialLexicon` scores every article in
   ``[-1, +1]`` (~1000+ items/sec). If the *mean* magnitude is at or above
   ``lexicon_threshold`` the case is already decided and the cheaper tiers win.
2. **BERT tier**     — only when the word tier is ambiguous (|mean| < threshold):
   :class:`ChineseBertSentiment` (FinBERT_zh) re-scores the articles, again in
   ``[-1, +1]``.
3. **critic tier**   — only for *high-dispersion* groups (cross-article std >
   ``bert_threshold`` and ≥ ``critic_min_articles``): :class:`DeepSeekCritic`
   reasons across the articles and the weighted fusion resolves the conflict.

Every tier keeps ``[-1, +1]`` internally; ``[0, +1]`` normalization happens
only at the factor boundary (``compute_emotion`` / ``compute_factor``) so the
cross-sectional factor is comparable with the Phase 8 factors' scale.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .bert import ChineseBertSentiment
from .critic import DeepSeekCritic
from .ingestion import NewsIngestor
from .lexicon import ChineseFinancialLexicon

logger = logging.getLogger(__name__)


def _mean01(x: list[float]) -> float:
    """Mean of a ``[-1, 1]`` list mapped to ``[0, 1]``."""
    return float((np.mean(x) + 1.0) / 2.0) if x else 0.5


def build_report_signal(
    reports: pd.DataFrame,
    scores: pd.DataFrame,
    trading_dates: Iterable[pd.Timestamp | str],
    symbols: Iterable[str],
    decay_days: int = 10,
) -> pd.Series:
    """PIT cross-sectional sentiment panel from dated report scores.

    For every trading date ``t`` and symbol ``s``, the signal is the sentiment
    of the **most recent** report with ``report_date <= t`` and
    ``report_date >= t - decay_days`` (carry-forward until the next report or
    expiry). Reports are unknown before their publish date, so no look-ahead.
    Symbols with no recent report are NaN and drop out of the daily rank-IC.
    Returns a ``(date, symbol)``-indexed Series of ``[0, 1]`` scores.
    """
    trading = pd.DatetimeIndex(sorted(pd.to_datetime(trading_dates)))
    symbols = list(symbols)
    sig = pd.Series(
        np.nan,
        index=pd.MultiIndex.from_product([trading, symbols], names=["date", "symbol"]),
    )
    sc = scores.set_index("title")["final"]
    sub = reports[reports["title"].isin(sc.index)].copy()
    if sub.empty:
        return sig
    sub["score"] = sub["title"].map(sc)
    sub["rd"] = pd.to_datetime(sub["date"])

    for symbol, group in sub.groupby("symbol"):
        if symbol not in symbols:
            continue
        g = group.sort_values("rd")
        rdates = g["rd"].to_numpy()
        rvals = g["score"].to_numpy()
        pos = np.searchsorted(rdates, trading.to_numpy(), side="right") - 1
        valid = pos >= 0
        if not valid.any():
            continue
        t = trading.to_numpy()[valid]
        rp = pos[valid]
        ages = (t - rdates[rp]) / np.timedelta64(1, "D")
        keep = ages <= decay_days
        tk = t[keep]
        if tk.size:
            sig.loc[(tk, symbol)] = rvals[rp[keep]]
    return sig


class TriAgentSentiment:
    """Per-symbol, per-date layered Chinese sentiment scorer."""

    def __init__(
        self,
        ingestor: Optional[NewsIngestor] = None,
        lexicon: Optional[ChineseFinancialLexicon] = None,
        bert: Optional[ChineseBertSentiment] = None,
        critic: Optional[DeepSeekCritic] = None,
        lexicon_threshold: float = 0.3,
        bert_threshold: float = 0.25,
        critic_min_articles: int = 3,
    ) -> None:
        self.ingestor = ingestor or NewsIngestor()
        self.lexicon = lexicon or ChineseFinancialLexicon()
        self.bert = bert or ChineseBertSentiment()
        self.critic = critic or DeepSeekCritic()
        self.lexicon_threshold = lexicon_threshold
        self.bert_threshold = bert_threshold
        self.critic_min_articles = critic_min_articles

    # ------------------------------------------------------------------ tiers
    def _tier_word(self, texts: list[str]) -> list[float]:
        return [self.lexicon.score(t) for t in texts]

    def _tier_bert(self, texts: list[str]) -> list[float]:
        return self.bert.predict_batch(texts)

    def _tier_critic(self, symbol: str, texts: list[str], bert_scores: list[float]) -> float:
        return self.critic.analyze(symbol, texts, bert_scores)

    # ------------------------------------------------------------- per-symbol
    def compute_emotion(self, symbol: str, date: str) -> tuple[float, str]:
        """``(sentiment [0,1], tier_used)`` for one symbol on one date.

        The tier string is returned alongside so tests and reports can audit
        which layers actually fired (cost control).
        """
        items = self.ingestor.get_news(symbol, date)
        texts = [it.to_text() for it in items]
        if not texts:
            return 0.5, "none"

        lex_scores = self._tier_word(texts)
        mean_lex = float(np.mean(lex_scores))

        if abs(mean_lex) >= self.lexicon_threshold:
            return _mean01(lex_scores), "word"

        bert_scores = self._tier_bert(texts)
        disp = float(np.std(bert_scores))
        if disp > self.bert_threshold and len(texts) >= self.critic_min_articles:
            critic_score = self._tier_critic(symbol, texts, bert_scores)
            final = (
                0.5 * critic_score
                + 0.3 * _mean01(bert_scores)
                + 0.2 * _mean01(lex_scores)
            )
            return float(np.clip(final, 0.0, 1.0)), "critic"

        return _mean01(bert_scores), "bert"

    # ---------------------------------------------------------- report titles
    def score_titles(self, titles: list[str]) -> pd.DataFrame:
        """Per-title sentiment in ``[0, 1]`` using the word→BERT ladder.

        Research-report titles are short texts where a single title rarely needs
        cross-article reasoning, so the DeepSeek critic (the cost lever) stays
        off this path — the lexicon decides high-confidence titles and BERT
        handles the rest. Returns a DataFrame with ``[title, lexicon, bert,
        final, tier]`` for caching/auditing.
        """
        lex_scores = [self.lexicon.score(t) for t in titles]
        need_bert = [abs(l) < self.lexicon_threshold for l in lex_scores]
        bert_scores: list[float | None] = [None] * len(titles)
        if any(need_bert):
            b = self.bert.predict_batch([t for t, n in zip(titles, need_bert) if n])
            it = iter(b)
            for i in range(len(titles)):
                if need_bert[i]:
                    bert_scores[i] = next(it)
        rows = []
        for title, ls, bs in zip(titles, lex_scores, bert_scores):
            if bs is None:
                rows.append({"title": title, "lexicon": round(ls, 4),
                             "bert": None, "final": round(_mean01([ls]), 4), "tier": "word"})
            else:
                rows.append({"title": title, "lexicon": round(ls, 4), "bert": round(bs, 4),
                             "final": round(_mean01([bs]), 4), "tier": "bert"})
        return pd.DataFrame(rows)

    # -------------------------------------------------------------- cross-sec
    def compute_factor(self, symbols: Iterable[str], as_of_date: str) -> pd.Series:
        """Cross-sectional sentiment factor (0..1) for ``symbols`` on ``date``.

        Symbols with no news on the date map to 0.5 (neutral) and remain
        cross-sectionally rankable — the Phase 9.1 factor output layer.
        """
        scores: dict[str, float] = {}
        for symbol in symbols:
            s, _ = self.compute_emotion(symbol, as_of_date)
            scores[symbol] = s
        return pd.Series(scores)
