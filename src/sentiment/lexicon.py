"""ChineseFinancialLexicon — word-level sentiment for the TriAgent bottom tier.

Chinese has no tokenizing spaces, so unlike the English VADER-style scorer in
:mod:`src.data.text_feeds` this lexicon does **forward maximum matching (FMM)**:
at each position it greedily matches the longest dictionary entry (sentiment
word, intensifier, or negator). Negators flip a sentiment word's sign only when
*adjacent* (no gap), which keeps ``不看好``/``未达标``/``无风险`` correct while
avoiding false flips from single characters embedded in longer words (e.g.
``无`` inside ``无形资产``). Intensifiers apply when adjacent to the word (or to
its negator), so ``非常不看好`` and ``大幅利好`` are both handled.

``score`` returns ``[-1, +1]`` via ``net / (1 + total)`` — direction and
strength both survive (``大幅利好`` ≈ +0.64 vs ``利好`` ≈ +0.50), unlike a pure
sign-ratio which would erase intensity. This tier is meant to be *cheap and
fast* (~1000+ items/sec); ambiguity is deliberately pushed up to the BERT tier.
"""

from __future__ import annotations

import re

# --- word lists (blueprint §3.2 + high-frequency A-share news terms) ---------

_POSITIVE = {
    "增长", "上涨", "盈利", "利好", "超预期", "买入", "推荐", "新高", "突破",
    "反弹", "反转", "改善", "提升", "加速", "放量", "活跃", "强劲", "乐观",
    "看好", "加仓", "增持", "跑赢", "领先", "龙头", "稀缺", "溢价", "低估",
    "价值", "分红", "回购", "业绩", "确定性", "高景气", "涨停", "预增", "扭亏",
    "中标", "获批", "签约", "投产", "量产", "扩张", "丰厚", "亮眼", "创纪录",
    "拟增持", "突破性", "高速增长", "双增", "净增",
}

_NEGATIVE = {
    "下跌", "亏损", "利空", "低于预期", "减持", "卖出", "警惕", "新低", "破位",
    "回调", "恶化", "下滑", "放缓", "萎缩", "缩量", "低迷", "悲观", "看空",
    "减仓", "回避", "跑输", "滞后", "风险", "泡沫", "高估", "踩踏", "恐慌",
    "崩盘", "停牌", "问询", "处罚", "诉讼", "违约", "退市", "暴雷", "跌停",
    "预亏", "立案", "调查", "违规", "质押", "爆仓", "失信", "债务", "逾期",
    "罚款", "警示函", "问询函", "终止", "中止", "亏损扩大", "商誉减值", "计提",
}

_INTENSIFIERS: dict[str, float] = {
    "非常": 1.5, "极度": 2.0, "轻微": 0.5, "大幅": 1.8, "小幅": 0.6,
    "持续": 1.2, "显著": 1.6, "温和": 0.7, "突然": 1.3, "连续": 1.4,
    "创": 1.2, "再度": 1.3, "或": 0.5,
}

_NEGATORS = {"不", "未", "无", "非", "不是", "不会", "尚未", "并未", "没有"}

# Multi-char first so FMM matches "不是" before bare "不".
_ALL_WORDS = sorted(
    _POSITIVE | _NEGATIVE | set(_INTENSIFIERS) | _NEGATORS, key=len, reverse=True
)


class ChineseFinancialLexicon:
    """Greedy-FMM word-level sentiment scorer over Chinese financial text."""

    def __init__(
        self,
        positive: set[str] | None = None,
        negative: set[str] | None = None,
        intensifiers: dict[str, float] | None = None,
        negators: set[str] | None = None,
    ) -> None:
        self.positive = set(positive or _POSITIVE)
        self.negative = set(negative or _NEGATIVE)
        self.intensifiers = dict(intensifiers or _INTENSIFIERS)
        self.negators = set(negators or _NEGATORS)
        self._words = sorted(
            self.positive | self.negative | set(self.intensifiers) | self.negators,
            key=len,
            reverse=True,
        )

    # ------------------------------------------------------------------ utils
    def _match(self, text: str, i: int) -> str | None:
        """Longest dictionary entry starting at ``i``, or None."""
        for w in self._words:
            if text.startswith(w, i):
                return w
        return None

    def _preceding_negator(self, text: str, start: int) -> str | None:
        """A negator whose *end* touches ``start`` (adjacent, no gap)."""
        for w in self.negators:
            s = start - len(w)
            if s >= 0 and text[s:start] == w:
                return w
        return None

    def _preceding_intensifier(self, text: str, end: int) -> float:
        """Multiplier for an intensifier ending exactly at ``end``."""
        for w, wgt in self.intensifiers.items():
            s = end - len(w)
            if s >= 0 and text[s:end] == w:
                return wgt
        return 1.0

    # ----------------------------------------------------------------- scoring
    def score(self, text: str) -> float:
        """Sentiment in ``[-1, +1]`` (negative..positive) for ``text``."""
        if not text:
            return 0.0
        pos_sum = neg_sum = 0.0
        i = 0
        n = len(text)
        while i < n:
            w = self._match(text, i)
            if w is None:
                i += 1
                continue
            base = 0.0
            if w in self.positive:
                base = 1.0
            elif w in self.negative:
                base = -1.0
            if base != 0.0:
                # Intensity anchors to the negator when one is present, so
                # 「非常不看好」scales 看好's flip (非常 before 不), while
                # 「大幅利好」scales 利好 directly.
                neg = self._preceding_negator(text, i)
                anchor = i if neg is None else i - len(neg)
                intensity = self._preceding_intensifier(text, anchor)
                if neg is not None:
                    base = -base
                adj = base * intensity
                if adj > 0:
                    pos_sum += adj
                else:
                    neg_sum += -adj
                i += len(w)
                continue
            i += len(w)
        total = pos_sum + neg_sum
        if total == 0:
            return 0.0
        return (pos_sum - neg_sum) / (1.0 + total)
