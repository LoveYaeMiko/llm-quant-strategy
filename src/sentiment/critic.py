"""DeepSeekCritic — cross-article reasoning tier for the Chinese TriAgent.

The blueprint (§3.4) opens its own ``OpenAI`` client; this implementation
instead **injects the existing backend** built by
:func:`src.llm_client.build_llm_backend`, so the key never leaves
``configs/llm_routing.yaml`` / ``.env`` and token usage is recorded into the
shared :class:`src.cost_tracker.CostTracker` — the same budget discipline as
every other LLM call in the repo (blueprint §3.6: "所有 LLM 调用使用 DeepSeek").

Only high-dispersion samples reach the critic (TriAgent routing), so the API
cost is bounded to <5% of articles per the blueprint's risk table.

``analyze`` returns ``[0, +1]`` (0 = extremely negative, 0.5 = neutral,
1 = extremely positive). With no backend (offline / no key) it degrades to the
mean of the BERT scores rather than inventing a neutral — the pipeline stays
runnable and the critic only ever adds signal on top.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_SYSTEM = (
    "你是金融情感分析专家。分析以下新闻组，给出整体情感倾向。\n"
    "只输出一个数字（0~1），不包含任何解释：\n"
    "- 0：极度负面（重大利空）\n"
    "- 0.5：中性/无明显倾向\n"
    "- 1：极度正面（重大利好）"
)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class DeepSeekCritic:
    """Cross-sentence/group sentiment critic (TriAgent top tier). Backend injected."""

    def __init__(self, backend=None) -> None:
        self.backend = backend

    @property
    def available(self) -> bool:
        return self.backend is not None

    def analyze(
        self,
        symbol: str,
        articles: list[str],
        bert_scores: list[float],
    ) -> float:
        """Overall sentiment in ``[0, 1]`` for ``articles`` of ``symbol``."""
        if self.backend is None or not articles:
            return float(np.clip(np.mean(bert_scores), -1.0, 1.0) / 2 + 0.5)
        prompt = (
            f"{_SYSTEM}\n\n股票：{symbol}\n\n新闻及初步分析：\n"
            + "\n".join(
                f"[{i + 1}] {a[:200]}… (初步情感: {s:+.2f})"
                for i, (a, s) in enumerate(zip(articles, bert_scores))
            )
            + "\n\n请给出整体情感裁定（只输出 0~1 数字）："
        )
        try:
            raw = self.backend.complete(prompt, temperature=0.1, max_tokens=16)
            match = _NUMBER_RE.search(raw.strip())
            if not match:
                return self._fallback(bert_scores)
            val = float(match.group())
            return float(np.clip(val, 0.0, 1.0))
        except Exception as exc:  # noqa: BLE001 — critic must never kill the run
            logger.warning("DeepSeekCritic failed (%s); using BERT mean", exc)
            return self._fallback(bert_scores)

    @staticmethod
    def _fallback(bert_scores: list[float]) -> float:
        return float(np.clip(np.mean(bert_scores), -1.0, 1.0) / 2 + 0.5)
