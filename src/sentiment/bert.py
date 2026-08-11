"""ChineseBertSentiment — sentence-level BERT tier for the Chinese TriAgent.

Loads **FinBERT_zh** (``yiyanghkust/finbert-tone-chinese``) by preference — the
blueprint's §3.3 recommendation, fine-tuned on Chinese financial text, labels
``{0: Neutral, 1: Positive, 2: Negative}`` — and falls back to plain
``bert-base-chinese`` when the FinBERT weights are absent (that fallback has an
*untrained* classifier head, so it only separates lexical polarity weakly;
``bert.py``'s role in the layered pipeline is the medium-confidence tier, with
the DeepSeek critic resolving what stays ambiguous).

Weights are read from ``data/models/`` with ``local_files_only=True`` (China
network blocks huggingface.co; the mirror files are downloaded at setup). The
model moves to CUDA automatically when ``torch.cuda.is_available()``; with the
CPU-only torch build that currently reports False and inference stays on CPU.

``score`` returns ``[-1, +1]`` as ``P(positive) - P(negative)`` (Neutral is
dropped from the signed axis), matching the :class:`SentimentScorer` protocol
used elsewhere in the repo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .lexicon import ChineseFinancialLexicon

logger = logging.getLogger(__name__)

# Local weight dirs, in preference order.
_LOCAL_MODELS = ("data/models/finbert_zh", "data/models/bert-base-chinese")

_LABEL_NEUTRAL, _LABEL_POSITIVE, _LABEL_NEGATIVE = 0, 1, 2


class ChineseBertSentiment:
    """Chinese BERT sentence-level sentiment classifier (TriAgent mid tier)."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 256,
    ) -> None:
        self.model_dir = model_dir or self._resolve_model_dir()
        self.max_length = max_length
        self._load(device)

    @staticmethod
    def _resolve_model_dir() -> str:
        for cand in _LOCAL_MODELS:
            if Path(cand, "model.safetensors").is_file() or Path(cand, "pytorch_model.bin").is_file():
                return cand
        raise FileNotFoundError(
            "no local BERT weights found under data/models/; run the setup "
            "download from hf-mirror.com (finbert_zh preferred)"
        )

    def _load(self, device: Optional[str]) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.model.eval()
        # Auto-detect GPU when the caller doesn't pin a device.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        logger.info("ChineseBertSentiment loaded from %s on %s", self.model_dir, self.device)

    # ----------------------------------------------------------------- scoring
    def score(self, text: str) -> float:
        """Sentiment in ``[-1, +1]`` for one text (P(pos) - P(neg))."""
        if not text:
            return 0.0
        inputs = self.tokenizer(
            text[: self.max_length],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=1)
        p = probs[0].cpu().numpy()
        pos = float(p[_LABEL_POSITIVE])
        neg = float(p[_LABEL_NEGATIVE])
        return float(np.clip(pos - neg, -1.0, 1.0))

    def predict_batch(self, texts: list[str]) -> list[float]:
        """Batch scores, falling back to the lexicon for empty strings."""
        if not texts:
            return []
        lex = ChineseFinancialLexicon()
        inputs = self.tokenizer(
            [t[: self.max_length] for t in texts],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=1).cpu().numpy()
        return [
            float(np.clip(r[_LABEL_POSITIVE] - r[_LABEL_NEGATIVE], -1.0, 1.0))
            if t else lex.score(t)
            for t, r in zip(texts, probs)
        ]
