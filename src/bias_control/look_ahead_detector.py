"""FinCAD context-aware decoding — suppress look-ahead at inference time.

FINSABER Defect 02 (LLM recalls the future from memory) is not fixed by a PIT
loader, because the leak lives *inside* the pretrained weights. FinCAD's
"Context-Aware Decoding" (review.md §2.4) therefore adapts at inference: during
generation we penalise the logits of every candidate token whose surface form
embeds a date strictly after the current timestamp T. The model can still say
"we raised guidance" — it just cannot ground that on a *future date*, which is
the vector through which memorised outcomes leak into a backtest.

Blueprint verification ("FinCAD Check"): a cheating factor — one that quietly
uses the future — must lose >50% of its IC once suppression is applied.
:class:`LookAheadAudit` measures exactly that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_YYYYMMDD = re.compile(r"(?<!\d)((?:19|20)\d{2})[-/](\d{1,2})[-/](\d{1,2})(?!\d)")
_YYYYMM = re.compile(r"(?<!\d)((?:19|20)\d{2})[-/](\d{1,2})(?![-/]?\d)")
_YYYY = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_MONTH_YEAR = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+((?:19|20)\d{2})\b", re.I)


def extract_dates(text: str) -> list[tuple[str, pd.Timestamp]]:
    """Return ``(mention, interpreted_date)`` pairs found in ``text``.

    A bare year ``2024`` is interpreted conservatively as 2024-12-31 — the
    latest instant that year could refer to — so any use of a future year is
    flagged. Full dates and month+year are interpreted directly.
    """
    found: list[tuple[str, pd.Timestamp]] = []
    low = text
    for m in _YYYYMMDD.finditer(low):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            found.append((m.group(0), pd.Timestamp(y, mo, d)))
        except ValueError:
            continue
    for m in _YYYYMM.finditer(low):
        y, mo = int(m.group(1)), int(m.group(2))
        try:
            found.append((m.group(0), pd.Timestamp(y, mo, 1)))
        except ValueError:
            continue
    for m in _MONTH_YEAR.finditer(low):
        mo, y = _MONTHS[m.group(1).lower()[:3]], int(m.group(2))
        found.append((m.group(0), pd.Timestamp(y, mo, 1)))
    for m in _YYYY.finditer(low):
        y = int(m.group(1))
        if not (1900 <= y <= 2100):
            continue
        found.append((m.group(0), pd.Timestamp(y, 12, 31)))
    # dedupe, keep first occurrence order
    seen: set[str] = set()
    out = []
    for mention, ts in found:
        if mention not in seen:
            seen.add(mention)
            out.append((mention, ts))
    return out


def future_mentions(text: str, as_of: str | pd.Timestamp) -> list[tuple[str, pd.Timestamp]]:
    """Mentions in ``text`` that are strictly after ``as_of``."""
    t = pd.Timestamp(as_of)
    return [(m, d) for m, d in extract_dates(text) if d > t]


# ---------------------------------------------------------------------------
# Token-level suppression (the FinCAD mechanism)
# ---------------------------------------------------------------------------


def token_future_penalty(token_strings: Sequence[str], as_of: str | pd.Timestamp) -> np.ndarray:
    """Per-token penalty in [0, 1] for tokens embedding a future date.

    Full future dates score 1.0; bare future-year mentions score 0.7 (a year
    alone is weaker evidence than an explicit date). Tokens with no future date
    score 0.0 and are untouched.
    """
    t = pd.Timestamp(as_of)
    pen = np.zeros(len(token_strings), dtype=float)
    for i, tok in enumerate(token_strings):
        worst = 0.0
        for mention, d in extract_dates(tok):
            if d > t:
                severity = 1.0 if _YYYYMMDD.search(mention) or _MONTH_YEAR.search(mention) else 0.7
                worst = max(worst, severity)
        pen[i] = worst
    return pen


def apply_penalty(
    logits: np.ndarray,
    penalty: np.ndarray,
    penalty_scale: float = 2.0,
) -> np.ndarray:
    """Subtract ``penalty_scale * penalty`` from the logits (in place or copy)."""
    if logits.shape != penalty.shape:
        raise ValueError(f"logits {logits.shape} != penalty {penalty.shape}")
    return logits - penalty_scale * penalty


@dataclass
class SuppressionStats:
    total_tokens: int
    penalised_tokens: int
    max_penalty: float
    mean_penalty: float

    @property
    def fraction_penalised(self) -> float:
        return self.penalised_tokens / max(self.total_tokens, 1)


def decode_with_lookahead_suppression(
    logits: np.ndarray,
    token_strings: Sequence[str],
    as_of: str | pd.Timestamp,
    penalty_scale: float = 2.0,
) -> tuple[np.ndarray, SuppressionStats]:
    """Return ``(suppressed_logits, stats)``.

    Implements FinCAD's "modify the logits during generation to penalise
    future-date mentions" (blueprint Phase 1 step 3 hint). ``logits`` has shape
    (vocab,) or (1, vocab); ``token_strings`` maps vocab index -> decoded token.
    """
    arr = np.asarray(logits, dtype=float)
    if arr.ndim == 2:
        pen = token_future_penalty(token_strings, as_of)
        new = apply_penalty(arr, pen[None, :], penalty_scale)
    else:
        pen = token_future_penalty(token_strings, as_of)
        new = apply_penalty(arr, pen, penalty_scale)
    stats = SuppressionStats(
        total_tokens=int(arr.shape[-1]),
        penalised_tokens=int((pen > 0).sum()),
        max_penalty=float(pen.max()) if pen.size else 0.0,
        mean_penalty=float(pen.mean()) if pen.size else 0.0,
    )
    return new, stats


class LookAheadDetector:
    """Object API for the FinCAD suppression, plus text-level auditing."""

    def __init__(self, as_of: str | pd.Timestamp | None = None, penalty_scale: float = 2.0) -> None:
        self.as_of: pd.Timestamp | None = pd.Timestamp(as_of) if as_of is not None else None
        self.penalty_scale = penalty_scale

    def suppress_logits(
        self, logits: np.ndarray, token_strings: Sequence[str]
    ) -> tuple[np.ndarray, SuppressionStats]:
        if self.as_of is None:
            raise ValueError("as_of must be set before suppress_logits()")
        return decode_with_lookahead_suppression(
            logits, token_strings, self.as_of, self.penalty_scale
        )

    def audit_output(self, text: str) -> list[tuple[str, pd.Timestamp]]:
        """All future-date mentions in a generated output (the leak surface)."""
        if self.as_of is None:
            raise ValueError("as_of must be set before audit_output()")
        return future_mentions(text, self.as_of)


# ---------------------------------------------------------------------------
# End-to-end "cheating factor" audit (blueprint verification)
# ---------------------------------------------------------------------------


def rank_ic(signal: pd.Series, forward: pd.Series) -> float:
    """Spearman rank IC between a signal and forward returns.

    Implemented as Pearson on ranks so no scipy is required on the critical path.
    """
    df = pd.concat([signal.rename("s"), forward.rename("f")], axis=1).dropna()
    if len(df) < 3 or df["s"].nunique() < 2 or df["f"].nunique() < 2:
        return 0.0
    return float(df["s"].rank().corr(df["f"].rank(), method="pearson"))


@dataclass
class LookAheadAuditResult:
    leaky_ic: float
    suppressed_ic: float
    relative_reduction: float

    @property
    def passes_check(self) -> bool:
        """Blueprint FinCAD check: reduction > 50%."""
        return self.relative_reduction > 0.50


class LookAheadAudit:
    """Quantify how context-aware decoding degrades a "cheating" signal.

    ``cheating_signal`` embeds the future (e.g. next-period return) — what a
    model with memorised outcomes would emit. ``suppressed_signal`` is what the
    same model emits once future-date tokens are penalised: it must rely only on
    past information, which in the audit's synthetic market carries ~zero IC.
    """

    def evaluate(
        self,
        dates: Sequence[pd.Timestamp],
        forward_returns: Sequence[float],
        cheating_signal: Sequence[float],
        suppressed_signal: Sequence[float],
    ) -> LookAheadAuditResult:
        idx = pd.DatetimeIndex(dates)
        leaky = rank_ic(pd.Series(list(cheating_signal), index=idx), pd.Series(list(forward_returns), index=idx))
        clean = rank_ic(pd.Series(list(suppressed_signal), index=idx), pd.Series(list(forward_returns), index=idx))
        reduction = (leaky - clean) / max(abs(leaky), 1e-12)
        return LookAheadAuditResult(leaky_ic=leaky, suppressed_ic=clean, relative_reduction=reduction)


def suppress_logits_torch(
    logits,
    tokenizer,
    as_of: str | pd.Timestamp,
    penalty_scale: float = 2.0,
):
    """Torch variant: same suppression, applied to a ``(B, vocab)`` tensor.

    Requires ``torch`` and a tokenizer exposing ``convert_ids_to_tokens`` /
    ``convert_tokens_to_string`` (transformers-style). Returns a copy of the
    tensor with penalties applied.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("suppress_logits_torch needs torch") from exc

    ids = torch.arange(logits.shape[-1], device=logits.device).tolist()
    toks = [tokenizer.convert_ids_to_tokens([i])[0] for i in ids]
    strings = [tokenizer.convert_tokens_to_string([t]) for t in toks]
    pen = token_future_penalty(strings, as_of)
    pen_t = torch.tensor(pen, dtype=logits.dtype, device=logits.device)
    return logits - penalty_scale * pen_t
