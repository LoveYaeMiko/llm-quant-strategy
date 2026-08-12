"""Research-report title sentiment factor — the Phase 9.1 backtest source.

Reports are the blueprint §3.1 "东方财富研报" channel and the only free
A-share sentiment source with full history (2017→present, verified), so the
2022-2025 IC>0.015 gate runs on **report-title sentiment**. The live news
channel (:class:`~src.sentiment.ingestion.NewsIngestor`) is forward-only and
feeds the same TriAgent at runtime.

``ensure_report_scores`` scores every *unique* report title through the
TriAgent word→BERT ladder and caches by title, so a re-run of the gate never
re-pays the BERT cost. ``run_report_gate`` builds the PIT carry-forward panel
(:func:`~src.sentiment.triagent.build_report_signal`), evaluates it with the
repo's standard :func:`~src.backtest.metrics.factor_eval` bundle and a
long-short portfolio, and applies the Phase 9.1 gate (max(ic, rank_ic) >= 0.015).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .triagent import TriAgentSentiment, build_report_signal

logger = logging.getLogger(__name__)

_BERT_CHUNK = 500  # bound the CPU batch per call


def _score_chunk_worker(chunk: list[str], model_dir: str, lexicon_threshold: float) -> list[dict]:
    """One scoring worker (top-level for Windows spawn): loads the model once,
    routes each title through word→BERT, returns title-level records."""
    from .bert import ChineseBertSentiment
    from .lexicon import ChineseFinancialLexicon

    agent = TriAgentSentiment(
        lexicon=ChineseFinancialLexicon(),
        bert=ChineseBertSentiment(model_dir),
        lexicon_threshold=lexicon_threshold,
    )
    out = agent.score_titles(chunk)
    return out[["title", "final", "tier"]].to_dict("records")


def ensure_report_scores(
    reports: pd.DataFrame,
    triagent: TriAgentSentiment,
    cache_path: str = "data/reports/report_sentiment.parquet",
    tier: str = "triagent",
    workers: int = 1,
) -> pd.DataFrame:
    """Title-level sentiment ``[final, tier]`` for every report, cached by title.

    ``tier="word"`` skips the BERT ladder (fast, lexicon-only — useful for a
    quick read on the gate before the full run); ``"triagent"`` routes
    ambiguous titles through FinBERT_zh. Scores depend only on the title text,
    so the cache dedupes across symbols and brokers. ``workers > 1`` scores the
    BERT pass across processes (each loads FinBERT_zh once) for CPU speedup.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached: dict[str, tuple[float, str]] = {}
    if cache_path.is_file():
        cdf = pd.read_parquet(cache_path)
        cached = dict(zip(cdf["title"], zip(cdf["final"], cdf["tier"])))

    titles = sorted(set(reports["title"]))
    todo = [t for t in titles if t not in cached]
    rows = [{"title": t, "final": s, "tier": tier_} for t, (s, tier_) in cached.items()]
    if todo and tier == "triagent":
        chunks = [todo[i : i + _BERT_CHUNK] for i in range(0, len(todo), _BERT_CHUNK)]
        if workers > 1 and len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(_score_chunk_worker, ch, triagent.bert.model_dir,
                              triagent.lexicon_threshold)
                    for ch in chunks
                ]
                for f in futures:
                    rows.extend(f.result())
        else:
            for chunk in chunks:
                out = triagent.score_titles(chunk)
                rows.extend(out[["title", "final", "tier"]].to_dict("records"))
        new = pd.DataFrame(rows).drop_duplicates("title").sort_values("title")
        new.to_parquet(cache_path, index=False)
    elif todo:
        # word-only path: lexicon directly, NOT triagent.score_titles (whose
        # BERT routing would load the model via TriAgent's ``bert or ...``
        # default even when the caller passed bert=None).
        rows.extend({"title": t, "final": round((triagent.lexicon.score(t) + 1) / 2, 4),
                     "tier": "word"} for t in todo)
    out = pd.DataFrame(rows).drop_duplicates("title")
    return out


def run_report_gate(
    reports: pd.DataFrame,
    scores: pd.DataFrame,
    market,
    symbols: list[str],
    decay_days: int = 10,
    gate: float = 0.015,
) -> dict:
    """PIT panel → factor_eval bundle + long-short portfolio → gate verdict."""
    from ..backtest.metrics import factor_eval

    forward = market.forward_returns
    trading_dates = sorted(forward.index.get_level_values(0).unique())
    sig = build_report_signal(reports, scores, trading_dates, symbols, decay_days=decay_days)
    m = factor_eval(sig, forward, n_trials=1)

    from ..backtest.engine import BacktestConfig, PointInTimeBacktest

    bt = PointInTimeBacktest(
        BacktestConfig(
            long_pct=0.10,
            short_pct=0.10,
            max_position_pct=0.05,
        )
    )
    tradable = getattr(market, "forward_returns_tradable", forward)
    pm = bt.run(sig, tradable).metrics
    top_ic = max(m["ic"], m["rank_ic"])
    return {
        "metrics": m,
        "portfolio": {k: pm.get(k) for k in ("sharpe", "max_drawdown", "annualized_return", "t_stat", "turnover")},
        "gate": {"ic_threshold": gate, "max_ic": top_ic, "passed": top_ic >= gate},
        "n_signal_cells": int(sig.notna().sum()),
    }
