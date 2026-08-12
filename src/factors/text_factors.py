"""Text dispersion / novelty factors (PHASE9_BERT_BLUEPRINT).

Pivots off the Phase 9.1 sentiment FAIL: instead of scoring report-title
*direction* (which is essentially priced-in lagging information on the HS300),
these factors measure *information structure* between institutional report
vectors:

* **dispersion** — the mean pairwise cosine *distance* among a symbol's report
  embeddings over a trailing window. High dispersion ⇒ institutional views
  disagree ⇒ more pricing uncertainty ⇒ (hypothesis) higher future excess
  return. Tests "disagreement", not "direction".
* **novelty** — the cosine distance between the *latest* report embedding and
  the centroid of the trailing-window history. High novelty ⇒ the newest report
  carries a new thesis / expectation gap ⇒ (hypothesis) higher future return.
  Tests "new information", not "already-priced information".

Both are **deterministic** — pure NumPy/SKLearn cosine math over the cached
FinBERT_zh CLS embeddings (``data/text/hs300_research_2022_2025.parquet``), with
**no LLM calls** (blueprint "Critical" constraint). PIT is preserved: at
``as_of`` a symbol only sees reports with ``report_date <= as_of``, so the panel
is as-of-valid and feeds the repo's standard ``factor_eval`` / portfolio
backtest unchanged (blueprint §3.2/3.3, adapted to the existing ``score_panel``
contract used by :class:`~src.factors.pead.PEADFactor`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


class TextFactorCalculator:
    """Deterministic text factors from cached BERT report embeddings.

    The embedding store is a Parquet of per-(symbol, date, title) 768-d
    ``title_embedding`` rows (FinBERT_zh CLS). We index it per symbol into a
    ``(report_date array, embedding matrix)`` pair so every cross-sectional
    query is a slice, not a full-table scan.
    """

    def __init__(
        self,
        data_dir: str = "data/text",
        cache_file: str = "hs300_research_2022_2025.parquet",
        min_dispersion_articles: int = 3,
        min_novelty_history: int = 5,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.min_disp = int(min_dispersion_articles)
        self.min_novelty = int(min_novelty_history)
        self._per_symbol: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._load(cache_file)

    # --------------------------------------------------------------- loading
    def _load(self, cache_file: str) -> None:
        path = self.data_dir / cache_file
        if not path.is_file():
            raise FileNotFoundError(
                f"embedding cache {path} missing — run the Phase 9.1 vector "
                "extraction first (data/text/hs300_research_2022_2025.parquet)"
            )
        df = pd.read_parquet(path)
        need = {"symbol", "date", "title_embedding"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"embedding cache missing columns {sorted(missing)}")
        for symbol, g in df.groupby("symbol"):
            g = g.sort_values("date")
            dates = pd.to_datetime(g["date"]).to_numpy(dtype="datetime64[ns]")
            emb = np.stack([np.asarray(v, dtype=np.float64) for v in g["title_embedding"]])
            self._per_symbol[symbol] = (dates, emb)
        self.n_symbols = len(self._per_symbol)

    @property
    def symbols(self) -> list[str]:
        return list(self._per_symbol.keys())

    def _slice(self, symbol: str, as_of: pd.Timestamp, window_days: int,
               include_today: bool = True) -> np.ndarray:
        """Embedding rows for ``symbol`` with report_date within the window.

        PIT window: ``as_of - window < report_date <= as_of`` (or ``< as_of``
        when ``include_today`` is False, for the novelty history side). Returns
        an ``(n, d)`` matrix (empty array when no report matches).
        """
        entry = self._per_symbol.get(symbol)
        if entry is None:
            return np.empty((0, 0))
        dates, emb = entry
        t = as_of.to_datetime64()
        lo = (as_of - pd.Timedelta(days=window_days)).to_datetime64()
        # searchsorted: right=True -> <= t; right=False -> < t
        i0 = int(np.searchsorted(dates, lo, side="right"))  # > lo (strictly after window start)
        i1 = int(np.searchsorted(dates, t, side="right" if include_today else "left"))
        return emb[i0:i1] if i1 > i0 else np.empty((0, emb.shape[1]))

    # ------------------------------------------------------------- dispersion
    @staticmethod
    def _mean_pairwise_distance(emb: np.ndarray) -> float:
        """Mean off-diagonal pairwise cosine distance (1 - similarity)."""
        n = len(emb)
        if n < 2:
            return float("nan")
        normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        sim = cosine_similarity(normed)  # n x n, diagonal = 1
        off = sim[~np.eye(n, dtype=bool)]
        return float(1.0 - off.mean())

    def dispersion(self, symbol: str, as_of: str | pd.Timestamp, window: int = 20) -> float:
        """Mean pairwise cosine distance of report vectors in the window.

        NaN when fewer than ``min_dispersion_articles`` reports fall in the
        window (the cross-section rank then drops the symbol for that date).
        """
        emb = self._slice(symbol, pd.Timestamp(as_of), window)
        if len(emb) < self.min_disp:
            return float("nan")
        return self._mean_pairwise_distance(emb)

    # --------------------------------------------------------------- novelty
    def novelty(self, symbol: str, as_of: str | pd.Timestamp, window: int = 180) -> float:
        """Distance of the latest report to the centroid of the window history.

        ``window`` is the *history* horizon: the centroid is the mean of report
        embeddings strictly before ``as_of`` within the window, and the "latest
        report" is the most recent report on-or-before ``as_of``. NaN unless
        both a latest report and ≥ ``min_novelty_history`` history rows exist.
        """
        entry = self._per_symbol.get(symbol)
        if entry is None:
            return float("nan")
        dates, emb = entry
        t = pd.Timestamp(as_of).to_datetime64()
        i_last = int(np.searchsorted(dates, t, side="right")) - 1
        if i_last < 0:
            return float("nan")
        # history = trailing window *before* the latest report batch, anchored at
        # the latest report's own date (not as_of) so the latest report never
        # pollutes its own reference centroid.
        hist = self._slice(symbol, pd.Timestamp(dates[i_last]), window, include_today=False)
        if len(hist) < self.min_novelty:
            return float("nan")
        latest = emb[i_last]
        normed_latest = latest / np.linalg.norm(latest)
        center = hist.mean(axis=0)
        normed_center = center / np.linalg.norm(center)
        sim = float(normed_latest @ normed_center)
        return float(1.0 - np.clip(sim, -1.0, 1.0))

    # ---------------------------------------------------------- cross-section
    def factor_snapshot(self, symbols: Iterable[str], as_of: str | pd.Timestamp,
                        kind: str, window: int) -> pd.Series:
        """Cross-sectional ``(symbol -> factor)`` at one date (NaN when no signal)."""
        fn = self.dispersion if kind == "dispersion" else self.novelty
        return pd.Series({s: fn(s, as_of, window) for s in symbols})

    def score_panel(
        self,
        dates: Iterable[str | pd.Timestamp],
        symbols: Iterable[str],
        kind: str,
        window: int,
    ) -> pd.Series:
        """Full ``(date, symbol)`` signal panel, percentile-ranked per date.

        Mirrors :meth:`PEADFactor.score_panel`: per date the factor is ranked
        cross-sectionally (pct=True) so the panel is directly consumable by
        ``factor_eval`` / the portfolio backtest. Symbols with no signal get NaN
        and drop out of that date's long/short book.
        """
        syms = list(symbols)
        records: list[tuple[pd.Timestamp, str, float]] = []
        for d in dates:
            t = pd.Timestamp(d)
            snap = self.factor_snapshot(syms, t, kind, window)
            r = snap.rank(pct=True)
            for s, v in r.items():
                records.append((t, s, float(v)))
        if not records:
            return pd.Series(dtype=float)
        idx = pd.MultiIndex.from_tuples([(a, b) for a, b, _ in records], names=["date", "symbol"])
        return pd.Series([v for _, _, v in records], index=idx)


# ---------------------------------------------------------------------------
# single-shot backtest helper (gate)
# ---------------------------------------------------------------------------


def run_text_gate(
    market,
    symbols: list[str],
    kind: str,
    window: int,
    gate: float = 0.015,
    cache_file: str = "hs300_research_2022_2025.parquet",
    data_dir: str = "data/text",
) -> dict:
    """PIT panel → factor_eval bundle + long-short portfolio → gate verdict.

    ``market`` must expose ``forward_returns``; trading dates come from its
    index so the panel aligns exactly with the backtest's forward returns.
    """
    from ..backtest.metrics import factor_eval

    calc = TextFactorCalculator(cache_file=cache_file, data_dir=data_dir)
    forward = market.forward_returns
    trading_dates = sorted(forward.index.get_level_values(0).unique())
    sig = calc.score_panel(trading_dates, symbols, kind, window)
    m = factor_eval(sig, forward, n_trials=1)

    from ..backtest.engine import BacktestConfig, PointInTimeBacktest

    bt = PointInTimeBacktest(
        BacktestConfig(long_pct=0.10, short_pct=0.10, max_position_pct=0.05)
    )
    tradable = getattr(market, "forward_returns_tradable", forward)
    pm = bt.run(sig, tradable).metrics
    top_ic = max(m["ic"], m["rank_ic"])
    return {
        "kind": kind,
        "window_days": int(window),
        "metrics": m,
        "portfolio": {k: pm.get(k) for k in ("sharpe", "max_drawdown", "annualized_return", "t_stat", "turnover")},
        "gate": {"ic_threshold": gate, "max_ic": top_ic, "passed": top_ic >= gate},
        "n_signal_cells": int(sig.notna().sum()),
        "n_symbols_loaded": calc.n_symbols,
    }
