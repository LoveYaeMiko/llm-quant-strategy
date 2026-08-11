"""Phase 9.1 — Chinese news sentiment (word lexicon -> Chinese BERT -> DeepSeek critic).

Real-time-forward acquisition: AKShare's free news feeds expose no historical
pagination (verified 2026-08-11: ``stock_news_em(symbol)`` returns the latest
~10 items with no ``date`` parameter; the whole-market ``stock_info_global_*``
feeds return only the current day). So the news store is filled forward from
ingestion time onward — there is no 2022-2025 backfill, per the Phase 9.1
blueprint decision (the historical window is closed for news).

The sentiment stack mirrors the blueprint's layered TriAgent:
1. word-level    — ChineseFinancialLexicon (dependency-free, high throughput)
2. sentence-level — Chinese BERT (bert-base-chinese, local weights, GPU if available)
3. cross-sentence — DeepSeek critic (only for high-dispersion samples)
"""
