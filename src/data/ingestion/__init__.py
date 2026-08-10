"""Ingestion pipeline: vendor adapters → convert → PIT records.

* :mod:`.convert` — pure functions turning vendor frames into PIT records;
* :mod:`.alphafeed_adapter` — AlphaFeed (primary, paid/unlimited) OHLCV + factors;
* :mod:`.baostock_adapter` — Baostock (free) universe snapshots + index constituents;
* :mod:`.akshare_adapter` — AKShare (free, flaky) news + fundamentals snapshot;
* :mod:`.ingestor` — orchestration + data-quality summary.
"""
