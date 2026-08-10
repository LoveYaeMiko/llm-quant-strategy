# ADR-0005: Full-A data, bounded research universe

Data ingestion covers the full A-share universe (~5600 symbols, 2010-2025), but the interactive research loops (`mine`/`backtest`/`monitor`) run on a bounded reference universe (`research.universe: "hs300_500"`), overridable per run with `--symbols`. Running research on ~22M rows would make each factor evaluation take minutes, making the LLM-driven mining loop unusable. Full-A final validation is an explicit, on-demand step after factors pass on the reference universe.
