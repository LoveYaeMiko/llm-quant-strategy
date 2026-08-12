"""Phase 10 — three-layer simulation portfolio (Alpha core → tactical tilt →
risk circuit-breaker) and its full-sample backtest runner.

Blueprint: ``blueprint/PHASE10_BLUEPRINT.md``. The layers are deliberately
small, deterministic classes: each exposes ``apply(weights, date)`` (or
``weights_on(date)``) over plain ``{symbol: weight}`` dicts so the orchestration
in :mod:`layer_integration` stays trivial and unit-testable.
"""
