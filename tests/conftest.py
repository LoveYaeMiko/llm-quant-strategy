"""Shared fixtures for the test suite."""

from __future__ import annotations

import os

import pandas as pd
import pytest

# Offline guard: an empty value is "set", so config._load_dotenv won't repopulate
# it from .env and no CLI/test path ever dials the real Postgres.
os.environ["PIT_DATABASE_URL"] = ""

from src.data.synthetic import make_synthetic_market
from src.factors.code_generator import FactorContext


@pytest.fixture(scope="session")
def market():
    """Small synthetic market — cheap enough for every module's tests."""
    return make_synthetic_market(symbols=24, days=160, seed=7)


@pytest.fixture(scope="session")
def fctx(market):
    """FactorContext over the synthetic (date, symbol) panel."""
    return FactorContext(market.long)


@pytest.fixture(scope="session")
def forward(market):
    return market.forward_returns


@pytest.fixture(scope="session")
def scores_panel(market):
    """A deterministic, mean-reverting cross-sectional signal."""
    return (
        market.long["close"]
        .groupby(level=1)
        .transform(lambda s: -s.rolling(5, min_periods=3).mean() / s)
    )
