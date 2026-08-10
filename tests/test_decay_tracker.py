"""DecayTracker tests — rolling-window ICIR decay detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.monitoring.decay_tracker import DecayTracker


def _panel():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=60)
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D", "E", "F"]], names=["date", "symbol"])
    fwd = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    noise = pd.Series(rng.normal(0, 0.004, len(idx)), index=idx)
    return fwd, noise


def test_positive_correlation_is_healthy():
    fwd, noise = _panel()
    res = DecayTracker(window_days=15, min_days=10, icir_threshold=0.30).monitor(fwd + noise, fwd)
    assert res.healthy
    assert res.recent_icir > 0.30


def test_anticorrelation_decays():
    fwd, noise = _panel()
    res = DecayTracker(window_days=15, min_days=10, icir_threshold=0.30).monitor(-fwd + noise, fwd)
    assert res.decayed
    assert res.recent_icir < 0
    assert res.first_decayed_at is not None
    assert res.windows and res.windows[-1].decayed


def test_from_config_reads_research_decay():
    cfg = Config({"research": {"decay": {"window_days": 45, "icir_threshold": 0.20}}})
    t = DecayTracker.from_config(cfg)
    assert t.window_days == 45
    assert t.icir_threshold == 0.20


def test_empty_inputs_returns_clean_result():
    res = DecayTracker().monitor(pd.Series(dtype=float), pd.Series(dtype=float))
    assert res.windows == []
    assert not res.decayed
    assert res.summary


def test_should_keep_alias():
    fwd, noise = _panel()
    t = DecayTracker(window_days=15, min_days=10, icir_threshold=0.30)
    assert t.should_keep(fwd + noise, fwd)
    assert not t.should_keep(-fwd + noise, fwd)
