"""FinCAD look-ahead detector tests — the anti-memorisation mechanism."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bias_control.context_decoder import FinCADWrapper, MockLLMBackend, sanitize_prompt
from src.bias_control.look_ahead_detector import (
    LookAheadAudit,
    LookAheadDetector,
    apply_penalty,
    decode_with_lookahead_suppression,
    extract_dates,
    future_mentions,
    rank_ic,
    token_future_penalty,
)


def test_extract_dates_full_and_year():
    dates = extract_dates("buy on 2024-06-15 then 2024, or Jun 2025")
    mentions = {d[0] for d in dates}
    assert "2024-06-15" in mentions
    assert "2024" in mentions  # bare year interpreted as 2024-12-31


def test_future_mentions_relative_to_as_of():
    as_of = pd.Timestamp("2024-01-01")
    # "2024-06-15" yields two mentions: the full date and the bare year 2024
    mentions = future_mentions("earnings confirm on 2024-06-15", as_of)
    assert any(m[0] == "2024-06-15" for m in mentions)
    assert all(m[1] > as_of for m in mentions)
    assert len(future_mentions("guidance for 2023 was fine", as_of)) == 0


def test_token_penalty_severity():
    as_of = pd.Timestamp("2024-01-01")
    pen = token_future_penalty(["2024-06-15", "2024", "hello", "2023"], as_of)
    assert pen[0] == 1.0          # explicit future date
    assert pen[1] == 0.7          # bare future year
    assert pen[2] == 0.0          # no date
    assert pen[3] == 0.0          # past year is fine


def test_apply_penalty():
    logits = np.array([1.0, 2.0, 3.0])
    pen = np.array([0.0, 1.0, 0.0])
    out = apply_penalty(logits, pen, penalty_scale=2.0)
    assert np.allclose(out, [1.0, 0.0, 3.0])


def test_decode_suppression_stats():
    as_of = pd.Timestamp("2024-01-01")
    tokens = ["the", "2024-06-15", "report", "is", "good"]
    logits = np.zeros(len(tokens))
    _, stats = decode_with_lookahead_suppression(logits, tokens, as_of)
    assert stats.penalised_tokens == 1
    assert stats.max_penalty == 1.0


def test_fincad_wrapper_scrubs_future():
    mock = MockLLMBackend({"leak": "Buy on 2024-06-15 because earnings confirm on 2024-06-15."})
    wrapper = FinCADWrapper(mock)
    result = wrapper.complete("leak", as_of=pd.Timestamp("2024-01-01"))
    assert result.leak_free
    assert "2024-06-15" not in result.text
    assert result.suppressed_count >= 1


def test_sanitize_prompt_redacts():
    clean, redacted = sanitize_prompt("quarter ends 2024-12-31 now", "2024-06-01")
    assert "2024-12-31" not in clean
    assert len(redacted) == 1


def test_look_ahead_audit_reduces_ic():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=100)
    fwd = rng.normal(0, 0.01, len(dates))
    audit = LookAheadAudit().evaluate(
        dates, list(fwd), list(fwd), list(np.zeros(len(dates)))
    )
    assert audit.leaky_ic > 0.9
    assert audit.relative_reduction > 0.5
    assert audit.passes_check


def test_rank_ic_zero_for_constant():
    assert rank_ic(pd.Series([1.0, 1.0, 1.0]), pd.Series([0.1, 0.2, 0.3])) == 0.0
