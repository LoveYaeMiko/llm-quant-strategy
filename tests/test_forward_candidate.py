"""Forward candidate wiring tests (audit item 2).

The candidate must differ from the incumbent in EXACTLY one rule. These tests pin
the three properties that make the paired comparison interpretable: same account
shape, the pre-registered parameter diff, and inheritance of the production
14:50 order list (so both books trade the same auction).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.config import Config  # noqa: E402

import forward_candidate as fc  # noqa: E402

SPEC = {
    "account_suffix": "_FWD_ATR",
    "ledger": "outputs/forward/candidate_atr_1p0_25_40/ledger.sqlite",
    "params": {"pb_atr_mult": 1.0, "pb_stop_lo": 0.025, "pb_stop_hi": 0.040},
    "switch_rule": {"window_days": 120, "paired_diff_gt": 0.0, "t_min": 1.5},
}


def _cfg() -> Config:
    return Config({
        "shadow": {
            "ledger_db": "outputs/shadow_ledger.sqlite",
            "accounts": [{
                "name": "D_5W", "alpha_source": "pullback", "universe": "hs300_500",
                "cash": 50000.0, "pb_k": 6, "pb_stop_lo": 0.035, "pb_stop_hi": 0.035,
                "pb_live_intraday_from": "2026-09-04",
            }],
        },
        "forward": {"candidates": {"atr_1p0_25_40": SPEC}},
    })


def test_candidate_account_differs_only_in_the_stop_width():
    cfg = _cfg()
    incumbent = fc._production_account(cfg, "D_5W")
    cand = fc._candidate_account(cfg, SPEC)
    diff = {k: (incumbent.get(k), cand.get(k)) for k in set(incumbent) | set(cand)
            if incumbent.get(k) != cand.get(k)}
    assert set(diff) == {"name", "pb_preclose_account", "pb_atr_mult", "pb_stop_lo", "pb_stop_hi"}
    assert cand["pb_stop_lo"] == 0.025 and cand["pb_stop_hi"] == 0.040
    assert cand["pb_atr_mult"] == 1.0
    assert cand["name"] == "D_5W_FWD_ATR"
    # everything else — cash, universe, k, live gate — is inherited unchanged
    assert cand["cash"] == incumbent["cash"]
    assert cand["pb_k"] == incumbent["pb_k"]
    assert cand["pb_live_intraday_from"] == incumbent["pb_live_intraday_from"]


def test_candidate_inherits_the_production_order_list():
    cand = fc._candidate_account(_cfg(), SPEC)
    assert cand["pb_preclose_account"] == "D_5W"


def test_ledger_paths_are_isolated():
    prod, cand = fc._ledger_paths(_cfg(), SPEC)
    assert prod.name == "shadow_ledger_D_5W.sqlite"
    assert "candidate_atr_1p0_25_40" in str(cand)
    assert prod != cand


def test_spec_missing_is_a_clear_error():
    cfg = Config({"forward": {"candidates": {}}})
    with pytest.raises(SystemExit):
        fc._spec(cfg, "atr_1p0_25_40")


def test_default_rule_is_the_recorded_candidate():
    assert fc.DEFAULT_RULE == "atr_1p0_25_40"
