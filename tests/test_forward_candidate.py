"""Forward candidate wiring tests (audit item 2 + the 2026-09-10 degeneracy fix).

The candidate experiment must be able to answer its own question. That requires
THREE properties, all pinned here:

1. the two arms differ in exactly one rule (the stop width);
2. both arms are REPLAYS of the frozen rule set — the previous version let the
   candidate inherit the production 14:50 order list on live dates, where the
   runner executes that list instead of computing its own targets and the
   intraday sweep is gated off, so the stop width drove nothing and the two
   ledgers came out byte-identical;
3. the paired comparison counts only days both arms advanced independently,
   never the seeded prefix.
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


def test_arms_differ_only_in_the_stop_width():
    cfg = _cfg()
    incumbent = fc._replay_account(cfg, SPEC, "incumbent")
    cand = fc._replay_account(cfg, SPEC, "candidate")
    diff = {k: (incumbent.get(k), cand.get(k)) for k in set(incumbent) | set(cand)
            if incumbent.get(k) != cand.get(k)}
    assert set(diff) == {"name", "pb_atr_mult", "pb_stop_lo", "pb_stop_hi"}
    assert cand["pb_stop_lo"] == 0.025 and cand["pb_stop_hi"] == 0.040
    assert cand["pb_atr_mult"] == 1.0
    assert incumbent["pb_stop_lo"] == incumbent["pb_stop_hi"] == 0.035
    # everything else — cash, universe, k — is inherited unchanged
    assert cand["cash"] == incumbent["cash"] == 50000.0
    assert cand["pb_k"] == incumbent["pb_k"] == 6


def test_arms_are_replays_not_live_inheritors():
    """The degeneracy fix: no order-list inheritance, no live gate."""
    cfg = _cfg()
    for arm in ("incumbent", "candidate"):
        acc = fc._replay_account(cfg, SPEC, arm)
        assert acc["pb_live_intraday_from"] == "", "the live gate must be cleared"
        assert "pb_preclose_account" not in acc, "an arm must compute its own close targets"


def test_ledger_paths_are_isolated():
    paths = fc._ledger_paths(_cfg(), SPEC)
    assert paths["production"].name == "shadow_ledger_D_5W.sqlite"
    assert "candidate_atr_1p0_25_40" in str(paths["candidate"])
    assert "incumbent_replay" in str(paths["incumbent"])
    assert len({str(p) for p in paths.values()}) == 3, "three distinct ledgers"


def test_spec_missing_is_a_clear_error():
    cfg = Config({"forward": {"candidates": {}}})
    with pytest.raises(SystemExit):
        fc._spec(cfg, "atr_1p0_25_40")


def test_default_rule_is_the_recorded_candidate():
    assert fc.DEFAULT_RULE == "atr_1p0_25_40"


def test_arm_meta_roundtrip(tmp_path):
    (tmp_path / "arm_meta.json").write_text('{"seed_cutoff": "2026-09-09"}', encoding="utf-8")
    assert fc._arm_meta(tmp_path / "ledger.sqlite")["seed_cutoff"] == "2026-09-09"
    assert fc._arm_meta(tmp_path / "missing" / "ledger.sqlite") == {}
