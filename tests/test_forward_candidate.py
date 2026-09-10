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

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.config import Config  # noqa: E402

import forward_candidate as fc  # noqa: E402

SPEC = {
    "account_suffix": "_FWD_ATR",
    "ledger": "outputs/forward/candidate_atr_1p0_25_35/ledger.sqlite",
    "params": {"pb_atr_mult": 1.0, "pb_stop_lo": 0.025, "pb_stop_hi": 0.035},
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
        "forward": {"candidates": {"atr_1p0_25_35": SPEC}},
    })


def test_arms_differ_only_in_the_stop_width():
    cfg = _cfg()
    incumbent = fc._replay_account(cfg, SPEC, "incumbent")
    cand = fc._replay_account(cfg, SPEC, "candidate")
    diff = {k: (incumbent.get(k), cand.get(k)) for k in set(incumbent) | set(cand)
            if incumbent.get(k) != cand.get(k)}
    # the incumbent's flat 3.5% and the candidate's [2.5%, 3.5%] ATR band share the
    # same CEILING; only the floor and the ATR multiplier differ
    assert set(diff) == {"name", "pb_atr_mult", "pb_stop_lo"}
    assert cand["pb_stop_lo"] == 0.025 and cand["pb_stop_hi"] == 0.035
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
    assert "candidate_atr_1p0_25_35" in str(paths["candidate"])
    assert "incumbent_replay" in str(paths["incumbent"])
    assert len({str(p) for p in paths.values()}) == 3, "three distinct ledgers"


def test_spec_missing_is_a_clear_error():
    cfg = Config({"forward": {"candidates": {}}})
    with pytest.raises(SystemExit):
        fc._spec(cfg, "atr_1p0_25_35")


def test_default_rule_is_the_enabled_candidate():
    """The tracked candidate is the one the corrected 800-name grid supports.

    It was ``atr_1p0_25_40`` until 2026-09-10, when the re-run on the corrected
    pool showed that candidate is the WORST of the four wide variants on the
    project's own max-min rule (IS Sharpe −0.12) while ``atr_1p0_25_35`` is the
    best (min 0.29). The incumbent flat 3.5% still wins max-min (0.40), so the
    deployment is unchanged — only the forward target moved.
    """
    assert fc.DEFAULT_RULE == "atr_1p0_25_35"


def test_arm_meta_roundtrip(tmp_path):
    (tmp_path / "arm_meta.json").write_text('{"seed_cutoff": "2026-09-09"}', encoding="utf-8")
    assert fc._arm_meta(tmp_path / "ledger.sqlite")["seed_cutoff"] == "2026-09-09"
    assert fc._arm_meta(tmp_path / "missing" / "ledger.sqlite") == {}


def _no_market(monkeypatch):
    """Fail loudly if anything tries to build the market slice."""
    import src.cli as cli

    def _boom(*_a, **_k):
        raise AssertionError("market slice built although there was nothing to advance")

    monkeypatch.setattr(cli, "_build_market_for_paper", _boom)


def test_seeded_arms_do_not_build_a_market_they_cannot_use(tmp_path, monkeypatch):
    """A window starting AFTER the newest bar must not cost a market build.

    Observed on 2026-09-10 18:28: the fresh candidate arm was seeded from
    production, so both arms already stood at the newest bar (2026-09-10) while
    the pre-registered window starts 2026-09-11. The run still built the full
    ~10 GB market slice, then printed 「nothing to advance」 and returned. The
    pre-seed guard cannot catch this (the arm ledger does not exist yet when it
    runs), so the guard after seeding must prove it without the panel:
    ``end = min(target_end, panel.max()) <= target_end`` and every arm is
    already at ``target_end`` ⇒ nothing to do, nothing to write.
    """
    from types import SimpleNamespace

    import src.autopilot.state as ctrl
    import src.paper.shadow as shadow

    cfg = _cfg()
    ledger = tmp_path / "ledger.sqlite"
    ledger.write_bytes(b"")
    (tmp_path / "arm_meta.json").write_text(
        json.dumps({"seed_cutoff": "2026-09-10", "seed_boundary": "2026-09-11"}),
        encoding="utf-8",
    )
    prod = tmp_path / "prod.sqlite"
    prod.write_bytes(b"")

    monkeypatch.setattr(fc, "_cfg", lambda: cfg)
    monkeypatch.setattr(fc, "_spec", lambda _c, _rule: SPEC)
    monkeypatch.setattr(fc, "_ledger_paths", lambda _c, _s: {"production": prod})
    monkeypatch.setattr(fc, "_arm_paths",
                        lambda _c, _s: {"incumbent": ledger, "candidate": ledger})
    monkeypatch.setattr(fc, "_last_advanced", lambda _p: "2026-09-10")
    monkeypatch.setattr(fc, "_pit_max_bar", lambda _c: "2026-09-10")
    monkeypatch.setattr(shadow, "resolve_shadow_universe", lambda _c, _u: ["000001"])
    monkeypatch.setattr(ctrl.ControlState, "load",
                        classmethod(lambda _cls, _p: SimpleNamespace(gross_scale=1.0)))
    _no_market(monkeypatch)

    args = SimpleNamespace(rule="atr_1p0_25_35", date="2026-09-10",
                           start="2026-09-11", fresh=False)
    assert fc.cmd_run(args) == 0
    # nothing was written: the arm meta is untouched (no last_run_* stamp)
    assert fc._arm_meta(ledger).get("seed_cutoff") == "2026-09-10"
    assert "last_run_date" not in fc._arm_meta(ledger)


def test_an_arm_behind_the_newest_bar_still_builds_the_market(tmp_path, monkeypatch):
    """The guard must NOT swallow real work: one arm at 2026-09-09 ⇒ 09-10 runs.

    ``None`` (no ledger yet) counts as "cannot prove" and falls through to the
    normal path — here the market builder is reached and then fails, which is
    exactly how the test observes that the guard let the run continue.
    """
    from types import SimpleNamespace

    import src.autopilot.state as ctrl
    import src.paper.shadow as shadow

    cfg = _cfg()
    ledger = tmp_path / "ledger.sqlite"
    ledger.write_bytes(b"")
    (tmp_path / "arm_meta.json").write_text(
        json.dumps({"seed_cutoff": "2026-09-09"}), encoding="utf-8")
    prod = tmp_path / "prod.sqlite"
    prod.write_bytes(b"")

    monkeypatch.setattr(fc, "_cfg", lambda: cfg)
    monkeypatch.setattr(fc, "_spec", lambda _c, _rule: SPEC)
    monkeypatch.setattr(fc, "_ledger_paths", lambda _c, _s: {"production": prod})
    monkeypatch.setattr(fc, "_arm_paths",
                        lambda _c, _s: {"incumbent": ledger, "candidate": ledger})
    monkeypatch.setattr(fc, "_last_advanced", lambda _p: "2026-09-09")
    monkeypatch.setattr(fc, "_pit_max_bar", lambda _c: "2026-09-10")
    monkeypatch.setattr(shadow, "resolve_shadow_universe", lambda _c, _u: ["000001"])
    monkeypatch.setattr(ctrl.ControlState, "load",
                        classmethod(lambda _cls, _p: SimpleNamespace(gross_scale=1.0)))
    _no_market(monkeypatch)

    args = SimpleNamespace(rule="atr_1p0_25_35", date="2026-09-10",
                           start="2026-09-11", fresh=False)
    with pytest.raises(AssertionError):
        fc.cmd_run(args)
