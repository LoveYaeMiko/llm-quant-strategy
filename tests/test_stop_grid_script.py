"""Offline unit tests for the ``scripts/d_stop_grid.py`` CLI plumbing.

No market build, no ledger, no network: only the pure helpers that decide WHERE
an artifact goes, WHICH variants run, how an older artifact is re-read, and what
the stamped payload must contain. The grid itself is a 10-30 minute, ~10 GB-RAM
run and is never started from the test suite.

The provenance assertions matter because this artifact is quoted in
``docs/D_TRACK_EVIDENCE.md`` §三: after the 2026-09-10 PIT-pool defect, a grid
number is only readable together with the pool it ran on — which is why the
artifact must carry both the five-field provenance block and the panel-health
block, and why ``--reuse`` must never invent provenance for numbers it did not
produce.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.d_stop_grid as g  # noqa: E402
from src.provenance import sha256_of  # noqa: E402


# --------------------------------------------------------------------------- #
# output path / variant selection
# --------------------------------------------------------------------------- #
def test_out_path_defaults_to_the_historical_artifact():
    assert g._out_path("", "") == ROOT / "outputs" / "d_stop_grid.json"


def test_out_path_uses_the_label_suffix_only_when_out_is_absent():
    assert g._out_path("is_2026_800", "") == ROOT / "outputs" / "d_stop_grid_is_2026_800.json"
    assert g._out_path("is_2026_800", "docs/evidence/x.json") == ROOT / "docs" / "evidence" / "x.json"


def test_out_path_keeps_an_absolute_out_as_given(tmp_path):
    abs_path = (tmp_path / "art.json").resolve()
    assert g._out_path("", str(abs_path)) == abs_path


def test_select_variants_all_or_subset():
    assert set(g._select_variants("")) == set(g.VARIANTS)
    assert list(g._select_variants("flat_3p5, atr_1p0_25_40")) == ["flat_3p5", "atr_1p0_25_40"]
    # an unknown label selects nothing (the caller turns that into a hard error
    # rather than writing an artifact with no numbers in it)
    assert g._select_variants("nope") == {}


def test_the_variant_table_is_the_documented_eight():
    assert list(g.VARIANTS) == [
        "flat_1p5", "flat_2p0", "flat_2p5", "flat_3p0", "flat_3p5",
        "atr_1p0_25_35", "atr_1p0_25_40", "atr_1p5_25_40",
    ]
    # the deployed width must stay a FLAT 3.5% (stop_lo == stop_hi)
    assert g.VARIANTS["flat_3p5"] == {"pb_stop_lo": 0.035, "pb_stop_hi": 0.035}


# --------------------------------------------------------------------------- #
# re-reading an older artifact (--only merges into it)
# --------------------------------------------------------------------------- #
def test_load_variants_prefers_the_nested_shape(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({
        "label": "is_2026_800",
        "panel": {"n_columns": 800},
        "variants": {"flat_3p5": {"sharpe": 0.67}},
        "provenance": {"window": {"start": "2026-01-01", "end": "2026-08-28"}},
    }), encoding="utf-8")
    assert g._load_variants(p) == {"flat_3p5": {"sharpe": 0.67}}


def test_load_variants_still_reads_the_legacy_flat_map(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({
        "flat_3p5": {"sharpe": 1.66},
        "atr_1p0_25_40": {"sharpe": 0.97},
    }), encoding="utf-8")
    assert set(g._load_variants(p)) == {"flat_3p5", "atr_1p0_25_40"}


def test_merge_existing_marks_only_the_untouched_variants():
    prev = {"flat_2p5": {"sharpe": 0.21}, "flat_3p5": {"sharpe": 0.67}}
    merged = g._merge_existing(prev, {"flat_3p5": g.VARIANTS["flat_3p5"]})
    assert merged["flat_2p5"] == {"sharpe": 0.21, "reused": True}
    # the variant that WAS re-run keeps its fresh row (no `reused` flag)
    assert merged["flat_3p5"] == {"sharpe": 0.67}
    assert prev["flat_2p5"] == {"sharpe": 0.21}, "the loaded artifact must not be mutated"


def test_load_variants_is_silent_about_a_missing_or_broken_file(tmp_path):
    assert g._load_variants(tmp_path / "absent.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert g._load_variants(broken) == {}


# --------------------------------------------------------------------------- #
# convention + stamped payload
# --------------------------------------------------------------------------- #
def test_convention_states_basis_execution_and_the_shared_slice():
    text = g._convention({"pb_stop_trigger": "close", "pb_stop_open_minutes": 30})
    for needle in (
        "adjusted-close", "close rebalance", "below 15:00", "T+1", "no leverage",
        "trigger=close", "open_minutes=30", "SAME market slice", "market_override",
    ):
        assert needle in text, needle


def _artifact():
    return g._artifact(
        label="is_2026_800",
        start="2026-01-01",
        end="2026-08-28",
        variants_out={"flat_3p5": {"sharpe": 0.67, "n_symbols": 800, "n_bars": 529}},
        panel={"n_columns": 800, "n_warm_20": 800, "effective_ratio": 1.0, "n_symbols": 800},
        assembly={"book_class": "PullbackPortfolio"},
        kill_switch={"mode": "normal", "gross_scale": 1.0},
        convention="adjusted-close price basis; T+1; no leverage",
        data_as_of="2026-09-10",
    )


def test_artifact_carries_the_five_fields_and_hashes_its_own_payload():
    art = _artifact()
    block = art["provenance"]
    assert block["window"] == {"start": "2026-01-01", "end": "2026-08-28"}
    assert block["data_as_of"] == "2026-09-10"
    assert block["convention"]
    payload = {k: v for k, v in art.items() if k != "provenance"}
    assert block["artifact_sha256"] == sha256_of(payload)
    assert art["label"] == "is_2026_800"
    assert art["panel"]["effective_ratio"] == 1.0
    assert art["variants"]["flat_3p5"]["n_symbols"] == 800


# --------------------------------------------------------------------------- #
# --reuse
# --------------------------------------------------------------------------- #
def _existing(tmp_path: Path) -> Path:
    src = tmp_path / "in.json"
    src.write_text(json.dumps({
        "label": "is_2026_800",
        "window": {"start": "2026-01-01", "end": "2026-08-28"},
        "variants": {"flat_3p5": {"sharpe": 0.67}},
        "panel": {"n_columns": 800},
        "provenance": {
            "window": {"start": "2026-01-01", "end": "2026-08-28"},
            "convention": "adjusted-close price basis",
            "data_as_of": "2026-09-10",
            "artifact_sha256": "0" * 64,
            "code_commit": "a" * 40,
            "code_dirty": True,
        },
    }), encoding="utf-8")
    return src


def test_reuse_re_emits_with_a_verified_hash_and_the_original_commit(tmp_path):
    src = _existing(tmp_path)
    out = tmp_path / "out.json"
    assert g._reemit(src, out, label="relabelled") == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["label"] == "relabelled"
    assert data["variants"] == {"flat_3p5": {"sharpe": 0.67}}
    # numbers were NOT recomputed, so the provenance must still name the code
    # that produced them (and the data cut-off they were produced on)
    assert data["provenance"]["code_commit"] == "a" * 40
    assert data["provenance"]["data_as_of"] == "2026-09-10"
    assert data["provenance"]["artifact_sha256"] == sha256_of(
        {k: v for k, v in data.items() if k != "provenance"}
    )


def test_reuse_refuses_to_invent_provenance(tmp_path):
    src = tmp_path / "legacy.json"
    src.write_text(json.dumps({"flat_3p5": {"sharpe": 1.66}}), encoding="utf-8")
    out = tmp_path / "out.json"
    assert g._reemit(src, out) == 2
    assert not out.exists()


# --------------------------------------------------------------------------- #
# CLI form
# --------------------------------------------------------------------------- #
def test_parse_args_keeps_the_historical_positional_form():
    args = g._parse_args(["2026-01-01", "2026-08-28", "--only=flat_3p5,flat_2p5"])
    assert (args.start, args.end) == ("2026-01-01", "2026-08-28")
    assert args.only == "flat_3p5,flat_2p5"
    assert (args.label, args.out, args.reuse) == ("", "", "")


def test_parse_args_defaults_match_the_documented_window():
    args = g._parse_args([])
    assert (args.start, args.end) == ("2026-01-01", "2026-08-28")


def test_parse_args_reads_the_new_flags():
    args = g._parse_args([
        "2025-09-01", "2025-12-31",
        "--label", "oos_2025h2_800", "--out", "outputs/x.json",
        "--reuse", "outputs/y.json",
    ])
    assert args.label == "oos_2025h2_800"
    assert args.out == "outputs/x.json"
    assert args.reuse == "outputs/y.json"


# --------------------------------------------------------------------------- #
# stop-width statistics
# --------------------------------------------------------------------------- #
def test_stop_width_stats_flat_vs_band():
    flat = g._stop_width_stats({"stop_lo": 0.035, "stop_hi": 0.035, "atr_mult": 1.5})
    assert flat["flat"] is True and flat["band"] == [0.035, 0.035]
    band = g._stop_width_stats({"stop_lo": 0.025, "stop_hi": 0.04, "atr_mult": 1.0})
    assert band["flat"] is False and band["band"] == [0.025, 0.04]
