"""记忆回路 — AlphaMemo-style residual / veto memory."""

from __future__ import annotations

from src.factors.residual_memory import ResidualMemory, edit_motif
from src.factors.semantic_space import SchemaPlan


def _plan(event="Momentum Breakout", qualities=("Momentum",), direction="long"):
    return SchemaPlan(event, "Bull Market", qualities, direction, "score")


def test_edit_motif_single_field_changes():
    parent = _plan(qualities=("Momentum",))
    child = _plan(qualities=("Low Volatility",))
    assert edit_motif(parent, child) == "quality:Momentum→Low Volatility"

    child_dir = _plan(direction="short")
    assert edit_motif(parent, child_dir) == "direction:long→short"

    child_ev = _plan(event="Mean Reversion")
    assert edit_motif(parent, child_ev) == "event:Momentum Breakout→Mean Reversion"


def test_edit_motif_identity():
    p = _plan()
    assert edit_motif(p, p) == "identity"


def test_residual_memory_positive_cell_gains_confidence():
    mem = ResidualMemory(n_conf=12, min_observations=3)
    for child_q in (0.06, 0.07, 0.08, 0.09, 0.10):
        mem.update("Low Volatility", "quality:→Momentum", child_q, 0.03, True)
    delta, conf = mem.query("Low Volatility", "quality:→Momentum")
    assert delta > 0
    assert 0.0 < conf <= 1.0


def test_residual_memory_negative_cell_is_excluded_from_top():
    mem = ResidualMemory()
    mem.update("Low Volatility", "quality:→Momentum", 0.06, 0.03, True)
    mem.update("Low Volatility", "quality:→Momentum", 0.07, 0.03, True)
    mem.update("Momentum", "direction:long→short", 0.01, 0.04, False)
    cells = mem.top_cells(k=10)
    assert cells, "positive cell should be present"
    assert all(c["mean_residual"] > 0 for c in cells)


def test_veto_triggers_on_repeated_failures():
    mem = ResidualMemory(min_observations=3, veto_threshold=0.80)
    for _ in range(4):
        mem.update("Momentum", "direction:long→short", 0.01, 0.04, False)
    vetoed, rate = mem.vetoed("Momentum", "direction:long→short")
    assert vetoed is True
    assert rate >= 0.80


def test_vetoed_motifs_collects_category_motifs():
    mem = ResidualMemory(min_observations=3)
    for _ in range(3):
        mem.update("Momentum", "direction:long→short", 0.0, 0.04, False)
    assert "direction:long→short" in mem.vetoed_motifs("Momentum")
    assert mem.vetoed_motifs("Low Volatility") == set()


def test_residual_memory_persistence_roundtrip(tmp_path):
    mem = ResidualMemory()
    mem.update("Low Volatility", "quality:→Momentum", 0.06, 0.03, True)
    mem.update("Low Volatility", "quality:→Momentum", 0.07, 0.03, True)
    path = tmp_path / "residual.json"
    mem.save(path)
    mem2 = ResidualMemory.load(path)
    assert mem2.size() == mem.size() == 1
    d1, c1 = mem.query("Low Volatility", "quality:→Momentum")
    d2, c2 = mem2.query("Low Volatility", "quality:→Momentum")
    assert d1 == d2 and c1 == c2
