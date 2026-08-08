"""AlphaMemo structured-memory tests."""

from __future__ import annotations

from src.factors.memory_manager import MemoryManager, Trajectory


def _mm():
    mm = MemoryManager()
    mm.record_result(0, {"schema": "s1"}, "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))", {"rank_ic": 0.05})
    mm.record_result(1, {"schema": "s2"}, "Neg(TS_ZScore(Close, 20))", {"rank_ic": 0.04})
    mm.record_result(2, {"schema": "s3"}, "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))", {"rank_ic": 0.06})
    return mm


def test_record_and_size():
    mm = _mm()
    assert mm.size() == 3


def test_subtree_frequency_grows_with_reuse():
    mm = _mm()
    f_repeated = "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))"
    f_novel = "Inv(TS_Std(Volume, 20))"
    assert mm.subtree_frequency(f_repeated) > mm.subtree_frequency(f_novel)


def test_avoidance_penalty_proportional():
    mm = _mm()
    f_repeated = "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))"
    f_novel = "Inv(TS_Std(Volume, 20))"
    assert mm.avoidance_penalty(f_repeated) > mm.avoidance_penalty(f_novel)


def test_top_performers_sorted():
    mm = _mm()
    tops = mm.top_performers(k=2, metric="rank_ic")
    assert tops[0].metrics["rank_ic"] == 0.06
    assert len(tops) == 2


def test_diversity_screen():
    mm = _mm()
    close = "Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))"
    far = "Inv(TS_Std(Volume, 20))"
    assert mm.diversity_screen(close) is False  # already present
    assert mm.diversity_screen(far) is True


def test_has_formula_by_canonical():
    mm = _mm()
    assert mm.has_formula("Rank_Mul(Rank(Close), Rank(TS_Return(Close, 10)))")
    assert not mm.has_formula("Inv(Volume)")


def test_save_load_roundtrip(tmp_path):
    mm = _mm()
    path = tmp_path / "memory.json"
    mm.save(path)
    mm2 = MemoryManager.load(path)
    assert mm2.size() == 3
    assert mm2.has_formula("Neg(TS_ZScore(Close, 20))")


def test_record_invalid_formula_does_not_crash():
    mm = MemoryManager()
    mm.record_result(0, {}, "not a formula ((((", {"rank_ic": 0.0})
    assert mm.size() == 1
    assert mm.subtree_frequency("not a formula ((((") == 0.0
