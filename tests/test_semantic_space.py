"""AlphaSchema semantic-space tests."""

from __future__ import annotations

import random

from src.factors.semantic_space import SchemaPlan, SemanticSpace


def _plan():
    return SchemaPlan(
        event="Earnings Surprise",
        context="Post-Earnings Drift",
        qualities=("Momentum",),
        direction="long",
        output="rank",
    )


def test_schema_plan_key_is_stable_and_canonical():
    assert _plan().key() == _plan().key()
    # ordering of qualities is part of the key
    a = _plan()
    b = SchemaPlan.from_dict({**a.to_dict(), "qualities": ("Mean Reversion",)})
    assert a.key() != b.key()


def test_natural_language_readable():
    text = _plan().natural_language()
    assert "Earnings Surprise" in text
    assert "Momentum" in text
    assert "long" in text
    assert "rank" in text


def test_space_sample_valid():
    space = SemanticSpace()
    rng = random.Random(0)
    for _ in range(20):
        p = space.sample(rng)
        assert space.validate(p)


def test_space_neighbors_differ_from_seed():
    space = SemanticSpace()
    p = _plan()
    rng = random.Random(0)
    neigh = space.neighbors(p, rng, k=5)
    assert len(neigh) >= 1
    keys = {n.key() for n in neigh}
    assert p.key() not in keys  # neighbours differ from the seed
    assert len(keys) == len(neigh)  # no duplicates


def test_from_dict_roundtrip():
    p = _plan()
    assert SchemaPlan.from_dict(p.to_dict()) == p


def test_invalid_plan_rejected():
    space = SemanticSpace()
    bad = SchemaPlan(
        event="not_a_real_event", context="x", qualities=("y",), direction="long", output="rank"
    )
    assert space.validate(bad) is False


def test_schema_plan_validates_direction():
    import pytest

    with pytest.raises(ValueError):
        SchemaPlan(event="Earnings Surprise", context="Bull Market", direction="diagonal")
