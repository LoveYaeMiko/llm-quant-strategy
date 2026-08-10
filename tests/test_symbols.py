"""Symbol canonicalization tests — one format across every data source."""

from __future__ import annotations

import pytest

from src.data.schema.symbols import (
    SymbolError,
    code6,
    exchange_of,
    from_bare_code,
    is_bj,
    normalize_symbol,
    to_baostock,
)


def test_normalize_canonical_passthrough():
    assert normalize_symbol("600519.SH") == "600519.SH"


def test_normalize_baostock_form():
    assert normalize_symbol("sh.600519") == "600519.SH"
    assert normalize_symbol("sz.000001") == "000001.SZ"
    assert normalize_symbol("bj.430047") == "430047.BJ"


def test_bare_code_is_ambiguous_and_raises():
    with pytest.raises(SymbolError):
        normalize_symbol("600519")


def test_to_baostock_roundtrip():
    assert to_baostock("600519.SH") == "sh.600519"


def test_code6():
    assert code6("600519.SH") == "600519"
    assert code6("000001.SZ") == "000001"


def test_exchange_of():
    assert exchange_of("600519.SH") == "SH"
    assert exchange_of("sh.600000") == "SH"
    assert exchange_of("sz.000001") == "SZ"


def test_is_bj():
    assert is_bj("430047.BJ")
    assert not is_bj("600519.SH")


def test_from_bare_code_infers_exchange():
    assert from_bare_code("600519") == "600519.SH"
    assert from_bare_code("000001") == "000001.SZ"
    assert from_bare_code("300750") == "300750.SZ"
    assert from_bare_code("430047") == "430047.BJ"


def test_from_bare_code_bj_920_renumbered_codes():
    # 920xxx is the 2025+ 北交所 renumbering — must map to BJ, not SH (900xxx B-share)
    assert from_bare_code("920001") == "920001.BJ"
    assert is_bj(from_bare_code("920002"))


def test_from_bare_code_pads():
    assert from_bare_code("1") == "000001.SZ"
