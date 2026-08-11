"""Phase 9.1 — ChineseFinancialLexicon tests (range, polarity, negation, degree)."""

from __future__ import annotations

from src.sentiment.lexicon import ChineseFinancialLexicon


def test_score_range():
    lx = ChineseFinancialLexicon()
    for text in ("非常看好后市", "公司大幅亏损", "今日窄幅震荡", "", "完全不相关的表述"):
        assert -1.0 <= lx.score(text) <= 1.0


def test_polarity_directions():
    lx = ChineseFinancialLexicon()
    assert lx.score("公司净利润大幅增长，业绩超预期") > 0.3
    assert lx.score("公司因违规被处罚，股价大跌") < -0.3
    assert abs(lx.score("今日大盘窄幅震荡，成交量持平")) < 0.15


def test_negation_flips_sign():
    lx = ChineseFinancialLexicon()
    base = lx.score("看好后市")
    negated = lx.score("不看好后市")
    assert base > 0.2 and negated < -0.2
    # 无亏损 = positive
    assert lx.score("业绩无亏损") > 0.0


def test_intensifier_amplifies():
    lx = ChineseFinancialLexicon()
    plain = lx.score("利好")
    strong = lx.score("大幅利好")
    assert strong > plain


def test_negator_plus_intensifier_chain():
    lx = ChineseFinancialLexicon()
    # 非常 + 不 + 看好 → negative and slightly stronger than 不看好
    assert lx.score("非常不看好") < lx.score("不看好") < 0


def test_empty_and_neutral():
    lx = ChineseFinancialLexicon()
    assert lx.score("") == 0.0
    assert lx.score("产品创新不断涌现") == 0.0  # 无金融词典词
