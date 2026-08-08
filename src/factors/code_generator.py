"""Code generator — schema → executable factor, via a hard-coded operator library.

Blueprint §4B requires the operator library to be **hardcoded** so the code agent
cannot hallucinate operators. This module therefore ships a fixed, closed set of
66+ operators (arithmetic, time-series, cross-sectional, compound) over a panel
indexed by ``(date, symbol)``, a small formula parser, a **safe evaluator** (no
Python ``eval`` — formulas are parsed to an AST and dispatched through the
library), an AST-distance function for the diversity gate, and the translator the
code agent calls (temperature 0.0 in :class:`~src.agents.code_agent.CodeAgent`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

DATE = "date"
SYMBOL = "symbol"

# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """Base class for the formula AST."""


@dataclass
class NodeLiteral(Node):
    value: float


@dataclass
class NodeVar(Node):
    name: str


@dataclass
class NodeCall(Node):
    name: str
    args: list[Node]


# ---------------------------------------------------------------------------
# Factor evaluation context
# ---------------------------------------------------------------------------


class FactorContext:
    """Evaluation context: a panel ``(date, symbol)`` of raw fields.

    Field lookup is case-insensitive so formulas written against ``Close`` work
    on a dataframe whose column is ``close``.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("factor data must be a DataFrame indexed by (date, symbol)")
        self.data = data

    def field(self, name: str) -> pd.Series:
        if name in self.data.columns:
            return self.data[name]
        lower = name.lower()
        for col in self.data.columns:
            if str(col).lower() == lower:
                return self.data[col]
        raise KeyError(f"unknown factor field {name!r}")

    def market_factor(self, s: pd.Series) -> pd.Series:
        """Cross-sectional mean per date — the default market factor."""
        return s.groupby(level=DATE).transform("mean")


# ---------------------------------------------------------------------------
# Operator definitions
# ---------------------------------------------------------------------------

DATE_LVL = DATE
SYM_LVL = SYMBOL


def _roll(s: pd.Series, window: int) -> pd.Series:
    return s.groupby(level=SYM_LVL).rolling(window).mean()


def _ts(s: pd.Series, window: int, method: str, **kw: Any) -> pd.Series:
    """Apply a rolling ``method`` per symbol.

    ``window`` is coerced to int: the parser stores numeric literals as floats,
    so a formula's ``20`` arrives here as ``20.0`` and pandas' ``rolling`` would
    reject it.
    """
    window = int(window)
    return s.groupby(level=SYM_LVL).transform(lambda g: getattr(g.rolling(window), method)(**kw))


@dataclass
class Operator:
    name: str
    arity: int
    fn: Callable[..., Any]
    description: str
    category: str


def _op(name: str, arity: int, fn: Callable[..., Any], description: str, category: str) -> Operator:
    return Operator(name=name, arity=arity, fn=fn, description=description, category=category)


# -- arithmetic -------------------------------------------------------------

def _add(a, b, ctx=None): return a + b
def _sub(a, b, ctx=None): return a - b
def _mul(a, b, ctx=None): return a * b
def _div(a, b, ctx=None): return a / b.replace(0, np.nan) if isinstance(b, pd.Series) else a / b
def _avg(a, b, ctx=None): return (a + b) / 2.0
def _max2(a, b, ctx=None): return np.maximum(a, b)
def _min2(a, b, ctx=None): return np.minimum(a, b)
def _power(a, k, ctx=None): return a ** k
def _abs(x, ctx=None): return np.abs(x)
def _sign(x, ctx=None): return np.sign(x)
def _log(x, ctx=None): return np.log(x.replace(0, np.nan) if isinstance(x, pd.Series) else np.maximum(x, 1e-12))
def _sqrt(x, ctx=None): return np.sqrt(np.maximum(x, 0))
def _square(x, ctx=None): return x * x
def _cube(x, ctx=None): return x * x * x
def _inv(x, ctx=None): return 1.0 / x
def _neg(x, ctx=None): return -x

# -- time series --------------------------------------------------------------

def _ts_mean(s, w, ctx=None): return _ts(s, w, "mean")
def _ts_std(s, w, ctx=None): return _ts(s, w, "std", ddof=0)
def _ts_median(s, w, ctx=None): return _ts(s, w, "median")
def _ts_sum(s, w, ctx=None): return _ts(s, w, "sum")
def _ts_max(s, w, ctx=None): return _ts(s, w, "max")
def _ts_min(s, w, ctx=None): return _ts(s, w, "min")
def _ts_skew(s, w, ctx=None): return _ts(s, w, "skew")
def _ts_kurt(s, w, ctx=None): return _ts(s, w, "kurt")
def _ts_rank(s, w, ctx=None): return _ts(s, w, "rank", pct=True)
def _ts_return(s, w, ctx=None): return s / s.groupby(level=SYM_LVL).shift(int(w)) - 1.0
def _ts_delay(s, w, ctx=None): return s.groupby(level=SYM_LVL).shift(int(w))
def _ts_delta(s, w, ctx=None): return s - s.groupby(level=SYM_LVL).shift(int(w))
def _ts_zscore(s, w, ctx=None): return (_ts(s, w, "mean") - s) / (_ts(s, w, "std", ddof=0) + 1e-12) * (-1.0)
def _ts_quantile(s, w, q, ctx=None): return _ts(s, w, "quantile", quantile=float(q))
def _ts_ema(s, w, ctx=None): return s.groupby(level=SYM_LVL).transform(lambda g: g.ewm(span=int(w), adjust=False).mean())
def _ts_wma(s, w, ctx=None): return s.groupby(level=SYM_LVL).transform(lambda g: _linear_weights(g, int(w)))
def _ts_decay_linear(s, w, ctx=None): return s.groupby(level=SYM_LVL).transform(lambda g: _linear_weights(g, int(w)))
def _ts_slope(s, w, ctx=None):
    def slope(g):
        return g.rolling(int(w)).apply(lambda x: _lin_slope(x), raw=True)
    return s.groupby(level=SYM_LVL).transform(slope)
def _ts_corr(a, b, w, ctx=None): return a.groupby(level=SYM_LVL).rolling(int(w)).corr(b).reset_index(level=0, drop=True)
def _ts_cov(a, b, w, ctx=None): return a.groupby(level=SYM_LVL).rolling(int(w)).cov(b).reset_index(level=0, drop=True)
def _ts_beta(a, b, w, ctx=None): return _ts_cov(a, b, w) / (_ts_std(b, w) ** 2 + 1e-12)
def _ts_mad(s, w, ctx=None): return (_ts(s, w, "mean") - s).abs().groupby(level=SYM_LVL).transform(lambda g: g.rolling(int(w)).mean())
def _ts_prod(s, w, ctx=None): return _ts(s, w, "apply", func=np.prod)
def _ts_count_pos(s, w, ctx=None): return (s > 0).astype(float).groupby(level=SYM_LVL).transform(lambda g: g.rolling(int(w)).sum())
def _ts_argmax(s, w, ctx=None):
    def am(g):
        return g.rolling(int(w)).apply(lambda x: int(np.nanargmax(x)) if np.any(~np.isnan(x)) else np.nan, raw=True)
    return s.groupby(level=SYM_LVL).transform(am)
def _ts_argmin(s, w, ctx=None):
    def am(g):
        return g.rolling(int(w)).apply(lambda x: int(np.nanargmin(x)) if np.any(~np.isnan(x)) else np.nan, raw=True)
    return s.groupby(level=SYM_LVL).transform(am)
def _ts_ratio(s, w, ctx=None): return s / (_ts_sum(s, w) + 1e-12)

# -- cross section ------------------------------------------------------------

def _cs_rank(s, ctx=None): return s.groupby(level=DATE_LVL).rank(pct=True)
def _cs_zscore(s, ctx=None):
    return s.groupby(level=DATE_LVL).transform(lambda g: (g - g.mean()) / (g.std() + 1e-12))
def _cs_scale(s, ctx=None):
    return s.groupby(level=DATE_LVL).transform(lambda g: g / (g.abs().sum() + 1e-12))
def _cs_tanh(s, ctx=None): return np.tanh(_cs_zscore(s))
def _cs_sum(s, ctx=None): return s.groupby(level=DATE_LVL).transform("sum")
def _cs_mean(s, ctx=None): return s.groupby(level=DATE_LVL).transform("mean")
def _cs_max(s, ctx=None): return s.groupby(level=DATE_LVL).transform("max")
def _cs_min(s, ctx=None): return s.groupby(level=DATE_LVL).transform("min")
def _cs_median(s, ctx=None): return s.groupby(level=DATE_LVL).transform("median")
def _cs_neutralize(s, *factors, ctx=None):
    """OLS-residualise ``s`` on named factor columns (default: market mean)."""
    if ctx is None:
        raise ValueError("cs_neutralize requires ctx")
    df = pd.DataFrame({"y": s})
    if not factors or factors == ("market",):
        df["mkt"] = s.groupby(level=DATE_LVL).transform("mean")
    else:
        for f in factors:
            df[f] = ctx.field(str(f))
    xcols = [c for c in df.columns if c != "y"]
    out = s.copy()
    for _, grp in df.groupby(level=DATE_LVL):
        y = grp["y"].values
        X = grp[xcols].values if xcols else np.ones((len(y), 1))
        X1 = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        out.loc[grp.index] = y - X1 @ beta
    return out

# -- compound rank operators (Alpha158 style) -------------------------------

def _rank_mul(a, b, ctx=None): return _cs_rank(a) * _cs_rank(b)
def _rank_add(a, b, ctx=None): return _cs_rank(a) + _cs_rank(b)
def _rank_sub(a, b, ctx=None): return _cs_rank(a) - _cs_rank(b)
def _rank_div(a, b, ctx=None): return _cs_rank(a) / (_cs_rank(b) + 1e-12)
def _rank_min(a, b, ctx=None): return np.minimum(_cs_rank(a), _cs_rank(b))
def _rank_max(a, b, ctx=None): return np.maximum(_cs_rank(a), _cs_rank(b))
def _cs_softmax(s, ctx=None):
    e = np.exp(_cs_zscore(s).clip(-20, 20))
    return e / (e.groupby(level=DATE_LVL).transform("sum") + 1e-12)

# -- logic ---------------------------------------------------------------------

def _cond(c, a, b, ctx=None): return np.where(c > 0, a, b)
def _greater(a, b, ctx=None): return (a > b).astype(float)
def _less(a, b, ctx=None): return (a < b).astype(float)
def _geq(a, b, ctx=None): return (a >= b).astype(float)
def _and(a, b, ctx=None): return (np.sign(a) * np.sign(b) > 0).astype(float)
def _or(a, b, ctx=None): return (np.sign(a) * np.sign(b) < 0).astype(float) * 0 + ((a != 0) | (b != 0)).astype(float)


def _lin_slope(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan
    idx = np.arange(n)
    b, a = np.polyfit(idx, x, 1)
    return b


def _linear_weights(g: pd.Series, window: int) -> pd.Series:
    """Weighted moving average with weights 1..window (most recent heaviest)."""
    weights = np.arange(1, window + 1, dtype=float)
    return g.rolling(window).apply(
        lambda x: float(np.dot(x, weights) / weights.sum()), raw=True
    )


def _register() -> dict[str, Operator]:
    lib: dict[str, Operator] = {}

    def reg(name: str, arity: int, fn: Callable[..., Any], desc: str, cat: str) -> None:
        lib[name] = _op(name, arity, fn, desc, cat)

    arith = "Arithmetic"
    ts = "TimeSeries"
    cs = "CrossSection"
    compound = "Compound"
    logic = "Logic"

    reg("add", 2, _add, "a + b", arith)
    reg("sub", 2, _sub, "a - b", arith)
    reg("mul", 2, _mul, "a * b", arith)
    reg("div", 2, _div, "a / b (guards divide-by-zero)", arith)
    reg("avg", 2, _avg, "(a + b) / 2", arith)
    reg("max2", 2, _max2, "element-wise maximum", arith)
    reg("min2", 2, _min2, "element-wise minimum", arith)
    reg("power", 2, _power, "a ** k", arith)
    reg("abs", 1, _abs, "absolute value", arith)
    reg("sign", 1, _sign, "sign of value", arith)
    reg("log", 1, _log, "natural log (guards zero)", arith)
    reg("sqrt", 1, _sqrt, "square root", arith)
    reg("square", 1, _square, "x^2", arith)
    reg("cube", 1, _cube, "x^3", arith)
    reg("inv", 1, _inv, "1 / x", arith)
    reg("neg", 1, _neg, "-x", arith)

    reg("ts_mean", 2, _ts_mean, "rolling mean over window", ts)
    reg("ts_std", 2, _ts_std, "rolling std over window", ts)
    reg("ts_median", 2, _ts_median, "rolling median", ts)
    reg("ts_sum", 2, _ts_sum, "rolling sum", ts)
    reg("ts_max", 2, _ts_max, "rolling max", ts)
    reg("ts_min", 2, _ts_min, "rolling min", ts)
    reg("ts_skew", 2, _ts_skew, "rolling skewness", ts)
    reg("ts_kurt", 2, _ts_kurt, "rolling kurtosis", ts)
    reg("ts_rank", 2, _ts_rank, "rolling percentile rank", ts)
    reg("ts_return", 2, _ts_return, "s / shift(w) - 1", ts)
    reg("ts_delay", 2, _ts_delay, "value w periods ago", ts)
    reg("ts_delta", 2, _ts_delta, "s - shift(w)", ts)
    reg("ts_zscore", 2, _ts_zscore, "(mean - s) / std (standardised, negated)", ts)
    reg("ts_quantile", 3, _ts_quantile, "rolling q-th quantile", ts)
    reg("ts_ema", 2, _ts_ema, "exponential moving average (span=w)", ts)
    reg("ts_wma", 2, _ts_wma, "linearly weighted moving average", ts)
    reg("ts_decay_linear", 2, _ts_decay_linear, "linear-decay weighted mean (Alpha101 decay_linear)", ts)
    reg("ts_slope", 2, _ts_slope, "rolling linear-regression slope", ts)
    reg("ts_corr", 3, _ts_corr, "rolling correlation of a, b", ts)
    reg("ts_cov", 3, _ts_cov, "rolling covariance of a, b", ts)
    reg("ts_beta", 3, _ts_beta, "rolling regression beta of a on b", ts)
    reg("ts_mad", 2, _ts_mad, "rolling mean absolute deviation", ts)
    reg("ts_prod", 2, _ts_prod, "rolling product", ts)
    reg("ts_count_pos", 2, _ts_count_pos, "rolling count of positive values", ts)
    reg("ts_argmax", 2, _ts_argmax, "rolling index of maximum", ts)
    reg("ts_argmin", 2, _ts_argmin, "rolling index of minimum", ts)
    reg("ts_ratio", 2, _ts_ratio, "s / rolling sum(s, w)", ts)

    reg("rank", 1, _cs_rank, "cross-sectional percentile rank (alias of cs_rank)", cs)
    reg("cs_rank", 1, _cs_rank, "cross-sectional percentile rank", cs)
    reg("cs_zscore", 1, _cs_zscore, "cross-sectional z-score", cs)
    reg("cs_scale", 1, _cs_scale, "scale to unit absolute sum per date", cs)
    reg("cs_tanh", 1, _cs_tanh, "tanh of cross-sectional z-score", cs)
    reg("cs_sum", 1, _cs_sum, "cross-sectional sum", cs)
    reg("cs_mean", 1, _cs_mean, "cross-sectional mean", cs)
    reg("cs_max", 1, _cs_max, "cross-sectional max", cs)
    reg("cs_min", 1, _cs_min, "cross-sectional min", cs)
    reg("cs_median", 1, _cs_median, "cross-sectional median", cs)
    reg("cs_neutralize", 2, _cs_neutralize, "OLS residual vs factor columns (market by default)", cs)
    reg("cs_softmax", 1, _cs_softmax, "softmax of cross-sectional z-score", cs)

    reg("rank_mul", 2, _rank_mul, "cs_rank(a) * cs_rank(b)", compound)
    reg("rank_add", 2, _rank_add, "cs_rank(a) + cs_rank(b)", compound)
    reg("rank_sub", 2, _rank_sub, "cs_rank(a) - cs_rank(b)", compound)
    reg("rank_div", 2, _rank_div, "cs_rank(a) / cs_rank(b)", compound)
    reg("rank_min", 2, _rank_min, "min of cross-sectional ranks", compound)
    reg("rank_max", 2, _rank_max, "max of cross-sectional ranks", compound)

    reg("cond", 3, _cond, "where(c > 0, a, b)", logic)
    reg("greater", 2, _greater, "a > b as 0/1", logic)
    reg("less", 2, _less, "a < b as 0/1", logic)
    reg("geq", 2, _geq, "a >= b as 0/1", logic)
    reg("logical_and", 2, _and, "both non-zero", logic)
    reg("logical_or", 2, _or, "either non-zero", logic)

    return lib


OPERATOR_LIBRARY: dict[str, Operator] = _register()

# blueprint §4B: hardcoded closed set so the code agent cannot hallucinate.
OPERATOR_NAMES: tuple[str, ...] = tuple(OPERATOR_LIBRARY)


def operator_count() -> int:
    return len(OPERATOR_LIBRARY)


# ---------------------------------------------------------------------------
# Formula parser (recursive descent — no eval)
# ---------------------------------------------------------------------------


class FormulaError(ValueError):
    """Raised for malformed / invalid formulas."""


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            tokens.append(("id", text[i:j]))
            i = j
        elif c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and (text[j].isdigit() or text[j] == "."):
                j += 1
            tokens.append(("num", text[i:j]))
            i = j
        elif c in "(),":
            tokens.append((c, c))
            i += 1
        else:
            raise FormulaError(f"unexpected character {c!r} at {i}")
    return tokens


def parse_expression(text: str) -> Node:
    """Parse a formula string into an AST.

    Grammar: ``expr := literal | var | name '(' expr (',' expr)* ')'``.
    """
    tokens = _tokenize(text)
    pos = 0

    def peek() -> tuple[str, str] | None:
        return tokens[pos] if pos < len(tokens) else None

    def advance() -> tuple[str, str]:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def parse_node() -> Node:
        nonlocal pos
        tok = peek()
        if tok is None:
            raise FormulaError("unexpected end of formula")
        if tok[0] == "num":
            advance()
            return NodeLiteral(float(tok[1]))
        if tok[0] == "id":
            name = advance()[1]
            nxt = peek()
            if nxt is not None and nxt[0] == "(":
                advance()
                args: list[Node] = []
                if peek() is not None and peek()[0] != ")":
                    args.append(parse_node())
                    while peek() is not None and peek()[0] == ",":
                        advance()
                        args.append(parse_node())
                if peek() is None or peek()[0] != ")":
                    raise FormulaError("expected ')'")
                advance()
                return NodeCall(name, args)
            return NodeVar(name)
        raise FormulaError(f"unexpected token {tok!r}")

    node = parse_node()
    if pos != len(tokens):
        raise FormulaError(f"trailing tokens after {pos}")
    return node


def validate(node: Node, *, require_operators: bool = True) -> None:
    """Structural validation: operator existence + arity (no data required)."""
    if isinstance(node, NodeLiteral):
        return
    if isinstance(node, NodeVar):
        return
    if isinstance(node, NodeCall):
        op = OPERATOR_LIBRARY.get(node.name.lower())
        if require_operators and op is None:
            raise FormulaError(f"unknown operator {node.name!r} (library is closed)")
        if op is not None and len(node.args) != op.arity:
            raise FormulaError(
                f"{node.name} expects {op.arity} arguments, got {len(node.args)}"
            )
        for arg in node.args:
            validate(arg, require_operators=require_operators)


def evaluate(node: Node, ctx: FactorContext) -> pd.Series:
    """Evaluate an AST against a context. The only Python ``eval``-free path."""
    if isinstance(node, NodeLiteral):
        return node.value
    if isinstance(node, NodeVar):
        return ctx.field(node.name)
    op = OPERATOR_LIBRARY[node.name.lower()]
    values = [evaluate(arg, ctx) for arg in node.args]
    return op.fn(*values, ctx=ctx)


def eval_expression(formula: str, ctx: FactorContext) -> pd.Series:
    node = parse_expression(formula)
    validate(node)
    return evaluate(node, ctx)


# ---------------------------------------------------------------------------
# AST utilities (diversity / memory)
# ---------------------------------------------------------------------------


def canonical(node: Node) -> str:
    """Canonical serialization — equal structures give equal strings."""
    if isinstance(node, NodeLiteral):
        return f"lit:{node.value:.6g}"
    if isinstance(node, NodeVar):
        return f"var:{node.name}"
    return f"call:{node.name}(" + ",".join(canonical(a) for a in node.args) + ")"


def ast_subtrees(node: Node) -> list[str]:
    """All canonical subtree strings (used by the frequent-subtree memory)."""
    subs: list[str] = []
    _collect(node, subs)
    return subs


def _collect(node: Node, out: list[str]) -> None:
    out.append(canonical(node))
    if isinstance(node, NodeCall):
        for arg in node.args:
            _collect(arg, out)


def ast_distance(a: Node, b: Node) -> float:
    """Normalised structural distance in [0, 1].

    Recursively compares node kinds, literals and operator names; identical
    trees give 0.0, any top-level mismatch gives 1.0. Used for the diversity
    gate (AlphaAgent AST-principle check, review.md §2.2).
    """
    if type(a) is not type(b):
        return 1.0
    if isinstance(a, NodeLiteral):
        return 0.0 if abs(a.value - b.value) < 1e-9 else 1.0
    if isinstance(a, NodeVar):
        return 0.0 if a.name == b.name else 1.0
    if isinstance(a, NodeCall):
        if a.name != b.name:
            return 1.0
        if len(a.args) != len(b.args):
            return 1.0
        if not a.args:
            return 0.0
        return sum(ast_distance(x, y) for x, y in zip(a.args, b.args)) / len(a.args)
    return 1.0


def node_to_python(node: Node) -> str:
    """Pretty-print an AST as a callable expression string (for export/audit)."""
    if isinstance(node, NodeLiteral):
        return f"{node.value:g}"
    if isinstance(node, NodeVar):
        return f"Field({node.name!r})"
    return f"{node.name}(" + ", ".join(node_to_python(a) for a in node.args) + ")"


def extract_lookbacks(node: Node) -> list[int]:
    """Numeric literal arguments — the lookback windows referenced by a formula."""
    lookbacks: list[int] = []
    if isinstance(node, NodeCall):
        for arg in node.args:
            if isinstance(arg, NodeLiteral) and arg.value == int(arg.value):
                lookbacks.append(int(arg.value))
            else:
                lookbacks.extend(extract_lookbacks(arg))
    return lookbacks


# ---------------------------------------------------------------------------
# Generated factor
# ---------------------------------------------------------------------------


@dataclass
class GeneratedFactor:
    name: str
    formula: str
    meaning: str
    category: str
    data_fields_used: list[str] = field(default_factory=list)
    operators_used: list[str] = field(default_factory=list)
    lookback_periods: list[int] = field(default_factory=list)
    expression: Optional[Node] = None
    python_code: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "formula": self.formula,
            "meaning": self.meaning,
            "category": self.category,
            "data_fields_used": self.data_fields_used,
            "operators_used": self.operators_used,
            "lookback_periods": self.lookback_periods,
            "python_code": self.python_code,
        }


class CodeGenerator:
    """Schema → executable factor. Deterministic (temperature 0.0 upstream)."""

    def __init__(self) -> None:
        self.library = OPERATOR_LIBRARY

    def parse(self, formula: str) -> Node:
        node = parse_expression(formula)
        validate(node)
        return node

    def generate(
        self,
        formula: str,
        *,
        name: Optional[str] = None,
        meaning: str = "",
        category: str = "",
    ) -> GeneratedFactor:
        node = self.parse(formula)
        fields = sorted({a.name for a in _walk(node) if isinstance(a, NodeVar)})
        ops = [c.name for c in _walk(node) if isinstance(c, NodeCall)]
        lookbacks = sorted({lb for lb in extract_lookbacks(node) if lb > 0})
        return GeneratedFactor(
            name=name or formula,
            formula=formula,
            meaning=meaning,
            category=category or (ops[0] if ops else "generic"),
            data_fields_used=fields,
            operators_used=ops,
            lookback_periods=lookbacks,
            expression=node,
            python_code=self.to_python(node),
        )

    @staticmethod
    def to_python(node: Node) -> str:
        """Compiled standalone source (exported by the online distillation)."""
        src = node_to_python(node)
        return (
            "import numpy as np\nimport pandas as pd\n\n"
            "def Field(name):\n"
            "    return lambda data: data[name]\n\n"
            f"def compiled(data):\n"
            f"    from src.factors.code_generator import FactorContext, evaluate\n"
            f"    ctx = FactorContext(data)\n"
            f"    return evaluate(parse_expression({node_to_python(node)!r}), ctx)\n"
        )

    def distance(self, a: Node | str, b: Node | str) -> float:
        pa = a if isinstance(a, Node) else self.parse(a)
        pb = b if isinstance(b, Node) else self.parse(b)
        return ast_distance(pa, pb)

    def within_lookback_bounds(self, formula: str, max_lookback: int, min_lookback: int) -> bool:
        node = self.parse(formula)
        lbs = extract_lookbacks(node)
        return all(min_lookback <= lb <= max_lookback for lb in lbs if lb > 0)


def _walk(node: Node):
    yield node
    if isinstance(node, NodeCall):
        for arg in node.args:
            yield from _walk(arg)


def default_formula_for(plan) -> str:
    """Deterministic schema → formula mapping (offline fallback).

    AlphaSchema decouples exploration from implementation: any LLM can do the
    translation with comparable factor quality. When no LLM is attached (tests,
    cheap offline runs) this pure function fills in, keyed deterministically off
    the plan so the same plan always yields the same formula.
    """
    from .semantic_space import SchemaPlan  # local import avoids a cycle

    if not isinstance(plan, SchemaPlan):
        plan = SchemaPlan.from_dict(plan if isinstance(plan, dict) else plan.to_dict())
    q = plan.qualities[0]
    w = [5, 10, 20, 30, 60][abs(hash(plan.key())) % 5]
    if q in ("Momentum", "Trend", "Carry"):
        return f"Rank_Mul(Rank(Close), Rank(TS_Return(Close, {w})))"
    if q in ("Mean Reversion", "Short-Term Reversal"):
        return f"Neg(TS_ZScore(Close, {w}))"
    if q == "Low Volatility":
        return f"Inv(TS_Std(Close, {w}))"
    if q == "Value":
        return "Rank(Close)"
    if q == "Liquidity":
        return "Rank(Volume)"
    return f"TS_Rank(TS_Return(Close, {w}), {w})"
