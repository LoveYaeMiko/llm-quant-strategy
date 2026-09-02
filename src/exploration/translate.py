"""Deterministic translator: Alpha101/GTJA191/AQML expressions → FQA grammar.

Parses the infix/prefix expressions from the factor zoo inventory
(``paper/factor_zoo/inventory.json``) into an AST and renders FQA's closed
operator grammar. Unsupported constructs defer the *whole* formula with a
reason — a partially-misinterpreted alpha is worse than an explicitly
deferred one.

Operator mapping (FQA names):
* time-series — MEAN/STD/SUM/MAX/MIN/TSRANK/DELTA/DELAY/CORR/COVIANCE/
  DECAYLINEAR/WMA/SMA/COUNT/VWAP/LOWDAY/HIGHDAY + AQML ``Ts_*`` aliases;
* cross-section — RANK/Scale;
* arithmetic/logic — ABS/LOG/SIGN/POW/SQRT/NOT/IF/AND/OR + ternary ``?:``.

Deferred categories: ``benchmark`` (index/HS300/market fields), ``amount``
(turnover/VWAP-as-field), ``undefined`` (unmapped token), ``parse`` (syntax).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# tokenizer / parser
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"(?P<id>\$?[A-Za-z_][A-Za-z0-9_]*)|"
    r"(?P<op>\*\*|&&|\|\||[+\-*/^(),<>?:=&|!]))"
)


def _tokenize(expr: str) -> list[tuple[str, str]]:
    expr = expr.replace("−", "-").replace("–", "-").replace("×", "*")
    toks: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise ValueError(f"unexpected character at {pos}: {expr[pos:pos+10]!r}")
        pos = m.end()
        if m.group("num") is not None:
            toks.append(("num", m.group("num")))
        elif m.group("id") is not None:
            toks.append(("id", m.group("id")))
        else:
            toks.append(("op", m.group("op")))
    return toks


class _Parser:
    def __init__(self, toks: list[tuple[str, str]]):
        self.toks = toks
        self.i = 0

    def peek(self) -> Optional[tuple[str, str]]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> tuple[str, str]:
        t = self.peek()
        if t is None:
            raise ValueError("unexpected end of expression")
        self.i += 1
        return t

    def expect_op(self, op: str) -> None:
        t = self.next()
        if t != ("op", op):
            raise ValueError(f"expected {op!r}, got {t!r}")

    def parse_expr(self) -> tuple:
        node = self.parse_logic()
        if self.peek() is not None and self.peek() == ("op", "?"):
            self.next()
            a = self.parse_expr()
            self.expect_op(":")
            b = self.parse_expr()
            node = ("call", "if", [node, a, b])
        return node

    def parse_logic(self) -> tuple:
        node = self.parse_cmp()
        while self.peek() is not None and self.peek()[0] == "op" and self.peek()[1] in ("&&", "||"):
            op = self.next()[1]
            node = ("bin", op, node, self.parse_cmp())
        return node

    def parse_cmp(self) -> tuple:
        node = self.parse_sum()
        if self.peek() is not None and self.peek()[0] == "op" and self.peek()[1] in "<>=":
            op = self.next()[1]
            if op == "=":
                node = ("bin", "=", node, self.parse_sum())
            else:
                if self.peek() is not None and self.peek() == ("op", "="):
                    self.next()
                    op += "="
                node = ("bin", op, node, self.parse_sum())
        return node

    def parse_sum(self) -> tuple:
        node = self.parse_term()
        while self.peek() is not None and self.peek()[0] == "op" and self.peek()[1] in "+-":
            op = self.next()[1]
            node = ("bin", op, node, self.parse_term())
        return node

    def parse_term(self) -> tuple:
        node = self.parse_factor()
        while self.peek() is not None and self.peek()[0] == "op" and self.peek()[1] in "*/":
            op = self.next()[1]
            node = ("bin", op, node, self.parse_factor())
        return node

    def parse_factor(self) -> tuple:
        node = self.parse_unary()
        if self.peek() is not None and self.peek()[0] == "op" and self.peek()[1] == "^":
            self.next()
            node = ("bin", "^", node, self.parse_factor())
        return node

    def parse_unary(self) -> tuple:
        if self.peek() is not None and self.peek()[0] == "op" and self.peek()[1] in "-!":
            op = self.next()[1]
            if op == "-":
                nxt = self.toks[self.i] if self.i < len(self.toks) else None
                if nxt is not None and nxt[0] == "num":
                    self.next()
                    return ("lit", -float(nxt[1]))
                return ("neg", self.parse_unary())
            return ("call", "not", [self.parse_unary()])
        return self.parse_primary()

    def parse_primary(self) -> tuple:
        t = self.peek()
        if t is None:
            raise ValueError("unexpected end of expression")
        if t[0] == "num":
            self.next()
            return ("lit", float(t[1]))
        if t[0] == "id":
            self.next()
            name = t[1]
            # Daic115 pseudo-code chain (LOWERCASE if/elif/else): a nested Cond.
            # Capitalised If(...) stays an ordinary 3-arg function call.
            if name in ("if", "elif"):
                branches: list[tuple[tuple, tuple]] = []
                self.expect_op("(")
                cond = self.parse_expr()
                self.expect_op(")")
                val = self.parse_expr()
                branches.append((cond, val))
                while self.peek() is not None and self.peek()[0] == "id" and self.peek()[1] == "elif":
                    self.next()
                    self.expect_op("(")
                    c2 = self.parse_expr()
                    self.expect_op(")")
                    v2 = self.parse_expr()
                    branches.append((c2, v2))
                if self.peek() is not None and self.peek()[0] == "id" and self.peek()[1] == "else":
                    self.next()
                    node: tuple = self.parse_expr()
                else:
                    node = ("lit", 0.0)
                for c, v in reversed(branches):
                    node = ("call", "if", [c, v, node])
                return node
            if self.peek() is not None and self.peek() == ("op", "("):
                self.next()
                args: list[tuple] = []
                if self.peek() != ("op", ")"):
                    while True:
                        args.append(self.parse_expr())
                        if self.peek() == ("op", ","):
                            self.next()
                            continue
                        break
                self.expect_op(")")
                return ("call", name, args)
            return ("field", name)
        if t == ("op", "("):
            self.next()
            node = self.parse_expr()
            self.expect_op(")")
            if self.peek() is not None and self.peek() == ("op", "?"):
                self.next()
                a = self.parse_expr()
                self.expect_op(":")
                b = self.parse_expr()
                node = ("call", "if", [node, a, b])
            return node
        raise ValueError(f"unexpected token {t!r}")


# --------------------------------------------------------------------------- #
# mapping + rendering
# --------------------------------------------------------------------------- #

#: WorldQuant / GTJA op names → FQA op names (all lowercase, arity validated)
_OP_MAP: dict[str, str] = {
    "mean": "ts_mean", "avg": "ts_mean",
    "std": "ts_std", "stddev": "ts_std", "ts_std": "ts_std",
    "sum": "ts_sum", "sumif": "ts_sum",
    "max": "ts_max", "min": "ts_min",
    "tsmax": "ts_max", "tsmin": "ts_min", "ts_max": "ts_max", "ts_min": "ts_min",
    "tsrank": "ts_rank", "ts_rank": "ts_rank",
    "delta": "ts_delta", "ts_delta": "ts_delta",
    "delay": "ts_delay", "ts_delay": "ts_delay",
    "corr": "ts_corr", "ts_corr": "ts_corr",
    "coviance": "ts_cov", "covariance": "ts_cov", "ts_cov": "ts_cov",
    "decaylinear": "ts_decay_linear", "ts_decay_linear": "ts_decay_linear",
    "wma": "ts_wma", "sma": "ts_sma",
    "count": "ts_count",
    "vwap": "ts_vwap",
    "lowday": "ts_argmin", "highday": "ts_argmax",
    "ts_argmax": "ts_argmax", "ts_argmin": "ts_argmin",
    "rank": "rank",
    "abs": "abs", "log": "log", "sign": "sign",
    "pow": "power", "power": "power", "sqrt": "sqrt",
    "if": "cond", "and": "logical_and", "or": "logical_or",
    "add": "add", "sub": "sub", "mul": "mul", "div": "div",
    "neg": "neg", "min2": "min2", "max2": "max2",
    "scale": "cs_scale",
    "regbeta": "ts_beta",
    "ts_return": "ts_return", "ts_mean": "ts_mean", "ts_min2": "min2",
    # AQML / WorldQuant camel-case aliases
    "ts_sum": "ts_sum", "ts_decaylinear": "ts_decay_linear",
    "decay_linear": "ts_decay_linear", "correlation": "ts_corr",
    "ts_product": "ts_prod", "product": "ts_prod",
    "ts_kurt": "ts_kurt", "ts_skew": "ts_skew", "ts_zscore": "ts_zscore",
    "ts_median": "ts_median", "ts_quantile": "ts_quantile", "ts_ema": "ts_ema",
    "ts_prod": "ts_prod",
    # qlib Alpha158 dialect (loader.py, MIT) — $field args, element-wise
    # Greater/Less, time-series Rank, linear-regression family
    "ref": "ts_delay", "quantile": "ts_quantile",
    "idxmax": "ts_argmax", "idxmin": "ts_argmin",
    "greater": "max2", "less": "min2",
    "rsquare": "ts_rsq", "slope": "ts_slope", "resi": "ts_resi",
    "med": "ts_median", "mad": "ts_mad",
    "kurt": "ts_kurt", "skew": "ts_skew", "wma": "ts_wma",
    "beta": "ts_beta", "cov": "ts_cov",
    "count": "ts_count",
}

#: field names → FQA fields (or None ⇒ defer category)
_FIELD_MAP: dict[str, Optional[str]] = {
    "close": "Close", "open": "Open", "high": "High", "low": "Low",
    "volume": "Volume", "vol": "Volume",
    "amount": "Amount",
    # Daic115 shorthand
    "c": "Close", "o": "Open", "h": "High", "l": "Low", "v": "Volume",
    "amt": "Amount", "a": "Amount",
    "vwap": "__intraday_vwap__",  # intraday VWAP = amount / volume
    "returns": "__daily_returns__",  # AQML "returns" = TS_Return(Close, 1)
    "ret": "__daily_returns__",      # Daic115 RET = CLOSE/DELAY(CLOSE,1)-1
}

_BENCHMARK_FIELDS = {
    "benchmarkindexclose", "banchmarkindexopen", "indexclose", "indexopen",
    "banchmarkindexclose",
    "csi300", "hs300", "bmk", "bench_c", "bench_o", "bmk_c", "bmk_o",
    "mkt", "sequence", "smb", "hml", "self", "self_", "turn", "turnover",
    "ewma", "filter", "regresi", "sumac",
    "dtm", "dbm", "s_dtm", "s_dbm", "cap", "indneutralize", "sector",
    "industry", "indclass",
}

_BINARY_RENDER = {
    "+": "Add", "-": "Sub", "*": "Mul", "/": "Div", "^": "Power",
    "<": "Less", ">": "Greater", ">=": "Geq",
    "&&": "Logical_And", "||": "Logical_Or",
}

def _fmt_num(v: float) -> str:
    """Render a non-negative literal in FQA-parseable form.

    FQA's parser rejects scientific notation like ``1e-12`` (the embedded
    minus reads as an operator) — tiny magnitudes expand to plain decimals.
    """
    if float(v).is_integer():
        return str(int(v))
    if 0.0 < v < 1e-4:
        return f"{v:.15f}".rstrip("0").rstrip(".")
    return f"{v:g}"


#: FQA operators and their arities (for validation before rendering)
_ARITY = {
    "ts_mean": 2, "ts_std": 2, "ts_sum": 2, "ts_max": 2, "ts_min": 2,
    "ts_rank": 2, "ts_delta": 2, "ts_delay": 2, "ts_corr": 3, "ts_cov": 3,
    "ts_decay_linear": 2, "ts_wma": 2, "ts_sma": 3, "ts_count": 2,
    "ts_vwap": 3, "ts_argmin": 2, "ts_argmax": 2, "ts_return": 2,
    "ts_beta": 3,
    "rank": 1, "cs_scale": 1,
    "abs": 1, "log": 1, "sign": 1, "power": 2, "sqrt": 1, "neg": 1,
    "cond": 3, "logical_and": 2, "logical_or": 2,
    "add": 2, "sub": 2, "mul": 2, "div": 2, "min2": 2, "max2": 2,
    "ts_prod": 2, "ts_kurt": 2, "ts_skew": 2, "ts_zscore": 2,
    "ts_median": 2, "ts_quantile": 3, "ts_ema": 2,
    "ts_rsq": 2, "ts_slope": 2, "ts_resi": 2, "ts_mad": 2,
}


@dataclass
class Translation:
    status: str  # "ok" | "deferred"
    fqa: Optional[str] = None
    reasons: list[str] = field(default_factory=list)


class _Renderer:
    def __init__(self) -> None:
        self.reasons: list[str] = []
        self.deferred = False

    def _defer(self, reason: str) -> str:
        self.deferred = True
        self.reasons.append(reason)
        return "X"

    def render(self, node: tuple) -> str:
        kind = node[0]
        if kind == "lit":
            v = node[1]
            if v < 0:
                return f"Neg({self.render(('lit', -v))})"  # FQA has no negative literals
            return _fmt_num(v)
        if kind == "field":
            name = node[1].lower().lstrip("$")
            if name in _FIELD_MAP:
                mapped = _FIELD_MAP[name]
                if mapped == "__daily_returns__":
                    return "TS_Return(Close, 1)"
                if mapped == "__intraday_vwap__":
                    return "Div(Amount, Volume)"
                return mapped
            # Alpha101 derived series: ADV20/ADV40 = 20/40-day mean volume
            m = re.fullmatch(r"adv(\d+)", name)
            if m:
                return f"Ts_Mean(Volume, {int(m.group(1))})"
            if name in _BENCHMARK_FIELDS:
                return self._defer(f"benchmark/amount field: {node[1]}")
            return self._defer(f"unknown field: {node[1]}")
        if kind == "neg":
            return f"Neg({self.render(node[1])})"
        if kind == "bin":
            op = node[1]
            if op in ("<=", "==", "="):
                return self._defer(f"comparison {op!r} has no FQA operator")
            render_op = _BINARY_RENDER.get(op, op.title())
            return f"{render_op}({self.render(node[2])}, {self.render(node[3])})"
        if kind == "call":
            fname, args = node[1], node[2]
            key = fname.lower().lstrip("_")
            # SignedPower(x, k) → Mul(Sign(x), Power(Abs(x), k)) (AQML dialect)
            if key == "signedpower" and len(args) == 2:
                x = self.render(args[0])
                return f"Mul(Sign({x}), Power(Abs({x}), {self.render(args[1])}))"
            # Alpha101 FILTER(x, n) → where(|x| > n, x, 0)
            if key == "filter" and len(args) == 2:
                x = self.render(args[0])
                return f"Cond(Greater(Abs({x}), {self.render(args[1])}), {x}, 0)"
            # boolean NOT(x) → 1 - x
            if key == "not" and len(args) == 1:
                return f"Sub(1, {self.render(args[0])})"
            op = _OP_MAP.get(key)
            if op is None:
                return self._defer(f"unknown operator: {fname}")
            # Daic115 REGBETA(x, SEQUENCE(n)) = regression beta on the time
            # index — exactly a rolling slope
            if op == "ts_beta" and len(args) == 2 and args[1][0] == "call" and args[1][1].lower() == "sequence":
                return f"Ts_Slope({self.render(args[0])}, {self.render(args[1][2][0])})"
            # qlib Rank(x, d) is a TIME-SERIES percentile (2 args); the
            # WorldQuant/GTJA rank(x) is cross-sectional (1 arg)
            if op == "rank" and len(args) == 2:
                op = "ts_rank"
            # MIN/MAX disambiguation: rolling form has a NUMERIC second arg;
            # element-wise form has two non-literal operands, or a literal
            # FIRST operand (MAX(0, x) → Max2(x, 0))
            if op in ("ts_min", "ts_max") and len(args) == 2:
                if args[1][0] == "lit":
                    pass  # rolling window form
                elif args[0][0] == "lit" and args[1][0] != "lit":
                    op = "max2" if op == "ts_max" else "min2"
                    args = [args[1], args[0]]
                elif not any(a[0] == "lit" for a in args):
                    op = "min2" if op == "ts_min" else "max2"
            if len(args) != _ARITY.get(op, -1):
                return self._defer(f"arity mismatch: {fname}({len(args)} args)")
            rendered = [self.render(a) for a in args]
            canonical = op.title()
            return f"{canonical}({', '.join(rendered)})"
        return self._defer(f"unhandled node: {kind}")


def _substitute(node: tuple, name: str, definition: tuple) -> tuple:
    """Replace ``('field', name)`` nodes with ``definition`` (Daic115 ``where``)."""
    if node[0] == "field" and node[1].lower() == name:
        return definition
    if node[0] in ("neg",):
        return ("neg", _substitute(node[1], name, definition))
    if node[0] == "bin":
        return ("bin", node[1], _substitute(node[2], name, definition),
                _substitute(node[3], name, definition))
    if node[0] == "call":
        return ("call", node[1], [_substitute(a, name, definition) for a in node[2]])
    return node


def translate(expr: str) -> Translation:
    """Translate one expression into FQA grammar (or defer with reasons)."""
    try:
        toks = _tokenize(expr)
        parser = _Parser(toks)
        ast = parser.parse_expr()
        # Daic115 "… where cond = C > DELAY(C, 1)" — named sub-expression
        if parser.peek() is not None and parser.peek()[0] == "id" and parser.peek()[1] == "where":
            parser.next()
            t = parser.next()
            parser.expect_op("=")
            definition = parser.parse_expr()
            ast = _substitute(ast, t[1].lower(), definition)
        if parser.peek() is not None:
            raise ValueError(f"trailing tokens: {toks[parser.i:]}")
    except ValueError as exc:
        return Translation(status="deferred", reasons=[f"parse: {exc}"])
    r = _Renderer()
    fqa = r.render(ast)
    if r.deferred:
        return Translation(status="deferred", reasons=r.reasons)
    return Translation(status="ok", fqa=fqa)


# --------------------------------------------------------------------------- #
# inventory runner
# --------------------------------------------------------------------------- #

def translate_inventory(
    inventory_path: str | Path = "paper/factor_zoo/inventory.json",
    out_path: str | Path = "paper/factor_zoo/translated.json",
) -> dict[str, Any]:
    """Translate every formula in the inventory; write per-id results + summary."""
    inv = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    summary: dict[str, dict[str, int]] = {}
    families = [k for k in inv if k != "summary"]
    for family in families:
        results: dict[str, dict] = {}
        counts: dict[str, int] = {}
        for e in inv.get(family, []):
            src = e.get("aqml") or e.get("original")
            if not src:
                results[e["id"]] = {"status": "deferred", "reasons": ["no formula"]}
                counts["no_formula"] = counts.get("no_formula", 0) + 1
                continue
            t = translate(src)
            results[e["id"]] = (
                {"status": "ok", "fqa": t.fqa}
                if t.status == "ok"
                else {"status": "deferred", "reasons": t.reasons}
            )
            counts[t.status] = counts.get(t.status, 0) + 1
        out[family] = results
        summary[family] = counts
    payload = {"summary": summary, **out}
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


__all__ = ["translate", "translate_inventory", "Translation"]
