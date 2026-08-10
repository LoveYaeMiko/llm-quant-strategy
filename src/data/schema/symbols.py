"""A-share symbol canonicalization — one format across every data source.

The canonical form is AlphaFeed's ``600519.SH`` (6-digit code + exchange suffix).
Baostock returns ``sh.600000`` / ``sz.000001`` / ``bj.430047``; AKShare wants a
bare 6-digit code. A bare 6-digit code is ambiguous without an exchange marker
(``000001`` is 平安银行 on SZ but 上证指数 on SH), so it raises.
"""

from __future__ import annotations

import re

_EXCHANGE_PREFIX = {"sh": "SH", "sz": "SZ", "bj": "BJ"}
_CANONICAL_RE = re.compile(r"^(\d{6})\.(SH|SZ|BJ)$")
_BAOSTOCK_RE = re.compile(r"^(sh|sz|bj)\.(\d{6})$")

# A-share code prefixes (canonical). Anything else is an index / B-share / misc
# instrument. Used by Baostock's fallback index filter and by :func:`from_bare_code`.
A_SHARE_PREFIXES = (
    "600", "601", "603", "605", "688",           # 沪主板 / 科创板
    "000", "001", "002", "003", "300", "301",    # 深主板 / 创业板
    "43", "83", "87", "92",                       # 北交所 (920xxx = 2025+ renumbering)
)


class SymbolError(ValueError):
    """Raised when a symbol cannot be canonicalized (missing exchange marker)."""


def exchange_of(code: str) -> str:
    """Return the exchange suffix ('SH'/'SZ'/'BJ') for a code or canonical symbol."""
    m = _CANONICAL_RE.match(code)
    if m:
        return m.group(2)
    m = _BAOSTOCK_RE.match(code)
    if m:
        return _EXCHANGE_PREFIX[m.group(1)]
    raise SymbolError(f"cannot determine exchange for {code!r} (need 600519.SH or sh.600000)")


def normalize_symbol(code: str) -> str:
    """Canonicalize to AlphaFeed form ``600519.SH``.

    Accepts ``sh.600000``, ``sz.000001``, ``bj.430047`` or a canonical passthrough.
    A bare 6-digit code (``600519``) is ambiguous and raises.
    """
    m = _CANONICAL_RE.match(code)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    m = _BAOSTOCK_RE.match(code)
    if m:
        return f"{m.group(2)}.{_EXCHANGE_PREFIX[m.group(1)]}"
    raise SymbolError(
        f"cannot normalize {code!r} — need an exchange marker (600519.SH or sh.600000); "
        "a bare code is ambiguous across exchanges"
    )


def to_alphafeed(code: str) -> str:
    """Alias of :func:`normalize_symbol` (the canonical form)."""
    return normalize_symbol(code)


def to_baostock(symbol: str) -> str:
    """Convert a canonical symbol to Baostock's form ``sh.600519``."""
    m = _CANONICAL_RE.match(symbol)
    if not m:
        raise SymbolError(f"cannot convert {symbol!r} to baostock form")
    suffix = m.group(2).lower()
    return f"{suffix}.{m.group(1)}"


def code6(symbol: str) -> str:
    """Bare 6-digit code (what AKShare / most web sources want)."""
    m = _CANONICAL_RE.match(symbol)
    if not m:
        raise SymbolError(f"cannot extract 6-digit code from {symbol!r}")
    return m.group(1)


def is_bj(symbol: str) -> bool:
    """True for 北交所 symbols (AlphaFeed serves none of them — returns empty)."""
    return symbol.endswith(".BJ")


def from_bare_code(code: str) -> str:
    """Best-effort canonicalization of a bare 6-digit code.

    Some feeds (e.g. AKShare's ``stock_zh_a_spot_em``) return bare codes without
    an exchange marker. Infer by prefix — a real 600519 means 沪市, 000001 深市 —
    which is unambiguous in practice because A-share codes are exchange-assigned
    by prefix. ``92xxxx`` is the 2025+ 北交所 renumbering, so it must be checked
    before the generic ``9`` (which otherwise means a 900xxx SH B-share).
    """
    c = str(code).strip().zfill(6)
    if c.startswith(("4", "8", "92")):
        return f"{c}.BJ"
    if c.startswith(("6", "9")):
        return f"{c}.SH"
    if c.startswith(("0", "2", "3")):
        return f"{c}.SZ"
    raise SymbolError(f"cannot infer exchange for bare code {c!r}")
