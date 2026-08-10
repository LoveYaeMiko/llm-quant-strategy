"""Baostock — free fallback: universe snapshots + index constituents (ADR-0004).

Baostock 0.9.30 still calls ``DataFrame.append``, removed in pandas 2.x, so we
monkeypatch it back at import time. Login is lazy (the handshake takes ~1-2 min
against the socket server) and every call is rate-limited gently.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..schema.rate_limiter import RateLimiters
from ..schema.symbols import A_SHARE_PREFIXES, normalize_symbol, to_baostock

logger = logging.getLogger(__name__)

# pandas 2.x backport — baostock's internals expect DataFrame.append.
if not hasattr(pd.DataFrame, "append"):

    def _append(self, other, ignore_index=False, **kwargs):
        return pd.concat([self, other], ignore_index=ignore_index)

    pd.DataFrame.append = _append


def _is_index_code(code: str) -> bool:
    """Exchange-aware index filter, used only when ``query_all_stock`` lacks a type column.

    The 000/001/002/003 prefixes are SZ *stocks* but SH *indexes* (``sh.000001``
    = 上证综指), so a single merged prefix set leaks SH indexes into the universe
    (the pre-fix "known gap"). Check against the exchange's own A-stock prefixes:
    SH 600/601/603/605/688/689, SZ 000/001/002/003/300/301 (B-shares 900/200 and
    index families like 399/880/950 are excluded).
    """
    c6 = code.split(".")[-1] if "." in code else code
    ex = code.split(".")[0].lower() if "." in code else ""
    if ex == "sh":
        return not c6.startswith(("600", "601", "603", "605", "688", "689"))
    if ex == "sz":
        return not c6.startswith(("000", "001", "002", "003", "300", "301"))
    if ex == "bj":
        return False  # all 北交所 codes are stocks; excluded separately via _is_bj_code
    return not c6.startswith(A_SHARE_PREFIXES)


def _is_bj_code(code: str) -> bool:
    """Baostock form ``bj.430047`` (AlphaFeed serves none of these → empty)."""
    return code.lower().startswith("bj.")


class BaostockAdapter:
    """Lazy, rate-limited facade over the baostock socket API."""

    def __init__(self, config=None, enabled: bool = True) -> None:
        self.enabled = enabled
        self._bs = None
        self._logged_in = False
        RateLimiters.configure(config)

    # -- session -----------------------------------------------------------

    def _ensure_login(self) -> None:
        if not self.enabled:
            raise RuntimeError("baostock is disabled (BAOSTOCK_ENABLED=false)")
        if self._logged_in:
            return
        import baostock as bs

        with RateLimiters.baostock:
            lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock login failed: {lg.error_msg}")
        self._bs = bs
        self._logged_in = True

    def _query(self, fn_name: str, **kwargs) -> pd.DataFrame:
        """Run a baostock query (by function name) and drain its ResultData.

        ``_bs`` is lazy and None until login, so the callable must be resolved
        AFTER :meth:`_ensure_login` — grabbing it at the call site crashes with
        ``'NoneType' object has no attribute ...`` before login ever runs.
        """
        self._ensure_login()
        fn = getattr(self._bs, fn_name)
        with RateLimiters.baostock:
            rs = fn(**kwargs)
        if rs.error_code != "0":
            raise RuntimeError(f"baostock {fn.__name__} failed: {rs.error_msg}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        return pd.DataFrame(rows, columns=rs.fields)

    # -- queries -----------------------------------------------------------

    def fetch_universe(self, date: str) -> pd.DataFrame:
        """Membership snapshot at ``date`` (single baostock call).

        Filters index codes via the authoritative ``type == '1'`` column when the
        feed provides it, else a prefix heuristic; excludes 北交所. Returns a
        frame with canonical ``symbol`` + ``date`` columns.
        """
        df = self._query("query_all_stock", day=date)
        if df.empty:
            return df
        if "type" in df.columns:
            df = df[df["type"] == "1"]
        else:
            df = df[~df["code"].map(_is_index_code)]
        df = df[~df["code"].map(_is_bj_code)]
        df["symbol"] = df["code"].map(normalize_symbol)
        df["date"] = pd.Timestamp(date)
        renames = {"code_name": "name", "ipoDate": "ipo_date", "outDate": "out_date"}
        df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
        keep = ["symbol", "date"]
        for col in ["name", "ipo_date", "out_date", "status"]:
            if col in df.columns:
                keep.append(col)
        return df[keep]

    def fetch_index_constituents(self, index: str, date: str) -> list[str]:
        """HS300/ZZ500 constituents at ``date`` (canonical symbols), for the research universe."""
        name = index.lower()
        if name == "hs300":
            fn_name = "query_hs300_stocks"
        elif name == "zz500":
            fn_name = "query_zz500_stocks"
        else:
            raise ValueError(f"unknown index {index!r} (use 'hs300' or 'zz500')")
        df = self._query(fn_name, date=date)
        if df.empty:
            return []
        code_col = "code" if "code" in df.columns else df.columns[0]
        return sorted(df[code_col].map(normalize_symbol))

    def fetch_stock_basic(self, symbol: str) -> pd.Series:
        """IPO/out dates + type for one symbol (delisted names included)."""
        df = self._query("query_stock_basic", code=to_baostock(normalize_symbol(symbol)))
        if df.empty:
            return pd.Series(dtype=object)
        return df.rename(columns={"code_name": "name", "ipoDate": "ipo_date", "outDate": "out_date"}).iloc[0]

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        if self._logged_in:
            try:
                self._bs.logout()
            except Exception:  # noqa: BLE001 — logout is best-effort
                pass
            self._logged_in = False
