"""News ingestion — real-time-forward, resumable, deduplicated (Phase 9.1).

Free news sources expose no historical pagination (verified 2026-08-11):
``ak.stock_news_em(symbol)`` returns the latest ~10 items for one symbol with
no ``date`` argument; the ``stock_info_global_*`` feeds are current-day only.
``stock_news_em`` is still the *only* free per-symbol news feed, so the
ingestor drives it per HS300 symbol and stores results per collection day.

Design (adapted from the blueprint's Week-1 item with the history-bounded
reality): the crawler keeps a ``crawl_state.json`` so a partial run resumes;
each collection run appends to a per-day Parquet shard (``news_YYYY-MM-DD``)
deduped on the article URL. There is deliberately **no 2022-2025 backfill** —
the source cannot serve it — so the store grows forward from the first ingest.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

_NEWS_COLUMNS = ["symbol", "title", "content", "publish_time", "source", "url"]
# AKShare's Chinese column names map onto these (they arrive as raw CJK labels
# regardless of locale; matched by position because the feed order is stable).
_FIELD_ORDER = ["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"]


@dataclass
class NewsItem:
    symbol: str
    title: str
    content: str
    publish_time: str
    source: str
    url: str

    def to_text(self) -> str:
        return f"{self.title}。{self.content}"


def _normalize_symbol(raw: str) -> str:
    """AKShare keys news by bare 6-digit code; convert to system form.

    ``stock_news_em`` returns ``关键词`` like ``600519`` — we append ``.SH``/
    ``.SZ`` from the 6-digit prefix the same way ``to_baostock``/universe lists
    do (000/001/002/003/300/301 -> SZ, else SH). BJ is never in HS300.
    """
    code = str(raw).split(".")[0].strip()
    if not (code.isdigit() and len(code) == 6):
        return str(raw)
    return f"{code}.{'SZ' if code[:3] in ('000', '001', '002', '003', '300', '301') else 'SH'}"


def fetch_symbol_news(symbol: str, pause: float = 0.0) -> list[NewsItem]:
    """Latest news items for one system symbol via AKShare stock_news_em.

    The feed returns at most ~10 of the most recent articles; there is no
    date filter, so this is inherently *forward* collection — every run grabs
    whatever is newest as of now.
    """
    code = symbol.split(".")[0]
    try:
        df = ak.stock_news_em(symbol=code)
    except Exception as exc:  # noqa: BLE001 — a flaky symbol must not kill the run
        logger.warning("stock_news_em(%s) failed: %s", symbol, exc)
        return []
    if df is None or df.empty:
        return []
    items: list[NewsItem] = []
    for _, row in df.iterrows():
        title = str(row.get("新闻标题", "") or "")
        content = str(row.get("新闻内容", "") or "")
        if not title and not content:
            continue
        items.append(
            NewsItem(
                symbol=_normalize_symbol(str(row.get("关键词", symbol) or symbol)),
                title=title,
                content=content,
                publish_time=str(row.get("发布时间", "") or ""),
                source=str(row.get("文章来源", "") or ""),
                url=str(row.get("新闻链接", "") or ""),
            )
        )
    if pause:
        time.sleep(pause)
    return items


class NewsIngestor:
    """Per-symbol real-time-forward news collection with resume state.

    State lives in ``<data_dir>/crawl_state.json`` (``last_run`` date and the
    per-run URL bloom); news shards are Parquet files ``news_<date>.parquet``
    holding deduped ``NewsItem`` rows.
    """

    def __init__(self, data_dir: str = "data/news") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "crawl_state.json"
        self._state: dict = self._load_state()

    def _load_state(self) -> dict:
        if self.state_file.is_file():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — corrupt state resets
                logger.warning("corrupt crawl_state.json, resetting")
        return {"last_run": None, "seen_urls": {}}

    def _save_state(self) -> None:
        self.state_file.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _shard(self, date: str) -> Path:
        return self.data_dir / f"news_{date}.parquet"

    def _seen_urls(self, date: str) -> set[str]:
        return set(self._state.get("seen_urls", {}).get(date, []))

    def collect(self, symbols: list[str], pause: float = 0.0) -> dict:
        """One real-time-forward sweep over ``symbols``; dedup on article URL.

        Returns per-symbol item counts. The sweep stores into the shard for the
        item's *publish date* (falling back to today for missing timestamps),
        so a single run can touch several recent days' shards.
        """
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        rows_by_day: dict[str, list[dict]] = {}
        seen = dict(self._state.get("seen_urls", {}))
        counts: dict[str, int] = {}

        for symbol in symbols:
            items = fetch_symbol_news(symbol, pause=pause)
            fresh = 0
            for item in items:
                day = item.publish_time[:10] or today
                if item.url and item.url in seen.get(day, set()):
                    continue
                rows_by_day.setdefault(day, []).append(
                    {
                        "symbol": item.symbol,
                        "title": item.title,
                        "content": item.content,
                        "publish_time": item.publish_time,
                        "source": item.source,
                        "url": item.url,
                    }
                )
                if item.url:
                    seen.setdefault(day, set()).add(item.url)
                fresh += 1
            counts[symbol] = fresh

        for day, rows in rows_by_day.items():
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=_NEWS_COLUMNS)
            shard = self._shard(day)
            if shard.is_file():
                old = pd.read_parquet(shard)
                df = pd.concat([old, df], ignore_index=True).drop_duplicates(
                    subset=["url"], keep="last"
                )
            df.to_parquet(shard, index=False)

        self._state["last_run"] = today
        self._state["seen_urls"] = {k: sorted(v) for k, v in seen.items()}
        self._save_state()
        return counts

    def get_news(self, symbol: str, date: str) -> list[NewsItem]:
        """All stored items for one symbol published on ``date`` (PIT text)."""
        shard = self._shard(date)
        if not shard.is_file():
            return []
        df = pd.read_parquet(shard)
        df = df[df["symbol"] == symbol]
        return [
            NewsItem(
                symbol=r.symbol,
                title=r.title,
                content=r.content,
                publish_time=r.publish_time,
                source=r.source,
                url=r.url,
            )
            for r in df.itertuples()
        ]

    def coverage(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Per-day article counts (for the ≥ coverage data gate)."""
        recs = []
        for shard in sorted(self.data_dir.glob("news_*.parquet")):
            day = shard.stem.removeprefix("news_")
            if start and day < start:
                continue
            if end and day > end:
                continue
            df = pd.read_parquet(shard)
            recs.append({"date": day, "articles": int(len(df)),
                         "symbols": int(df["symbol"].nunique())})
        return pd.DataFrame(recs, columns=["date", "articles", "symbols"])


# ---------------------------------------------------------------------------
# Research reports — the backfillable sentiment channel (Phase 9.1 gate source)
# ---------------------------------------------------------------------------

_REPORT_COLS = ["symbol", "title", "rating", "broker", "industry", "date", "url"]
# Eastmoney report column order (positions are stable across versions):
#   [0]序号 [1]股票代码 [2]股票简称 [3]报告名称 [4]东财评级 [5]机构
#   [6]近一月个股研报数 [7..12]盈利预测 [13]行业 [14]日期 [15]报告PDF链接
_REPORT_POS = {"title": 3, "rating": 4, "broker": 5, "industry": 13, "date": 14, "url": 15}


class ReportIngestor:
    """Historical per-symbol research reports (Eastmoney via AKShare).

    Unlike news, ``ak.stock_research_report_em(symbol)`` returns the **full
    per-symbol history** (verified 2017-08 → present; 760 reports for 600519),
    which is what makes the 2022-2025 sentiment gate testable. Reports are the
    blueprint §3.1 "东方财富研报" channel and carry two sentiment signals:
    the report *title* (scored by the TriAgent) and the analyst *rating*
    (买入/增持/持有 — used as a cross-check, kept raw here).

    One Parquet per symbol under ``data/reports/``; a full HS300 sweep is a
    one-time ~15-20 min background job (the endpoint has no date param and
    always returns everything, so per-symbol caching makes re-runs free).
    """

    def __init__(self, data_dir: str = "data/reports") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _symbol_cache(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol.replace('.', '_')}.parquet"

    def cached_symbols(self) -> set[str]:
        # A 0-byte file is a failed write (Windows pyarrow errno-22) — not a
        # cache; the symbol must be re-fetched.
        return {p.stem.replace("_", ".") for p in self.data_dir.glob("*.parquet")
                if p.stat().st_size > 0}

    def collect(self, symbols: list[str], pause: float = 0.2, force: bool = False) -> dict[str, int]:
        """Fetch every uncached (or forced) symbol's full report history.

        Returns per-symbol row counts. A flaky symbol logs a warning and is
        skipped, never killing the sweep (blueprint risk table mitigation).
        """
        counts: dict[str, int] = {}
        for symbol in symbols:
            cache = self._symbol_cache(symbol)
            if not force and cache.is_file():
                continue
            code = symbol.split(".")[0]
            try:
                df = ak.stock_research_report_em(symbol=code)
            except Exception as exc:  # noqa: BLE001 — a flaky symbol must not kill the run
                logger.warning("stock_research_report_em(%s) failed: %s", symbol, exc)
                continue
            rows = self._normalize(df, symbol)
            if not rows:
                logger.warning("no report rows for %s", symbol)
                continue
            # Atomic write (temp + rename) — Windows pyarrow can throw a
            # transient errno-22 on direct writes; a flaky symbol must never
            # kill the sweep (resume skips already-cached symbols). A symbol
            # that fails to persist stays uncached and is retried next run.
            tmp = cache.with_suffix(".parquet.tmp")
            try:
                pd.DataFrame(rows, columns=_REPORT_COLS).to_parquet(tmp, index=False)
                tmp.replace(cache)
            except OSError as exc:  # noqa: BLE001
                logger.warning("persist failed for %s (%s); skipping", symbol, exc)
                tmp.unlink(missing_ok=True)
                continue
            counts[symbol] = len(rows)
            if pause:
                time.sleep(pause)
        return counts

    @staticmethod
    def _normalize(df: pd.DataFrame, fallback_symbol: str) -> list[dict]:
        """Map AKShare CJK columns to the standard report schema."""
        if df is None or df.empty:
            return []
        try:
            pos = {"title": 3, "rating": 4, "broker": 5, "symbol": 1,
                   "industry": 13, "date": 14, "url": 15}
        except IndexError:  # pragma: no cover — defensive against AKShare reordering
            pos = {"title": 3, "rating": 4, "broker": 5, "symbol": 1,
                   "industry": 13, "date": 14, "url": 15}
        rows: list[dict] = []
        for _, r in df.iterrows():
            title = str(r.iloc[pos["title"]] or "").strip()
            if not title:
                continue
            rows.append({
                "symbol": _normalize_symbol(str(r.iloc[pos["symbol"]] or fallback_symbol)),
                "title": title,
                "rating": str(r.iloc[pos["rating"]] or "").strip(),
                "broker": str(r.iloc[pos["broker"]] or "").strip(),
                "industry": str(r.iloc[pos["industry"]] or "").strip(),
                "date": str(r.iloc[pos["date"]] or "")[:10],
                "url": str(r.iloc[pos["url"]] or "").strip(),
            })
        return rows

    def load(self, symbols: list[str] | None = None,
             start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Union of per-symbol report caches, filtered to symbols/dates."""
        parts = []
        for p in sorted(self.data_dir.glob("*.parquet")):
            if p.stat().st_size == 0:
                continue  # failed write; not a valid cache shard
            sym = p.stem.replace("_", ".")
            if symbols is not None and sym not in set(symbols):
                continue
            df = pd.read_parquet(p)
            if start:
                df = df[df["date"] >= start]
            if end:
                df = df[df["date"] <= end]
            if not df.empty:
                parts.append(df)
        if not parts:
            return pd.DataFrame(columns=_REPORT_COLS)
        return pd.concat(parts, ignore_index=True)
