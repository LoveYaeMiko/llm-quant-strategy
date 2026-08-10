"""Ingestion orchestrator — universe → prices → fundamentals → news.

One pipeline run writes PIT records into the shared ``pit_records`` store
(SQLite or Postgres) and prints a data-quality summary (Q6): coverage days,
max gap, and a survivorship self-check — so the human can flip
``data.real_data: true`` with eyes open.

Phases mirror the blueprint §6 flow but keep the Q1-Q4 bounded defaults:
full-A data is ingested, research runs on ``research.universe``; fundamentals
and news are opt-in pilots.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from ...config import Config
from ..point_in_time_loader import SYMBOL, VALID_FROM, from_url
from ..schema.symbols import is_bj, normalize_symbol
from .akshare_adapter import AkshareAdapter
from .alphafeed_adapter import AlphaFeedAdapter, to_price_records
from .baostock_adapter import BaostockAdapter
from .convert import (
    FUNDAMENTAL,
    PRICE,
    TEXT,
    UNIVERSE,
    fundamental_records,
    price_records,
    text_records,
    universe_records,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    """Tally + quality metrics reported by :meth:`Ingestor.ingest`."""

    bars: int = 0
    universe_records: int = 0
    fundamentals: int = 0
    news: int = 0
    symbols_ok: int = 0
    symbols_failed: int = 0
    symbols_empty: int = 0  # legitimately no history (new/quiet names) — not an error
    errors: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    coverage_days: Optional[int] = None
    min_date: object = None
    max_date: object = None
    max_gap_days: Optional[int] = None
    delisted: int = 0
    survivorship_date: str = ""

    def summary(self) -> str:
        lines = [
            "═══ 数据入库摘要 ═══",
            f"价格K线记录    : {self.bars:,}",
            f"成分股快照记录  : {self.universe_records:,}",
            f"基本面记录     : {self.fundamentals:,}",
            f"新闻/公告记录  : {self.news:,}",
            f"成功 / 失败个股 : {self.symbols_ok} / {self.symbols_failed}",
            f"无历史(新/停牌): {self.symbols_empty}",
            f"耗时           : {self.elapsed:.1f}s",
        ]
        if self.coverage_days is not None:
            lines += [
                f"价格覆盖天数    : {self.coverage_days}",
                f"价格区间        : {self.min_date} ~ {self.max_date}",
                f"最大缺口(工作日): {self.max_gap_days} 天",
            ]
        lines.append(
            f"幸存者自查      : {self.delisted} 只在 {self.survivorship_date} 存在、现已退市/缺席"
        )
        if self.errors:
            lines.append(f"失败明细（{len(self.errors)} 条，最多列前 10）:")
            lines += [f"  - {e}" for e in self.errors[:10]]
        return "\n".join(lines)


class Ingestor:
    """Ties the three adapters to the PIT store and drives the four phases."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.store = from_url(config.require("data.pit_database_url"))
        self.batch_size = int(config.get("data.alphafeed.batch_size", 100))
        self.universe_dir = Path(config.get("data.universe_dir", "data/universe"))
        self.primary = AlphaFeedAdapter(config.require("data.alphafeed.api_key"), config)
        # ${BAOSTOCK_ENABLED} interpolates to "" when unset — treat that as
        # enabled (the free fallback is on by default), not silently disabled.
        baostock_enabled = str(config.get("data.baostock.enabled", "true")).lower() in ("1", "true", "yes", "")
        self.baostock = BaostockAdapter(config, enabled=baostock_enabled)
        akshare_enabled = str(config.get("data.akshare.enabled", "true")).lower() in ("1", "true", "yes", "")
        self.akshare = AkshareAdapter(config, enabled=akshare_enabled)

    # -- public ------------------------------------------------------------

    def ingest(
        self,
        *,
        symbols: Optional[list[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        fundamentals: bool = False,
        news: bool = False,
        resume: bool = False,
        limit: Optional[int] = None,
    ) -> IngestStats:
        """Run the pipeline; returns an :class:`IngestStats` quality summary."""
        t0 = time.time()
        stats = IngestStats(survivorship_date=str(self.config.get("data.checks.survivorship_date", "")))
        try:
            if symbols is None:
                self._ingest_universe(stats)
            self._ingest_prices(symbols, start, end, limit, resume, stats)
            if fundamentals:
                self._ingest_fundamentals(symbols, stats)
            if news:
                self._ingest_news(stats)
            stats = self._fill_quality(stats)
        finally:
            self.baostock.close()
        stats.elapsed = time.time() - t0
        return stats

    # -- phase 1: universe ---------------------------------------------------

    def _ingest_universe(self, stats: IngestStats) -> None:
        anchors = []
        surv = str(self.config.get("data.checks.survivorship_date", ""))
        if surv:
            anchors.append(surv)
        today = str(pd.Timestamp.today().normalize().date())
        if not anchors or anchors[-1] != today:
            anchors.append(today)
        for day in anchors:
            try:
                frame = self.baostock.fetch_universe(day)
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"universe {day}: {exc}")
                logger.warning("universe snapshot %s failed: %s", day, exc)
                continue
            if frame.empty:
                continue
            recs = universe_records(frame)
            self.store.upsert(recs)
            stats.universe_records += len(recs)
            self.universe_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(self.universe_dir / f"universe_{day}.csv", index=False)
        self._ingest_index_universe(stats)

    def _ingest_index_universe(self, stats: IngestStats) -> None:
        """Cache HS300/ZZ500 constituents as JSON for the research universe."""
        self.universe_dir.mkdir(parents=True, exist_ok=True)
        for name in ("hs300", "zz500"):
            try:
                syms = self.baostock.fetch_index_constituents(name, str(pd.Timestamp.today().normalize().date()))
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"index {name}: {exc}")
                logger.warning("index %s fetch failed: %s", name, exc)
                continue
            (self.universe_dir / f"{name}.json").write_text(
                json.dumps(syms, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # -- phase 2: prices -----------------------------------------------------

    def _resolve_price_symbols(
        self, symbols: Optional[list[str]], limit: Optional[int]
    ) -> list[str]:
        if symbols:
            targets = [normalize_symbol(s) for s in symbols]
        else:
            latest = self.store.max_date(UNIVERSE)
            if not pd.isna(latest):
                targets = self.store.universe_as_of(latest, UNIVERSE)
            else:
                quotes = self.primary.fetch_quotes()
                col = "symbol" if "symbol" in quotes.columns else quotes.columns[0]
                targets = [normalize_symbol(str(s)) for s in quotes[col]]
        targets = [s for s in targets if not is_bj(s)]
        if limit:
            targets = targets[:limit]
        return targets

    def _ingest_prices(
        self,
        symbols: Optional[list[str]],
        start: Optional[str],
        end: Optional[str],
        limit: Optional[int],
        resume: bool,
        stats: IngestStats,
    ) -> None:
        targets = self._resolve_price_symbols(symbols, limit)
        if not targets:
            logger.warning("price pass: no symbols to ingest")
            return
        start_ts = pd.Timestamp(start or self.config.get("project.start_date", "2010-01-01"))
        end_ts = pd.Timestamp(end or pd.Timestamp.today().normalize())
        latest = self.store.max_valid_from(PRICE) if resume else pd.NaT
        present = set(self.store.symbols(PRICE)) if resume and not pd.isna(latest) else set()
        if resume and not pd.isna(latest):
            # Resume skips only symbols ALREADY stored (they re-fetch from the
            # global newest bar to capture restatements); brand-new symbols must
            # still get their full history — a global start would silently drop it.
            groups = []
            fresh = [s for s in targets if s not in present]
            if fresh:
                groups.append((fresh, start_ts))
            existing = [s for s in targets if s in present]
            if existing:
                groups.append((existing, max(start_ts, latest)))
            if not groups:
                logger.warning("price pass: nothing new to ingest (all symbols already stored)")
                return
        else:
            groups = [(targets, start_ts)]
        for group, group_start in groups:
            self._ingest_price_chunks(group, group_start, end_ts, stats)

    def _ingest_price_chunks(
        self, targets: list[str], start_ts: pd.Timestamp, end_ts: pd.Timestamp, stats: IngestStats
    ) -> None:
        """Fetch + upsert one symbol group in batches, isolating batch failures."""
        chunks = [targets[i : i + self.batch_size] for i in range(0, len(targets), self.batch_size)]
        step = max(1, len(chunks) // 10)
        for idx, chunk in enumerate(chunks, 1):
            try:
                klines = self.primary.fetch_klines(chunk, start=start_ts, end=end_ts, adjust="none")
                factors = self.primary.fetch_ex_factors(chunk)
                frame = to_price_records(klines, factors)
                if frame.empty:
                    # legitimately no history (new listing / long-suspended) —
                    # not an error, but worth surfacing so it isn't silent
                    stats.symbols_empty += len(chunk)
                    logger.info("price pass: no data for %s", chunk)
                    continue
                recs = price_records(frame)
                self.store.upsert(recs)
                stats.bars += len(recs)
                stats.symbols_ok += len(recs[SYMBOL].unique())
            except Exception as exc:  # noqa: BLE001 — a bad batch must not kill the run
                stats.symbols_failed += len(chunk)
                stats.errors.append(f"{chunk[0]}..{chunk[-1]}: {exc}")
                logger.error("price batch %s failed: %s", chunk, exc)
            if idx % step == 0 or idx == len(chunks):
                logger.info("price pass %d/%d batches, %d bars so far", idx, len(chunks), stats.bars)

    # -- phase 3: fundamentals (Q4 pilot) ------------------------------------

    def _ingest_fundamentals(self, symbols: Optional[list[str]], stats: IngestStats) -> None:
        pilot_size = int(self.config.get("data.fundamentals.pilot_size", 50))
        frame = self.akshare.fetch_fundamental_snapshot()
        if frame.empty:
            stats.errors.append("fundamentals: akshare snapshot empty")
            logger.warning("fundamentals snapshot empty")
            return
        if symbols:
            want = set(normalize_symbol(s) for s in symbols[:pilot_size])
            frame = frame[frame[SYMBOL].isin(want)]
        else:
            frame = frame.head(pilot_size)
        if frame.empty:
            return
        self.store.upsert(fundamental_records(frame))
        stats.fundamentals += len(frame)

    # -- phase 4: news (Q3 watchlist) -----------------------------------------

    def _ingest_news(self, stats: IngestStats) -> None:
        watch = self.config.get("data.news.watchlist") or []
        if not watch:
            logger.warning("news: empty watchlist in config")
            return
        per_symbol = int(self.config.get("data.news.per_symbol", 20))
        for sym in watch:
            try:
                frame = self.akshare.fetch_news(normalize_symbol(sym), limit=per_symbol)
            except Exception as exc:  # noqa: BLE001
                stats.errors.append(f"news {sym}: {exc}")
                continue
            if frame.empty:
                continue
            self.store.upsert(text_records(frame, kind="news"))
            stats.news += len(frame)

    # -- quality summary (Q6) -------------------------------------------------

    def _fill_quality(self, stats: IngestStats) -> IngestStats:
        lo = self.store.min_date(PRICE)
        hi = self.store.max_valid_from(PRICE)
        if pd.isna(lo) or pd.isna(hi):
            return stats
        stats.min_date = lo
        stats.max_date = hi
        dates = self.store.distinct_dates(PRICE)
        stats.coverage_days = len(dates)
        if len(dates) >= 2:
            # business days strictly between adjacent bars → 0 means back-to-back
            gaps = [len(pd.bdate_range(a, b, inclusive="neither")) for a, b in zip(dates[:-1], dates[1:])]
            stats.max_gap_days = max(gaps)
        if stats.survivorship_date:
            stats.delisted = len(self.store.delisted_symbols(stats.survivorship_date, UNIVERSE))
        return stats


def resolve_research_universe(config: Config, store=None) -> list[str]:
    """Resolve ``research.universe`` to a concrete symbol list.

    * ``"all"`` → every symbol in the store's latest price pass;
    * a cached name under ``data.universe_dir`` (e.g. ``hs300_500``) → union of
      the ``hs300.json`` / ``zz500.json`` constituents;
    * anything else → treated as a literal comma-separated symbol list.
    """
    name = str(config.get("research.universe", "hs300_500"))
    if name == "all":
        if store is None:
            store = from_url(config.require("data.pit_database_url"))
        latest = store.max_date(PRICE)
        return store.universe_as_of(latest, PRICE) if not pd.isna(latest) else []
    universe_dir = Path(config.get("data.universe_dir", "data/universe"))
    if name == "hs300_500":
        symbols = []
        for sub in ("hs300", "zz500"):
            path = universe_dir / f"{sub}.json"
            if path.is_file():
                symbols.extend(json.loads(path.read_text(encoding="utf-8")))
        if symbols:
            return sorted(set(symbols))
        raise FileNotFoundError(
            f"research.universe={name!r} but {universe_dir} has no hs300.json/zz500.json "
            "— run `python -m src.cli ingest` first"
        )
    # literal list, e.g. "600519.SH,000001.SZ"
    return sorted(set(normalize_symbol(s.strip()) for s in name.split(",") if s.strip()))
