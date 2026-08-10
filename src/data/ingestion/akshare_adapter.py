"""AKShare — free news / fundamentals scraping (flaky → every call retried).

AKShare hits EastMoney's website; endpoints break on site changes and time out.
Every call goes through :func:`retry_call` so a transient failure degrades to a
graceful empty result instead of killing the whole ingestion.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..schema.retry import retry_call
from ..schema.symbols import code6, from_bare_code
from ..schema.rate_limiter import RateLimiters

logger = logging.getLogger(__name__)


class AkshareAdapter:
    """Retried facade over the AKShare news/fundamentals endpoints."""

    def __init__(self, config=None, enabled: bool = True) -> None:
        self.enabled = enabled
        RateLimiters.configure(config)

    def _ak(self):
        import akshare as ak

        return ak

    def fetch_news(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        """Recent news for one symbol → frame with symbol/date/title/content/source."""
        if not self.enabled:
            return pd.DataFrame()
        ak = self._ak()

        def _fetch():
            df = ak.stock_news_em(symbol=code6(symbol))
            if df is None or df.empty:
                return df
            return pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": pd.to_datetime(df["发布时间"]),
                    "title": df["新闻标题"],
                    "content": df["新闻内容"],
                    "source": df["文章来源"],
                }
            )

        df = retry_call(_fetch, retries=2, base_delay=1.0, backoff=2.0, on_error=lambda exc: pd.DataFrame())
        return df.head(limit) if limit and not df.empty else df

    def fetch_fundamental_snapshot(self) -> pd.DataFrame:
        """One-call current fundamentals snapshot for the whole market (Q4 pilot).

        Columns: symbol, name, date, pe_ttm, pb, market_cap, turnover.
        """
        if not self.enabled:
            return pd.DataFrame()
        ak = self._ak()
        df = retry_call(
            ak.stock_zh_a_spot_em,
            retries=3,
            base_delay=2.0,
            backoff=2.0,
            on_error=lambda exc: pd.DataFrame(),
        )
        if df is None or df.empty:
            return df
        out = pd.DataFrame()
        out["symbol"] = df["代码"].map(from_bare_code)
        out["name"] = df["名称"]
        out["date"] = pd.Timestamp.today().normalize()
        out["pe_ttm"] = pd.to_numeric(df.get("市盈率-动态"), errors="coerce")
        out["pb"] = pd.to_numeric(df.get("市净率"), errors="coerce")
        out["market_cap"] = pd.to_numeric(df.get("总市值"), errors="coerce")
        out["turnover"] = pd.to_numeric(df.get("换手率"), errors="coerce")
        return out

    # -- reserved endpoints (kept for interface parity; flaky, so honest stubs) --

    def fetch_announcements(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        """Announcement feed — unimplemented for now (Q3: news first, extend later)."""
        if not self.enabled:
            return pd.DataFrame()
        logger.info("announcements endpoint not yet wired (akshare flaky) — skipping %s", symbol)
        return pd.DataFrame()

    def fetch_research_reports(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        """Research-report feed — unimplemented for now."""
        if not self.enabled:
            return pd.DataFrame()
        logger.info("research-reports endpoint not yet wired — skipping %s", symbol)
        return pd.DataFrame()
