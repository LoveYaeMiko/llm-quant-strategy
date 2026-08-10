# Implementation Blueprint: Production-Grade Data Foundation with AlphaFeed + Free Data Stack

> **Target**: Claude Code (or AI Agent)  
> **Goal**: Implement all Phase 1 (Data) and Phase 2 (Research) requirements from `requirements.md` using **AlphaFeed as primary market data source** and **Baostock + AKShare as free fundamental/text data sources**.  
> **Current State**: Framework exists (107 tests pass). `verify` passes with synthetic data. DeepSeek connected.  
> **End State**: Real PIT data loaded, walk-forward enabled, verify checks passing with real data.

---

## 1. Environment & Dependencies Setup

### 1.1 Update `pyproject.toml`

```toml
dependencies = [
    "alphafeed>=1.0.0",          # Market data (commercial, API key)
    "baostock>=0.8.8",           # Free A-share fundamentals, index constituents, delistings
    "akshare>=1.16.0",           # Free news, announcements, alternative data
    "psycopg2-binary>=2.9.0",    # PostgreSQL driver
    "sqlalchemy>=2.0.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "pytest>=8.0.0",
    "click>=8.1.0",
    "python-dotenv>=1.0.0",
]
```

### 1.2 Environment Variables (`.env`)

```bash
# AlphaFeed (Primary Market Data)
ALPHAFEED_API_KEY="your_key_here"

# Baostock & AKShare (No keys required, but we keep flags)
BAOSTOCK_ENABLED=true
AKSHARE_ENABLED=true

# Database
DATABASE_URL="postgresql://user:pass@localhost:5432/quant_pit"

# Optional: Fallback/Verification
TUSHARE_TOKEN=""  # Keep optional, not required for core flow
```

### 1.3 Database Setup

```sql
-- Run once to create schema
CREATE DATABASE quant_pit;
-- Tables will be auto-created by SQLAlchemy ORM or migration script.
```

---

## 2. Architecture: Three-Layer Data Ingestion

```
┌─────────────────────────────────────────────────────────────────────┐
│                       cli.py ingest                                │
├─────────────────────────────────────────────────────────────────────┤
│                         │                                          │
│          ┌──────────────┼──────────────────────────────┐          │
│          ▼              ▼                              ▼          │
│  ┌───────────────┐ ┌──────────────┐ ┌──────────────────────┐    │
│  │ AlphaFeed     │ │ Baostock     │ │ AKShare              │    │
│  │ Adapter       │ │ Adapter      │ │ Adapter              │    │
│  ├───────────────┤ ├──────────────┤ ├──────────────────────┤    │
│  │ • Daily K     │ │ • Stock Basic│ │ • News (Eastmoney)  │    │
│  │ • Quotes      │ │ • Financials │ │ • Announcements     │    │
│  │ • Min K       │ │ • HS300/SH50 │ │ • Research Reports  │    │
│  │ • Adjust Fact │ │ • Delistings │ │                      │    │
│  │ • Ticks       │ │ • Industry   │ │                      │    │
│  └───────────────┘ └──────────────┘ └──────────────────────┘    │
│                         │                                          │
│                         ▼                                          │
│              ┌─────────────────────┐                              │
│              │ PointInTimeStore    │                              │
│              │ (upsert / get_as_of)│                              │
│              └─────────────────────┘                              │
│                         │                                          │
│                         ▼                                          │
│                   verify (4 checks)                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Implementation: Rate Limiter (Critical)

AlphaFeed has strict limits. Must implement token bucket **before any API call**.

**File**: `src/data/schema/rate_limiter.py`

```python
import time
import threading
from typing import Optional

class RateLimiter:
    """Thread-safe token bucket for AlphaFeed API limits"""
    
    def __init__(self, rate_per_minute: int, burst: Optional[int] = None):
        self.rate = rate_per_minute / 60.0  # tokens per second
        self.burst = burst or rate_per_minute
        self.tokens = self.burst
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> float:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_refill = now
            
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.rate
                time.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= tokens
            return time.time()
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, *args):
        pass

# Singleton instances for each endpoint
class RateLimiters:
    daily = RateLimiter(300)        # 300/min for single
    daily_batch = RateLimiter(120)  # 120/min for batch (200 per call)
    quote = RateLimiter(300)        # 300/min for quotes
    minute = RateLimiter(120)       # 120/min for single
    minute_batch = RateLimiter(60)  # 60/min for batch
    tick = RateLimiter(120)         # 120/min for single
    tick_batch = RateLimiter(60)    # 60/min for batch
    adjust = RateLimiter(120)       # 120/min for adjust factors
```

---

## 4. AlphaFeed Adapter (Primary Market Data)

**File**: `src/data/ingestion/alphafeed_adapter.py`

```python
import pandas as pd
from typing import List, Optional
from alphafeed import AlphaFeed
from src.data.schema.rate_limiter import RateLimiters
from src.data.pit_store import PointInTimeStore

class AlphaFeedAdapter:
    """Primary market data source using AlphaFeed"""
    
    def __init__(self, api_key: str):
        self.af = AlphaFeed(api_key=api_key)
        self.store = PointInTimeStore()
        self.limiters = RateLimiters()
    
    def fetch_daily_batch(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        adjust: str = "forward"
    ) -> dict:
        """
        Fetches daily OHLCV in batches of 200.
        Rate: 120 calls/min, 200 symbols/call = 24,000 symbols/min.
        
        For A-share 5000 stocks, takes ~13 seconds for full history.
        """
        batch_size = 200
        results = {}
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            self.limiters.daily_batch.acquire()
            
            df = self.af.klines.batch(
                batch,
                period="1d",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                to_dataframe=True,
                show_progress=False
            )
            
            # AlphaFeed returns dict: {symbol: DataFrame}
            for sym, data in df.items():
                if not data.empty:
                    # Standardize columns: date, open, high, low, close, volume, adj_close
                    data["symbol"] = sym
                    data["valid_from"] = data["date"]
                    data["valid_to"] = "2099-12-31"
                    results[sym] = data
            
        return results
    
    def fetch_universe(self, market: str = "CN_Stock") -> List[str]:
        """Fetch all active stocks in given market"""
        self.limiters.quote.acquire()
        df = self.af.quotes.get(universes=market, to_dataframe=True)
        return df["symbol"].tolist()
    
    def fetch_adjust_factors(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch adjustment factors (splits/dividends) in batches of 200"""
        batch_size = 200
        all_factors = []
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            self.limiters.adjust.acquire()
            df = self.af.klines.get_adjust_factor(
                symbols=batch,
                to_dataframe=True
            )
            all_factors.append(df)
        
        return pd.concat(all_factors, ignore_index=True) if all_factors else pd.DataFrame()
    
    def ingest_full_history(self, start_date: str = "2010-01-01", end_date: str = "2025-12-31"):
        """Full pipeline: fetch universe -> fetch daily -> upsert to PIT store"""
        print("Fetching universe...")
        symbols = self.fetch_universe("CN_Stock")
        print(f"Found {len(symbols)} symbols. Fetching daily data...")
        
        daily_data = self.fetch_daily_batch(symbols, start_date, end_date)
        
        print(f"Ingesting {len(daily_data)} symbols into PointInTimeStore...")
        for sym, df in daily_data.items():
            self.store.upsert(df, symbol=sym)
        
        print("Done. Running verify...")
        # verify will be called separately
        return len(daily_data)
```

---

## 5. Baostock Adapter (Free Fundamentals, Index, Delistings)

**File**: `src/data/ingestion/baostock_adapter.py`

```python
import baostock as bs
import pandas as pd
from src.data.pit_store import PointInTimeStore

class BaostockAdapter:
    """
    Free A-share data source.
    Covers: Stock basic info (with delisting dates), HS300/SSE50 constituents,
    financials (8 categories), industry classification.
    No API keys, no rate limits (but be gentle).
    """
    
    def __init__(self):
        bs.login()
        self.store = PointInTimeStore()
    
    def fetch_stock_basic(self) -> pd.DataFrame:
        """
        Returns all A-share stocks with delisting dates.
        Fields: code, tradeStatus, outDate (delisting date).
        """
        rs = bs.query_all_stock()
        df = rs.get_data()
        # Convert to PIT format
        df["valid_from"] = df["ipoDate"] if "ipoDate" in df.columns else "1990-01-01"
        df["valid_to"] = df["outDate"].fillna("2099-12-31")
        return df
    
    def fetch_hs300_constituents(self, date: str = None) -> pd.DataFrame:
        """Fetch HS300 constituents as of given date. If None, latest."""
        rs = bs.query_hs300_stocks(date)
        return rs.get_data()
    
    def fetch_ss50_constituents(self, date: str = None) -> pd.DataFrame:
        """Fetch SSE50 constituents as of given date."""
        rs = bs.query_sz50_stocks(date)
        return rs.get_data()
    
    def fetch_financials(self, code: str, year: int, quarter: int) -> dict:
        """
        Fetch quarterly financials.
        Available: profit_data, growth_data, cash_flow, du_point, etc.
        """
        # Profit data (income statement)
        profit = bs.query_profit_data(code, year, quarter).get_data()
        # Growth data
        growth = bs.query_growth_data(code, year, quarter).get_data()
        # Cash flow
        cashflow = bs.query_cash_flow_data(code, year, quarter).get_data()
        # Dupont analysis (ROE decomposition)
        dupont = bs.query_dupont_data(code, year, quarter).get_data()
        return {
            "profit": profit,
            "growth": growth,
            "cashflow": cashflow,
            "dupont": dupont
        }
    
    def fetch_industry(self, code: str) -> str:
        """Fetch GICS-like industry classification."""
        rs = bs.query_stock_industry(code)
        df = rs.get_data()
        return df["industry"].iloc[0] if not df.empty else "Unknown"
    
    def ingest_delistings(self):
        """Update delisting events into PIT store."""
        basic = self.fetch_stock_basic()
        delisted = basic[basic["outDate"].notna()]
        # Mark these symbols with valid_to = outDate
        for _, row in delisted.iterrows():
            self.store.update_valid_to(row["code"], row["outDate"])
        return len(delisted)
```

---

## 6. AKShare Adapter (News, Announcements, Alternative Data)

**File**: `src/data/ingestion/akshare_adapter.py`

```python
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

class AKShareAdapter:
    """
    Free alternative data source.
    Covers: Financial news, announcements, research reports.
    Note: AKShare uses web scraping, may break. Implement retries and caching.
    """
    
    def fetch_news(self, symbol: str = None, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch Eastmoney news. If symbol provided, fetch specific stock news.
        """
        try:
            if symbol:
                df = ak.stock_news_em(symbol=symbol)
            else:
                # Latest market news (no symbol filter)
                df = ak.stock_news_em(symbol="000001")  # Placeholder
            # Filter by date
            cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            df = df[df["publish_time"] >= cutoff]
            return df
        except Exception as e:
            print(f"AKShare news fetch failed: {e}")
            return pd.DataFrame()
    
    def fetch_announcements(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch stock announcements from Eastmoney."""
        try:
            df = ak.stock_notice_report(symbol=symbol)
            # Filter by date range
            df = df[(df["notice_date"] >= start_date) & (df["notice_date"] <= end_date)]
            return df
        except Exception as e:
            print(f"AKShare announcements failed: {e}")
            return pd.DataFrame()
    
    def fetch_research_reports(self, symbol: str) -> pd.DataFrame:
        """Fetch research reports from Eastmoney."""
        try:
            df = ak.stock_research_report_em(symbol=symbol)
            return df
        except Exception as e:
            print(f"AKShare research reports failed: {e}")
            return pd.DataFrame()
```

---

## 7. PointInTimeStore Enhancements

**File**: `src/data/pit_store.py` (Update existing)

```python
# Add these methods to existing PointInTimeStore

def upsert(self, df: pd.DataFrame, symbol: str):
    """
    Insert or update records for a symbol.
    Assumes df has columns: date, open, high, low, close, volume, adj_close, valid_from, valid_to
    """
    # Implementation uses SQLAlchemy or direct SQL
    # For SQLite/PostgreSQL:
    # ON CONFLICT (symbol, date) DO UPDATE SET ...
    pass

def get_as_of(self, symbols: List[str], date: str) -> pd.DataFrame:
    """
    Get all records active on a specific date.
    Includes delisted stocks if date < delisting_date.
    """
    # SELECT * FROM prices 
    # WHERE symbol IN :symbols 
    # AND valid_from <= :date AND valid_to >= :date
    pass

def has_future_leak(self, symbol: str, date: str) -> bool:
    """
    Verify no data from after 'date' is used in backtest.
    Returns True if leak detected.
    """
    # Check if any record has valid_from > date
    pass
```

---

## 8. CLI Integration

**File**: `cli.py` (Update existing)

```python
import click
from src.data.ingestion.alphafeed_adapter import AlphaFeedAdapter
from src.data.ingestion.baostock_adapter import BaostockAdapter
from src.data.ingestion.akshare_adapter import AKShareAdapter

@click.group()
def cli():
    pass

@cli.command()
@click.option('--start', default='2010-01-01')
@click.option('--end', default='2025-12-31')
@click.option('--source', default='alphafeed', help='alphafeed | baostock | all')
def ingest(start, end, source):
    """Ingest real data from configured sources"""
    
    if source in ['alphafeed', 'all']:
        print("🚀 Ingesting from AlphaFeed...")
        adapter = AlphaFeedAdapter(api_key=os.getenv('ALPHAFEED_API_KEY'))
        count = adapter.ingest_full_history(start, end)
        print(f"✅ AlphaFeed: {count} symbols ingested")
    
    if source in ['baostock', 'all']:
        print("🚀 Ingesting from Baostock...")
        adapter = BaostockAdapter()
        basic = adapter.fetch_stock_basic()
        # Store delistings
        delisted_count = adapter.ingest_delistings()
        print(f"✅ Baostock: {delisted_count} delistings updated")
    
    if source in ['akshare', 'all']:
        print("🚀 Ingesting from AKShare...")
        adapter = AKShareAdapter()
        # Fetch sample news (or full range if configured)
        news = adapter.fetch_news(days_back=30)
        print(f"✅ AKShare: {len(news)} news articles fetched")
    
    print("🔍 Running verify...")
    # Call verify programmatically
    from tests.test_verify import run_verify
    run_verify()

@cli.command()
def verify():
    """Run all verification checks"""
    # Import and run existing verify
    pass

if __name__ == '__main__':
    cli()
```

---

## 9. Verification Checks (Requirement B1-B5)

**File**: `tests/test_verify.py`

```python
import pytest
from src.data.pit_store import PointInTimeStore

def test_no_future_leak():
    """B1: Ensure no data from future dates leak into past"""
    store = PointInTimeStore()
    # For any backtest date T, ensure all valid_from <= T
    df = store.get_as_of(symbols=['all'], date='2020-01-01')
    assert (df['valid_from'] <= '2020-01-01').all(), "Future leak detected!"

def test_survivorship_bias_included():
    """B4: Ensure delisted stocks are included in historical queries"""
    store = PointInTimeStore()
    # Check that we have delisted stocks in 2015
    df_2015 = store.get_as_of(symbols=['all'], date='2015-01-01')
    basic = store.get_basic_info()
    delisted_in_2015 = basic[basic['delist_date'] > '2015-01-01']
    # Ensure at least some delisted symbols appear
    assert len(set(delisted_in_2015['symbol']) & set(df_2015['symbol'])) > 0

def test_adjustment_consistency():
    """B3: Ensure adjusted close is consistent with adjustment factors"""
    # Check that close * factor[t-1]/factor[t] ≈ adjusted_close
    pass

def test_data_freshness():
    """B5: Monitor data staleness"""
    store = PointInTimeStore()
    last_date = store.get_last_date()
    assert (pd.Timestamp.now() - pd.Timestamp(last_date)).days < 3, "Data is stale"
```

---

## 10. Walk-Forward Configuration (Requirement C1-C2)

**File**: `configs/master_config.yaml` (Update)

```yaml
# Research time periods
research:
  train_start: "2010-01-01"
  train_end: "2019-12-31"
  val_start: "2020-01-01"
  val_end: "2021-12-31"
  test_start: "2022-01-01"
  test_end: "2025-12-31"
  
  # Factor mining parameters
  mining:
    iterations: 50          # C1: Increased from default 3
    walks: 20               # C1: More walk-forward rounds
    ic_threshold: 0.02
    rank_ic_threshold: 0.035
    bonferroni_correction: true  # C1: Multiple testing correction

# Data sources
data:
  primary: "alphafeed"
  fallback: "baostock"
  pit_database_url: "${DATABASE_URL}"
  
# Industry neutralization (C4)
factor_processing:
  neutralize_industry: true
  gics_level: 1            # 1=GICS Sector, 2=Industry Group
  neutralization_method: "pca"  # pca | zscore

# Market state bins (C5)
market_states:
  - "bull"
  - "bear"
  - "sideways"
  - "high_vol_low_ret"     # C5: Extended from 3 to 4 states
```

---

## 11. Decay Monitoring (Requirement C3)

**File**: `src/monitoring/decay_tracker.py`

```python
import pandas as pd
from src.data.pit_store import PointInTimeStore
from src.factors.metrics import compute_rank_ic

class DecayTracker:
    """Monitor factor performance over rolling windows"""
    
    def __init__(self, window_days: int = 90):
        self.window = window_days
        self.store = PointInTimeStore()
    
    def monitor(self, factor_name: str, factor_values: pd.Series):
        """Compute rolling RankIC and ICIR"""
        # Get price data for rolling window
        df = self.store.get_as_of(symbols=factor_values.index, date='latest')
        # Compute forward returns
        returns = df.groupby('symbol')['adj_close'].pct_change().shift(-1)
        
        # Rolling RankIC
        rank_ic = compute_rank_ic(factor_values, returns, self.window)
        icir = rank_ic.mean() / rank_ic.std()
        
        # Alert if below threshold
        if icir < 0.02:
            print(f"⚠️  Factor {factor_name} decayed: ICIR={icir:.4f}")
            # Auto-archive factor
        return rank_ic, icir
```

---

## 12. Implementation Order for Claude Code

### Step 1: Environment & Rate Limiter (30 min)
- [ ] Add dependencies to `pyproject.toml`
- [ ] Create `.env` with `ALPHAFEED_API_KEY`
- [ ] Implement `RateLimiter` with tests

### Step 2: AlphaFeed Adapter (1 hour)
- [ ] Implement `fetch_daily_batch` with batch logic
- [ ] Implement `fetch_universe`
- [ ] Implement `fetch_adjust_factors`
- [ ] Test with 10 symbols, verify response

### Step 3: PointInTimeStore Enhancement (30 min)
- [ ] Implement `upsert`
- [ ] Implement `get_as_of`
- [ ] Implement `has_future_leak`

### Step 4: Baostock Adapter (45 min)
- [ ] Implement `fetch_stock_basic` (includes delistings)
- [ ] Implement `fetch_hs300_constituents`
- [ ] Implement `fetch_financials`

### Step 5: AKShare Adapter (30 min)
- [ ] Implement `fetch_news`
- [ ] Implement `fetch_announcements`
- [ ] Add retry logic (AKShare can be flaky)

### Step 6: Verify Checks (30 min)
- [ ] Update `tests/test_verify.py` with B1-B5 checks
- [ ] Ensure all tests pass

### Step 7: Walk-Forward Config (15 min)
- [ ] Update YAML with train/val/test splits
- [ ] Update CLI to use periods

### Step 8: Integration Test (30 min)
- [ ] Run `cli.py ingest --start 2010-01-01 --end 2020-01-01 --source all`
- [ ] Run `cli.py verify` → must be **GREEN**

---

## 13. Success Criteria

After implementation, these commands must work:

```bash
# 1. Full data ingestion (should complete in < 5 minutes for 10 years of data)
python cli.py ingest --start 2010-01-01 --end 2020-01-01 --source alphafeed

# 2. Verify all checks pass (B1-B5)
python cli.py verify
# Expected output: ✅ All 107 tests passed

# 3. Backtest with real data
python cli.py backtest --start 2020-01-01 --end 2021-12-31 --factors ./outputs/factors.json

# 4. Factor mining with walk-forward
python cli.py mine --config configs/master_config.yaml --iterations 50 --walks 20
```

---

## 14. Troubleshooting Common Issues

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| AlphaFeed rate limit hit | Too many calls in short window | Increase batch size or add sleep |
| AKShare returns empty DataFrame | Website structure changed | Update AKShare (`pip install -U akshare`) or use fallback |
| Delisting dates missing | Baostock outDate may be NaN | Cross-check with AKShare or manual list |
| verify fails on survivorship | PIT store missing delisted stocks | Re-run Baostock `fetch_stock_basic` |

---

## 15. Final Prompt to Claude Code

> **Claude Code**, execute this blueprint in order. Do not skip steps.  
> 1. Install all dependencies.  
> 2. Implement RateLimiter first and test it.  
> 3. Implement AlphaFeedAdapter and test with a small batch (10 symbols).  
> 4. Enhance PointInTimeStore.  
> 5. Implement BaostockAdapter and AKShareAdapter.  
> 6. Run the full ingestion pipeline with `--source all`.  
> 7. Run `verify` and confirm all 107 tests pass with **real data** (not synthetic).  
> 8. If any test fails, fix immediately before proceeding.  
> 9. Commit working state.  
> 10. Report back with metrics (number of symbols ingested, time taken, verify results).  
>  
> **Critical**: The `verify` command must output "All 107 tests passed" before you consider this phase complete.