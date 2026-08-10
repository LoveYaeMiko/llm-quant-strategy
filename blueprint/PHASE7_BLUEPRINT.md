# Implementation Blueprint: Phase 7.1 → 7.5 — Production Readiness & Factor Mining Launch

> **Target**: Claude Code (or AI Agent)  
> **Current State**: Phase 7 complete — 158 tests pass, PIT storage ready, dual-column adjustment validated on 6 symbols × 2010–2025.  
> **Goal**: Move from "foundation built" to "research-ready" — commit the codebase, resolve blocking issues, ingest full A-share universe, flip `data.real_data`, and launch first factor mining run.  
> **Prerequisite**: Read `PHASE7_REPORT.md` first. This blueprint picks up exactly where it left off.

---

## 1. Immediate Actions (Phase 7.1 — Commit & Clean)

### 1.1 Workspace Status
Current working directory has **大量未提交改动** (including `PROJECT_BLUEPRINT.md` / `resource/*` moved to `blueprint/`). Need to commit logically.

**Step-by-step commit plan**:

```bash
# Step 1: Check current status
git status --short

# Step 2: Stage and commit by logical grouping

# Group 1: ADR documents
git add docs/adr/
git commit -m "docs: add ADR-0001 through ADR-0005 (closed-interval, dual-column, postgres-jsonb, baostock-universe, bounded-research)"

# Group 2: Core data source adapters + schema
git add src/data/ingestion/alphafeed_adapter.py
git add src/data/ingestion/baostock_adapter.py
git add src/data/ingestion/akshare_adapter.py
git add src/data/ingestion/convert.py
git add src/data/schema/symbols.py
git add src/data/schema/rate_limiter.py
git add src/data/schema/retry.py
git commit -m "feat: data source adapters (AlphaFeed primary, Baostock/AKShare supplementary) with symbol normalization"

# Group 3: PIT storage (Postgres + SQLite backends)
git add src/data/point_in_time_loader.py
git add src/data/postgres_loader.py
git commit -m "feat: PointInTimeStore with Postgres JSONB backend, SQLite fallback"

# Group 4: Ingestor orchestration + CLI
git add src/data/ingestion/ingestor.py
git add src/cli.py
git commit -m "feat: orchestrated ingestor (universe → price → fundamentals → news) with resume, CLI commands"

# Group 5: Verification checks (B1-B5 + 4 standing checks)
git add src/checklist.py
git commit -m "feat: B1-B5 real-data checks + standing pipeline checks"

# Group 6: Research support (decay tracker + walk-forward config)
git add src/monitoring/decay_tracker.py
git add configs/master_config.yaml
git commit -m "feat: DecayTracker, walk-forward config (train/val/test splits)"

# Group 7: Tests
git add tests/
git commit -m "test: 158 passed tests, including regression for multi-symbol factor swap and backward adjustment"

# Group 8: Blueprint migration (moved from root to blueprint/)
git add blueprint/
git rm PROJECT_BLUEPRINT.md  # if still at root
git rm -r resource/          # if still at root
git commit -m "chore: move blueprints to blueprint/ directory"

# Group 9: Docker/Compose
git add docker-compose.yml
git commit -m "chore: docker-compose for Postgres 17"

# Tag milestone
git tag phase7-data-foundation
```

### 1.2 Update CONTEXT.md
After commit, update `CONTEXT.md` with:

```markdown
## 已定架构决策 (ADR Summary)

| ADR | Decision |
|-----|----------|
| ADR-0001 | Price bars: closed interval (`valid_to = valid_from + 1D`) |
| ADR-0002 | Dual-column adjustment: `close` (adjusted) + `raw_close` + `adjust_factor` |
| ADR-0003 | Postgres JSONB PIT backend: `pit_records(symbol, valid_from, valid_to, payload, updated_at)` |
| ADR-0004 | Universe: Baostock `query_all_stock(day)`, filtered (exclude indices/Beijing) |
| ADR-0005 | Research universe: bounded (`hs300_500` / `all` / literal list) |

## 术语表 (Glossary)

| Term | Definition |
|------|------------|
| `valid_to = NaT` | Open interval (current/latest record) |
| `record_type` | Discriminator column: `price` / `universe` / `fundamental` / `text` |
| `close` | Post-adjusted close (forward-adjusted, anchored to latest) |
| `adjust_factor` | Cumulative adjustment factor (backward-propagated from ex-events) |
| `research.universe` | Config key for bounded research universe (default: `hs300_500`) |
| `data.real_data` | Gate switch — must be manually set `true` after verifying real data |

## 测试基线

- **158 tests passed** (pytest)
- Key regression tests:
  - `test_to_price_records_multisymbol_factors_not_swapped`
  - `test_to_price_records_applies_factor_backward`
```

---

## 2. Resolve Blocking Issues (Phase 7.2)

### 2.1 B4 — Survivorship Bias (Universe History)

**Problem**: Baostock port 10030 is blocked. `query_all_stock(day)` for historical universe snapshots is unavailable. Current `quotes.get()` only returns **current** constituents.

**Solution A (Recommended)**: Implement daily cumulative universe snapshots using AlphaFeed `quotes.get()`.

```python
# src/data/ingestion/universe_snapshot.py

def build_universe_history(start_date: str, end_date: str, step: str = "1D"):
    """
    Build historical universe by accumulating daily snapshots from AlphaFeed.
    This is forward-looking at runtime but becomes PIT-correct once stored.
    """
    date_range = pd.date_range(start_date, end_date, freq=step)
    universe_records = []
    
    for date in date_range:
        # Fetch current constituents as of this date
        df = alphafeed_adapter.fetch_quotes(as_of=date)
        for _, row in df.iterrows():
            universe_records.append({
                "symbol": row["symbol"],
                "valid_from": date,
                "valid_to": date + pd.Timedelta(days=1),  # closed interval
                "record_type": "universe",
                "payload": {"name": row.get("name"), "exchange": row.get("exchange")}
            })
    
    # Bulk upsert into PIT store
    pit_store.bulk_upsert(universe_records, record_type="universe")
```

**Implementation tasks**:
1. Create `src/data/ingestion/universe_snapshot.py`
2. Add to `Ingestor` pipeline (after price, before fundamentals)
3. Run for 2010–2025 with weekly snapshots (not daily, to save API calls)
4. B4 verification: query 2015-01-01, assert delisted stocks appear

**Solution B (Fallback)**: Use AKShare `stock_zh_a_hist` with `adjust=''` to get historical listings — but AKShare is web-scraped and less reliable.

**Decision**: Implement A first. If AlphaFeed `quotes` cannot provide historical as-of, fall back to B.

---

### 2.2 B5 — Data Freshness Semantics

**Problem**: `ingest --end 2025-12-31` (historical backfill) is inherently "stale". B5 checks freshness against `now()`, causing false failure.

**Fix**: Add `--mode` flag to `verify`:

```python
# src/checklist.py

def check_data_freshness(mode: str = "live"):
    """
    mode: 'backfill' | 'live'
    - backfill: skip freshness check (or check against --end date)
    - live: check against now() - 3 days
    """
    if mode == "backfill":
        # Freshness relative to config.project.end_date, not now()
        threshold_days = 7
        last_date = store.max_date()
        end_date = config["project"]["end_date"]
        gap = (pd.Timestamp(end_date) - pd.Timestamp(last_date)).days
        assert gap < threshold_days, f"Data gap: {gap} days vs {threshold_days} expected"
    else:
        # Standard live freshness check
        ...
```

**Update CLI**:

```bash
python -m src.cli verify --mode backfill   # for historical runs
python -m src.cli verify --mode live       # for daily operations
```

---

## 3. Full A-Share Ingestion (Phase 7.3)

### 3.1 Pre-flight Check

```bash
# 1. Confirm database is ready
docker ps | grep postgres

# 2. Confirm AlphaFeed API key is set
echo $ALPHAFEED_API_KEY  # or check .env

# 3. Run a small batch to validate
python -m src.cli ingest \
    --start 2010-01-01 \
    --end 2025-12-31 \
    --limit 100 \
    --symbols all

# Expected: 100 symbols ingested, ~390,000 rows, no errors
```

### 3.2 Full Ingestion Command

```bash
# Background execution recommended
nohup python -m src.cli ingest \
    --start 2010-01-01 \
    --end 2025-12-31 \
    --symbols all \
    > logs/ingest_full_$(date +%Y%m%d).log 2>&1 &

# Monitor progress
tail -f logs/ingest_full_*.log
```

**Estimated Resources**:
- Symbols: ~5,600
- Bars/symbol: ~3,900 (2010–2025, ~252 trading days/year × 15 years + 2010 partial + 2025 partial = ~3,886)
- Total rows: ~21.8 million
- AlphaFeed API calls: ~56 (28 batches × 2 calls: K-line + ex-factor)
- Time: 4–6 hours (network + Postgres write)
- Storage: ~2-3 GB (Postgres with JSONB)

### 3.3 Post-Ingestion Validation

```bash
# 1. Verify counts
python -m src.cli verify --mode backfill

# Expected output:
# B1 (no_future_leak)   : ✅ PASS
# B3 (adjustment)       : ✅ PASS (0 violations)
# B4 (survivorship)     : ✅ PASS (if universe snapshots done)
# B5 (freshness)        : ✅ PASS (backfill mode)

# 2. Check database directly
psql -U pit -d pit_data -c "
    SELECT 
        COUNT(*) as total_records,
        COUNT(DISTINCT symbol) as unique_symbols,
        MIN(valid_from) as earliest_date,
        MAX(valid_from) as latest_date
    FROM pit_records
    WHERE payload->>'record_type' = 'price';
"
```

---

## 4. Flip `data.real_data` (Phase 7.4)

**After** full ingestion completes and `verify --mode backfill` passes:

```yaml
# configs/master_config.yaml
data:
  real_data: true  # ⬅️ Manually change from false to true
  pit_database_url: "postgresql://pit:pit@localhost:5432/pit_data"
  # If using production, use env var:
  # pit_database_url: "${DATABASE_URL}"
```

**Verification**:

```bash
# Now verify with real_data=true
python -m src.cli verify --mode backfill

# All checks should pass. If B4 still fails, re-run universe snapshot.
```

**Commit the config change**:

```bash
git add configs/master_config.yaml
git commit -m "chore: flip data.real_data to true after full A-share ingestion"
git tag phase7-real-data
```

---

## 5. Launch Factor Mining (Phase 7.5)

### 5.1 Pre-mining Checklist

```bash
# 1. Confirm real_data is true
grep "real_data" configs/master_config.yaml
# Should output: real_data: true

# 2. Confirm DeepSeek API key is set
echo $DEEPSEEK_API_KEY

# 3. Run a quick smoke test of the factor pipeline
python -m src.cli mine --iterations 1 --walks 1
# Should complete in < 2 minutes, produce at least 1 candidate factor
```

### 5.2 First Real Mining Run

```bash
# Full mining: 50 iterations, 20 walk-forward rounds
python -m src.cli mine \
    --config configs/master_config.yaml \
    --iterations 50 \
    --walks 20 \
    --memory-path ./outputs/memory_state.pkl \
    --output ./outputs/factor_pool.json

# Expected output:
# - 50 iterations × 10 candidates = 500 factor proposals
# - ~50-100 pass IC > 0.02 (training set)
# - ~10-20 pass validation set IC > 0.02
# - Final factor pool: 5-10 high-quality factors
```

### 5.3 Backtest the Factor Pool

```bash
python -m src.cli backtest \
    --factor-pool ./outputs/factor_pool.json \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --output ./outputs/backtest_report.html

# Expected: Sharpe > 1.0 on out-of-sample period
```

### 5.4 Monitor First Factors

```bash
python -m src.cli monitor \
    --watchlist ./outputs/factor_pool.json \
    --window 90 \
    --threshold 0.30

# Expected: initial ICIR values, no decay alerts
```

---

## 6. Cost Tracking (Standing)

Update `.env` with monthly budget:

```bash
# .env
LLM_MONTHLY_BUDGET_USD=500
LLM_ALERT_THRESHOLD=0.80   # Alert at 80% of budget
```

```bash
# Check current spend
python -m src.cli cost --monthly

# Output:
# Current month spend: $12.34 / $500.00 (2.47%)
# Projected monthly: $49.36 (9.87%)
# ✅ Within budget
```

---

## 7. Success Criteria Checklist

After completing this blueprint, the following should be true:

- [ ] `git status` clean, all changes committed
- [ ] `git tag` shows `phase7-data-foundation` and `phase7-real-data`
- [ ] `CONTEXT.md` updated with ADRs and glossary
- [ ] `verify --mode backfill` passes **all** B1-B5 checks
- [ ] `data.real_data: true` in `master_config.yaml`
- [ ] Full A-share universe ingested (≥ 5,000 symbols, ≥ 2010–2025)
- [ ] At least 1 factor candidate generated by `mine --iterations 1`
- [ ] Cost tracking reports < $500/month projected

---

## 8. Troubleshooting Guide

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| `ingest` fails mid-run | Network timeout | Use `--resume` — ingestor groups by symbol, resumes where left off |
| B4 survivor check fails | No universe snapshots | Run `universe_snapshot.py` for 2010–2025 (weekly intervals) |
| AlphaFeed rate limit (despite "unlimited") | Burst protection | Adjust `RateLimiter` sleep intervals |
| Postgres write slow | Work_mem too low | `SET work_mem = '256MB';` before ingestion |
| LLM returns malformed factor | Model drift | Increase temperature? Check structured output parser |
| DeepSeek cost exceeding budget | Too many iterations | Reduce `--iterations`, increase `--early_stop` sensitivity |

---

## 9. Next Milestone (Phase 8 — Factor Pool Maturity)

**When**: After 2-3 weeks of mining runs (when factor pool has 20+ validated factors)

**Goal**: Build multi-factor combination with market-neutral constraints

**Commands** (not yet):

```bash
# 1. Combine factors with PCA neutralization
python -m src.cli combine \
    --factor-pool ./outputs/factor_pool.json \
    --method icir_weighted \
    --neutralize industry,size

# 2. Generate daily signals for online deployment
python -m src.cli export \
    --factor-pool ./outputs/final_factors.json \
    --output ./online/compiled_factors.py

# 3. Simulate online trading (paper)
python -m src.cli paper \
    --config configs/paper_trading.yaml \
    --start 2025-01-01
```

---

## 10. Final Prompt to Claude Code

> **Claude Code**, execute this blueprint in order. Each section depends on the previous.
>
> 1. **First**, commit all changes in logical groups (Section 1). Do not skip this — we need a clean baseline.
> 2. **Second**, implement universe snapshot (Section 2.1) and freshnes mode flag (Section 2.2). These unblock B4/B5.
> 3. **Third**, run full A-share ingestion (Section 3). Use `--limit 100` first to validate, then full.
> 4. **Fourth**, flip `data.real_data` (Section 4) and confirm `verify --mode backfill` passes all checks.
> 5. **Fifth**, run the first mining iteration (Section 5.1) as a smoke test.
> 6. **Finally**, after all sections pass, tag the milestone and update `PHASE8_REPORT.md`.
>
> **Critical**: If any step fails, stop, diagnose, and report before proceeding. Do not skip verification checks.

---

**Blueprint version**: 1.0  
**Created**: 2026-08-10  
**Based on**: PHASE7_REPORT.md, Grilling Round-1 decisions (Q1-Q6)  
**Next review**: After Section 5 completes