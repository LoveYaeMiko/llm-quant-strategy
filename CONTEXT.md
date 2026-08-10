# FQA — LLM-driven quantitative trading system

Offline R&D pipeline (LLM-heavy factor mining) strictly separated from online execution (deterministic). Data foundation ingests real A-share market data, point-in-time (PIT) safe, into a Postgres backend.

## Language

### Data foundation

**Point-in-time record**:
A fact visible at time T iff `valid_from <= T < valid_to`; `valid_to` = NaT means the record is still valid. The single anti-look-ahead primitive every query builds on.
_Avoid_: snapshot, row, fact table row

**Closed-interval price bar**:
A price bar dated d is valid over `[d, d+1d)`. Querying date T returns only bars born on T — a query never sees a bar from the future.
_Avoid_: open-ended bar, "2099-12-31" expiry

**复权 (adjustment)**:
Dual-column price storage: `close` = backward-adjusted (PIT-stable, no ex-dividend gaps), `raw_close` = unadjusted, `adjust_factor` = the ex-factor in force at that date.
_Avoid_: single adjusted price, forward-adjustment

**record_type**:
A PIT payload discriminator separating `price`, `universe`, `fundamental`, and `text` records that share one `pit_records` table.
_Avoid_: separate tables per data kind

**Universe**:
The set of tradeable names (survivorship-safe: includes names that later delisted). Built from Baostock yearly snapshots, not from AlphaFeed (which has no universe endpoint).
_Avoid_: symbol pool, stock list

**Survivorship bias**:
Structurally avoided because a query at T returns only names alive at T; delisted names remain present for earlier dates.

### Research

**Walk-forward window**:
The train / validation / test time splits (`research.train_start … test_end`) a factor must not leak across.
_Avoid_: single train/test cut

**Research universe**:
The bounded subset (`research.universe: "hs300_500"`) the mine/backtest/monitor loops run on, while the underlying data stays full A-share. Full-A final validation is run on demand via `--symbols`.
_Avoid_: researching on the full universe by default

**Factor decay**:
A monitored factor whose rolling rank-IC / ICIR falls below `icir.keep_threshold` (0.30) is flagged and retired.
_Avoid_: "factor died", "strategy stopped working"
