# FQA — LLM-driven quantitative trading system

Offline R&D pipeline (LLM-heavy factor mining) strictly separated from online execution (deterministic). Data foundation ingests real A-share market data, point-in-time (PIT) safe, into a Postgres backend.

## 已定架构决策 (ADR 索引)

| ADR | 决策 |
|-----|------|
| ADR-0001 | 价格 bar 闭区间：`valid_to = valid_from + 1D` |
| ADR-0002 | 双列复权：`close`（后复权，锚定最新）+ `raw_close`（未复权）+ `adjust_factor` |
| ADR-0003 | Postgres JSONB PIT 后端：`pit_records(symbol, valid_from, valid_to, payload, updated_at)`，主键 `(symbol, valid_from)` |
| ADR-0004 | Universe 来自 Baostock `query_all_stock(day)`，过滤指数/北交所/B 股（exchange-aware 前缀） |
| ADR-0005 | 研究 universe 有界（`hs300_500` / `all` / 字面列表） |

> **网络状态（2026-08-10 实测）**：Baostock 端口 10030 **已恢复可连通**（TCP + `login` + `query_all_stock` 均正常）——此前「端口被阻断」的约束不再成立，universe 快照可直接用 Baostock 历史数据，无需 AlphaFeed 累积快照替代方案。

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

## 测试基线

- **159 passed**（`python -m pytest tests/`）
- 关键回归测试：
  - `test_to_price_records_multisymbol_factors_not_swapped` — 多符号因子不互换（索引对齐）
  - `test_to_price_records_applies_factor_backward` — 逐事件 ex_factor → 后复权（反向累计逆积）
  - `test_baostock_lazy_login_resolves_fn_after_login` — baostock 惰性登录后按名解析函数
  - `test_baostock_fetch_universe_prefix_fallback` — exchange-aware 指数过滤（sh.000001 不泄漏）
