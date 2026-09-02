# FQA — LLM-driven quantitative trading system

Offline R&D pipeline (LLM-heavy factor mining) strictly separated from online execution (deterministic). Data foundation ingests real A-share market data, point-in-time (PIT) safe, into a Postgres backend.

## 已定架构决策 (ADR 索引)

| ADR | 决策 |
|-----|------|
| ADR-0001 | 价格 bar 闭区间：`valid_to = valid_from + 1D` |
| ADR-0002 | 双列复权：`close`（后复权，锚定最新）+ `raw_close`（未复权）+ `adjust_factor` |
| ADR-0003 | Postgres JSONB PIT 后端：`pit_records(symbol, valid_from, valid_to, record_type, payload, updated_at)`，主键 `(symbol, valid_from, record_type)` — record_type 入键，价格与 universe 同日不互斥 |
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

- **165 passed**（`python -m pytest tests/`）
- 关键回归测试：
  - `test_to_price_records_multisymbol_factors_not_swapped` — 多符号因子不互换（索引对齐）
  - `test_to_price_records_applies_factor_backward` — 逐事件 ex_factor → 后复权（反向累计逆积）
  - `test_baostock_lazy_login_resolves_fn_after_login` — baostock 惰性登录后按名解析函数
  - `test_baostock_fetch_universe_prefix_fallback` — exchange-aware 指数过滤（sh.000001 不泄漏）
  - `test_price_and_universe_records_coexist_same_date` — 复合主键下价格与 universe 同日共存
  - `test_ingest_universe_backs_off_when_today_empty` — 当天空快照回退
  - `test_run_all_research_loop_skips_real_data_audit` — 研究循环只跑 4 项常开校验

## 数据地基状态（2026-08-10，Phase 7.1–7.5 完成）

- **全 A 全量入库**：12,480,696 价格 bar / 5,162 只 / 0 失败，覆盖 2010-01-04 ~ 2025-12-31（3,886 天），universe 快照 7,799 条；落库 Postgres 17（`postgresql://pit:pit@localhost:5432/pit_data`）。
- **`data.real_data: true`**（已翻转）；`verify --mode backfill` B1–B5 全绿（2,594@2015 → 5,205@2026-08-07，230 只已退市保留）。
- **研究循环校验门**：mine/backtest/evolve/monitor 跑窗口切片 store，只跑 4 项常开校验；`verify` 才追加 B1–B5（`real_data_audit` 参数）。
- 里程碑标签：`phase7-data-foundation`、`phase7-real-data`。

## 研究状态（2026-08-12，Phase 9 关闭 → Phase 10 启动）

- **Phase 9 全部关闭**：PEAD（9.2）、研报情绪（9.1a）、文本分歧度/新颖性（9.1b）三次单因子门禁均 FAIL（rank_ic < 0.015）。统一根因：HS300 上"已披露信息"在发布时已被充分定价。详见 `PHASE9_CLOSURE.md`。
- **资产重定位**：TriAgent 情绪缓存 → 风控熔断层；PEAD SUE 基建 → 战术倾斜层；文本因子 → 不再参与 Alpha 打分；BERT 向量缓存保留（`data/text/`）。
- **Phase 8 因子池**（Alpha 核心，`outputs/factors.json` 5 个低波+低换手公式）：组合回测 Sharpe 1.58–1.90 / maxDD ~10.5–11%（训练窗）。
- **Phase 10 进行中**（`blueprint/PHASE10_BLUEPRINT.md`）：三层融合（Alpha → 战术倾斜 → 风控熔断），2010–2025 全样本回测，门禁 Sharpe > 1.6 且 maxDD < 10%。
- 里程碑标签：`phase7-data-foundation`、`phase7-real-data`、`phase9-closed`、`phase9.1-sentiment-rejected`、`phase9.1-text-backtest-complete`、`phase9.1-text-rejected`。

## 提高收益三轨（2026-08-31 起，进行中）

> 影子盘毛 alpha≈0（成本主导亏损）→ 优先找收益来源。详见 `docs/ML_DATA_TRACKS.md`。

- **轨道 A — GitHub 公式动物园**：`paper/repos/aurumq-rl` + qlib 存档；237 条公式（alpha101×55 / gtja191×24 / alpha158×158）翻译进闭式算子库（`src/exploration/translate.py`）并全部可求值；walk-forward 扫描结论：HS300 上唯一稳定存活家族=短周期价格均值回归（与生产 min_lookback 60 纪律冲突，需组合级可交易性门）。
- **轨道 B — 新数据域**：融资融券（`src/data/margin.py`，192k 条 PIT 记录）与龙虎榜（`src/data/lhb.py`，73k 条，前视列按构造剥离）；门禁分段验证（发现/验证窗）——sl_mix 2026 OOS 符号翻转（负结果），sl_growth 边缘。
- **轨道 C — ML 双轨道**：LightGBM + GPU MLP（`src/ml/`，RTX 4060），Purged-KFold/embargo + walk-forward，冻结确定性工件 + `src/ml/promote.py` 生产硬门；最佳候选 = GPU MLP（test rank_ic 0.0558 / ICIR 8.28）；三方对决（vs 在任池）见 `outputs/ml_showdown.log`。
