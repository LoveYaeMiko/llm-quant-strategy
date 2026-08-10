# Phase 7 报告 — LLM 量化交易系统数据地基

> 项目：FQA（LLM 量化交易系统） · 阶段：Phase 1（数据）+ Phase 2（研究）地基
> 日期：2026-08-09 · 分支：main · 测试：**158 passed**

---

## 1. 目标与范围

按 `blueprint/IMPLEMENTATION_BLUEPRINT.md` 实现**生产级数据地基**，数据主源为 **AlphaFeed（已付费、理论无限额度）**，免费源 Baostock / AKShare 作为补充。交付内容：

- 四个数据源适配器（AlphaFeed / Baostock / AKShare + 统一符号规范化）
- PIT 时序存储（PointInTimeStore，Postgres 生产后端 + SQLite 离线后端）
- 入库编排器（universe → 价格 → 基本面 → 新闻）与 CLI
- B1–B5 真实数据校验 + walk-forward 配置 + 策略衰减追踪器（DecayTracker）
- 真实验证：6 只全 A 样本 × 2010–2025 全历史，落库 Postgres 并通过 B 类校验

**范围决策（Grilling Round-1）**：全 A 数据入库（研究层面用 `research.universe: "hs300_500"`，`--symbols` 做全量最终验证）；价格双列复权；新闻入库（watchlist ≈ HS300 前 30）；基本面默认关闭（`--fundamentals` 试点 ~50 只）；衰减阈值对齐 `icir.keep_threshold=0.30`；`data.real_data` 由人工手动翻转（入库先打印质量摘要）。

---

## 2. 架构决策（ADR）

| ADR | 决策 |
|-----|------|
| [ADR-0001](docs/adr/0001-closed-interval-price-bars.md) | 价格 bar 用**闭区间**：`valid_to = valid_from + 1D`，查询某日 T 只返回 T 当日产生的 bar |
| [ADR-0002](docs/adr/0002-dual-column-adjustment.md) | **双列复权**：`close`=后复权（前复权锚定最新），`raw_close`=未复权，`adjust_factor`=复权因子 |
| [ADR-0003](docs/adr/0003-postgres-jsonb-pit-backend.md) | Postgres **JSONB** 后端：`pit_records(symbol, valid_from, valid_to, payload JSONB, updated_at)`，主键 `(symbol, valid_from)`，`ON CONFLICT DO UPDATE` |
| [ADR-0004](docs/adr/0004-baostock-universe-source.md) | universe 来自 Baostock 年度 `query_all_stock(day)`，过滤指数/北交所 |
| [ADR-0005](docs/adr/0005-bounded-research-universe.md) | 研究 universe 有界（`hs300_500` / `all` / 字面列表），防止全 A 上跑研究 |

**存储契约**：`upsert / query / universe / universe_as_of / latest / has_future_leak / min_date / max_date / max_valid_from / delisted_symbols / distinct_dates / symbols / snapshot / history / from_url / build_price_bars`。`valid_to=NaT` 表示开区间。`record_type` 特征列区分 `price` / `universe` / `fundamental` / `text`。

---

## 3. 实现内容

### 3.1 数据源适配器

| 模块 | 职责 |
|------|------|
| `src/data/ingestion/alphafeed_adapter.py` | 主源。`fetch_klines`（显式 `count`，规避 API 静默 100-bar 上限）、`fetch_ex_factors`（长表归一化为 `{symbol: frame}`）、`fetch_quotes` |
| `src/data/ingestion/baostock_adapter.py` | universe 快照 + 指数成分（HS300/ZZ500）+ 基本基本面。pandas `append` monkeypatch 回退 |
| `src/data/ingestion/akshare_adapter.py` | 新闻（`stock_news_em`）+ 基本面快照（`stock_zh_a_spot_em`） |
| `src/data/schema/symbols.py` | 符号规范化：`600519.SH` 为规范形；`to_baostock` / `from_bare_code`（6/9→SH，0/2/3→SZ，4/8/92→BJ） |
| `src/data/schema/rate_limiter.py` | 限速器（对 AlphaFeed 保持礼貌，尽管额度无限） |
| `src/data/schema/retry.py` | 网络重试 |

### 3.2 存储

- `src/data/point_in_time_loader.py` — 内存/SQLite 实现
- `src/data/postgres_loader.py` — Postgres JSONB 实现（`execute_values` 批量 upsert）
- `src/data/ingestion/convert.py` — 记录类型与转换（`price_records` / `universe_records` / `fundamental_records` / `text_records`）

### 3.3 编排与 CLI

- `src/data/ingestion/ingestor.py` — 四阶段流水线 + `IngestStats` 质量摘要 + 断点续传（resume 按符号分组）
- `src/cli.py` — `ingest / verify / mine / backtest / evolve / monitor / export`，`--symbols / --limit / --resume / --fundamentals / --news`
- `src/checklist.py` — B1–B5 真实数据校验 + 4 项流水线常开校验（pit/fincad/diversity/cost）

### 3.4 B1–B5 校验门

| 检查 | 内容 | 触发条件 |
|------|------|---------|
| B1 `no_future_leak` | 训练/验证边界处查询不暴露此后产生的事实 | `real_data: true` |
| B3 `adjustment_consistency` | 后复权日收益不超出涨跌停带；因子变更日伴随原始价跳变 | `real_data: true` |
| B4 `survivorship` | 在 `survivorship_date` 存活、现已退市的名称被保留 | `real_data: true` |
| B5 `data_freshness` | 最新 bar 新鲜、历史覆盖度达标 | `real_data: true` |
| 常开 4 项 | pit 时序 / fincad 财务规范 / diversity 公式多样性 / cost 预算 | 恒 |

### 3.5 研究支撑

- `src/monitoring/decay_tracker.py` — 策略衰减追踪（整数交易日窗口，对齐 `icir.keep_threshold=0.30`）
- `configs/master_config.yaml` — walk-forward 窗口（train 2010–2019 / val 2020–2021 / test 2022–2025）、`survivorship_date: "2015-01-05"`（工作日锚点）

---

## 4. 关键发现与修复（真实数据驱动的 Bug）

在真实数据试点中暴露并修复了两个**确定性 Bug**，均发生在 `to_price_records` 双列复权路径：

### 4.1 多符号因子错位（索引对齐 Bug）

- **症状**：`600519.SH` 与 `000858.SZ` 一起抓取时，复权因子在两只股票间**互换**。
- **根因**：`out = concat(...).sort_values(["symbol","date"])` 留下**被打乱的 RangeIndex**；`out["adjust_factor"] = merged["ex_factor"].fillna(1.0)` 按**索引标签**对齐，而 `merged` 是全新 RangeIndex —— 两者位置错位，600519 的行拿到 000858 的因子。
- **修复**：`sort_values` 后 `reset_index(drop=True)`，使按标签赋值与 `merged` 位置对齐。
- **回归测试**：`test_to_price_records_multisymbol_factors_not_swapped`。

### 4.2 复权因子语义错误（per-event vs 累计）

- **症状**（B3 式扫描发现）：`000001.SZ` 2014-06-12 后复权日收益 **-37.9%**、`300750.SZ` 2024-04-30 **-45.3%**、`601318.SH` 2015-09-09 **-49.2%** —— 远超涨跌停带，本应是除权日却出现复权断层。
- **根因**：AlphaFeed 的 `ex_factor` 是**逐事件因子**（每个除权/拆股一行），**不是累计因子**——601318 序列 2.0125（2015 年 10 转 10）之后接 1.0061，累计因子不可能下降。旧实现把逐事件因子当累计因子 forward-fill，当下一笔小分红到来时因子「重置」1.81→1.02，制造出虚假的 -45% 断层；而原始价格在该日实际仅波动 -3.4%（真实分红）。
- **修复**：后复权因子 = 该 bar 日期**之后所有事件**的 `1/ex_factor` 的**反向累计积**（`merge_asof direction="forward", allow_exact_matches=False` 查最近的下一个事件）。除权日 bar 本身已是除权后价格，故排除自身事件。
- **回归测试**：`test_to_price_records_applies_factor_backward`（1:1 拆股：除权前 bar 因子 0.5、除权日及之后 1.0，复权序列连续）。

---

## 5. 真实验证结果

### 5.1 试点入库（Postgres 17，Docker）

```
价格K线记录    : 18,508
成功 / 失败个股 : 6 / 0
耗时           : 15.7s
价格覆盖天数    : 3,886
价格区间        : 2010-01-04 ~ 2025-12-31
最大缺口(工作日): 6 天        ← 春节休市，符合预期
```

样本覆盖：`600519.SH`(沪主板) / `000858.SZ`(深主板) / `300750.SZ`(创业板) / `601318.SH`(沪主板) / `000001.SZ`(深主板) / `688981.SH`(科创板)。

### 5.2 双列复权一致性（全量 18,508 bar）

- `close == raw_close × adjust_factor` 违反数：**0**
- 修复前最差复权日收益：**-49.2% / -45.3% / -37.9%**（因子伪断层）→ 修复后最差均为**合法市场波动**，全部落在涨跌停带内（2015 股灾跌停、2018-10-29 茅台跌停、2024-10 创业板/科创板涨停）
- 大额除权日（原始价跳 >8%）中，复权收益 <3% 的「干净除权」共 **6 例**（000001 2014/2015/2016 送转、600519 2011/2014/2015 分红）

### 5.3 B1–B5 实测（真实 Postgres 数据）

| 检查 | 结果 | 明细 |
|------|------|------|
| B1 无未来泄漏 | ✅ PASS | 边界 2019-12-31 查询返回 5 条事实，0 条出生于边界之后 |
| B3 复权一致性 | ✅ PASS | 6 符号 18,508 天，0 次复权收益超 ±30%（最差 20.0%）；95 个因子变更日，82 个伴随原始价跳变 |
| B4 幸存者 | ⚠️ 预期失败 | 0 条 universe 记录 —— 试点用 `--symbols` 跳过 universe 阶段；且 Baostock 端口 10030 被阻断，历史成分快照无法拉取（详见 §6） |
| B5 数据新鲜度 | ⚠️ 预期失败 | 最新 bar 2025-12-31（回填窗口按 `project.end_date` 封顶），新鲜度 221 天 > 阈值 7 天；覆盖度 93% > 70% 达标 |

> 注：B4/B5 的「失败」是试点/设计使然，非数据损坏。B4 逻辑有独立单元测试覆盖（audit_store 携带 universe 记录 → survivorship 通过）。

---

## 6. 已知问题与限制

1. **Baostock 数据端口 10030 被阻断** → 实时 universe 快照与历史成分股无法拉取。入库器已有回退路径（AlphaFeed `quotes` 提供**当前**成分），但无历史快照 → **B4 幸存者校验依赖 universe 数据，在当前网络下无法全量做实盘验证**。
2. **AlphaFeed `quotes.get` 仅返回当前成分**，无 as-of 历史 —— 需另选方案做历史 universe（见 §7）。
3. **B5 新鲜度与历史回填矛盾**：`ingest --end 2025-12-31` 的回填天然「陈旧」。新鲜度检查面向持续运营（每日增量），对一次性回填不适用。若需通过，日常增量运行不加 `--end` 即可。
4. **Docker 镜像拉取**：`postgres:16-alpine` 的二进制层在所有可用 CN 镜像被阻断（blob `2f537278bc74` 停滞），已改用 **`postgres:17-alpine`**（层集不同、可拉取），功能等价（JSONB + ON CONFLICT + execute_values）。
5. **北交所 920 重编号**已纳入 `from_bare_code`（920→BJ），但全 A 回填前需确认 AlphaFeed 对该板块的覆盖率。
6. **工作区尚未提交**：存在大量未提交改动（含 `PROJECT_BLUEPRINT.md`/`resource/*` 移至 `blueprint/` 的删除+新增），建议按逻辑点提交。

---

## 7. 建议的下一步

1. **全 A 全量回填**：`python -m src.cli ingest --start 2010-01-01 --end 2025-12-31`（约 5,600 只 × ~3,900 bar ≈ 2,200 万行），之后 `--resume` 做日常增量。AlphaFeed 额度无限，放心大胆跑。
2. **universe 历史快照**：绕开 Baostock —— 用 AlphaFeed `quotes` 每日累积快照（前瞻式），或引入替代历史 universe 源，使 B4 可做实盘验证。
3. **提交工作区**：将 Phase 1+2 地基作为逻辑里程碑提交（含 ADR 文档、docker-compose、蓝图迁移）。
4. **`data.real_data` 翻转**：待全 A 入库完成并人工审阅质量摘要后，按 Q6-A 手动置 `true`，随后 `verify` 触发 B1/B3/B4/B5 全门。
5. **走查 `verify` 门**：确认 B1–B5 全绿后，进入因子研究 / 回测流程。

---

*报告完。测试基线：`python -m pytest tests/` → 158 passed。真实数据试点落库 Postgres 17（`postgresql://pit:pit@localhost:5432/pit_data`）。*

---

# 附录 A：Phase 7.1 → 7.5 蓝图执行报告（2026-08-10）

> 按 `blueprint/PHASE7_BLUEPRINT.md` 执行（提交基线 → 解除 B4/B5 阻塞 → 全 A 全量入库 → 翻转 `data.real_data` → 因子挖掘冒烟）。
> **网络状态更新**：Baostock 数据端口 **10030 已恢复连通**，故全程直接使用 Baostock 拉取 universe，未采用蓝图针对端口阻断的 AlphaFeed 回退方案。
> 测试基线：**165 passed**。

---

## 7.1 逻辑提交与里程碑（phase7-data-foundation）

工作区大量未提交改动按逻辑分组落库：

| 提交 | 内容 |
|------|------|
| `446eaf1` | 蓝图迁移到 `blueprint/` 目录 + 新增 `CONTEXT.md` |
| `9c0d5dd` | docker-compose（Postgres 17）+ Phase 7 报告 + env 占位符 |
| `97ff496` | ingestion 包初始化 |
| `cdb7781` | CONTEXT.md：ADR 索引 + 网络状态 + 测试基线 |
| `1d06a44` | LLM 量化交易系统核心（前一里程碑） |

`CONTEXT.md` 更新：ADR 汇总表、测试基线、网络状态。

## 7.2 解除 B4/B5 阻塞（commit `fd956be`）

### B5 数据新鲜度 — `verify --mode backfill|live`
历史回填的存储天然「陈旧」，新鲜度必须按回填窗口而非"今天"度量：

- `verify --mode backfill` → `freshness_as_of = project.end_date`（回填封顶日）
- `verify --mode live` → 以 `now()` 度量（面向日常增量运营）
- 回归测试：`test_verify_backfill_mode_offline` / `test_verify_offline`

### B4 幸存者校验 — Baostock universe 快照
- 直接用 `query_all_stock(day)` 拉取全市场成分快照（过滤指数 / 北交所），写 `universe_{day}.csv` + `hs300.json` / `zz500.json` 研究 universe 缓存。
- **关键修复 — 「当天空快照」回退**：`query_all_stock("当天")` 在当日数据未定稿（且周末/节假日永远为空）时返回 0 行。`_latest_universe_snapshot` 最多回退 10 个自然日取最近的非空快照（`2026-08-10` → 回退到 `2026-08-07`，5,205 只）。
- 新增 `ingest --universe-only`：只跑 universe 阶段，B4 前置准备，不触碰价格。
- 回归测试：`test_ingest_universe_backs_off_when_today_empty` / `test_ingest_universe_only_skips_price_pass`。

## 7.3 全 A 全量入库（commit `4b322bb`）

流程：`ingest --limit 100` 验证 → 全量回填 2010-01-01 ~ 2025-12-31。

**全量入库摘要（`logs/ingest_full_20260810.log`）**：

```
价格K线记录    : 12,480,696
成分股快照记录  : 7,799
基本面/文本记录 : 0 / 0
成功 / 失败个股 : 5,162 / 0
耗时           : 1516.3s（约 25 分钟）
价格覆盖天数    : 3,886
价格区间        : 2010-01-04 ~ 2025-12-31
最大缺口(工作日): 6 天   ← 春节休市，符合预期
幸存者自查      : 230 只 2015-01-05 存在、现已退市/缺席
```

- **瞬时网络中断自愈**：全量期间出现一次 `WinError 10053`（连接被重置），重试逻辑接管，**0 失败**。
- **数据完整性 Bug 修复（ADR-0003 更新）**：全量价格阶段发现价格 bar 与 universe 快照在**同一 `(symbol, valid_from)`** 上互相覆盖（`record_type` 无差别），2015 cohort 一度从 2,594 缩水到 444。修复：
  - PIT 主键升级为复合主键 `(symbol, valid_from, record_type)`（Postgres + SQLite + 内存三端一致）；
  - Postgres 就地迁移（清库重建 + universe 阶段重跑）；
  - 回归测试：`test_price_and_universe_records_coexist_same_date` / `test_sqlite_price_and_universe_coexist_same_date`。

## 7.4 翻转 `data.real_data` + 全门验证（commit `827bc20`，tag `phase7-real-data`）

`configs/master_config.yaml` → `data.real_data: true`。`verify --mode backfill` **8/8 全 PASS**：

| 检查 | 结果 | 明细 |
|------|------|------|
| pit | ✅ | query(2023-12-31) 0 条未来事实 |
| fincad | ✅ | 输出净化 100% IC 归零，抑制率 100% > 50% |
| diversity | ✅ | 最小 AST 距离 1.00 ≥ 0.40 |
| cost | ✅ | 模拟月度 $0.53 ≤ $500 |
| B1 no_future_leak | ✅ | 边界 2019-12-31 返回 **3,530** 条事实，0 条未来泄漏 |
| B3 adjustment_consistency | ✅ | 20 符号 73,200 天；0 次超 ±30% 复权收益（最差 17.25%）；209 个因子变更日 187 个伴随原始价跳变 |
| B4 survivorship | ✅ | **2,594** 只 @2015-01-05 → **5,205** 只 @2026-08-07；其中 **230** 只已退市仍被保留 |
| B5 data_freshness | ✅ | 最新 bar 2025-12-31（0 天陈旧）；3,886/4,173 工作日覆盖 **93%** ≥ 70% |

> 与 8-09 试点报告对比：B4/B5 由「预期失败」翻转为**实盘全绿** —— 这正是「universe 历史快照」+「backfill 模式」两项解除工作达成的目标。

## 7.5 首次因子挖掘冒烟（commit `b667e3e`）

`python -m src.cli mine --iterations 1`：全链路端到端跑通，**exit 0**，4 项常开校验 PASS。

- 生成 3 个 LLM 假设（deepseek-v4-flash），公式翻译、因子计算、回测评估、风险门控、审计落盘全流程无异常。
- **accepted 0/3**：三个因子（`Inv(TS_Std(Close,10))` / `Neg(TS_ZScore(Close,5))` / `Neg(TS_ZScore(Close,30))`）方向均为 short / long_short，Sharpe 为负、回撤 > 15%，被 `reject_high_risk` 门正确拒绝。
- **诊断结论**：冒烟测试目的是验证流水线可运行，此结果证明「LLM 假设 → 因子生成 → 评估 → 风险门控」闭环完整，且**风险门在真实工作**（拒绝高回撤空头策略）。非缺陷，不改阈值强凑通过。

### 研究循环校验门修复（同一提交）
冒烟测试暴露一个设计缺口：研究循环（mine/backtest/evolve/monitor）跑在**窗口切片、universe 有界**的 store 上，B5 在训练窗口上必然「陈旧」、B4 的 universe 快照也被切出范围。修复：

- `run_all(..., real_data_audit: bool = False)` —— 研究循环只跑 4 项常开校验；`verify` 显式传 `real_data_audit=True` 才追加 B1–B5；
- 回归测试：`test_run_all_research_loop_skips_real_data_audit`；
- 同时提交 `data/universe/hs300.json` / `zz500.json`（研究 universe 缓存），`.gitignore` 排除 `universe_*.csv` 快照与 `logs/`。

## 测试与提交

- **165 passed**（`python -m pytest tests/`，45s）
- 提交链：`fd956be`(7.2) → `4b322bb`(7.3) → `827bc20`(7.4) → `b667e3e`(7.5 + 门修复)
- 标签：`phase7-data-foundation`、`phase7-real-data`，工作区干净

## 结论与下一步

Phase 7.1→7.5 蓝图全部完成：生产级数据地基就绪，12.48M 条真实 A 股价格 + 全市场 universe 快照落库 Postgres，B1–B5 实盘全绿，挖掘流水线可运行。**下一步（Phase 8 研究）**：正式因子挖掘 `mine --iterations 50 --trials 20`，以 walk-forward 产出首批候选因子并评估衰减。

*附录完。测试基线：165 passed。真实数据全量落库 Postgres 17（`postgresql://pit:pit@localhost:5432/pit_data`），12,480,696 价格 bar / 5,162 只。*
