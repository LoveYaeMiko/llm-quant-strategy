# FQA — A 股 LLM 量化研究 / 交易系统

> **当前通道：模拟盘（`deployment.mode: observe`、`real_money_enabled: false`）。**
> 全部成交都是系统内纸面撮合，**没有向任何券商发单、没有一分钱真实资金**。
> 接入实盘的放行清单见 [docs/LIVE_READINESS.md](docs/LIVE_READINESS.md)。

一套面向 A 股的量化系统：**离线研发（LLM 重）与在线执行（纯确定性）严格分离**，
全链路建立在 **point-in-time（PIT）数据地基** 之上；研究结论、部署配置与对外数字
都被「预注册 + provenance + 证据分级」的纪律约束，避免把调参样本当外样本、把回放
当实时、把模拟当实盘。

---

## 0. 状态速览（2026-09-10）

| 维度 | 状态 |
|---|---|
| 部署通道 | `observe`（模拟撮合）；`real_money_enabled: false` |
| 生产轨 | **单轨 `D_5W`**（5 万虚拟资金，回撤买入 + 实时日内止损）。A/B/C 三轨已于 2026-09-08 退役归档 |
| 数据地基 | 全 A 历史入库 12,480,696 bar / 5,162 只（2010-01-04 → 2025-12-31）；PostgreSQL 17 |
| 数据缺口 | 全 A 增量**实质停更**：2026-08-07 快照的 5,205 只中 **4,364 只** 在 2025-12-31 后无 bar，仅 D 轨的 800 只保持新鲜 |
| 可引用业绩 | IS 2026：+5.72% / Sharpe 0.67（t=0.53）；OOS 2025H2：+1.64% / Sharpe 0.40（t=0.23）——**均无统计显著性** |
| 前向期 | 窗口 **2026-09-11 → 2027-03-11**；预注册 v4 已冻结，闸门当前 verdict = `fail`（5 项因窗口未开始而「未测量」） |
| 实盘就绪 | A1–A11、B1–B6 全绿；C1/C2/C5/C6/C7、D2/D3/D9/D10/D12 未完成，其中 **D9 为阻断项** |
| 测试 | `python -m pytest tests/` → **752 tests**（全离线、全确定性） |

---

## 1. 快速开始

```bash
# 1. 依赖（numpy / pandas / scipy / scikit-learn / PyYAML 为最小集）
python -m pip install -e ".[dev]"          # 离线可跑
# 可选扩展：.[llm] openai  · .[dl] torch+transformers  · .[data] baostock/akshare/psycopg2

# 2. 环境变量（.env 已被 .gitignore 忽略，密钥永不入库）
cp .env.example .env                        # 填 DEEPSEEK_API_KEY / PIT_DATABASE_URL ...

# 3. PIT 数据库（Postgres 17；也可用 sqlite:///data/pit_data.db 纯本地跑）
docker compose up -d

# 4. 跑测试（752 用例，不需要数据库、不需要 LLM）
python -m pytest tests/ -q

# 5. 蓝图自检清单
python cli.py verify --mode live            # 历史回填后核验用 --mode backfill
```

**没有 `DEEPSEEK_API_KEY` 时系统完全离线确定性运行**（智能体走内置规则）；
没有 `PIT_DATABASE_URL` 时自动退化为合成市场。测试与 `verify` 都不依赖 LLM。

常用入口（`python cli.py <cmd>` 与 `python -m src.cli <cmd>` 等价）：

```bash
# --- 研究（离线层） ---
python cli.py ingest   --resume                         # 增量入库行情
python cli.py mine     --window train --iterations 50   # 多智能体因子挖掘
python cli.py backtest --factor-pool outputs/factors.json --weights icir --window test
python cli.py pool     filter --min-ic 0.02 --min-icir 0.30
python cli.py monitor  --watchlist outputs/factors.json --window-days 90
python cli.py evolve   --formula "Neg(TS_ZScore(Close, 10))"
python cli.py explore                                   # 探索轨迹（不触碰生产池）

# --- 部署（在线层 / 影子盘） ---
python cli.py export   --formula "..." --name mom_10    # 编译为确定性 JSON 工件
python cli.py paper    --start 2026-01-01               # 模拟盘日度循环（可续跑）
python cli.py shadow   --accounts D_5W                  # 影子盘：逐日记录目标持仓与 PnL
python cli.py preclose                                  # 收盘竞价下单层（15:00 竞价成交）
python cli.py live                                      # 盘中实时止损（09:30–15:00 轮询）
python cli.py autopilot                                 # 自动闭环：shadow → 风险闸门 → 周期任务
python cli.py dcycle   refit|challenger|decide|audit-cost
python cli.py weekly                                    # 周度重训 + 尾部 Sharpe 晋升闸门
```

### 日度运行由谁驱动

本仓库只提供 CLI 入口；**真实的日度调度与面板在独立项目 PAICC 中**
（APScheduler + Electron 面板，见 [docs/PAICC_INTEGRATION.md](docs/PAICC_INTEGRATION.md)）。
工作日任务序列大致为：`09:25 live` → `14:40 盘口快照` → `14:50 preclose` →
`15:02 分钟特征刷新` → `15:10 autopilot` → `15:20 前向候选` → `17:45 挑战者`，
另有每 5 分钟的 `live_watchdog` 存活巡检。**直接裸跑 `python cli.py shadow` 也能工作**
（会自愈数据、续跑账本），与调度器读写同一份账本。

---

## 2. 系统架构

```
┌─ OFFLINE 研发层（LLM 重） ────────────────────────────────────────────
│  SignalAgent → CodeAgent → EvalAgent → RiskAgent   多智能体流水线
│  语义空间五元组 · 83 个闭式算子 · IC/ICIR/Sharpe 评估 · 五段门控
│  (AlphaSchema)   (EvoQuant)        Bonferroni 校正  危机期/多假设
│
│  FinCADWrapper       解码时抑制未来日期 token + LookAheadAudit 审计
│  MemoryManager       结构化记忆（AlphaMemo 频繁子树回避）
│  EvoQuant            自我进化：诊断 → 候选 → 门控 → 蒸馏
│  CostTracker         月度 $500 预算闸门，逐 token 记账
│  ExperimentAuditor   全量留痕（config / 路由快照 / PIT 窗口）
└──────────────────────────────────────────────────────────────────────
         │ 通过全部门控的因子被 compile_factor() 编译为 JSON 工件（无 Python eval）
         ▼
┌─ ONLINE 执行层（纯确定性，不 import torch/openai） ────────────────────
│  signal_calculator   CompiledFactor 求值（白名单算子 + 字段访问）
│  portfolio_optimizer 逐日 PCA 中性化 → 排名 → 多空 → 相关性聚类限仓
│  order_executor      现价撮合、滑点/佣金/印花税、整手、涨跌停、T+1
└──────────────────────────────────────────────────────────────────────
         │
         ▼
┌─ 部署通道（当前 observe：只纸面撮合，不发单） ─────────────────────────
│  PaperRunner + SQLite 账本   逐日推进、可续跑、成本真实计费
│  autopilot 风险闸门          normal / de_risk(×0.5) / halt(×0)
│  preclose + live trader      收盘竞价下单层 + 分钟级实时止损
└──────────────────────────────────────────────────────────────────────
```

三条铁律：

1. **PIT**：一切历史查询走 `PointInTimeStore`（`valid_from / valid_to`）；`valid_to` = NaT
   表示仍然有效。价格 bar 是闭区间 `[d, d+1d)`（ADR-0001），查询永不看见未来 bar。
2. **离线/在线分离**：在线模块的 import 链不含 torch / transformers / openai，
   物理上不可能接触 LLM；在线求值只做白名单算子，无 `eval` 路径。
3. **确定性**：组合优化逐日独立（无随机数）；下单执行用固定种子，两次执行输出逐字节一致。

---

## 3. 目录结构

```
FQA/
├── cli.py                  # 薄启动器（无需 pip install -e）
├── pyproject.toml          # 打包 + 可选依赖分组（llm / dl / data / dev）
├── docker-compose.yml      # 本地 PIT Postgres（17-alpine，restart: unless-stopped）
├── configs/                # 单点配置源：改 YAML 即改全部闸门
│   ├── master_config.yaml      # 项目/数据/风控/研究窗/三层组合/影子账户/自动闭环（唯一权威）
│   ├── factor_thresholds.yaml  # IC/ICIR 阈值、多样性下限
│   ├── llm_routing.yaml        # 模型映射 + 成本 + AgenticAITA 触发器
│   ├── exploration.yaml        # 探索轨迹参数
│   └── forward_policy.yaml     # 前向期预注册/闸门/候选切换规则（预注册的唯一权威）
├── src/
│   ├── cli.py                  # argparse 入口（21 个子命令）
│   ├── config.py               # YAML 加载 + ${ENV} 插值 + stdlib .env
│   ├── agents/                 # signal / code / eval / risk / debate / 路由 / 推理闸门
│   ├── factors/                # 语义空间、算子库、记忆、PEAD、文本因子、Schema 探索
│   ├── bias_control/           # FinCAD 上下文解码 + LookAheadAudit
│   ├── backtest/               # PIT 感知回测 + 指标 + 涨跌停锁定掩码
│   ├── data/                   # PIT loader / Postgres / 摄取适配器 / 分钟 / 两融 / 龙虎榜
│   ├── ml/                     # LightGBM + torch MLP：标签 / Purged-KFold / 训练 / 晋升硬门
│   ├── portfolio/              # 三层组合：alpha_core / seasonal_tilt / risk_overlay / 集成
│   ├── paper/                  # 账本 + 日度运行器 + 影子状态 + ML 簿 + 回撤买入簿
│   ├── online/                 # 确定性信号计算 / 组合优化 / 下单执行
│   ├── forward/                # 前向期风险闸门 + 预注册（哈希冻结）
│   ├── live/                   # 实时盘中交易者 + 券商适配器（占位）
│   ├── autopilot/              # 控制状态 + kill-switch 风险闸门
│   ├── exploration/            # 公式动物园翻译 + 受控 LLM 假设 + 扫描
│   └── monitoring/ reporting/ sentiment/ calibration.py 等
├── tests/                  # 752 个用例（80 个文件），全离线
├── scripts/                # 一次性研究/审计/回补脚本（网格、OOS、前向、对账、看门狗）
├── docs/                   # 证据与协议文档 + evidence/（可引用工件快照）+ adr/
├── blueprint/              # 蓝图与阶段设计（PROJECT_BLUEPRINT / PHASE7-10 / review.md）
├── paper/                  # 论文索引、网页存档、factor_zoo 公式库存；repos/ 与 papers/ 不入库
├── data/                   # universe 缓存入库；行情/研报/模型/分钟等大缓存不入库
└── outputs/                # 运行期产物（账本、状态、日报、工件）；整体不入库
```

---

## 4. 数据地基（PIT）

**存储**：`pit_records(symbol, valid_from, valid_to, record_type, payload, updated_at)`，
主键 `(symbol, valid_from, record_type)`——`record_type` 入键使得价格与 universe
可以在同一天共存（ADR-0003）。JSONB payload 承载 `price` / `universe` / `fundamental` / `text`。

**双列复权**（ADR-0002）：`close` = 后复权（锚定最新，PIT 稳定无除权跳空）、
`raw_close` = 未复权、`adjust_factor` = 当日生效的除权因子。
分钟接口返回的本身就是前复权价，**不再二次乘因子**（这是一处曾导致止损提前触发的缺陷）。

**Universe**：来自 Baostock `query_all_stock(day)` 的年度快照，exchange-aware 过滤
指数 / 北交所 / B 股（ADR-0004）。幸存者偏差**结构性避免**：查询 T 日只返回 T 日
存续的标的，退市标的在更早的日期仍然存在。

**研究 universe 有界**（ADR-0005）：`research.universe: "hs300_500"`，
`--symbols` 是跑全 A 验证的逃生门。

### 当前数据缺口（必须知道）

| 缺口 | 现状 | 影响 |
|---|---|---|
| 2026 年价格面板只有 301 只（声明 800） | **已回补 500 只、0 报错**；硬闸门 `effective_universe` 由 0.376 → **1.00** | 此前公布的 IS 2026 业绩（+16.36% / Sharpe 1.66）**已作废**，见 §7 |
| 全 A 增量停更 | 2026-08-07 快照 5,205 只中 **4,364 只**在 2025-12-31 后无 bar | 全市场/成分研究已过期；D 轨的 800 只不受影响 |
| 2025-10-27 → 12-12 分钟数据洞 | 已回补（`scripts/backfill_minute_gap.py` + `fetch_intraday.py`） | OOS 窗口因此从 `citable=false` 变为 `true` |

---

## 5. CLI 命令总览（21 个）

| 分组 | 命令 | 作用 |
|---|---|---|
| 数据 | `ingest` | universe → 行情 →（可选）财务/新闻；`--resume` 增量、`--universe-only` 只刷快照 |
| 研究 | `mine` | 多智能体因子挖掘（signal→code→eval→risk，含拒绝反馈与组合模板槽位） |
| | `evolve` | EvoQuant 自进化一轮（诊断→候选→门控→蒸馏） |
| | `explore` | 探索轨迹：新算子 + 受控 LLM 假设的并行研究，**不触碰生产池** |
| | `backtest` | 单公式或因子池组合回测（`--weights equal\|icir\|dynamic`） |
| | `pool` | 因子池管理：`filter` / `diversify`（AST 距离）/ `report`（HTML）/ `promote` / `flag` |
| | `monitor` | 因子 IC/ICIR 滚动衰减监控（`--watchlist` 整池） |
| 单因子门禁 | `pead` | PEAD（季节 SUE）单因子验证，`--direction drift\|reversal` |
| | `sentiment-ingest` / `report-ingest` / `sentiment-factor` | 舆情三元组：实时新闻累积 / 研报历史回填 / 研报标题情绪 IC 门禁 |
| 在线 | `export` | 公式编译为 `CompiledFactor` JSON 工件（无动态代码） |
| | `verify` | 蓝图自检清单（PIT / FinCAD / 多样性 / 成本 / B1–B5） |
| 部署 | `paper` | 三层组合模拟盘，SQLite 账本可续跑 |
| | `shadow` | 影子盘：逐日记录目标持仓与 PnL，输出 `shadow_status*.json` + 日报 + CSV |
| | `preclose` | 收盘竞价下单层：用收盘前已知数据定委托清单，15:00 竞价价成交 |
| | `live` | 盘中实时交易：09:30–15:00 轮询最新分钟成交价，破位当场执行 |
| | `autopilot` | 自动闭环：shadow → 风险闸门（normal/de_risk/halt）→ 周期回校/衰减监控/重挖 |
| | `calibrate` | §7 三项回校：PEAD 倾斜幅度 / 舆情阈值 / 成本模型 |
| | `weekly` | 周度重训 + 尾部 Sharpe 改善才 promote |
| | `dcycle` | D 轨模型自优化：`refit` / `challenger` / `decide` / `audit-cost` |

---

## 6. 当前生产：D 轨（回撤买入 + 实时日内止损）

**账户** `D_5W`：5 万虚拟资金、`hs300_500` 截面、逐日评估、单槽上限 0.40、6 个等权槽位。
信号源是 **LightGBM 工件的截面排名**（强势股扫描器：2026 年 A 股原始动量回撤已失效，
ML 扫描为改良版；`pb_rank_source: "ml"` → `_resolve_artifact("lgbm", "")` 取最新工件，
当前为 `outputs/models/ml_20260901_004119.json`）。
入场需要「回调到 EMA21 ±2% 内 + 距 10 日高点回落 ≥3% + 缩量确认 +
尾盘 30 分钟量比 ≤ 0.5」，风险控制为固定 3.5% 止损 + 分钟收盘确认 + 开盘 30 分钟豁免 +
+1R 保本 / +1.5R 追踪 / +3R 冲高离场 / 最长持有 40 日，并叠加市场趋势门
（60 日均权趋势 > 0 才开新仓，< −3% 全簿清仓）。

### 证据分级（不可混用）

| 级别 | 含义 | 样本 | 可引用性 |
|---|---|---|---|
| **IS** | 参数就是在这段数据上选的 | 2026-01-01 → 08-28 | 只说明「被拟合过」，不是业绩预期 |
| **OOS** | 参数冻结后在另一段数据上跑 | 2025 段 | 唯一可作前瞻参考的历史证据 |
| **SHADOW** | 逐日向前推进，不重放历史 | 2026-01-01 → 今 | 最接近真实运行，但仍是模拟成交 |
| **LIVE-EXEC** | 影子盘中由实时交易者真实时刻执行的子集 | 2026-09-04 → 今 | 执行机制证据，**样本 n = 1** |

### 当前可引用的数字（`citable=true`）

| 窗口 | 级别 | 交易日 | 累计 | 年化 | Sharpe | t | maxDD | 成交 |
|---|---|---|---|---|---|---|---|---|
| 2026-01-01 → 08-28（800 只真实股票池） | IS | 159 | **+5.72%** | +9.28% | **0.67** | 0.53 | 8.42% | 236 |
| 2025-09-01 → 12-31（回补后） | OOS | 82 | **+1.64%** | +5.20% | **0.40** | 0.23 | 5.00% | 106 |
| ~~2026-01-01 → 08-28（仅 301 只截面）~~ | IS | 159 | ~~+16.36%~~ | ~~+27.34%~~ | ~~1.66~~ | 1.32 | 6.12% | 200 |

> **⚠️ 中间那行作废**：2026 年 PIT 价格面板当时每天只有 301 只标的（声明 800），
> 补成真实 800 只后同一窗口同一参数从 +16.36% 掉到 +5.72%——**约三分之二的「业绩」
> 来自被缩小的截面，而不是策略**。这正是新增硬闸门 `effective_universe` 的由来。

生产影子账本 `outputs/shadow_status_D_5W.json`（2026-09-10）：
净值 62,278.45（自 5 万，**+24.58%**、Sharpe 1.86、maxDD 10.20%、167 交易日、253 笔），
同期 HS300 基准 −3.08%，超额 +27.66%。**该账本的 2026 年历史是在上述 301 只截面缺口
下推进的**，因此它属于 SHADOW 级别、不可与上表直接相加比较。

### 模型自优化闭环：机制齐备但按闸门休眠

D 轨的月度滚动重训 + 平行挑战者 + 前向晋升闸门（`cli.py dcycle refit|challenger|decide`）
**代码/调度/测试齐备，但生产开关是关的**。历史三挑战者证据（2026-09-05）：唯一正面的
CH_A（+13.24pp，对上失效现役的同窗重放）是"现役失效时滚动重训有价值"的证据，而两个
**对比真实账本**的决定性窗口均大幅落后（CH_B −16.36pp、CH_C −17.07pp，且各含 1 笔
涨跌停违规）。裁决：**不启用生产循环，D 轨继续使用现役工件**——因为现役工件本身是
"2026 段最佳"的选择产物，且全部 `pb_*` 参数是在现役排名上网格调优的（模型-参数耦合）。
这与论文结论一致：*训练频繁、部署有选择*。详见 [docs/D_MODEL_CYCLE.md](docs/D_MODEL_CYCLE.md)。

### 统计功效：不要把噪声当 alpha

日频 Sharpe 标准误 ≈ `√(252/N)`：165 日 → **≈1.24**；252 日 → 1.00。
要检出真实 Sharpe 1.0（α=0.05、power=0.8）约需 **1,556 个交易日 ≈ 6.2 年**，检出 0.5
约需 **6,225 天 ≈ 25 年**。因此 `scripts/d_oos.py` 输出 `sharpe_standard_error` 与
`sharpe_t_stat`，而不是只报一个 Sharpe——**当前 IS t=0.53、OOS t=0.23，
没有任何统计显著的正 alpha 证据**。

其它已披露的口径问题（全部保留历史、不回溯改写）：生产账本有 **12 天负现金**
（最低 2026-03-17 −4,850.46，5 万账户 ≈ 8.7% 隐式杠杆），已由「先卖后买 + 买入按可用
现金裁剪」修复，历史天数由 `equity.cash_guard` 持续披露。

---

## 7. 研究结论存档

### Phase 7 — 数据地基（已完成）

全 A 全量入库 12,480,696 价量 bar / 5,162 只 / 0 失败，覆盖 2010-01-04 → 2025-12-31
（3,886 天），universe 快照 7,799 条，落库 PostgreSQL 17；`verify --mode backfill`
的 B1–B5 全绿（2,594@2015 → 5,205@2026-08-07，230 只已退市标的保留）。
详见 [docs/PHASE7_REPORT.md](docs/PHASE7_REPORT.md)。

### Phase 8 — 低波 + 低换手因子池（有效，进入 Alpha 核心）

`outputs/factors.json` 的 5 条公式，训练窗组合 Sharpe 1.58–1.90 / maxDD ~10.5–11%。

### Phase 9 — 三次失败与资产重定位（🔴 已关闭）

| 试验 | 门禁 | 最佳结果 | 结论 |
|---|---|---|---|
| PEAD（盈余公告漂移） | rank_ic > 0.015 | 负漂移（Q4−Q0 fwd20 = −2.6%），反转仍不过门 | ❌ |
| 研报情绪（TriAgent） | rank_ic > 0.015 | rank_ic = 0.0094 | ❌ |
| 文本分歧度 / 新颖性 | rank_ic > 0.015 | dispersion@20 = 0.0136（组合 Sharpe −0.95）；novelty 全弱 | ❌ |

统一根因：**HS300 上「已披露文本/基本面信息」在发布时已被充分定价**（三条独立证据线
收敛：研报情绪 0.0094 / 分析师评级 0.0047 / 事件研究无漂移）。BERT 层把情绪 IC 从
0.0015 提升到 0.0094（6 倍）——**架构有效、输入源受限**。
重定位：情绪管线 → 风控熔断层；PEAD SUE 基建 → 战术倾斜层；文本因子 → 不再参与
Alpha 打分。详见 [PHASE9_CLOSURE.md](PHASE9_CLOSURE.md)。

### Phase 10 — 三层组合（✅ 门禁 PASS）

Alpha 核心（Phase 8 价量因子）→ 战术倾斜（PEAD 反转）→ 风控熔断（舆情 Z-score）。
窗口 2011-01-18 → 2025-12-30（3,633 交易日），HS300，市场中性 ±10%，gross 1，单票上限 5%；
门禁 Sharpe > 1.6 且 maxDD < 10%：

| 场景 | Sharpe | 年化 | maxDD | 换手 | t |
|---|---|---|---|---|---|
| baseline（纯 Alpha 核心） | 1.695 | 13.3% | 9.2% | 0.18 | 6.44 |
| + 风控（舆情熔断） | 1.704 | 13.3% | 9.2% | 0.19 | 6.47 |
| + 倾斜（PEAD 反转） | 1.694 | 13.3% | 9.2% | 0.18 | 6.43 |
| 三层全开 | **1.704** | 13.3% | 9.2% | 0.19 | 6.47 |

与等权市场相关性 **0.047**。关键是两条「论文处方」：动量截面中性化
（arXiv:2507.07107）把 Sharpe 1.02 → 1.58、maxDD 20.1% → 12.7%；regime 自适应空腿
控制（AlphaCrafter γ）再推到 1.70 / 9.2%。覆盖层真实贡献很小（舆情仅 0.33pp、
PEAD 为负向噪声，默认关闭幅度）。详见 [docs/PHASE10_REPORT.md](docs/PHASE10_REPORT.md)。

> ⚠️ **Phase 10 的 PASS 不是生产状态**：它跑的是 HS300 截面 + 月度调仓 +
> 固定换手率成本模型的三层组合，与当前生产的 D 轨（`hs300_500` + 逐日评估 +
> 真实成本 + 分钟级日内止损）是**两套不同的系统**。Phase 10 是研究里程碑
> （Alpha 核心有效），D 轨的成绩单见 §6。

### 提高收益三轨（2026-08-31）

> 注意：这里的「三轨」是**研究渠道**，与已退役的四条资金轨（A_200W/B_10W/C_5W/D_5W）
> 不是同一件事，勿混淆。

| 轨 | 内容 | 结论 |
|---|---|---|
| **A** GitHub 公式动物园 | 四方方言翻译器（`src/exploration/translate.py`）+ 闭式算子库；`paper/factor_zoo/translated.json` 实测可译 **316 条**（alpha101 84 / gtja191 73 / alpha158 159），另有 141 条因缺算子 deferred。GTJA 第二来源补齐 88 条后，动物园由初版 237 → **316 条** | 扫描出 32 条「IC 可靠」的短周期候选，送生产风险门 → **1/32 通过**（唯一幸存 `Div(Ts_Min(Low,5), Close)`，贴门通过）。**结论：`min_lookback=60` 纪律是赚来的，不是保守**——31/32 在 IC 门上可靠，一到可交易收益的组合级门全部翻车（全样本回撤 16%–99%，多数 Sharpe 为负） |
| **B** 新数据域 | 融资融券（`src/data/margin.py`，788,304 条，2010-03→2026-08）、龙虎榜（`src/data/lhb.py`，72,876 条，2022-01→2026-08） | 分段门禁（发现窗选方向 → 验证窗纯 OOS）：两融信号弱但方向稳定，**低于 0.015 单因子门**，适合作 ML 特征而非独立因子；龙虎榜 `net_buy_5d` 2026 段**符号翻转** → **负结果**，不进入影子盘 |
| **C** ML 双轨道 | LightGBM + GPU MLP/XGBoost（`src/ml/`，RTX 4060）。特征集 = 34 基线 + 316 动物园公式 + 8 两融 ≈ **345 特征** | 见下方「ML 对决」小节 |

### ML 对决：规模口径比模型选择更关键（2026-08-31 / 09-01）

严格同路径对决（同一 `PaperRunner`、真实 A 股成本、10 日调仓、十分位多空 + regime 空腿），
**test 段 2022–2025** 与 **2026 实盘段（全部模型从未见过的样本外）** 两个时段 × 两个资金口径。
训练协议：walk-forward（train 2010–2019 拟合 + **Purged-KFold/embargo** 早停选型 →
val 2020–2021 → **test 2022–2025 一次性 OOS**）：

| 簿 | test 年化 | test Sharpe | **2026 段年化** |
|---|---|---|---|
| **LightGBM v2（345 特征）** | +18.4% | 2.04 | **+22.7%** |
| MLP v2（345） | +14.3% | 1.77 | +19.5% |
| rank 集成簿 | +19.4% | 2.09 | +11.0% |
| 在任 5 因子池 | +11.6% | 1.63 | **−1.7%** |
| MLP v1（108） | **+19.5%** | **2.35** | **−4.5%** |
| XGBoost-GPU（345） | +19.1% | 1.89 | −8.5% |

模型级指标（`src/ml/promote.py` 的地板：净 Sharpe ≥1.0、maxDD ≤15%、|rank_ic| ≥0.02，
且必须**双轴击败在任池**）：GPU MLP v1（108 特征）test rank_ic **0.0558** / ICIR **8.28**；
XGBoost-GPU（345）rank_ic **0.0566** / ICIR 6.28。

三条结论：

1. **泛化性排名反转**：MLP v1 在 test 段最强却在 2026 段 −4.5%；**LightGBM v2 两段都进前三
   且 2026 段第一**——宽特征 + GBDT 是更稳健的部署选择。这正是生产 D 轨扫描器加载
   LightGBM 工件而非 MLP 的原因。
2. **10 万口径下没有任何模型达标**：60 名 × 5 元最低佣金的「微尘成本」结构在该规模碾压
   一切（在任池 −12.4% 已是最不差）。**这是规模约束，不是 alpha 问题**——修复路径是成本
   治理（持仓数减半 / 调仓下限 2,000 元 / 带宽调仓），不是换模型。
3. **在任池在两个时段都被 4 个模型超越**，替换有充分证据；但
   `ml_promote` 周期任务**目前只存在于设计文档**（[docs/ML_DATA_TRACKS.md](docs/ML_DATA_TRACKS.md)
   §影子盘接线设计），代码中零命中——`evaluate_promotion()` 只被离线评估脚本
   `scripts/ml_portfolio_eval.py` 调用，**自动替换门尚未接线**，生产仍是人工装配。

> ⚠️ 上表 2026 段数字产出**早于** 2026-09-10 的「301 只 vs 800 只股票池」更正，
> 因此继承同一截面口径风险，未经在真实 800 只池上复跑验证。

详见 [docs/SHOWDOWN_V3_REPORT.md](docs/SHOWDOWN_V3_REPORT.md)、
[docs/ML_DATA_TRACKS.md](docs/ML_DATA_TRACKS.md)。

### A/B/C 三轨退役（2026-09-08）

原多资金轨影子盘（A_200W / B_10W / C_5W，LightGBM 截面长多簿）在
2026-01 → 08-28 的表现是 A +7.40% / Sharpe 0.67 / maxDD 16.90%，
B +0.82% / Sharpe 0.18 / maxDD 26.13%——**均劣于 D 轨的回撤买入纪律**。
三轨已退役，账本/状态/日报归档到 `outputs/archive/2026-09-08_retire_ABC/`，
`shadow.accounts` 现在只注册 `D_5W`。历史结论保留在
[docs/DUAL_TRACK_REPORT.md](docs/DUAL_TRACK_REPORT.md)。

---

## 8. 工程纪律与门禁

### 证据 provenance（2026-09-09 起强制）

对外 JSON 必须带五项：`window / convention / data_as_of / artifact_sha256 / code_commit`，
由 `scripts/check_provenance.py` 校验（缺项或哈希不符 → 退出码 1）。
**`citable` 字段是唯一的引用许可**——只有「fresh ledger + 全部断言通过 +
`params_match_production` + 无 `--set` 覆盖」时为 `true`。
`data_ok` 与 `citable` 分开：候选参数跑可以在健全数据上比较，但不能当作部署配置的数字。
所有数字引用时必须同时给出：级别（IS/OOS/SHADOW/LIVE-EXEC）· 窗口 · 触发约定 ·
止损宽度 · 样本量 · Sharpe 标准误 · 数据出处。

### 预注册（改规则前先冻结）

`scripts/prereg.py` 生成六字段只追加记录（含 `policy_sha256` + 代码指纹）；
改任何阈值 = **一次新试验**，必须升版本并写 `supersedes`。现行版本 v4
（`d_forward_2026h2`，`frozen_at 2026-09-10T16:41:41`）。
绑定检查已从「commit 号」改为「**行为代码内容指纹**」——只改注释不再误报漂移。

### 前向期风险闸门（替代能力闸门）

前向窗口在统计上**无法证明 alpha**（见 §6 功效计算），因此前向期只回答三类高信噪比
问题：**管线跟踪误差 / 成本模型标定 / 运行可靠性**。**10 项硬闸门**（文档表格拆成 11 行，
`tracking_error_min_days` 是跟踪误差两项的内部条件）任一失败即判定前向期失败并停机排查，
**「未测量」一律判失败**（不是「完美」）；Sharpe / maxDD / 超额只记录、不参与判定。
当前因为窗口尚未开始（2026-09-11 起），5 项读数「未测量」→ verdict = `fail`，
这是设计行为。协议见 [docs/FORWARD_PROTOCOL.md](docs/FORWARD_PROTOCOL.md)。

十个闸门：`prereg_binding` / `tracking_error_daily_pp` / `tracking_error_sign_bias` /
`cost_fee_deviation` / `cost_price_integrity` / `violations` / `availability` /
`data_freshness` / `symbol_minute_coverage` / `effective_universe`。
任一硬闸门连续 3 个交易日失败 → 停机排查管线（不是「策略不行」）。

### 前向候选：两条重放臂的配对检验

现役（固定 3.5% 止损）vs 候选（ATR 1.0 截断 [2.5%, 4.0%]），**唯一差异是止损宽度**，
两者都是重放账本、同一数据同一代码同一执行制度。切换规则：120 个配对交易日 +
日差均值 > 0 且配对 t > 1.5。干净窗口实测两臂日收益相关性 0.884、t = 0.376，
按观测效应量要达到 t > 1.5 需 **~1,291 个配对交易日 ≈ 5.1 年** → 当前结论 **HOLD**，
且接受「永不分离」（`may_never_separate: true`）。

> v1 设计曾让候选继承生产的订单清单、并关掉日内扫描，结果**两条账本逐位相同**——
> 2026-09-10 实测两臂状态文件净值 `62,278.45` / 成交 `253` 笔 / Sharpe `1.8553`
> 完全一致（仅 `last_run` 差 1 秒）。该实验"结构上无法回答自己的问题"，
> 按协议计为新试验，窗口自 2026-09-11 重新起算。

### 实盘接入前置条件

见 [docs/LIVE_READINESS.md](docs/LIVE_READINESS.md)。A（规则合规）/ B（执行可靠性）
全绿；C（资金与风控）与 D（证据与验证）尚有余项：

| 未完成项 | 说明 |
|---|---|
| C1 / C2 / C5 / C6 / C7 | 券商适配器（必须先过 `src/deploy.assert_simulated_only()`）、账户与资金划拨、面板一键熔断、券商 vs 账本对账、灾备回滚 |
| D2 / D3 | 实时执行样本仍偏少（n=1）、Sharpe 置信区间仍宽 |
| **D9（阻断）** | 偏差压力测试：退市通道 1.75pp（0.22×α，不阻断），但**截面构成通道实测 9.60pp/年 = 1.2× 目标 α** > 阈值 2.4pp → `bias_blocking_evolution: true`，**自进化闭环保持关闭**。另有一条**未测通道**（指数成分前瞻：`resolve_shadow_universe()` 把今天的 HS300+ZZ500 名单套用到 2025 窗口，PIT 库无历史成分数据）——按纪律「未测」永不当作「没问题」 |
| D10 | 纸面账户**不可测**市场冲击成本，实盘前必须用券商成交回填 `market_impact_bps` |
| D12 | 有效股票池：301 只缺口已回补、闸门读数 1.00，但文档勾选状态与工件窗口口径尚未同步 |

---

## 9. 论文 → 组件映射

蓝图骨架见 [blueprint/PROJECT_BLUEPRINT.md](blueprint/PROJECT_BLUEPRINT.md)，
2026 SOTA 综述见 [blueprint/review.md](blueprint/review.md)，论文/网页/仓库索引见
[paper/README.md](paper/README.md)。实现同时参考两者，蓝图未覆盖的偏误控制、
成本审计与多假设检验被逐一并入：

| 论文 | 并入的机制 | 位置 |
|---|---|---|
| FINSABER | Bias Traps：PIT 可见性窗口；Bonferroni 多假设 Sharpe 校正 | `data/point_in_time_loader.py`, `backtest/metrics.py` |
| FinCAD | 上下文感知解码：未来日期 logits 惩罚、prompt 清洗、`LookAheadAudit`（要求 >50% IC 下降） | `bias_control/` |
| AlphaSchema | 语义空间五元组 Event/Context/Qualities/Direction/Output | `factors/semantic_space.py` |
| AlphaJungle | SchemaPlan 邻域探索（EvoQuant 候选生成用 `space.neighbors`） | `factors/schema_explorer.py`, `evolver.py` |
| AlphaMemo | 结构化记忆 + 频繁子树回避 | `factors/memory_manager.py` |
| EvoQuant | 验证器引导的自进化：诊断 → 候选 → 门控 → 蒸馏 | `evolver.py`, `agents/risk_agent.py` |
| AgenticAITA | Adaptive Z-Score Trigger + Inference Gating | `configs/llm_routing.yaml`, `agents/inference_gate.py` |
| Sleipnir | 动态路由：按 bull/bear/sideways 排序智能体执行次序 | `agents/dynamic_router.py` |
| TriAgent | 分层情感（sentence → document → market），浅层用廉价模型 | `sentiment/triagent.py`, `configs/llm_routing.yaml` |
| MLMultiFactorBiasCorrection | 动量截面中性化（Phase 10 的关键处方一） | `portfolio/alpha_core.py` |
| AlphaCrafter | regime 自适应空腿控制 γ（关键处方二） | `portfolio/alpha_core.py` |
| Beyond Agent Arch | 成本/延迟纳入设计：$500 月预算闸门 + 代码缓存 + 确定性在线层 | `cost_tracker.py`, `online/` |

---

## 10. 验证

```bash
python cli.py verify --mode live        # 全库审计：常开 4-5 项 + 4 项真实数据审计
python -m pytest tests/ -q              # 752 用例，全离线
python scripts/check_provenance.py      # 工件 provenance 五项校验
python scripts/forward_health.py        # 前向期风险闸门评估
python scripts/d_oos.py                 # OOS 同构重放（14 项断言）
```

`verify` 跑的检查（`src/checklist.py`）：

| 检查 | 断言 |
|---|---|
| `pit` | 在 `pit.validation_timestamp` 查询不到任何 born-after 的事实 |
| `fincad` | 上下文感知解码真的生效——作弊因子的 IC 被压制到 0（要求降幅 > 50%） |
| `diversity` | 因子池两两 AST 距离不低于配置下限（默认 0.4） |
| `cost` | 模拟一个月的 LLM 调用投影不超过 `$500` |
| `causality` | 对每个公式做探针日扰动，读数不因未来数据改变 |
| `no_future_leak` | 在 train/val 边界查询，0 条越界事实（B1） |
| `adjustment_consistency` | 后复权日收益全部落在涨跌停带内、因子变更日伴随原始价跳变（B3） |
| `survivorship` | 2015-01-05 存续、此后退市的标的仍在库中（B4） |
| `data_freshness` | 行情截止日落后不超过阈值、覆盖率达标（B5） |

> `verify` 用**全库**（`bound_to_universe=False`）并追加真实数据审计；`mine` /
> `backtest` / `evolve` / `monitor` 跑的是窗口切片 + 有界 universe，只跑常开检查——
> 否则 B5 在训练窗切片上必然显示「陈旧」，B4 的 universe 快照也被切出范围。

---

## 11. LLM 配置与成本

- 后端：DeepSeek `https://api.deepseek.com/v1`，模型 `deepseek-v4-pro`
  （OpenAI 兼容，`src/llm_client.py` 懒加载 `openai`）。
- 密钥：`.env` 中 `DEEPSEEK_API_KEY`（`.gitignore` 已忽略）；`load_config` 用 stdlib
  读取 `.env` 并做 `${VAR}` 插值。
- 分层路由（quick / deep / generator / code / critic / sentiment）+ AgenticAITA
  自适应 Z-score 触发器：只在统计异常时上调重模型，其余保持廉价层。
- 每一次调用按 prompt/completion token 记账到 `CostTracker`；超过
  `budget.monthly_llm_cost_usd`（默认 $500）即拒绝继续调用。
- `factor_mining.force_combination_templates: true`：免费槽位也走确定性组合模板，
  实测 LLM 无法产出通过 15% 回撤门的单因子（0/118，100% `reject_high_risk`），
  因此默认挖掘路径**零 token**、完全确定性。

---

## 12. 关键文档索引

| 文档 | 内容 |
|---|---|
| [CONTEXT.md](CONTEXT.md) | 领域词汇表 + ADR 索引 + 数据地基/研究状态快照 |
| [docs/D_TRACK_EVIDENCE.md](docs/D_TRACK_EVIDENCE.md) | **D 轨证据分级与全部口径说明（引用数字前必读）** |
| [docs/FORWARD_PROTOCOL.md](docs/FORWARD_PROTOCOL.md) | 前向期协议：预注册、冻结、10 项硬闸门、候选切换 |
| [docs/LIVE_READINESS.md](docs/LIVE_READINESS.md) | 实盘接入放行清单（A/B/C/D 四组） |
| [docs/LANDING_PLAN.md](docs/LANDING_PLAN.md) | 落地清单与逐项验收 |
| [docs/EXECUTION_INVARIANTS.md](docs/EXECUTION_INVARIANTS.md) | 执行器不变式（含 long_only 与裁剪语义） |
| [docs/evidence/README.md](docs/evidence/README.md) | 可引用工件快照与 `citable` 口径 |
| [docs/PHASE7_REPORT.md](docs/PHASE7_REPORT.md) · [docs/PHASE10_REPORT.md](docs/PHASE10_REPORT.md) | Phase 7 / 10 报告 |
| [PHASE9_CLOSURE.md](PHASE9_CLOSURE.md) | Phase 9 关闭报告（三次失败与重定位） |
| [docs/ML_DATA_TRACKS.md](docs/ML_DATA_TRACKS.md) · [docs/SHOWDOWN_V3_REPORT.md](docs/SHOWDOWN_V3_REPORT.md) | 提高收益三轨 / ML 六方对决（双时段 × 双资金口径） |
| [docs/BIAS_STRESS.md](docs/BIAS_STRESS.md) · [docs/ADJUST_ANCHOR.md](docs/ADJUST_ANCHOR.md) · [docs/BASIS_CONTRACT.md](docs/BASIS_CONTRACT.md) | 偏差压力测试 / 复权锚点 / 价格口径契约 |
| [docs/PAICC_INTEGRATION.md](docs/PAICC_INTEGRATION.md) | 与外部调度面板 PAICC 的集成（日度任务序列） |
| [docs/adr/](docs/adr/) | 架构决策记录 0001–0005 |

---

## 13. 已知阻断项与下一步（按优先级）

1. **全市场入库恢复**：4,364 只标的在 2025-12-31 后停更，全市场/成分研究已过期；
   或明确把研究范围收敛到 D 池。
2. **用 800 只真实股票池复跑止损宽度 A/B 与网格**：`docs/D_TRACK_EVIDENCE.md` §三的
   绝对数字已作废，**相对排序也尚未验证**。
3. **前向积累样本**：干净窗口自 2026-09-11 起算；实时执行样本要从 1 笔积累到至少几十笔。
4. **偏差压力测试（D9，阻断）**：截面构成通道 9.60pp/年 = 1.2× alpha，未降到
   0.3× α（2.4pp）以下前不得开启自进化闭环。退市通道 1.75pp 单独看不阻断，
   但**判定只取最差通道**；另有「指数成分前瞻」通道因缺历史成分名单**无法测量**，
   需要补齐成分股历史或明确接受 L=−80% 情形。
5. **实盘通道补齐**（C1/C2/C5/C6/C7）与**成本标定**（D10，需真实成交回填冲击成本）。
6. **文档/口径一致性**（引用前先核对，以工件与 config 为准）：
   - `preclose` 在同一仓库内有 **14:50 / 14:55** 两种命名（模块 docstring 与 CLI help 写
     14:55，LIVE_READINESS 与 FORWARD_PROTOCOL 写 14:50；时间守卫实为 14:45–15:10）。
   - `LIVE_READINESS.md` 的 **D12** 仍是未勾选（引 0.376 判失败），而工件该闸门已读
     **1.00**——回补已完成，勾选状态未同步。
   - 配对工件 `outputs/forward/paired_atr_1p0_25_40.json` 仍把**已作废的 2026-09-10**
     计为一个配对日（其 `code_commit` 早于 v4 冻结）。
   - `LANDING_PLAN.md` 阶段 6 的「2025-10-27→12-12 分钟数据回补」仍标 `[ ]`，
     但该回补已完成、对应 OOS 窗口已 `citable=true`。
   - `CONTEXT.md` 的数字滞后：融资融券写 192k 条（报告为 788,304）、测试基线写
     165 passed（当前 752）。该文件作为**词汇表**仍权威，数字请以本文与工件为准。
   - Phase 10 报告里的覆盖层阈值（舆情 −2.5σ / 5 天冻结）与当前 `configs/master_config.yaml`
     （`zscore_threshold: -1`、`freeze_days: 11`）不同——报告自身把该项列为部署前遗留，
     之后由 §7 回校写回；引用时以 config 为准。

---

## 14. 开发约定

- **领域词汇**：命名概念时使用 `CONTEXT.md` 的词条，不要漂移到它明确避免的同义词
  （如「因子衰减」而非「策略失效」）。改动触碰 ADR 时显式指出冲突，不要静默覆盖。
- **Issue / 规格**：以 markdown 落在 `.scratch/<feature-slug>/`，一个 ticket 一个文件；
  见 [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md)。
- **提交**：不回溯改写已记录的模拟成交与历史工件；口径修正一律「新增版本 + 标注作废」。
- **配置优先**：任何阈值都放在 `configs/*.yaml`，不要在源码里硬编码。

---

## 15. 许可

MIT，见 [LICENSE](LICENSE)。

论文 PDF、克隆的参考仓库与运行期产物（`outputs/`、`logs/`、多数 `data/` 子目录）
不入版本库，见 [.gitignore](.gitignore)。
