# 实盘接入前置条件（LIVE READINESS）

> 原则（用户长期约束，逐字保留）：
> **"一切都建立在所有改动还有整套系统没有违规操作能够直接接入实盘的基础上"**
> **"检查D轨日内操作应采取实盘实时操作的方式，不允许对过去已知时点进行买卖，应该实时更新持仓股票的盈亏情况，交易记录时间需要精确到分钟"**
>
> 当前通道：`deployment.mode = observe`、`deployment.real_money_enabled = false`
> （`configs/master_config.yaml`）——**全部成交为系统内纸面撮合，未向任何券商发单**。
> 本文是「从 observe 切到 live」的放行清单；**未全部勾选前不得开启
> `real_money_enabled`**。切换只约束切换之后的行为，绝不回溯改写任何已记录的
> 模拟/影子成交（`src/deploy.py` 只提供闸门与状态，不触碰账本）。

## A. 规则合规（硬性）

| # | 项 | 状态 | 依据 / 落地 |
| --- | --- | --- | --- |
| A1 | 不得对过去已知时点买卖：日内决策只读当前成交价 | [x] | `src/live/trader.py` 轮询最新分钟成交价；`pb_live_intraday_from` 之后的日期禁止分钟回放（`tests/test_no_retroactive_trades.py`） |
| A2 | 交易记录精确到分钟 | [x] | `Fill.time` = `HH:MM:SS`（实时）/ `15:00`（收盘竞价）/ 分钟 bar 时刻（回放）；面板与日报均展示 |
| A3 | T+1 | [x] | 日内只卖不买；买入一律在收盘/竞价层，`tests/test_no_retroactive_trades.py::t_plus_1` 与 `d_oos.py` 断言 |
| A4 | 涨跌停/停牌不可成交 | [x] | 收盘层用 `board_limit`（主板 10%/创业板 20%/科创 20%/北交所 30%，含 2020-08-24 前后差异）；实时层跌停价不成交（`tests/test_order_executor_limit_band.py`、`tests/test_live_trader_guards.py`） |
| A5 | 集合竞价前提交委托 | [x] | 14:50 `cli.py preclose` 生成委托清单 → 15:00 竞价价成交；错过即当日零成交（不补单） |
| A6 | 收盘后不再做日内决策 | [x] | 实时层决策窗口 09:30–11:30 / 13:00–15:00（`_in_trading_hours`） |
| A7 | 成本模型为监管真实值 | [x] | 印花税 5bps 卖出、过户费 0.1bps 双边、佣金 2.5bps（最低 5 元）、滑点 2bps；周六 `dcycle audit-cost` 校验 |
| A8 | 不得出现隐式杠杆（负现金） | [x] | `OrderExecutor` 先卖后买 + 买入按可用现金裁剪（`_affordable_shares`）；历史 12 天负现金原样保留并由 `equity.cash_guard` 披露（`tests/test_cash_guard.py`） |
| A9 | 报价时间可信 | [x] | 分钟 bar 先归一化到本地时间；无时间戳 fail-closed；超前 >10 分钟视为时钟/数据偏移、不决策（`tests/test_live_trader_guards.py`、`scripts/clock_offset.py`） |
| A10 | 交易日历 | [x] | PAICC `trading_calendar.py`（2026 已观测休市日 + `quant_holidays` 覆盖）；节假日不再拉起 live/preclose/日度闭环 |
| A11 | 实时进程健康 | [x] | PAICC 每 5 分钟 `live_watchdog`：会话内检测 pid + 命令行，死亡则前向重启；`/quant/stop` 默认保护 live 进程 |

## B. 执行可靠性

| # | 项 | 状态 | 依据 / 落地 |
| --- | --- | --- | --- |
| B1 | 单实例互斥（不重复下单） | [x] | pid 锁 + 进程命令行校验（`_is_live_process`） |
| B2 | 报价陈旧/断线不决策 | [x] | `live.max_quote_age_minutes`（默认 5 分钟）；轮询失败不退出、下一轮重试 |
| B3 | 后端/调度重启后可续跑 | [x] | 09:25 live 启动 + 盘中「前向恢复」（只读当前价）；15:02/17:45/14:40 启动补跑；14:50 preclose 有自身时间守卫，**不补跑** |
| B4 | 账本并发安全 | [x] | `protect_after` 水位线：收盘 run 不删实时交易者并发写入的成交 |
| B5 | 数据面依赖可自愈 | [x] | PAICC `ensure_pit_db_up`（Docker daemon/健康检查等待）；分钟特征 15:02 刷新 + 日度自愈 |
| B6 | 可观测 | [x] | 面板：实时盘中（到秒）、委托清单、任务调度（8 job 计划/上次/下次/状态）、进程、日志 |

## C. 资金与风控（切 live 前必须补）

| # | 项 | 状态 | 说明 |
| --- | --- | --- | --- |
| C1 | 券商接口与下单适配器 | [ ] | 需新增 `src/live/broker.py`，**必须调用 `src.deploy.assert_simulated_only()`** 后才可发单 |
| C2 | 实盘账户与资金划拨 | [ ] | 券商账户、三方存管、单笔/单日限额 |
| C3 | kill-switch 接入实盘通道 | [x] | autopilot 档位已接入 D 簿（`scale_getter`，只缩不加；`tests/test_pullback_control_scale.py`）；实盘适配器必须复用同一 `ControlState` |
| C4 | 风控红线告警通道 | [x] | 红线仪表盘 + `autopilot_alerts.jsonl` + 可选 webhook |
| C5 | 人工熔断开关 | [ ] | 需面板「一键停止」→ 取消 live 任务并平掉挂单 |
| C6 | 对账（券商 vs 账本） | [ ] | 需每日收盘后比对成交/持仓/资金，差异即告警 |
| C7 | 灾备与回滚 | [ ] | 断网/券商故障时的降级方案与回滚流程 |

## D. 证据与验证（切 live 前必须完成）

| # | 项 | 状态 | 说明 |
| --- | --- | --- | --- |
| D1 | OOS 验证（与生产同构口径） | [x] | `scripts/d_oos.py`（14 项断言，含符号级窗口覆盖）→ `outputs/d_oos_*.json`；结论见 `docs/D_TRACK_EVIDENCE.md` |
| D2 | 实时执行样本量 | [ ] | 实时执行的日内止损样本仍偏少（见 D_TRACK_EVIDENCE 的 provenance 统计），需继续累积 |
| D3 | 统计显著性 | [ ] | Sharpe 标准误 √(252/N)：当前样本量下置信区间仍宽，见 D_TRACK_EVIDENCE |
| D4 | 回放/实时口径分离 | [x] | `fills.source` ∈ {live, replay, close, auction}，面板与日报分别计数 |
| D5 | 工件 provenance 完整 | [x] | 每个对外 JSON 必含 `window / convention / data_as_of / artifact_sha256 / code_commit`；`scripts/check_provenance.py` 校验（缺项/哈希不符即失败） |
| D6 | 前向风险闸门 | [x] | `scripts/forward_health.py`：跟踪误差/成本/违规/可用率/新鲜度/覆盖率；**未测量即判失败**。见 `docs/FORWARD_PROTOCOL.md` |
| D7 | 预注册（改规则前先冻结） | [x] | `scripts/prereg.py`（六字段 + 哈希 + 只追加）；改规则须升版本 + `supersedes` |
| D8 | 复权锚点可复现 | [x] | `scripts/check_adjust_anchor.py`（基线冻结 + 漂移检测）；见 `docs/ADJUST_ANCHOR.md` |
| D9 | 数据偏差压力测试 | [ ] | **判定为 BLOCKING**（2026-09-10 口径修正）：退市通道 1.75pp，但同一工件的截面构成通道实测 **9.60pp/年 = 1.2×α** → `bias_blocking_evolution: true`，**自进化闭环保持关闭**。见 `docs/BIAS_STRESS.md` |
| D10 | 成本模型标定（真实成交） | [ ] | 市场冲击成本在纸面账户**不可测**；实盘前必须用券商成交回填 `market_impact_bps`（`docs/FORWARD_PROTOCOL.md` §1.3） |
| D11 | 前向测量绑定预注册 | [x] | `prereg_gate` 五项检查（记录/窗口/`frozen_at`/`policy_sha256` 运行时重算/`code_commit` + 干净工作区）；首次实评 `prereg_binding: PASS` |
| D12 | 部署股票池 == 声明股票池 | [ ] | **2026 年 PIT 价格面板只有 301 只**（声明 800；2025 年 5,163）；新增硬闸门 `effective_universe` 实测 **0.376 → 判失败**；回补脚本 `scripts/backfill_price_gap.py`（499 只零覆盖），补完需重跑 IS 并重发工件 |
| D13 | 执行器不得静默开空头 | [x] | 卖出按「卖出 pass 开始时持仓」累计裁剪，超出部分记入 `OrderResult.skipped` 并进入运行结果；`long_only` 默认开启（`docs/EXECUTION_INVARIANTS.md`） |

## 切换流程（草案）

1. 本文 A/B/D 全绿，C 全部补齐并完成一次纸面演练；
2. 在 `configs/master_config.yaml` 设置 `deployment.mode: "live"`、
   `deployment.real_money_enabled: true`（两步分开，先观察一轮日度闭环）；
3. 券商适配器首次发单前打印并记录 `deployment_status(cfg)` 与 kill-switch 档位；
4. 首日以最小额度（建议 ≤ 5 万）运行，收盘后按 C6 对账；
5. 任何差异 → 立即回到 `observe`（只改通道，不改历史成交）。

> D5–D8 是 2026-09-09 审计项 1.1–1.4 的落地检查点；D9/D10 是尚未满足的**阻断项**，
> 未完成前不得把「前向期跑过 N 个月」当作 alpha 证据（前向窗口对 alpha 无功效）。
