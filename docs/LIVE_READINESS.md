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
| D1 | OOS 验证（与生产同构口径） | [x] | `scripts/d_oos.py`（11 项断言）→ `outputs/d_oos_*.json`；结论见 `docs/D_TRACK_EVIDENCE.md` |
| D2 | 实时执行样本量 | [ ] | 实时执行的日内止损样本仍偏少（见 D_TRACK_EVIDENCE 的 provenance 统计），需继续累积 |
| D3 | 统计显著性 | [ ] | Sharpe 标准误 √(252/N)：当前样本量下置信区间仍宽，见 D_TRACK_EVIDENCE |
| D4 | 回放/实时口径分离 | [x] | `fills.source` ∈ {live, replay, close, auction}，面板与日报分别计数 |

## 切换流程（草案）

1. 本文 A/B/D 全绿，C 全部补齐并完成一次纸面演练；
2. 在 `configs/master_config.yaml` 设置 `deployment.mode: "live"`、
   `deployment.real_money_enabled: true`（两步分开，先观察一轮日度闭环）；
3. 券商适配器首次发单前打印并记录 `deployment_status(cfg)` 与 kill-switch 档位；
4. 首日以最小额度（建议 ≤ 5 万）运行，收盘后按 C6 对账；
5. 任何差异 → 立即回到 `observe`（只改通道，不改历史成交）。
