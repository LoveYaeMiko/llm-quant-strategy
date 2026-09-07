# PAICC × FQA 多资金轨影子盘集成报告

> 2026-09-02 起：PAICC 量化面板全面切换到 FQA 多资金轨影子盘（A_200W / B_10W / C_5W）；
> 原三因子池策略退出部署（仅保留在研究侧 `paper`/`calibrate` 扫描路径，不再进入影子部署）。

## 一、策略部署（FQA 侧）

| 项 | A_200W | B_10W | C_5W |
|---|---|---|---|
| 信号源 | LightGBM v2（345 特征 + 融资融券，ML 工件） | 同左 | 同左 |
| 截面 | hs300 | hs300_500 | hs300_500 |
| 簿形 | 10% 长多（30 名） | 5% 长多（15 名，单票 5% 上限） | 5% 长多（15 名，单票 5% 上限） |
| 调仓/治理 | 10 日 / 无治理 | 10 日 / 下限 2000 元 + 带宽 0.1% | 10 日 / 下限 2000 元 + 带宽 0.1% |
| 档位 | 逐账户风险闸门（normal/de_risk/halt ×0.5/×0） | 同左 | 同左 |
| 模型迭代 | 周日 `cli.py weekly`（重训 + 尾部 Sharpe promote 闸门） | 同左 | 同左 |

最新影子成绩（截至 2026-09-02）：A 2,556,885（+27.90%，Sharpe 1.84，maxDD 8.0%）；
B 127,327（+27.44%，Sharpe 1.93，maxDD 10.1%）；C 62,027（+24.26%，Sharpe 1.75，maxDD 10.3%）。

## 二、FQA 引擎改动

- `cmd_autopilot` → 双账户闭环：逐账户影子推进（ML 工件簿形）→ 逐账户风险闸门 →
  逐账户 `autopilot_state_<name>.json` + `autopilot_report_<name>.md`；
  因子池专属的衰减监控/重挖只对 `alpha_source: pool` 账户生效，ML 账户由周度重训负责迭代。
- §7 成本回校：聚合全部影子账本（`outputs/shadow_ledger*.sqlite`）的累计成交计算真实成本模型。
- `shadow_status_<name>.json` 新增 `account_config`（截面/簿形/调仓/治理/工件）供面板展示。

## 三、PAICC 后端改动

- `/quant/status` → 双账户负载：`{overall, accounts: {A_200W|B_10W: {overall, red_lines,
  last_trading_date, data_freshness_days, equity}}}`（总体状态取两账户最差）。
- 红线历史 → 按账户持久化：`quant_redline_history` 增加 `account` 列（启动时自动迁移），
  10s 轮询按账户去重入库；`/quant/redline-history?account=X&limit=N`。
- `/quant/autopilot` → 双账户档位映射（`autopilot_state_<name>.json`）。
- **`/quant/live`（2026-09-04 新增）** → D 轨实时盘中状态（`outputs/live_<account>.json`，
  `account` 缺省取配置 `live.account`）：`{ts(到秒), equity_live, cash, invested_pct,
  positions[{symbol, shares, last, entry, stop, pnl, pnl_pct}]}`；交易者未运行过返回 `null`。
- 调度新增 `quant_live_start`：工作日 09:25 拉起 `python cli.py live`（detached，
  pid 锁防双开，15:10 自退出）。
- 调度邮件：`run_shadow_daily` / `run_autopilot_daily` 发送「双资金轨」合并日报
  （各账户报表分节 + LLM 点评）；周日 18:00 `run_weekly_cycle` 不变；超时 7200s。

## 四、PAICC 前端改动（QuantPage / DualShadowPanel）

- **影子模式（长期测试 · 多资金轨）**：四账户 Tab —— 档位/敞口/账户配置标签 +
  净值/收益/Sharpe/回撤/成本统计 + 净值 vs HS300 与回撤/超额图 + 红线 + 当日目标持仓
  （Top20，含**买入价/现价**）+ 最近 50 笔详细交易记录（含分钟级成交时间；卖出笔同时显示
  **买入价**（移动加权成本，与 FQA `enrich_positions` 同口径）与成交价）；卡片头部带
  「立即运行」与调度时刻。
- **D 轨盘中实时卡片（2026-09-04 新增）**：pullback 账户 Tab 内 30 秒轮询
  `/quant/live`，展示实时权益/现金/仓位占比 + 逐仓现价/入场价/止损价/盈亏（更新时刻到秒）。
- **红线仪表盘**：按账户分节展示四条红线卡片（总体状态取最差）。
- **红线历史**：账户切换器（Segmented）+ 每日红线热力图（按账户查询）。
- **自动闭环**：逐账户档位/敞口/原因/周期时间戳 + 「运行闭环」+ 新增「周度闭环」按钮
  （`/quant/weekly/run`）+ 调度与最近运行摘要。
- **§7 三项回校**：保留（成本回校现基于双轨聚合成交；PEAD/舆情扫描为研究侧信息）。
- 旧单账户「影子模式（长期测试）」卡片移除，全部功能并入双轨面板。

## 五、运行方式

- 后端：`PAICC\backend\.venv\Scripts\python.exe run.py`（127.0.0.1:8000，已重启加载新代码）。
- 前端：`PAICC\frontend\npm start`（electron-vite preview 运行已构建产物，应用窗口已启动）；
  源码改动后需 `npm run build` 重建。
- 调度（进程内 APScheduler）：
  - 工作日 09:25 `cli.py live`（D 轨实时盘中交易）· **14:50 盘口快照** ·
    **14:55 `cli.py preclose`**（收盘竞价下单层：14:55 决策委托清单，15:00 竞价价成交；
    错过即当日收盘零成交，绝不事后补单）· **15:02 日内特征刷新**（`scripts/refresh_intraday_daily.py`）·
    **15:10 `cli.py autopilot`**（四轨闭环，收盘竞价成交后立即执行；运行内
    `ensure_intraday_current` 自愈双保险）· **17:45 `cli.py dcycle challenger`**
    （D 轨模型挑战者平行影子推进，`quant_d_cycle_enabled` 闸门控制）· 周日 18:00
    **D 轨模型月度循环**（每月第一个周日：`dcycle decide` 前向晋升闸门 + `dcycle refit`
    滚动重训，非首周跳过）· 周六 18:00 **成本模型一致性检查**（`dcycle audit-cost`，
    替代原 §7 回校——真实成本结构为监管固定值，仅校验不调参），均可在面板手动触发；
  - **启动补跑（catch-up）**：后端每次启动时检查当日 scheduled 任务是否已运行
    （operation_logs / 补跑标记），未运行且已过点时立即补跑一次——应用被关闭导致
    错过 15:10 时，重新打开 PAICC 会自动补上；
  - **错过容忍**：日度任务 misfire_grace=24h——机器在 17:30 处于睡眠、稍后唤醒时
    也会执行当次循环（FQA 账本可续跑，自动回补漏掉的交易日）；
  - **运行反馈**：各任务开始/结束推送 `quant_<x>_started` / `quant_<x>_ran`
    WebSocket 事件，面板按钮显示运行中状态、结束后自动刷新；
  - 机器电源计划已设「从不睡眠」（STANDBYIDLE=0），17:30 触发不再依赖人机交互。
- 性能：ML 账户部署下日度运行跳过研报全量重取（仅因子池账户需要），
  一次日度循环约 15-25 分钟；特征矩阵按工件+universe 缓存增量构建。
