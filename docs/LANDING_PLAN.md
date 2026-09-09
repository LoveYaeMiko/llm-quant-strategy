# D 轨单轨落地计划（LANDING PLAN）

> 状态：**执行中** · 建立于 2026-09-08 · 本文件是"所有已探讨方案实际落地"的唯一权威清单。
> 每完成一项，把 `[ ]` 改为 `[x]` 并补上验证证据（命令 / 文件 / 数字）。
> **2026-09-09 更新**：阶段 0–5 全部落地；**第二轮独立审计的 15 项处置见阶段 6**
> （`docs/AUDIT_2026-09-09_FIXES.md`）。当前结论见 `docs/D_TRACK_EVIDENCE.md` §九：
> **没有可引用的样本外证据（2025 窗口 citable=false），系统保持 observe（模拟盘）**。
>
> 总原则（用户长期约束，逐字保留）：
> **"一切都建立在所有改动还有整套系统没有违规操作能够直接接入实盘的基础上"**
> **"检查D轨日内操作应采取实盘实时操作的方式，不允许对过去已知时点进行买卖，应该实时更新持仓股票的盈亏情况，交易记录时间需要精确到分钟"**

---

## 阶段 6 — 第二轮独立审计（2026-09-09）处置

| # | 事项 | 状态 | 落地动作 / 验收 |
| --- | --- | --- | --- |
| V-1 | 生产账本 12 天负现金（隐式杠杆） | [x] | `OrderExecutor` 先卖后买 + 买入按可用现金裁剪；`tests/test_cash_guard.py` 7 用例；`equity.cash_guard` 持续披露 |
| A-1 | `fills.source` 历史为空、断言恒真 | [x] | 断言改为"列必须存在"+样本存在；`tests/test_fill_time_precision.py` 6 用例 |
| A-2 | 实时 `_limit_pct` 日期盲 / 北交所 10% / 无时间戳 fail-open | [x] | 委托 `board_limit`；无时间戳 fail-closed；`tests/test_live_trader_guards.py` 15 用例 |
| A-3 | D-1/D-2 调用点未锁定 | [x] | `tests/test_call_sites.py` 2 用例（真实 `build_preclose_orders` / `_build_account_portfolio`） |
| A-4 | 闸门无执行点 + `bool("false")` | [x] | `src/live/broker.py`（`RealBroker` 构造即闸门）；`_as_bool` + `mode=observe` 一票否决；`test_deployment_gate.py` 20 用例 |
| P-1 | 符号级覆盖探针缺失、旧工件误报 | [x] | `d_oos.py` 新增 `minute_symbol_coverage_ok` / `no_fills_inside_data_hole` / `citable`；2025 窗口实测 `citable=false`（最小覆盖 34%、45 笔洞内成交）；5 个旧工件标 `data_coverage_invalid` |
| P-2 | OOS 被用于选参数 | [x]（披露） | 文档明确 `selection_window: OOS-2025H2`；干净窗口重定义为前向 live-only |
| P-3/P-4 | PAICC 无交易日历 / 无看门狗 | [x] | `trading_calendar.py` + `quant_holidays` 设置；每 5 分钟 `live_watchdog`；`tests/test_quant_calendar.py` 9 用例 |
| P-5/P-6 | 指纹自比较、OOS 无 kill-switch | [x] | 对照探针改由**配置账户**构建；OOS 读取 ControlState 并记录档位 |
| PAICC-1..5 | UI 时间戳误标 / 缺 `quant_weekly_time` / preclose 到 15:10 / 午休 resume / `/quant/stop` 可杀 trader | [x] | 面板改「成交时刻」；补默认值；preclose 收到 14:56；resume 排除午休；`stop_command(force=False)` 保护 live |
| D-8/D-9 | 挑战者制度/评估窗、调仓相位 | [x] | `run_challenger` 接入 14:50 委托层；`decide_promotion` 只数窗内成交；相位锚定窗口首日 |
| 弱测试 | 3 处恒真/空转断言 | [x] | 全部重写（内容比较 + 样本存在 + 波动率调至带内） |
| 数据回补 | 2025-10-27→12-12 分钟数据 | [ ] | **收盘后**运行 `python scripts/fetch_intraday.py --start 2025-10-20 --end 2025-12-15` → `refresh_intraday_daily.py` → 重跑 `d_oos.py` 至 `citable=true`（盘中抓取会与 14:40/14:50 争用限流） |

---

## 阶段 0 — 单轨收敛（本次请求）

| # | 事项 | 状态 | 证据 / 验收 |
| --- | --- | --- | --- |
| 0.1 | `configs/master_config.yaml` 仅注册 `D_5W`，A/B/C 退役 | [x] | `shadow.accounts == [D_5W]`（YAML 解析验证通过） |
| 0.2 | A/B/C 账本/状态/日报/成交/档位/挑战者账本归档 | [x] | `outputs/archive/2026-09-08_retire_ABC/`（含 README） |
| 0.3 | `scripts/audit_tracks.py` 改为读配置账户（去掉硬编码 A/B/C/D） | [x] | 读 `shadow.accounts` 按 priority 降序；实测 `[('D_5W', 100, 'pullback')]`；D 回放已加载 `intraday=`/`minute_provider` |
| 0.4 | 文档标注 D-only 部署（A/B/C 退役） | [x] | `DUAL_TRACK_REPORT.md` 横幅、`PAICC_INTEGRATION.md` 横幅、新增 `D_TRACK_EVIDENCE.md` / `LIVE_READINESS.md` / `LANDING_PLAN.md` |
| 0.5 | PAICC 后端：调度状态暴露 job 列表（含 D 轨 live/preclose/intra/challenger）；文案改单轨 | [x] | `GET /api/quant/schedule` 返回 8 条 `jobs[]`（静态表 + 真实 next_run 覆盖）；新增 `/api/quant/preclose`；`pytest` 46 passed |
| 0.6 | PAICC 前端重写：仅保留 D 轨有用组件 | [x] | 新增 `DTrackPanel.tsx`、重写 `QuantPage.tsx`、删除 `DualShadowPanel.tsx`；`typecheck`+`build` 通过 |

## 阶段 1 — D 轨缺陷修复（用户独立审计 D-1~D-9）

| # | 缺陷 | 状态 | 落地动作 / 验收 |
| --- | --- | --- | --- |
| D-1 | 配置注释引用 grid6 无日内止损口径（+33.2%/1.59）而非部署口径 | [x] | 注释改为 grid7 `D_close_skip30`（+12.82%/+21.22%/1.158/10.23%），并显式标注 D_close_only 非部署口径 |
| D-2 | 触发约定敏感性 3.4×（插针最低价 vs 分钟收盘确认） | [x] | `docs/D_TRACK_EVIDENCE.md` §二 全表 + 面板「触发约定 分钟收盘确认」标签 + 配置注释标注出处 |
| D-3 | 唯一 OOS 证据为负（2025 段 −1.65%，close-only 且无 tail_vol 门槛） | [~] | `scripts/d_oos.py`（11 断言）已落地；OOS 运行结果见 §阶段 3.2（窗口受分钟缓存覆盖限制） |
| D-4 | 96% 的日内止损来自收盘回放而非实时执行 | [x] | `fills.source` ∈ {live,replay,close,auction}（含账本迁移）；日报/面板分别计数；实测 55 笔日内成交中 54 笔回放、1 笔实时（`D_TRACK_EVIDENCE.md` §四） |
| D-5 | preclose 无法卖出"已退出"标的（targets 只含 `compute_weights` 的键） | [x] | `merge_targets()` 先把当前持仓钉成 0.0；`tests/test_preclose_exits.py` 5 用例 |
| D-6 | kill-switch（autopilot gross_scale）未接入 pullback 簿 | [x] | `PullbackPortfolio(scale_getter=)` 只缩不加、0 = 清仓；`cli.py` 与 `preclose.py` 均接入；`tests/test_pullback_control_scale.py` 7 用例 |
| D-7 | 实时交易无跌停/停牌/陈旧报价守卫；轮询窗口到 15:10 与收盘竞价层冲突 | [x] | 决策窗口改为 09:30–11:30/13:00–15:00；跌停价不成交、陈旧报价（>5 分钟）不决策、状态文件记录 blocked；`tests/test_live_trader_guards.py` 14 用例 |
| D-8 | ATR 混合口径（PIT 面板 open/high/low 原始、close 前复权）+ TR 被压成一维 | [x] | `_adjust_factor_frame` 统一口径、`np.maximum.reduce` 保留二维；分钟回放成交价换算到面板口径；`tests/test_pullback_atr_basis.py` 6 用例；影响见 `outputs/d_atr_impact.json` |
| D-9 | 杂项（按代码复核逐条列出） | [x] | 见下方「D-9 复核清单」 |
| D-8c | 修复后口径的止损宽度再调优（`atr_mult × stop_hi` 网格 + OOS 复核） | [x] | `scripts/d_stop_grid.py` 8 变体 IS + 4 候选 OOS（11 断言全通过）：max-min 规则选 **flat 3.5%**（IS Sharpe 1.57 / OOS 1.60），旧 2.5% 在 OOS 最差（−1.10）；证据 `outputs/d_stop_grid.json`、`d_oos_oos_2025h2_*.json`；限制（短样本、|t|<1）已在 `D_TRACK_EVIDENCE.md` §三/§九 标注 |
| D-10 | 挑战者晋升改为人工确认（`auto_promote: false`），避免 30 日窗自动换模型 | [x] | `src/d_cycle.py` + `tests/test_challenger_promotion_gate.py` 3 用例；`docs/D_MODEL_CYCLE.md` §1.4 |
| D-11 | 红线按 alpha_source 过滤（D 轨不再显示无关的 PEAD/空腿线） | [x] | `_applicable_red_lines` + `tests/test_red_lines_applicable.py` 3 用例 |

## 阶段 2 — 网格/生产同构与回放合规

| # | 事项 | 状态 | 落地动作 / 验收 |
| --- | --- | --- | --- |
| 2.1 | `scripts/d_track_grid5/6/7`、`d_track_tune2/3`、`compliant_grid`、`audit_tracks` 未加载 `intraday=`（无 tail_vol 门槛） | [x] | 全部加载 `intraday=`；`tune2/tune3/compliant_grid` 的 BASE 补 `tail_vol_max=0.5`；**grid7 已用修复后代码重跑**（`outputs/d_track_grid7.json`，触发约定结论已更新为日内止损不损害收益） |
| 2.2 | `audit_tracks.py` D 轨回放缺 `intraday=` | [x] | 同上（与 0.3 合并） |
| 2.3 | `refresh_intraday_daily` 文档与代码不一致（docstring 说 15:05 前不含当日，代码 `end = today`） | [x] | 新增 `_resolve_end_date`（15:00 前不含当日）+ `tests/test_intraday_refresh.py` 6 用例 |
| 2.4 | `order_executor` 涨跌停带宽硬编码、与日期/板块无关 | [x] | 两处调用点改用共享 `board_limit`（主板 10%/创业板 20%（2020-08-24 前 10%）/科创 20%/北交所 30%）；`tests/test_order_executor_limit_band.py` 4 用例 |
| 2.5 | preclose(14:50) 与 depth(14:50) 任务同点冲突 | [x] | depth 移到 **14:40**（PAICC `quant_scheduler` + job 目录 + 文档）；preclose 保持 14:50 + 自身时间守卫 |

## 阶段 3 — OOS 验证方法论落地

| # | 事项 | 状态 | 落地动作 / 验收 |
| --- | --- | --- | --- |
| 3.1 | 建立 `scripts/d_oos.py`：新账本、生产同构装配、11 项断言 | [x] | `_shadow_cycle(ledger_override=, write_artifacts=False, probe=)` + `_book_fingerprint`；断言含 `fresh_ledger_not_resumed`、`contiguous_window_no_gap`、`intraday_frames_loaded`、`minute_provider_loaded`、`strategy_fingerprint_is_production`、`live_dates_not_replayed`、`t_plus_1_respected`、`no_fill_on_limit_locked_bar`、`cost_model_consistent` 等 |
| 3.2 | 运行 OOS（2025 段 + 2026 段），记录结论 | [x] | 2026-01-01→08-28：**11/11 通过**，+14.91%/Sharpe 1.53/SE 1.26/**t 1.22**；2025-09-01→12-31：**11/11 通过**，−6.14%/Sharpe −1.10/SE 1.75/**t −0.63**（35 天分钟覆盖仅 274/800 标的）。两窗口均无统计显著证据，见 `D_TRACK_EVIDENCE.md` §六/§九 |
| 3.3 | 证据分级文档：in-sample / OOS / 实时影子 | [x] | `docs/D_TRACK_EVIDENCE.md`：四类证据、触发约定敏感性、ATR 影响、成交来源、统计功效、引用规范 |

## 阶段 4 — 实盘接入合规前置

| # | 事项 | 状态 | 落地动作 / 验收 |
| --- | --- | --- | --- |
| 4.1 | `deployment_status`（observe）仅约束未来实盘通道，不回溯模拟成交 | [x] | `configs/deployment` 段 + `src/deploy.py`（`assert_simulated_only` 硬闸门）；影子状态/实时状态均上报通道；`tests/test_deployment_gate.py` 6 用例 |
| 4.2 | 禁止回溯已知时点交易的自动化检查 | [x] | `tests/test_no_retroactive_trades.py`：实时日期不回放、`protect_after` 并发安全、`source` 标注；`tests/test_fill_time_precision.py` 审计生产账本时间精度 |
| 4.3 | 实盘接入前置条件清单（合规/风控/券商接口/资金） | [x] | `docs/LIVE_READINESS.md`（A/B/D 全绿、C 项待补，未全绿不得开启 `real_money_enabled`） |

## 阶段 5 — 验证与交付

| # | 事项 | 状态 | 验收 |
| --- | --- | --- | --- |
| 5.1 | FQA 测试全绿 | [x] | 新增/受影响测试全绿；全量 `pytest tests/ --ignore=tests/test_phase8_remedy.py` exit 0；`test_phase8_remedy` 为**既有**状态依赖失败（`outputs/factors.json` 只有 5 条，断言 ≥6，与本次改动无关） |
| 5.2 | PAICC 后端测试 + 前端 typecheck/build | [x] | `pytest` 46 passed；`npm run typecheck` exit 0；`npm run build` 成功 |
| 5.3 | 后端重启与端点验证 | [x] | 后端已重启（PID 33752）；`/quant/schedule` 8 jobs（depth 14:40）、`/quant/accounts` 仅 D_5W、`/quant/live`、`/quant/preclose`、`/quant/trades`、`/quant/status` 全部正常 |
| 5.4 | 双仓提交推送 | [x] | FQA `48d6504`、PAICC `1bbcc34` 已推送 main |

---

## D-9 复核清单（逐条核对后勾选）

- [x] D-9.1 `scripts/replay_missed_morning.py` 标注「诊断用途，不作为实时成交依据」，成交标记 `source="replay"`
- [x] D-9.2 交易记录时间精度：`Fill.time` 为 `HH:MM:SS`（实时/回放）或 `15:00`（竞价）；`tests/test_fill_time_precision.py` 审计生产账本
- [x] D-9.3 收益口径统一：复权净值 + 含费含滑点（`D_TRACK_EVIDENCE.md` §八 引用规范）
- [x] D-9.4 账本 `seq` 水位线与实时写入并发安全（`tests/test_no_retroactive_trades.py`）
- [x] D-9.5 挑战者账本/晋升闸门只在 D 轨内运行（`src/d_cycle.py` 只取 `D_5W`）
