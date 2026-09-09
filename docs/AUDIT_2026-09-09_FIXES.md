# 2026-09-09 独立审计处置表（FQA + PAICC）

> 来源：用户方的五份深审（代码缺陷 / OOS 工件 / PAICC / 新测试 / 生产数据）。
> 逐条给出**处置**与**验证证据**。原则不变：
> **没有统计显著的正 alpha 证据之前，`deployment.mode = observe`（模拟盘）不得切换。**

## 一、已修复（本轮）

| # | 审计发现 | 严重度 | 处置 | 验证 |
| --- | --- | --- | --- | --- |
| **V-1** | 生产账本 12 天负现金（2026-03-17 −4,850，≈8.7% 隐式杠杆）；`order_executor` 扣款无条件、无购买力检查 | 🔴 已造成实际后果 | `OrderExecutor.execute/execute_orders` 改为**先卖后买**，每笔买入经 `_affordable_shares()` 按可用现金裁剪（预留费用），不足一手整手则跳过；`PaperLedger.cash_stats()` + 影子状态 `equity.cash_guard` 持续披露历史负现金天数 | `tests/test_cash_guard.py` 7 用例；部署口径重跑 `flat_3p5` **+15.35%/1.57 → +16.36%/1.66**（无崩溃、无退化） |
| **A-2** | 实时 `_limit_pct` 日期盲、北交所按 10% → 10–30% 跌幅被误判"跌停不可卖" | 🔴 会误判真实卖单 | `_limit_pct` 委托共享的 `limit_locked.board_limit(symbol, date)`；`_limit_down_price` 传入日期 | `tests/test_live_trader_guards.py`：北交所 0.295、创业板 2020-08-24 前 0.095/后 0.195；主板跌停价 10.86 |
| **A-2b** | 无时间戳的报价 fail-open（放行），且测试把该行为钉成正确 | 🟠 | 改为 **fail-closed**：无可用时间戳 → 不决策；测试改为断言"必须拒绝" | 同上 |
| **A-1** | `fills.source` 迁移代码存在但**历史 252 笔永久为空**；`test_provenance_labels_are_known` 因此恒真 | 🟠 | 账本列已在（下次打开自动补），断言改为**要求列存在**且区分"有 intraday / 有 close 成交"再审计 | `tests/test_fill_time_precision.py` 6 用例（全部先断言样本存在） |
| **A-3** | D-1/D-2 的**调用点**未被测试锁定（删掉 `merge_targets()` 或 `scale_getter` 接线 → 全套绿） | 🟠 回归风险 | 新增 `tests/test_call_sites.py`：真实调用 `build_preclose_orders`（mock 重 I/O）断言退出持仓出现在 15:00 委托里；真实调用 `_build_account_portfolio` 断言 `gross_scale_wired=True` 且 halt 下权重全 0 | 2 用例 |
| **A-4** | `assert_simulated_only` 在 `src/` 零调用者 → 闸门只是声明；且 `bool("false") is True` 会打开闸门 | 🟠 | 新增 `src/live/broker.py`（`PaperBroker` / `RealBroker`）：`RealBroker` 构造即调用闸门 → observe 下**无法实例化**；`deployment_status` 用 `_as_bool` 解析字符串、且 `mode=observe` 一票否决 | `tests/test_deployment_gate.py` 20 用例 |
| **P-1** | 35 日分钟洞（525/800 标的 0 覆盖，全部沪市）但覆盖探针是**日期级**→ `minute_window_coverage=1.0`、`all_passed=true` | 🔴 阻断 | `d_oos.py` 新增**符号级**探针：`minute_min_symbol_coverage`（阈值 0.90）、`low_coverage_days`、`fills_in_low_coverage_days`，并作为断言 `minute_symbol_coverage_ok` / `no_fills_inside_data_hole`；`intraday_frames_loaded` 改用 all+tail_vol；5 个旧工件标记 `data_coverage_invalid: true` | 见下节 OOS 复跑输出 |
| **P-2** | 用 OOS 窗口选止损宽度 → OOS 被消耗，且"先定规则"无预注册 | 🔴 方法论 | 不改数字，改**表述与字段**：`d_oos.py` 输出 `is_candidate_run` / `params_match_production` / `citable`；文档明确 flat_3p5 的 `selection_window: OOS-2025H2`；干净窗口重新定义为**前向 live-only（2026-09-09 起）** | `docs/D_TRACK_EVIDENCE.md` §六/§九 |
| **P-3** | PAICC 无交易日历（节假日照跑） | 🟠 | 新增 `backend/app/services/trading_calendar.py`：2026 已观测休市日（从 HS300 基准日历推导）+ `quant_holidays` 设置覆盖；调度器全部 `weekday()<5` 改为 `is_trading_day()` | `tests/test_quant_calendar.py` 5 用例 |
| **P-4** | 无 live 看门狗（"watchdog" 只是日志 tailer） | 🟠 | 新增每 5 分钟 `live_watchdog` job：会话内检查 `outputs/live_<acc>.pid` + 进程命令行，死亡则前向重启；`quant_manager.live_trader_alive()` | 同上 4 用例 |
| **P-5** | 指纹只覆盖 book 级（缺 cash/成本/code_commit/data_fingerprint） | 🟡 | `d_oos.py` 的对照探针改为**从配置账户构建**（不再用同一份 `--set` 自比较），并纳入 `gross_scale_wired`；kill-switch 档位与生产一致（读 `ControlState`） | `params_match_production` / `assembly_is_production` 两个字段 |
| **P-6** | OOS 跑在无 kill-switch 状态 | 🟡 | 同上：`d_oos.py` 读取账户 ControlState 并传入 `control_scale`，工件记录 `kill_switch.mode/gross_scale` | 同上 |
| **P-7** | `src/deploy.py` 是声明式、无执行点 | 🟡 | 见 A-4（`RealBroker` 成为真实调用点） | 同上 |
| **PAICC-1** | 面板把时间戳当价格渲染（"成交价 09:31:00"） | 🟡 | 改为「成交时刻」 | `npm run typecheck` |
| **PAICC-2** | `quant_weekly_time` 不在 DEFAULTS → 静默用硬编码 18:00 | 🟡 | 补默认值 + 注释 | PAICC 56 测试 |
| **PAICC-3** | preclose 守卫允许到 15:10（竞价后 10 分钟） | 🟠 | 收紧到 **14:40–14:56**（与重试窗口一致） | 同上 |
| **PAICC-4** | 午休 12:00 重启后端会拉起 trader（resume 窗口连续 09:30–15:00） | 🟠 | resume 窗口改为 09:30–11:30 / 13:00–15:00 | 同上 |
| **PAICC-5** | `/quant/stop` 可杀掉正在交易的 trader | 🟠 | `stop_command(force=False)` 默认**保护** `cli.py live` 进程并在返回中报告 `protected_live_trader`；需 `force=true` 才杀 | 同上 |
| **D-8** | 挑战者没有 preclose 制度、晋升闸门数**整本账**成交 | 🟡 | `run_challenger` 接入与生产相同的 14:50 委托层（`preclose_provider`）；`decide_promotion` 只数**评估窗内**成交 | `tests/test_d_cycle.py` + `test_challenger_promotion_gate.py` |
| **D-9** | 重启后调仓相位重置 | 🟡 | `runner` 的相位锚改为窗口首日（`i % rebalance_days`） | 全量测试 |
| **弱测试** | `test_red_lines_applicable` 恒真；`test_fill_time_precision` 大面积空跑；`test_pullback_atr_basis` 因被 clip 到 4% 而空转 | 🟡 | 三个测试全部重写：内容比较 + 输入不可变；先断言样本存在；调整合成波动率使止损宽度落在带内（并断言 `< 0.04`） | 各自用例 |
| **杂项** | `scripts/check_gross_zero.py` 指向已退役 A 轨并会创建空账本；`_basis_factor` 被当作死代码 | 🟡 | 删除该一次性脚本；`_basis_factor` 保留（测试与诊断用）并修正 docstring | — |

## 二、未修复（明确留作后续，附理由）

| # | 项 | 为何本轮不做 |
| --- | --- | --- |
| 1 | ~~回补 2025-10-27→12-12 的分钟数据~~ | ✅ **已完成**（2026-09-09 15:25，收盘后）：`scripts/backfill_minute_gap.py` 80 次 API 调用约 5 分钟，最小覆盖 **273 → 794/800**、低覆盖日 **35 → 0**；重跑 2025 OOS 得 **+1.64%/Sharpe 0.40（t=0.23），13/13 断言通过、`citable=true`**。回补前同一配置为 +8.12%/1.58 —— **数据洞把 OOS 抬高了约 6.5pp**。 |
| 2 | ML 排序器的 `test_window` 覆盖整个 OOS 窗口 | 属模型训练口径，改动会改变现役工件 → 需要重训 + 重新走晋升闸门，不能在盘中做。 |
| 3 | 指纹缺 `code_commit` / `data_fingerprint`（hash 级） | `cash`/成本/上限/universe/数据切片**已补齐**（`_book_fingerprint` 的 `execution` / `universe_size` / `data`）；commit-hash 级指纹留作收尾项。 |
| 4 | `read_trade_records` 仍读整本账计算移动成本 | 与 D-4 相关但不影响正确性；属性能/口径优化。 |
| 5 | `atr_1p0_25_35` / `atr_1p0_25_40` 的 OOS 对比 | 那两个备选是在**回补前**的残缺窗口上比较的；要重评需用回补后数据重跑（下一步）。 |

## 二之二、本轮追加（写完后补充的项）

| 项 | 说明 |
| --- | --- |
| 报价时钟偏移 | `scripts/clock_offset.py`：实测 AlphaFeed 分钟 bar 标签比本地（已与互联网校时，偏差 0 秒）超前约 1–5 分钟；守卫新增"超前 > `max_future_quote_minutes`(10) 不决策"，并保留小幅超前容忍。 |
| 任务卡可读性 | PAICC 的 live/depth/preclose/intraday/challenger/watchdog 六个 job 原先只写操作日志 → 面板 `last_run/last_status` 恒为「—」；现由 `_recording()` 包装记录结果，watchdog 也已 `_stamp`。 |
| PAICC 测试入口 | `README.md` 补充两条零安装跑法；`.venv\Scripts\python.exe -m unittest discover -s tests -t .` 实测 **56 tests OK**。 |

## 三、结论（不变）

- **代码正确性**：本轮修完审计确认的 4 个功能性问题（负现金、实时限价、fail-open 陈旧报价、挑战者制度）与 3 个证据完整性问题（自比较指纹、日期级覆盖探针、恒真断言）。
- **证据可信度**：2025 窗口的 OOS 工件已标记 `data_coverage_invalid`；2026 与 2025H2 两个窗口都已被用于选择 → **部署配置的前向观测数 = 0**。
- **判定**：`deployment.mode = observe`、`real_money_enabled = false` 保持不变；在分钟数据回补 + 符号级覆盖达标 + 前向样本积累之前，任何 D 轨正收益数字（IS、OOS、影子）都不得作为资金决策依据。
