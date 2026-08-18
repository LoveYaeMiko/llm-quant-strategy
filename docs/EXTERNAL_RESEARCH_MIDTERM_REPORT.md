# 中长期三项落地 + 模拟盘长期测试可行性评估

> 提交：`73e797a` · 全套测试 **334 passed / 1 failed**（唯一失败为 Phase 8 预存在
> 验收项 `test_phase8_remedy_success_criteria`，依赖静态 `outputs/factors.json` 产物，
> 与本次无关）。

---

## 一、中长期三项落地（对应 EXTERNAL_RESEARCH §六.5）

三项全部 **config-gated 默认 OFF**，保持既有行为与既有测试全绿，与 A1/A4/A2 的
落地方式一致——这是本项目的落地纪律：**新门控默认不改变存量决策**，仅当在
`configs/factor_thresholds.yaml` 显式打开时才介入。

### 1. 记忆回路（AlphaMemo residual/veto memory）

**问题**：此前闭环只有「负向 rejection feedback」——被拒因子被记忆并反哺，但
「什么样的编辑能提升 IC」这一正向信号没有回到 writer 侧。

**落地**：

| 文件 | 内容 |
| --- | --- |
| `src/factors/residual_memory.py` | 以 `(category, edit_motif)` 为键的残差记忆：`residual = child_rank_ic − parent_baseline`；置信度 = `count_gate × certainty(entropy) × variance_penalty`；高失败 cell 进入 veto |
| `src/evolver.py` | `propose_edits` 阶段 veto 过滤 + `evolve` 阶段写入残差，关闭 writer→judge→feedback **双向**回路 |
| `src/agents/signal_agent.py` | `_build_positive_feedback()` 把 `residual_memory.top_cells` 的胜出 edit motif 反哺进 prompt |

### 2. 邻域选参（A3, EP004 reject-argmax）

**问题**：EvoQuant 纯 argmax——只看「IC 最高的那个候选」，无法区分「高原」与
「尖峰」，也未记录 in/out-sample 排名可外推性。

**落地**：

| 文件 | 内容 |
| --- | --- |
| `src/backtest/metrics.py` | `neighborhood_plateau`（与最高 IC 相差 < band 的邻近变体占比） + `in_out_ic_correlation`（train/valid IC 排名相关性 r，过拟合早期信号） |
| `src/evolver.py` | config-gated argmax 拒绝；`result.metrics` 新增 `plateau_support / is_plateau / in_out_ic_r` |

### 3. 辩论评审（TradingAgents Bull/Bear 对抗 + AgenticAITA）

**问题**：RiskAgent 是单趟确定性门控，没有「空方质疑」这一步——一个过了 risk gate
的因子，可能只是因为多方证据恰好够线。

**落地**：

| 文件 | 内容 |
| --- | --- |
| `src/agents/debate_agent.py` | 确定性多空证据 net 出 `margin = bull − bear`；LLM（deep tier，B2）仅做可选自然语言综合，**pass/fail 永不依赖 LLM**（保持离线可跑） |
| `src/cli.py` | 接受环内 config-gated `debate.require_pass` 门控，未过判 `reject_debate` |

**测试**：新增 22 项（residual_memory 7 + neighborhood 8 + debate 7），全绿。

---

## 二、模拟盘长期测试可行性评估

### 2.1 已具备（架构层面 ready）

| 能力 | 现状 |
| --- | --- |
| 因子生成→评估→门控→池 | 全链路就绪；A1/A4/A2 + 本次三项补齐了严谨性缺口 |
| 三层组合仿真 | Phase 10 gate **PASS**（`ec4bec7`，tag `phase10-ready-for-deployment`）：Sharpe **1.704**、ann 13.3%、maxDD **9.2%**、换手 0.19、中性 book β≈0；全样本复用现有缓存，**无需新增数据接口** |
| 在线确定性执行组件 | `src/online/`：`signal_calculator.compute_signal`（公式→日度信号）→ `portfolio_optimizer.optimize`（PCA 中性化 + 相关断裂分散，date×symbol 权重）→ `order_executor.execute`（滑点 2bps / 佣金 5bps / 仓位上限 5%，确定性成交） |
| 数据源 | AlphaFeed 主行情 + AKShare 新闻/舆情 + Baostock/AKShare 免费备用，PIT store 因果约束 |
| 运维 | C1 事件流 + C2 结构化日报 + webhook 通知（已落地） |

### 2.2 缺口（模拟盘的最后一块拼图）

1. **无模拟盘日度运行器/调度器**。`src/online/` 三模块是「积木」，但**没有一个
   `paper` 命令**把 signal→optimize→execute→PnL 串成日度循环并按日推进。这是
   「回测通过」→「模拟盘长期测试」之间唯一缺失的一公里。

2. **持仓/资金状态不跨日持久化**。`OrderExecutor` 的 `positions/cash` 是内存态，
   进程退出即失；模拟盘需要跨日持久化（JSON 或 SQLite）+ 恢复/审计。

3. **实时时点对齐**。舆情风控与 PEAD 依赖真实时点数据流（AKShare 实时舆情流 +
   真实 SUE 财报时点）；当前回测用 PIT 缓存，模拟盘需切换到 T+0/T+1 实际可用时点。

### 2.3 结论与建议

**结论：架构层面已具备启动模拟盘长期测试的条件，但缺一个「模拟盘日度运行器」
组件。** 该组件由三块组成，体量小、复用现有确定性模块，无新数据接口：

```
paper runner  =  日度循环  (signal_calculator → portfolio_optimizer → order_executor)
               + 持仓/资金持久化  (JSON / SQLite，重启可恢复)
               + 日度 PnL/日报  (复用 C2 build_digest + WebhookNotifier)
```

**§7 三项遗留不阻塞启动，反而正是「长期测试」要解决的问题**：

1. PEAD 倾斜幅度默认关闭 → 模拟盘月度调仓 + 真实 SUE 时点重估；
2. 舆情风控阈值（-2.5σ / 5 天冻结）→ 按实盘舆情流校准；
3. 成本模型固定换手率（0.18-0.19）→ 真实佣金/冲击重估。

**建议**：补 `src/paper/`（paper runner + 持久化）后即可启动模拟盘长期测试。
启动本身不涉及新的决策风险——三层组合已过 gate，运行器只是把已验证的确定性
路径接到日度时钟上。

---

## 三、下一步

- [ ] 落地 `src/paper/`（paper runner + 持仓持久化 + 日度调度），补齐最后一公里；
- [ ] 模拟盘先跑「影子模式」（不实盘下单，只记录每日目标持仓与 PnL），积累
      §7 三项校准所需数据；
- [ ] 用积累数据重评 PEAD 倾斜幅度 / 舆情阈值 / 成本模型。
