# Phase 8 报告 — IC-Sharpe 背离修复（LIMIT_DOWN 蓝图）

> 项目：FQA · 阶段：Phase 8（因子挖掘 · 修复专项）
> 日期：2026-08-10 · 分支：main · 测试：208 passed

## 0. 关键发现：IC 为正 ≠ Sharpe 为正（A股跌停连板的系统性杀伤）

Phase 8.1 冒烟批 3/3 因子、首轮 5 迭代验证批 20 个因子**全部**被风控门拒绝
（`reject_high_risk`：回撤 > 15%）。初步怀疑为数据缺陷，但经四层证据证明
**不是代码 bug，而是 A 股微观结构的系统性杀伤**：

| 证据层 | 发现 | 结论 |
|--------|------|------|
| **索引与对齐** | `(date, symbol)` 无重复；每日 IC 与分位价差相关性 ≈ 0.80 | 无系统性符号反转，代码对齐正确 |
| **平静期正常** | 2015-02~04 小样本：IC +0.06，价差 +0.022 | 因子在正常市有效 |
| **崩溃期反转** | 2015-05~09：IC +0.024 但价差 −0.0016；最差日 74% 股票次日跌停 | 股灾段少数极值主导等权均值 |
| **决定性测试** | 单日收益截断 ±1% 后价差由负转正 | 正是 −10%~−30% 前向收益拖累均值 |

**机制**：反转因子买「跌得最惨」的股票。A 股跌停后无法卖出、次日常连板，
顶部分位的少数 −10% 前向收益主导等权均值；而 Rank-IC 只排名不计量级。
于是 IC 为正（排名正确）但 Sharpe 为负（收益亏损）。
附带发现：挖掘闭环无反馈——LLM 每次从零提议，不知道上一批为什么全被拒，
会持续生产反转因子（这是比评估口径更关键的命门）。

> 备注：早期版本中 `pct_change(fill_method='pad')` 的停牌缺口填充缺陷（复牌日
> 伪造 +1970% 离群值）**是真实 bug**（已修复，`fill_method=None`，新增回归测试），
> 但它**不是**系统性 0/20 拒绝的根因——修复后 20 个因子仍全部被拒。

## 1. LIMIT_DOWN 蓝图实施方案（B + C + D）

按 `blueprint/LIMIT_DOWN_BLUEPRINT.md`（决策框架：B 剔除涨跌停日 / C 挖掘反馈 /
D 延长窗口组合，明确不采纳 A 截断 ±1%——伪造回撤）：

| 方案 | 实施位置 | 内容 |
|------|---------|------|
| **B** | `src/backtest/limit_locked.py`（新） | `limit_lock_mask` 识别涨跌停锁定 bar（open/close 触及昨收 ± 阈值；主板 0.095 / 创业板 0.195 自 2020-08-24 / 科创板 0.195 / 北交所 0.295，按板块按日期动态）。`tradeable_forward_returns` 将「入场日锁定 **或** 出场日锁定」的 bar 置 NaN（不可真实成交的 −10% 连板不得主导组合）。IC/rank_ic 仍用原始前向收益，保留排序信息。 |
| **B 接线** | `src/cli.py`、`src/agents/eval_agent.py`、`src/pool.py` | `_attach_tradable_forward` 在 `_market_data` 后挂载 `forward_returns_tradable`；`evaluate(..., forward_tradable=...)` 组合收益用可交易序列；危机窗口切片同样用可交易序列。 |
| **C** | `src/agents/signal_agent.py` | 每次被拒因子（formula/verdict/IC/Sharpe/回撤）写入 `outputs/rejection_history.json` 并重放进下一轮 LLM Prompt（最近 3 轮），附「硬约束指令」：严禁 `TS_Rank/TS_ZScore(Close, N)` 反转逻辑、优先低波/PEAD/资金流背离/质量、TS_Return lookback ≥ 60、必须附带股灾压力测试说明。 |
| **D** | `src/factors/code_generator.py`、`src/agents/code_agent.py` | `default_formula_for` 窗口改为 60/120/240 三档；LLM 提交的 `TS_*` lookback < 60 由 `bump_lookbacks` 自动升级到 60（`auto_upgrade_lookback: true`）。 |
| **股灾压力测试** | `src/agents/eval_agent.py`、`configs/master_config.yaml` | 组合回撤必须在 2015 股灾（2015-05-01→2015-09-30）、2018 熊市、2024 微盘股三个危机窗口内均 < 20%，否则 `reject_crisis`。 |

配置：`evaluation.portfolio.exclude_limit_locked: true`、
`factor_mining.{min_lookback: 60, max_lookback: 240, default_lookback: 120,
allowed_lookbacks: [60,120,240], auto_upgrade_lookback: true,
enable_mining_feedback: true, feedback_rounds: 3}`。
CLI：`mine --enable-feedback/--feedback-rounds/--crisis-test`。

**单元测试（蓝图成功标准前两条）**：
- `tests/test_limit_locked.py`（新增 6 例）：识别涨跌停锁定 bar、仅剔除入场/出场
  锁定 bar、2015-08-31 式「74% 跌停日」整体剔除、板块动态阈值、eval-agent 集成
  （rank_ic 两路相同 + 可交易序列组合收益更高 / 回撤更低）。
- `tests/test_feedback.py`（新增 4 例）：拒绝历史持久化 + 重载、反馈文本含拒绝
  公式与「严禁」/「≥ 60」、反馈注入 LLM Prompt（启/停）。
- `tests/test_code_generator.py`（新增 5 例）：`bump_lookbacks` 升级短窗口、嵌套
  TS_* 逐层升级、非 TS 数字字面量不动、≥60 不动；默认公式窗口 ∈ {60,120,240}。

## 2. 5 迭代验证批（带反馈，含股灾压力测试）

验证命令：`mine --iterations 5 --trials 10 --crisis-test`
（5 迭代 × 4 假设 = 20 因子；`--trials 10` = Bonferroni 多重检验；成本 $0.023，25 次 LLM 调用）

- 候选因子总数：**20**
- 通过风控门（IC > 0.02，Sharpe > 0.5，回撤 < 15%）：**0 / 20** ← 触发蓝图 Critical 规则
- 拒绝原因：**20/20 全部 `reject_high_risk`**（可交易长-短组合回撤 > 15%），无一进入 Sharpe/显著性门。
- 通过因子中纯反转逻辑：0（无通过因子）。

**最接近通过的 4 个因子**（IC 与 Sharpe 均为正，仅回撤 22%~29% 超限）：

| 因子 | IC | Sharpe | 最大回撤 | 备注 |
|------|-----|--------|---------|------|
| `Neg(TS_ZScore(Close, 240))` | 0.019 | 0.48 | 22% | 长「跌得深」反转，被提 3 次 |
| `Neg(TS_ZScore(Close, 120))` | 0.023 | 0.40 | 27% | 反转 |
| `Inv(TS_Std(Close, 240))` | 0.012 | 0.36 | 27% | 低波 |
| `Neg(TS_ZScore(Close, 60))` | 0.026 | 0.19 | 29% | 反转 |

**诊断（蓝图 Critical 规则的「反馈闭环是否被摄入」检查）**：
1. **方案 B 机制验证通过**：剔除率 2.0%（真实 A 股水平），可交易序列保留 96.6% bar；
   股灾窗口回撤（如 2015 反转因子 5.2%）远低于修复前的崩溃段——跌停连板杀伤已从
   组合收益中移除。**修复前**同样的因子回撤 82%~94%，**修复后**降至 22%~29%。
2. **瓶颈是 15% 主回撤门**：2010-2019 十年 A 股长-短十分位价差上，全部 20 个因子
   回撤 22%~97%。15% 门对十年窗口过严（见「结论与下一步」）。
3. **方案 C 反馈未被有效摄入**：LLM（deepseek-v4-flash）无视硬约束，跨迭代**逐字重复**
   已被拒的公式——`Neg(TS_ZScore(Close, 240))`×3、`TS_Rank(TS_Return(Close,240),240)`×4、
   `TS_Rank(TS_Return(Close,60),60)`×3、`Inv(TS_Std(Close,240))`×2（20 槽位仅 ~7 个不同公式）；
   **从未提出** PEAD / 质量 / 资金流背离因子。20 个因子中纯反转/动量/流动性占绝大多数，
   `TS_Rank(TS_Return(...),N)`（纯反转）与 `Rank_Mul(Rank(Close),Rank(TS_Return(...)))`（动量）
   均为负 IC、负 Sharpe、回撤 79%~97%。LLM 路径无公式去重（`memory.has_formula` 只拦离线路径）。

## 3. 结论与下一步

- **0/20 通过 → 按蓝图 Critical 规则停止**，不启动更长挖掘。
- 修复本身（方案 B + D）机制验证有效，是评估口径的正确修正；失败点在**方案 C 的执行**
  （反馈未被摄入）与**主回撤门阈值**。
- 待选补救（蓝图 §9 风险表预置的惩罚机制 + 阈值复核），需用户拍板后执行。

---
## 附录 A：蓝图关键规则执行记录

- **冒烟批（修复前）**：3/3 被拒（回撤 82%~94%）→ 触发诊断。
- **根因**：见 §0（跌停连板 + 无挖掘反馈）。非代码 bug。
- **修复验证批**：0/20 通过（全部 `reject_high_risk`）→ 触发 Critical 规则，停止并上报。
- **正式挖掘**：未启动（等待用户决策补救方案）。
