# Phase 8 补救验证报告（validation_BLUEPRINT 融合版）

> **日期**: 2026-08-10
> **状态**: ❌ **未通过门控（5/20 = 25% < 30%）** —— 按蓝图规则停止，不打 tag `phase8-remedy-complete`，不进入 Phase 9，不自行调参。
> **蓝图依据**: `blueprint/validation_BLUEPRINT.md` §3.5/§3.6/§7（通过率 ≥ 30%（6/20）→ commit+tag；< 30% → 停止并报告）

---

## 一、背景

Phase 8 因子挖掘初版验证 **0/20** 全被 `reject_high_risk` 拒绝。根因双重：

1. **回撤门结构性矛盾**：15% 绝对回撤门与十年 A 股长-短组合（beta≈0）矛盾；
2. **LLM 失控**：DeepSeek-v4-flash 无视 Prompt 硬约束，跨迭代重复提交被拒公式（20 槽位仅 ~7 个不同公式）。

融合版补救（validation_BLUEPRINT）吸收了两项改进：
- **超额回撤口径**（§3.3）：相对基准计算超额回撤；
- **因子组合模板**（§3.2）：确定性双因子等权组合模板槽 + 自由 LLM 槽。

---

## 二、本阶段决策链（均经用户确认）

| 轮次 | 决策 | 结果 |
|---|---|---|
| 1 | 「修复重复并重跑」 | 模板槽禁止重提已测公式；修后 0/20 → 9/20 |
| 2 | 「降 floor 到 0.25 并去重」 | 修复自由槽/模板槽双向重复 + diversity 地板 0.40→0.25 |
| 3（本次） | 「修 Bug 并重跑」 | 修复跨轮 rejection_history 污染锁死模板池；run7 重跑 |

### 本阶段关键修复

1. **超额回撤门回退为绝对回撤门**（`master_config.yaml` 注释 + `cli.py`）：超额回撤指标对市场中性多空组合结构上不可通过——纯现金仓位也会因基准自身波动（HS300 2013 年谷 -41% / 2015 年峰 +51%）而测得 excess_dd ≈ 1.03。移除 `benchmark`/`max_excess_drawdown` 后 eval/risk 门回退到绝对 15% 门。
2. **模板重定向**：alpha 引擎实证只有「低波 + 低换手」家族在真实 HS300 下有 alpha。模板池 4 个方向全部 Volume 锚定，槽 0 每轮保留给已验证的 `Avg(Neg(Rank(TS_Std(Close,·))), Neg(Rank(TS_Mean(Volume,·))))`。
3. **双向去重**：`accepted_ever` 进模板 `skip`（模板→自由方向）+ 验收时 `continue`（自由→模板方向），杜绝同一公式二进池。
4. **diversity 地板 0.40 → 0.25**（`factor_thresholds.yaml`，因 merge 顺序后文件覆盖前文件必须改这里而非 master_config）。
5. **跨轮 rejection_history 污染修复**（本次）：见第四节。

---

## 三、run6：清单全绿但契约缩水（5/14）

首次修复后重跑：

- 验证清单 4 项全绿：pit ✅ / fincad ✅ / diversity ✅（0.25）/ cost ✅，EXIT=0。
- 但**模板槽在第 2/4 轮返回 0 个候选** → 只测了 14 个而非契约的 20 个 → 接受 5/14 = 35.7%（若按实际候选分母计算）。
- `test_phase8_remedy_success_criteria`（§3.5，按 20 契约硬断言 ≥6）**失败**。

---

## 四、run7：契约恢复，通过率仍不足（5/20）

### 4.1 根因诊断：跨轮 rejection_history 污染（代码 Bug，非阈值问题）

- `outputs/rejection_history.json` **跨 `cmd_mine` 调用持久化**：run5+run6 累积 **95 条 / 43 个唯一公式**。
- `SignalAgent.__init__` 启动时加载该文件（`signal_agent.py:62-68`），`generate_template_formulas` 把它**全部**加入模板池 blocked（`signal_agent.py:192`）。
- 结果：**run6 启动时 24 个模板池已有 20 个被陈旧历史锁死**，第 2 轮起池子耗尽 → 模板槽返回 0。
- 复现：用真实 `rejection_history_path` 重放，得到与 run6 一致的 4/10 模板槽产出（2,2,0,0,0）。

### 4.2 修复（约 3 行，非阈值调参）

`src/cli.py` `cmd_mine` 迭代循环前：

```python
signal.rejection_history = []
```

拒绝历史改为**当轮作用域**：模板池阻塞与挖掘反馈 prompt 都只反映本轮拒绝；`record_rejection` 每次覆写文件，磁盘历史同步自愈。

回归测试：`tests/test_phase8_remedy.py::test_stale_rejection_history_must_not_starve_template_pool`
（先复现陈旧历史锁死池，再验证重置后 5 轮 × 2 模板槽 = 10/10 全产出）。

### 4.3 run7 结果

| 检查项 | 结果 |
|---|---|
| 候选契约 | ✅ **20/20**（修复生效，无饿死） |
| pit | ✅ PASS（2023-12-31 无未来事实） |
| fincad | ✅ PASS（作弊因子 IC 1.000 → 0.000，抑制 100%） |
| diversity | ✅ PASS（最小成对 AST 距离 0.25 ≥ 0.25） |
| cost | ✅ PASS（月成本 $0.02 ≤ $500） |
| **通过率（§3.5）** | ❌ **5/20 = 25% < 30%（需 ≥6）** |
| 反转因子 | ✅ 0 个 |
| 组合模板来源 | ✅ 2 个（≥2） |

被接受 5 因子全部来自已验证的低波+低换手家族（rank_ic 0.0205–0.0225，icir 2.5–2.8），成对 AST 距离恰为 0.25（同家族 lookback 变体）。

---

## 五、诊断：为什么只接受 5/20

- **只有「低波 + 低换手」家族能同时通过 IC 门（≥0.02）与 Sharpe/回撤门**。
- 模板池另两个方向全部 `reject_high_risk`：
  - **低波+缩量**（`Avg(Neg(Rank(TS_Std)), Neg(Rank(TS_Delta(Volume))))`）：rank_ic 高达 0.030–0.033，但 **Sharpe 仅 -0.07~0.25**（缩量不是有效信号），被回撤门拒；
  - **双低换手**（Volume-only）：rank_ic 0.013–0.016 **低于 IC 门**。
- 自由 LLM 槽 10 个里 3 个命中该家族并过门，其余多为反转/动量垃圾（`Rank_Mul`、`Rank(Volume)`）被拒。
- 该家族 lookback 变体恰好压在 0.02 门线上方/下方，5/10 模板槽里 2 个、自由槽里 3 个过线。

**结论**：这是真实的边际 alpha，不是机制故障。任何「追满 6 个」都只能靠降 IC 门（0.02）——那正是蓝图明令禁止的自行调参。

---

## 六、门控判定与行动

- **§3.5 通过率不足：5/20 < 6/20** → **门控未通过**。
- 按蓝图 §7：「通过率 < 30% → 停止并报告（不要自行调参）」。
- 行动：**不打 tag `phase8-remedy-complete`，不进入 Phase 9**；本报告存档，代码修复保留在工作区。

---

## 七、本阶段文件变更

| 文件 | 变更 |
|---|---|
| `src/cli.py` | `accepted_ever` 双向去重；`signal.rejection_history = []` 当轮作用域（跨轮污染修复） |
| `configs/factor_thresholds.yaml` | `diversity.min_ast_distance` 0.40 → 0.25（必须改此文件，merge 后文件覆盖前文件） |
| `configs/master_config.yaml` | 删除被遮蔽的 diversity 块，加 merge 顺序说明；回退超额回撤门 |
| `src/checklist.py` | diversity_check 默认地板 0.25 |
| `src/agents/signal_agent.py` | 模板槽保留槽 0 给已验证家族 |
| `src/factors/schema/validator.py` | COMBINATION_TEMPLATES 全部 Volume 锚定 |
| `tests/test_phase8_remedy.py` | 双向去重 + 跨轮污染回归测试 |

---

## 八、下一步选项（供用户决策）

1. **接受现状停止**：Phase 8 补救未过门控，存档报告，暂缓 Phase 9。可另行调研低波+低换手家族的表达空间。
2. **人工复核门控口径**：若认为「实际候选分母」更合理（run7 5/14 亦为 5，未到 6），或需调整验证契约，需用户明确授权改蓝图。
3. **换策略继续**：不改阈值的前提下，可探索合法提升通过率的机制手段（如模板池增加其他**实证有 alpha** 的家族——需先跑验证），再重跑。这属于机制改进而非调参，但需用户批准后执行。

---
*报告完毕。按蓝图规则，未打 tag、未进入 Phase 9、未调参。*
