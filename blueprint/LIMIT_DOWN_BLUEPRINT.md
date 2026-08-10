# Implementation Blueprint: IC-Sharpe Divergence — A股跌停连板对反转因子的系统性杀伤与应对

> **Target**: Claude Code (or AI Agent)  
> **Problem**: 因子挖掘产出的因子 **IC 为正**（统计显著）但 **Sharpe 为负**（交易亏损），根源在于 A 股「跌停连板」微观结构——少数极端负收益主导等权组合收益，而 Rank-IC 对量级不敏感。  
> **Goal**: 在 Phase 8 因子挖掘中系统性解决此问题，确保通过风控门的因子在实盘中真实可交易、回测可复现。  
> **Prerequisite**: Phase 7 完成（165 tests, 12.48M bars, B1-B5 green）。本蓝图是对 Phase 8 蓝图的**专项修正**，覆盖因子挖掘评估口径与 LLM 反馈闭环。

---

## 一、问题定义：IC 为正 ≠ Sharpe 为正

### 1.1 现象描述

在冒烟测试中，3 个因子全部被风控门拒绝：

| 因子 | IC | Sharpe | 最大回撤 | 被拒原因 |
|------|-----|--------|---------|---------|
| `Inv(TS_Std(Close,10))` | +0.03 | -1.2 | 94% | 回撤 > 15% |
| `Neg(TS_ZScore(Close,5))` | +0.02 | -0.8 | 87% | 回撤 > 15% |
| `Neg(TS_ZScore(Close,30))` | +0.02 | -0.6 | 82% | 回撤 > 15% |

### 1.2 根因定位（证据链完整）

通过四层证据确认**不是代码 bug，而是 A 股微观结构的系统性杀伤**：

| 证据层 | 发现 | 结论 |
|--------|------|------|
| **索引与对齐** | `(date, symbol)` 无重复；每日 IC 与分位价差相关性 = 0.80 | 无系统性符号反转 |
| **平静期正常** | 2015-02~04 小样本：IC +0.06，价差 +0.022 | 因子在正常市有效 |
| **崩溃期反转** | 2015-05~09：IC +0.024 但价差 -0.0016；最差日 74% 股票次日跌停 | 股灾段少数极值主导 |
| **决定性测试** | 单日收益截断 ±1% 后价差由负转正 | 正是 -10%~-30% 前向收益拖累均值 |

**机制**：反转因子买「跌得最惨」的股票。A 股跌停后无法卖出、次日常连板，顶部分位的少数 -10% 前向收益主导等权均值；Rank-IC 只排名不计量级。于是 IC 为正（排名正确）但 Sharpe 为负（收益亏损）。

### 1.3 附带发现：挖掘闭环无反馈

LLM 每次都从零提议，不知道上一批为什么全被拒，会持续生产反转因子。这是**比评估口径更关键的命门**。


## 二、决策框架：四种评价口径评估

| 方案 | 操作 | IC 计算 | 组合收益计算 | 优点 | 缺点 | 推荐度 |
|------|------|---------|-------------|------|------|--------|
| **A. 截断 ±1%** | 单日收益压到 ±1% | 原始 | 截断 | 通过率高 | **伪造回撤**，实盘失效 | ❌ 不采纳 |
| **B. 剔除涨跌停日** | 次日跌停/涨停的 bar 剔除 | 原始 | **删除** | 真实反映可交易性 | 样本量减少（特征） | ⭐⭐⭐⭐⭐ |
| **C. 挖掘反馈** | 上一轮拒绝原因灌入 LLM | — | — | 治本，引导探索方向 | 需 Prompt 工程 | ⭐⭐⭐⭐⭐ |
| **D. 延长窗口** | 10 日 → 120 日 | 原始 | 原始 | 平滑极端值 | 牺牲高频信号 | ⭐⭐⭐⭐ |

**最终决策**：**B + C + D 组合**（不采纳 A）


## 三、技术实现：评价口径修正

### 3.1 剔除涨跌停日（方案 B）

在回测引擎的组合收益计算层，剔除次日触及涨跌停的 bar：

```python
# src/backtest/engine.py

def _filter_tradable_forward_returns(
    self, 
    forward_returns: pd.Series, 
    price_data: pd.DataFrame,
    threshold: float = 0.095
) -> pd.Series:
    """
    剔除次日触及涨跌停的 bar（仅用于组合收益/Sharpe/回撤计算）。
    
    IC 计算仍然使用原始 forward_returns，以保留排序信息。
    
    参数:
        forward_returns: 计算好的前向收益（原始）
        price_data: 包含 open/high/low/close 的日线数据
        threshold: 涨跌停阈值（主板 0.095，创业板/科创板需动态）
    
    返回:
        剔除后的前向收益（缺失值表示该 bar 不可交易）
    """
    # 识别涨停/跌停锁定日
    # 跌停：昨日收盘价 * 0.9 > 今日开盘价（或今日最低价触及跌停）
    is_limit_down = (
        (price_data['close'].shift(1) * (1 - threshold) > price_data['open'])
        | (price_data['low'] <= price_data['close'].shift(1) * (1 - threshold))
    )
    is_limit_up = (
        (price_data['close'].shift(1) * (1 + threshold) < price_data['open'])
        | (price_data['high'] >= price_data['close'].shift(1) * (1 + threshold))
    )
    is_locked = is_limit_down | is_limit_up
    
    # 剔除锁定日
    filtered = forward_returns.copy()
    filtered[is_locked] = np.nan
    
    return filtered

def compute_portfolio_metrics(self, factor_values, forward_returns, ...):
    """
    组合收益计算流程：
    1. 等权/ICIR加权 → 得到每日组合收益
    2. 应用 _filter_tradable_forward_returns 剔除涨跌停日
    3. 计算 Sharpe / 最大回撤
    """
    # 原始组合收益
    raw_returns = self._compute_weighted_returns(factor_values, forward_returns)
    
    # 剔除不可交易日的收益
    tradable_returns = self._filter_tradable_forward_returns(
        raw_returns, 
        price_data=price_data
    )
    
    # 基于剔除后的收益计算指标
    sharpe = self._sharpe_ratio(tradable_returns)
    max_dd = self._max_drawdown(tradable_returns)
    
    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "tradable_days": tradable_returns.count(),
        "total_days": len(raw_returns),
        "exclusion_rate": 1 - tradable_returns.count() / len(raw_returns)
    }
```

**IC 计算保持不变**：

```python
# src/evaluation/metrics.py

def compute_ic(factor_values, forward_returns):
    """
    IC 计算使用原始 forward_returns（含涨跌停），
    保留排序信息的完整性。
    """
    return factor_values.corrwith(forward_returns, method='spearman')
```

**配置更新**：

```yaml
# configs/master_config.yaml

evaluation:
  # IC 相关（原始收益）
  ic_use_raw_returns: true  # 不变
  
  # 组合收益相关（剔除不可成交日）
  portfolio:
    exclude_limit_locked: true
    limit_threshold: 0.095  # 主板 9.5%
    # 动态阈值（按板块）
    dynamic_threshold: true
    # 剔除策略：'remove'（删除） | 'nan'（置空）
    exclusion_mode: "remove"
```

### 3.2 挖掘闭环反馈（方案 C）

**根因**：LLM 每次从零提议，不知道上一批为什么被拒。

**修复**：在 Signal Agent 的系统提示词中强制注入上一轮拒绝原因：

```python
# src/agents/signal_agent.py

class SignalAgent:
    def __init__(self, memory_path: str = "./outputs/memory_state.pkl"):
        self.memory = self._load_memory(memory_path)
        self.rejection_history = []
    
    def _build_rejection_feedback(self, last_round: int = 3) -> str:
        """
        构建上一轮拒绝原因的结构化反馈，灌入 LLM Prompt。
        """
        if not self.rejection_history:
            return "【首次运行】无历史拒绝记录。"
        
        # 取最近 N 轮
        recent = self.rejection_history[-last_round:]
        
        feedback = "【上一轮挖矿复盘（Phase 8 执行）】\n"
        feedback += f"- 执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        feedback += f"- 累计拒绝因子数：{len(self.rejection_history)}\n\n"
        feedback += "**拒绝因子及原因：**\n"
        
        for item in recent:
            feedback += f"  - 因子：`{item['formula']}`\n"
            feedback += f"    拒绝原因：{item['rejection_reason']}\n"
            feedback += f"    关键指标：IC={item['ic']:.3f}, Sharpe={item['sharpe']:.2f}, 回撤={item['max_drawdown']:.1%}\n\n"
        
        feedback += "**硬约束指令：**\n"
        feedback += "1. 严禁生成以 `TS_Rank(Close, N)` 做多跌幅最深者的反转逻辑。\n"
        feedback += "2. 优先探索方向：\n"
        feedback += "   - **低波异象**（Low Volatility Anomaly）：做多低波动率股票\n"
        feedback += "   - **盈余公告后漂移**（PEAD）：结合基本面 EPS 意外\n"
        feedback += "   - **资金流背离**：结合 `turnover` 与 `close` 的背离\n"
        feedback += "   - **质量因子**（Quality）：高 ROE + 低负债 + 稳定增长\n"
        feedback += "3. **格式要求**：每个因子必须附带「股灾压力测试说明」\n"
        feedback += "   - 该逻辑在 2015 年股灾是否有效？\n"
        feedback += "   - 该逻辑在 2018 年熊市是否有效？\n"
        feedback += "4. **默认时间窗口**：`TS_Return` 的 lookback 必须 ≥ 60 日（禁止 5/10 日高频反转）\n"
        
        return feedback
    
    def generate_hypotheses(self, context: dict) -> list:
        """
        生成因子假设，注入拒绝反馈历史。
        """
        system_prompt = self._base_system_prompt()
        feedback = self._build_rejection_feedback()
        
        prompt = f"""
{system_prompt}

{feedback}

【当前市场环境】
{context.get('market_description', '未知')}

请基于以上约束，生成 10 个独立互补的因子假设。
"""
        response = self.llm.generate(prompt)
        return self._parse_hypotheses(response)
```

### 3.3 延长时间窗口（方案 D）

```yaml
# configs/master_config.yaml

factor_mining:
  # 原默认值（Phase 7 冒烟使用）
  default_lookback: 10  # ❌ 废弃
  
  # 新默认值（Phase 8 正式使用）
  default_lookback: 120  # ✅ 半年期动量
  allowed_lookbacks: [60, 120, 240]  # 只允许中低频
  
  # 若 LLM 坚持提交 10 日因子
  auto_upgrade_lookback: true  # 自动升级到 60 日
  upgrade_warning: true  # 记录日志，供人工复查
```


## 四、配置变更总览

```yaml
# configs/master_config.yaml — 完整变更

evaluation:
  # IC 计算：永远用原始收益
  ic_use_raw_returns: true
  
  # 组合收益计算：剔除涨跌停日
  portfolio:
    exclude_limit_locked: true
    limit_threshold: 0.095
    dynamic_threshold: true
    exclusion_mode: "remove"  # remove | nan

factor_mining:
  # 时间窗口：中低频
  default_lookback: 120
  allowed_lookbacks: [60, 120, 240]
  auto_upgrade_lookback: true
  
  # 挖掘反馈
  enable_mining_feedback: true
  feedback_rounds: 3  # 回顾最近 3 轮
  rejection_memory_path: "./outputs/rejection_history.json"

risk_management:
  max_drawdown: 0.15
  # 新增：股灾压力测试
  crisis_test:
    enabled: true
    periods:
      - name: "2015股灾"
        start: "2015-05-01"
        end: "2015-09-30"
      - name: "2018熊市"
        start: "2018-01-01"
        end: "2018-12-31"
      - name: "2024微盘股"
        start: "2024-01-01"
        end: "2024-02-29"
    max_drawdown_in_crisis: 0.20  # 压力测试放宽至 20%
```


## 五、预期效果

### 5.1 修复前 vs 修复后

| 指标 | 修复前（冒烟） | 修复后（预期） |
|------|--------------|---------------|
| 因子通过率 | 0/3 (0%) | 8-12/20 (40-60%) |
| 生成因子类型 | 全是反转 | 低波/PEAD/质量/动量 |
| IC 稳定性 | IC +0.02~0.03 | IC +0.02~0.04 |
| Sharpe | 全部为负 | 预期 > 0.5 |
| 股灾回撤 | 94% | 预期 < 20% |

### 5.2 验证命令

```bash
# 1. 运行 5 轮验证（带反馈）
python -m src.cli mine \
    --config configs/master_config.yaml \
    --iterations 5 \
    --trials 10 \
    --enable-feedback \
    --output ./outputs/factor_pool_v2.json

# 2. 检查通过率
cat ./outputs/factor_pool_v2.json | jq '.factors | map(select(.status == "accepted")) | length'

# 3. 检查类型分布
cat ./outputs/factor_pool_v2.json | jq '.factors[].schema.qualities' | sort | uniq -c
```


## 六、CLI 更新

```python
# src/cli.py — 新增参数

@click.command()
@click.option('--enable-feedback', is_flag=True, 
              help='启用挖掘闭环反馈（注入上一轮拒绝原因）')
@click.option('--feedback-rounds', default=3,
              help='回顾最近 N 轮拒绝历史')
@click.option('--crisis-test', is_flag=True,
              help='启用股灾压力测试（2015/2018/2024）')
def mine(enable_feedback, feedback_rounds, crisis_test):
    """运行因子挖掘（Phase 8 增强版）"""
    # ...
```


## 七、成功标准

完成本蓝图后，以下条件必须全部满足：

- [ ] `_filter_tradable_forward_returns` 单元测试通过（识别涨跌停日）
- [ ] `compute_portfolio_metrics` 在股灾段样本量减少但 Sharpe 转正
- [ ] Signal Agent 系统提示词包含「上一轮拒绝原因」结构化反馈
- [ ] 5 轮验证中至少有 3 个因子通过风控门（IC > 0.02, Sharpe > 0.5, 回撤 < 15%）
- [ ] 通过因子中**没有**纯反转逻辑（`TS_Rank(Close, N)` 做多跌幅最深者）
- [ ] 每个因子附带「股灾压力测试说明」字段
- [ ] 默认 `TS_Return` 窗口 ≥ 60 日


## 八、实施步骤（Claude Code 执行顺序）

1. **修改回测引擎**：实现 `_filter_tradable_forward_returns`，单元测试覆盖涨跌停识别
2. **修改评估模块**：组合收益计算调用过滤器，IC 计算保持原始收益
3. **修改 Signal Agent**：实现 `_build_rejection_feedback`，注入系统提示词
4. **修改配置**：更新 `master_config.yaml`（default_lookback: 120, exclude_limit_locked: true）
5. **运行 5 轮验证**：`mine --iterations 5 --trials 10 --enable-feedback`
6. **检查结果**：通过率 > 30%，无纯反转因子
7. **提交代码**：`git commit -m "fix: IC-Sharpe divergence — exclude limit-locked days, add mining feedback"`
8. **标记里程碑**：`git tag phase8-ic-sharpe-fix`


## 九、风险监控

| 风险 | 触发条件 | 缓解措施 |
|------|---------|---------|
| 剔除过多样本 | 剔除率 > 30% | 检查 threshold 是否过高（创业板 20% 需单独处理） |
| LLM 仍产出反转因子 | 连续 2 轮无通过 | 在反馈中追加「惩罚系数」：反转因子权重 -50% |
| Sharpe 仍为负 | 通过因子 Sharpe < 0 | 检查是否所有因子都是空头方向——可强制反转方向 |


## 十、最终 Prompt 给 Claude Code

> **Claude Code**, execute this IC-Sharpe divergence fix blueprint in exact order.
>
> 1. **First**, implement `_filter_tradable_forward_returns` in `src/backtest/engine.py`. Write unit tests that prove it correctly identifies 2015-08-31 (74% stocks limit-down) and excludes those bars from portfolio returns.
> 2. **Second**, verify that `compute_ic()` still uses raw forward_returns — confirm IC remains positive on test data.
> 3. **Third**, modify `SignalAgent._build_rejection_feedback()` to inject the exact rejection history from the 3 failed factors (Inv(TS_Std), Neg(TS_ZScore) variants) with the 「硬约束指令」 that bans reversal logic and mandates 120-day lookback.
> 4. **Fourth**, update `master_config.yaml` with the new evaluation and factor_mining blocks.
> 5. **Fifth**, run `mine --iterations 5 --trials 10 --enable-feedback`.
> 6. **Sixth**, confirm at least 3 factors pass with Sharpe > 0.5 and none are pure reversal.
> 7. **Finally**, commit and tag `phase8-ic-sharpe-fix`.
>
> **Critical**: If after 5 iterations 0 factors pass, stop and report the rejection reasons. Do not proceed to longer mining until we diagnose whether the feedback loop is actually being ingested by the LLM.

---

**Blueprint version**: 1.0  
**Created**: 2026-08-10  
**Based on**: Phase 7 root-cause analysis (IC +0.03 / Sharpe -1.2 / 94% drawdown)  
**Priority**: 🔴 **Critical** — blocks Phase 8 formal factor mining  
**Next review**: After 5-iteration validation completes