# Implementation Blueprint: Phase 10 — 模拟盘部署与三层融合

> **Target**: Claude Code (or AI Agent)
> **Current State**: Phase 9 正式关闭。三次失败（PEAD、研报情绪、分歧度/新颖性）确认：HS300 上“已披露信息”单因子预测力均低于 0.015 门槛。文本管线保留为风控模块，PEAD 保留为战术倾斜模块。
> **Goal**: 将 Phase 8 价量因子（低波+低换手）作为 Alpha 核心，叠加文本舆情熔断（风控层）和 PEAD 反转季节性倾斜（战术层），运行 2010–2025 全样本回测，验证三层融合后的 Sharpe > 1.6、最大回撤 < 10%，启动模拟盘部署。
> **Prerequisite**: Phase 8 完成（5 个低波+低换手因子，已验证有效）。Phase 9 代码已提交（TriAgent 缓存、PEAD 基建、文本因子回测框架）。209 tests passed。

---

## 一、Phase 9 归档确认

### 1.1 三次失败总结

| Phase | 因子 | 结果 | 根因 |
|-------|------|------|------|
| 9.2 | PEAD（盈余公告） | ❌ FAIL | A 股 HS300 呈负漂移，反转仍不过门 |
| 9.1a | 研报情绪 | ❌ FAIL | rank_ic=0.0094 < 0.015 |
| 9.1b | 分歧度/新颖性 | ❌ FAIL | dispersion@20=0.0136，Sharpe=-0.95；novelty 全弱 |

### 1.2 统一根因

> **HS300 上“已披露文本/基本面信息”在发布时已被充分定价**。三条证据线一致，非管线问题，是 A 股机构化的市场特征。

### 1.3 资产重定位

| 组件 | 原用途 | 新用途 |
|------|--------|--------|
| TriAgent + 缓存 | Alpha 生成 | **风控输入**（极端舆情触发熔断） |
| PEAD 基建 | Alpha 生成 | **战术倾斜**（财报季权重微调） |
| 文本因子 | 单因子选股 | **不再参与 Alpha 打分** |


## 二、Phase 10 最终架构：三层融合

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 10: 模拟盘部署架构                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  第一层：Alpha 核心（长期，月度调仓）                      │   │
│  │  ┌───────────────────────────────────────────────────────┐ │   │
│  │  │ 低波 + 低换手 双因子等权组合（5 个因子家族）         │ │   │
│  │  │ • 来源：Phase 8 唯一有效方向                         │ │   │
│  │  │ • 权重：等权（各 20%）                               │ │   │
│  │  │ • 调仓：月度                                        │ │   │
│  │  │ • 目标：赚取风险溢价（年化 ~10%，Sharpe ~1.5）      │ │   │
│  │  └───────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  第二层：战术倾斜（中期，财报季生效）                     │   │
│  │  ┌───────────────────────────────────────────────────────┐ │   │
│  │  │ PEAD 反转季节性调整（±20% 权重倾斜）                  │ │   │
│  │  │ • 触发：财报季（1-2月/4月/8月/10月）                 │ │   │
│  │  │ • 窗口：公告后 5 个交易日                             │ │   │
│  │  │ • 高 SUE → 降权 20%，低 SUE → 升权 20%              │ │   │
│  │  │ • 作用对象：持仓权重 > 3% 的股票                     │ │   │
│  │  └───────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  第三层：风控熔断（短期，任意交易日触发）                  │   │
│  │  ┌───────────────────────────────────────────────────────┐ │   │
│  │  │ 文本舆情极端事件触发                                  │ │   │
│  │  │ • 触发：研报情绪 Z-score < -2.5（历史 5% 分位）     │ │   │
│  │  │ • 动作：该持仓股减仓 50%，持续 5 个交易日            │ │   │
│  │  │ • 优先级：最高（覆盖战术倾斜）                       │ │   │
│  │  └───────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 三层优先级

| 场景 | 执行动作 |
|------|----------|
| 舆情熔断触发 | **覆盖一切**，立即减仓 50%，持续 5 天 |
| 仅 PEAD 倾斜触发 | ±20% 权重调整 |
| 两者同时触发 | 舆情熔断优先，PEAD 倾斜被覆盖 |
| 都不触发 | 保持 Alpha 核心权重 |


## 三、代码实现

### 3.1 文件结构（新增）

```
src/portfolio/
├── __init__.py
├── alpha_core.py          # Phase 8 低波+低换手组合（已有）
├── risk_overlay.py        # 文本舆情熔断（新增）
├── seasonal_tilt.py       # PEAD 反转战术倾斜（新增）
├── layer_integration.py   # 三层融合引擎（新增）
└── backtest_runner.py     # 全样本回测运行器（新增）
```

### 3.2 风险覆盖层（舆情熔断）

```python
# src/portfolio/risk_overlay.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.sentiment.triagent import TriAgentSentiment

class SentimentRiskOverlay:
    """
    文本舆情风控覆盖层
    当持仓股出现极端负面研报时，临时减仓 50%，持续 5 个交易日
    """
    
    def __init__(self, sentiment_store: TriAgentSentiment):
        self.sentiment = sentiment_store
        self.freeze_days = 5
        self.zscore_threshold = -2.5
    
    def apply(self, weights: Dict[str, float], date: str) -> Dict[str, float]:
        """
        对持仓权重应用舆情熔断
        """
        modified_weights = weights.copy()
        
        for symbol, weight in weights.items():
            # 获取该股票历史情绪分布
            hist_scores = self.sentiment.get_historical_scores(symbol, window=252)
            if len(hist_scores) < 20:
                continue
            
            # 最近 3 天情绪得分
            recent_scores = self.sentiment.get_scores(symbol, date, window=3)
            if not recent_scores:
                continue
            
            # 计算 Z-score
            mean = np.mean(hist_scores)
            std = np.std(hist_scores)
            if std == 0:
                continue
            
            zscore = (np.mean(recent_scores) - mean) / std
            
            # 触发熔断：极端负面
            if zscore < self.zscore_threshold:
                # 减仓 50%
                modified_weights[symbol] = weight * 0.5
                # 记录触发日志
                self._log_trigger(symbol, date, zscore, weight, modified_weights[symbol])
        
        return modified_weights
    
    def _log_trigger(self, symbol: str, date: str, zscore: float, 
                     old_weight: float, new_weight: float):
        """记录熔断触发日志"""
        # 写入 audit_store
        pass
```

### 3.3 战术倾斜层（PEAD 反转）

```python
# src/portfolio/seasonal_tilt.py

import pandas as pd
from typing import Dict, List, Optional
from src.factors.pead import PEADFactor

class PEADSeasonalTilt:
    """
    PEAD 反转战术倾斜
    财报季：高 SUE 降权 20%，低 SUE 升权 20%
    """
    
    def __init__(self, pead_factor: PEADFactor):
        self.pead = pead_factor
        self.tilt_amplitude = 0.20  # ±20%
        self.min_weight_for_tilt = 0.03  # 权重 > 3% 才调整
        self.window_days = 5  # 公告后 5 个交易日
    
    def is_earnings_season(self, date: str) -> bool:
        """判断是否处于财报季窗口"""
        # 财报季月份：1-2月（年报）、4月（一季报）、8月（中报）、10月（三季报）
        month = pd.Timestamp(date).month
        if month in [1, 2, 4, 8, 10]:
            return True
        # 具体到日期：公告后 5 个交易日内
        # 简化：财报季月份全月生效
        return False
    
    def apply(self, weights: Dict[str, float], date: str) -> Dict[str, float]:
        """
        应用 PEAD 战术倾斜
        """
        if not self.is_earnings_season(date):
            return weights
        
        # 获取该日期所有股票的 SUE
        sues = self.pead.get_all_sues(date)
        if not sues:
            return weights
        
        modified_weights = weights.copy()
        
        for symbol, weight in weights.items():
            if weight < self.min_weight_for_tilt:
                continue
            
            if symbol not in sues:
                continue
            
            # 计算 SUE 在全市场的分位
            percentile = self._get_percentile(sues[symbol], list(sues.values()))
            
            # 高 SUE（>80% 分位）降权
            if percentile > 0.8:
                modified_weights[symbol] = weight * (1 - self.tilt_amplitude)
            # 低 SUE（<20% 分位）升权
            elif percentile < 0.2:
                modified_weights[symbol] = weight * (1 + self.tilt_amplitude)
        
        return modified_weights
    
    def _get_percentile(self, value: float, all_values: List[float]) -> float:
        """计算 value 在所有值中的分位"""
        if not all_values:
            return 0.5
        return sum(1 for v in all_values if v < value) / len(all_values)
```

### 3.4 三层融合引擎

```python
# src/portfolio/layer_integration.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class ThreeLayerPortfolio:
    """
    三层融合引擎
    Alpha 核心 → 战术倾斜 → 风控熔断
    """
    
    def __init__(self, alpha_core, seasonal_tilt, risk_overlay):
        self.alpha = alpha_core
        self.tilt = seasonal_tilt
        self.risk = risk_overlay
    
    def compute_weights(self, symbols: List[str], date: str) -> Dict[str, float]:
        """
        逐层叠加，返回最终权重
        """
        # 第一层：Alpha 核心
        weights = self.alpha.compute_weights(symbols, date)
        
        # 第二层：战术倾斜
        weights = self.tilt.apply(weights, date)
        
        # 第三层：风控熔断（最高优先级）
        weights = self.risk.apply(weights, date)
        
        # 归一化
        return self._normalize(weights)
    
    def _normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        """权重归一化（总和 = 1）"""
        total = sum(weights.values())
        if total == 0:
            return weights
        return {k: v / total for k, v in weights.items()}
```

### 3.5 配置更新

```yaml
# configs/master_config.yaml

# Phase 10: 模拟盘配置
simulation:
  enabled: true
  start_date: "2010-01-01"
  end_date: "2025-12-31"
  rebalance_frequency: "monthly"
  universe: "hs300"

# Alpha 核心（Phase 8）
alpha_core:
  factors:
    - name: "low_vol_low_turnover"
      count: 5
      weights: "equal"
      rebalance: "monthly"

# 战术倾斜（Phase 9.2 产物）
seasonal_tilt:
  enabled: true
  type: "pead_reversal"
  amplitude: 0.20
  min_weight: 0.03
  window_days: 5
  months: [1, 2, 4, 8, 10]

# 风控熔断（Phase 9.1 产物）
risk_overlay:
  enabled: true
  type: "sentiment_risk"
  zscore_threshold: -2.5
  position_cut: 0.50
  freeze_days: 5
  min_trigger_samples: 20
```


## 四、回测验证

### 4.1 四种对比场景

```bash
# 1. Baseline（纯 Alpha 核心）
python -m src.cli backtest \
    --layers alpha_only \
    --start 2010-01-01 \
    --end 2025-12-31 \
    --output ./outputs/backtest_baseline.html

# 2. Alpha + 风控
python -m src.cli backtest \
    --layers alpha,risk \
    --start 2010-01-01 \
    --end 2025-12-31 \
    --output ./outputs/backtest_alpha_risk.html

# 3. Alpha + 战术倾斜
python -m src.cli backtest \
    --layers alpha,tilt \
    --start 2010-01-01 \
    --end 2025-12-31 \
    --output ./outputs/backtest_alpha_tilt.html

# 4. 三层全开（最终版本）
python -m src.cli backtest \
    --layers alpha,tilt,risk \
    --start 2010-01-01 \
    --end 2025-12-31 \
    --output ./outputs/backtest_three_layers.html
```

### 4.2 成功标准

| 指标 | Baseline | 三层全开（目标） |
|------|----------|-----------------|
| 年化收益 | ~10% | ≥ 10.5% |
| 夏普比率 | ~1.5 | **≥ 1.6** |
| 最大回撤 | ~10.5% | **< 10%** |
| 财报季回撤 | 基准 | **降低 ≥ 1.5%** |

### 4.3 门控判定

```python
# src/portfolio/backtest_runner.py

def check_gate(results: dict) -> tuple[bool, str]:
    """
    门控判定：
    - Sharpe > 1.6
    - Max Drawdown < 10%
    """
    passed = True
    reasons = []
    
    if results['sharpe'] < 1.6:
        passed = False
        reasons.append(f"Sharpe {results['sharpe']:.2f} < 1.6")
    
    if results['max_drawdown'] > 0.10:
        passed = False
        reasons.append(f"最大回撤 {results['max_drawdown']:.1%} > 10%")
    
    return passed, "; ".join(reasons) if reasons else "全部达标"
```


## 五、执行清单

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 10 执行清单                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Phase 9 归档（30分钟）                                    │
│  □ 1.1 更新 PHASE9_CLOSURE.md（三次失败 + 文本管线重定位）         │
│  □ 1.2 git tag phase9-closed                                      │
│  □ 1.3 更新 docs/RESEARCH_LOG.md                                  │
│                                                                     │
│  Step 2: 代码实现（2小时）                                         │
│  □ 2.1 src/portfolio/risk_overlay.py（舆情熔断）                  │
│  □ 2.2 src/portfolio/seasonal_tilt.py（PEAD战术倾斜）             │
│  □ 2.3 src/portfolio/layer_integration.py（三层融合引擎）          │
│  □ 2.4 src/portfolio/backtest_runner.py（回测运行器）              │
│  □ 2.5 单元测试：触发逻辑、优先级、归一化                          │
│                                                                     │
│  Step 3: 全样本回测（2010-2025）（2小时，后台运行）                │
│  □ 3.1 Baseline（纯低波组合）                                     │
│  □ 3.2 Alpha + 舆情熔断                                           │
│  □ 3.3 Alpha + PEAD倾斜                                           │
│  □ 3.4 三层全开（最终版本）                                       │
│                                                                     │
│  Step 4: 结果分析与报告（1小时）                                   │
│  □ 4.1 对比四项回测指标（Sharpe/收益/回撤/换手）                  │
│  □ 4.2 验证熔断触发场景（降低回撤 > 0）                           │
│  □ 4.3 验证财报季贡献（倾斜生效期表现）                           │
│  □ 4.4 生成 PHASE10_SIMULATION_BLUEPRINT.md                       │
│  □ 4.5 如果达标：git tag phase10-ready-for-deployment             │
│  □ 4.6 如果未达标：报告诊断，不推进实盘                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


## 六、里程碑

```bash
# Phase 9 关闭
git tag phase9-closed

# 代码实现完成
git tag phase10-code-complete

# 回测完成
git tag phase10-backtest-complete

# 如果达标
git tag phase10-ready-for-deployment
```


## 七、最终 Prompt 给 Claude Code

> **Claude Code**, execute Phase 10 simulation deployment blueprint:
>
> 1. **First**, archive Phase 9: Update `PHASE9_CLOSURE.md` with the three failures (PEAD, sentiment, dispersion). Tag `phase9-closed`.
>
> 2. **Second**, implement the three-layer architecture:
>    - `risk_overlay.py`: Sentiment risk cut (Z-score < -2.5 → 50% position cut for 5 days). Use existing TriAgent cache.
>    - `seasonal_tilt.py`: PEAD reversal tilt (±20% weight adjustment during earnings seasons).
>    - `layer_integration.py`: Orchestrate all three layers with correct priority (risk > tilt > alpha).
>
> 3. **Third**, run four full-sample backtests (2010–2025) on HS300:
>    - Baseline: Alpha core only
>    - Alpha + Risk
>    - Alpha + Tilt
>    - Alpha + Tilt + Risk (Final)
>
> 4. **Fourth**, verify success criteria:
>    - Final Sharpe > 1.6
>    - Final max drawdown < 10%
>    - Risk overlay reduces drawdown when triggered
>    - Seasonal tilt improves earnings-season performance
>
> 5. **Fifth**, if all gates pass, generate `PHASE10_SIMULATION_BLUEPRINT.md` and tag `phase10-ready-for-deployment`.
>
> 6. **Critical**: The risk overlay has the highest priority. If a stock triggers risk cut, the PEAD tilt on that stock must be overridden.
> **Critical**: Do not lower the gate thresholds. 1.6 Sharpe and 10% max drawdown are hard requirements for deployment.