# Implementation Blueprint: Phase 9.1 — 文本分歧度/新颖性因子（确定性回溯）

> **Target**: Claude Code (or AI Agent)  
> **Current State**: Phase 9.1 研报情绪单因子门禁 FAIL（rank_ic=0.0094 < 0.015）。诊断确认：不是管线问题（BERT 提升 6 倍证明架构有效），而是研报标题情绪在 HS300 上信噪比天然不足。NLP 管线保留，转向文本分歧度/新颖性方向。  
> **Goal**: 用已缓存的研报 BERT 向量，以**确定性脚本（无 LLM 调用）** 计算文本分歧度/新颖性因子，回测 2022–2025 HS300，门控标准 rank_ic > 0.015。  
> **Prerequisite**: Phase 9.1 研报数据已缓存（2022–2025 HS300，约 7,200 篇研报标题 + BERT 向量）。Phase 8 价量因子（低波+低换手家族，5 个）已确认有效。

---

## 一、背景与决策

### 1.1 研报情绪失败原因（已确认，非 Bug）

| 证据 | 结论 |
|------|------|
| 门禁 rank_ic=0.0094 < 0.015 | 未通过 |
| 最优 5 天窗口 rank_ic=0.0139，仍不足 | 结构弱，非阈值问题 |
| 分析师评级独立验证 ord_rank_ic=0.0047 | 研报整体信息量有限 |
| 分年 IC 不稳定（2022✓, 2023✗, 2024✗, 2025✓） | 信号不稳定 |
| 事件研究 LS t ∈ [-1.42, +0.33] | 发布后无信息漂移 |
| BERT 层贡献：0.0015 → 0.0094（提升 6 倍） | **架构有效，输入源受限** |

### 1.2 转向决策

| 转向方向 | 逻辑 | 与情绪差异 |
|----------|------|-----------|
| **文本分歧度** | 机构研报向量距离越大 → 定价不确定性越高 → 未来超额收益越高 | 测“差异”而非“方向” |
| **文本新颖性** | 新研报与历史研报向量距离越大 → 新逻辑/预期差 → 未来超额收益 | 测“新信息”而非“已定价信息” |

**执行方式**：确定性脚本（无 LLM 调用），复用已缓存研报 BERT 向量。


## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│              Phase 9.1: 文本分歧度/新颖性因子                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    数据层（已有）                          │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │ 研报缓存: data/text/hs300_research_2022_2025.parquet│   │   │
│  │  │ 字段: symbol, date, title, title_embedding (BERT)  │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    因子计算层（新增）                      │   │
│  │                                                             │   │
│  │  分歧度：过去 N 天所有研报向量的平均余弦距离                │   │
│  │  新颖性：新研报向量与过去 M 天历史向量中心的距离            │   │
│  │                                                             │   │
│  │  输出：截面排名因子 [0,1]                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    回测验证层                              │   │
│  │                                                             │   │
│  │  门控：rank_ic > 0.015（与研报情绪相同标准）               │   │
│  │  窗口扫描：5/10/20/30 天                                   │   │
│  │  样本：2022-2025 HS300                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


## 三、核心实现

### 3.1 数据加载

```python
# src/factors/text_factors.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity

class TextFactorCalculator:
    """
    文本因子计算器（确定性，无 LLM 调用）
    复用已缓存的 BERT 向量
    """
    
    def __init__(self, data_dir: str = "data/text"):
        self.data_dir = Path(data_dir)
        self.embeddings_cache = {}  # {(symbol, date): embedding}
        self._load_embeddings()
    
    def _load_embeddings(self):
        """加载缓存的研报 BERT 向量"""
        # 从 Parquet 加载
        df = pd.read_parquet(self.data_dir / "hs300_research_2022_2025.parquet")
        # 构建索引: (symbol, date) -> embedding
        for _, row in df.iterrows():
            key = (row["symbol"], row["date"])
            self.embeddings_cache[key] = np.array(row["title_embedding"])
        print(f"加载 {len(self.embeddings_cache)} 条研报向量")
```

### 3.2 分歧度因子

```python
# src/factors/text_factors.py

def calc_dispersion_factor(
    self,
    symbols: List[str],
    as_of_date: str,
    window: int = 20
) -> pd.Series:
    """
    计算截面分歧度因子。
    
    逻辑：过去 window 天内，每只股票所有研报向量的平均余弦距离。
    距离越大 → 机构分歧越大 → 定价不确定性越高 → 预期收益越高。
    
    返回：截面排名（0~1），分歧度越高排名越高。
    """
    scores = {}
    
    for symbol in symbols:
        # 获取过去 window 天的研报向量
        embeddings = self._get_embeddings(symbol, as_of_date, window)
        
        if len(embeddings) < 3:
            scores[symbol] = np.nan
            continue
        
        # 归一化
        normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # 余弦相似度矩阵
        sim_matrix = normed @ normed.T
        
        # 两两余弦距离（1 - 相似度），不含对角线
        n = len(embeddings)
        upper_indices = np.triu_indices(n, k=1)
        distances = 1 - sim_matrix[upper_indices]
        
        # 平均分歧度
        scores[symbol] = np.mean(distances)
    
    # 截面排名
    series = pd.Series(scores)
    return series.rank(pct=True).fillna(0.5)
```

### 3.3 新颖性因子

```python
# src/factors/text_factors.py

def calc_novelty_factor(
    self,
    symbols: List[str],
    as_of_date: str,
    window: int = 180
) -> pd.Series:
    """
    计算截面新颖性因子。
    
    逻辑：最新研报向量与过去 window 天历史向量中心的余弦距离。
    距离越大 → 新研报内容越“新”→ 预期差越大 → 预期收益越高。
    
    返回：截面排名（0~1），新颖性越高排名越高。
    """
    scores = {}
    
    for symbol in symbols:
        # 获取历史向量（过去 window 天，不含当日）
        history_embeddings = self._get_embeddings(symbol, as_of_date, window, include_today=False)
        
        if len(history_embeddings) < 5:
            scores[symbol] = np.nan
            continue
        
        # 当日新研报（取最近一篇）
        today_embedding = self._get_latest_embedding(symbol, as_of_date)
        if today_embedding is None:
            scores[symbol] = np.nan
            continue
        
        # 历史中心
        history_center = np.mean(history_embeddings, axis=0)
        history_center = history_center / np.linalg.norm(history_center)
        
        # 新向量与历史中心的余弦距离
        normed = today_embedding / np.linalg.norm(today_embedding)
        similarity = normed @ history_center
        distance = 1 - similarity
        
        scores[symbol] = distance
    
    # 截面排名
    series = pd.Series(scores)
    return series.rank(pct=True).fillna(0.5)
```

### 3.4 辅助方法

```python
# src/factors/text_factors.py

def _get_embeddings(
    self,
    symbol: str,
    as_of_date: str,
    window: int,
    include_today: bool = True
) -> np.ndarray:
    """
    获取指定股票在 as_of_date 之前 window 天内的所有研报向量。
    """
    end_date = pd.Timestamp(as_of_date)
    start_date = end_date - pd.Timedelta(days=window)
    
    embeddings = []
    for (sym, date), emb in self.embeddings_cache.items():
        if sym != symbol:
            continue
        if date < start_date or date > end_date:
            continue
        if not include_today and date == end_date:
            continue
        embeddings.append(emb)
    
    return np.array(embeddings) if embeddings else np.array([])

def _get_latest_embedding(self, symbol: str, as_of_date: str) -> Optional[np.ndarray]:
    """获取 as_of_date 当日的最新研报向量"""
    # 按时间倒序取最近一条
    candidates = []
    for (sym, date), emb in self.embeddings_cache.items():
        if sym == symbol and date == as_of_date:
            candidates.append((date, emb))
    if not candidates:
        return None
    # 取最新的（实际存储中同一天可能多条，取任意一条）
    return candidates[-1][1]
```

### 3.5 回测 CLI

```python
# src/cli.py — 新增 text 因子回测命令

@cli.command()
@click.option('--factor', type=click.Choice(['dispersion', 'novelty']), required=True)
@click.option('--window', default=20, help='时间窗口（天）')
@click.option('--start', default="2022-01-01")
@click.option('--end', default="2025-12-31")
@click.option('--symbols', default="hs300")
def backtest_text_factor(factor, window, start, end, symbols):
    """
    回测文本因子（分歧度/新颖性）
    确定性计算，无 LLM 调用。
    """
    # 1. 加载因子计算器
    from src.factors.text_factors import TextFactorCalculator
    calculator = TextFactorCalculator()
    
    # 2. 加载 Universe
    universe = load_universe(symbols)
    
    # 3. 逐日计算因子
    factor_values = {}
    for date in pd.date_range(start, end, freq="D"):
        # 只取交易日
        if not is_trading_day(date):
            continue
        
        if factor == "dispersion":
            values = calculator.calc_dispersion_factor(universe, date.strftime("%Y-%m-%d"), window)
        else:
            values = calculator.calc_novelty_factor(universe, date.strftime("%Y-%m-%d"), window)
        
        factor_values[date] = values
    
    # 4. 回测（复用 Phase 8 回测框架）
    results = run_backtest(factor_values, start, end)
    
    # 5. 打印结果
    print(f"\n=== {factor} 因子回测结果 ===")
    print(f"rank_ic: {results['rank_ic']:.4f}")
    print(f"icir: {results['icir']:.2f}")
    print(f"Sharpe: {results['sharpe']:.2f}")
    print(f"max_drawdown: {results['max_drawdown']:.2%}")
    
    # 6. 门控判定
    if results['rank_ic'] > 0.015:
        print("✅ 通过门控 (rank_ic > 0.015)")
    else:
        print("❌ 未通过门控 (rank_ic <= 0.015)")
    
    # 7. 保存报告
    save_report(results, f"outputs/text_{factor}_{window}d.html")
```


## 四、配置更新

```yaml
# configs/master_config.yaml

# Phase 9.1 文本因子配置（分歧度/新颖性）
text_factors:
  enabled: true
  
  # 数据路径
  data_dir: "data/text"
  cache_file: "hs300_research_2022_2025.parquet"
  
  # 分歧度因子
  dispersion:
    enabled: true
    window: 20          # 推荐扫描: 10/20/30/60
    min_articles: 3     # 至少 3 篇才计算
  
  # 新颖性因子
  novelty:
    enabled: true
    window: 180         # 半年历史窗口
    min_articles: 5     # 至少 5 篇历史才计算
  
  # 门控（与 Phase 9 保持一致）
  gate_threshold: 0.015

# 因子池（最终）
factor_combination:
  factors:
    # Phase 8: 价量因子
    - source: "phase8"
      family: "low_vol_low_turnover"
      count: 5
      weight_cap: 0.25
    
    # Phase 9.1: 文本因子（如果通过门控）
    - source: "phase9.1"
      family: "dispersion"
      enabled: false  # 回测通过后改为 true
      weight_cap: 0.20
    
    - source: "phase9.1"
      family: "novelty"
      enabled: false
      weight_cap: 0.20
    
    # Phase 9.2: PEAD（已关闭）
```


## 五、执行清单

```
┌─────────────────────────────────────────────────────────────────────┐
│              Phase 9.1 分歧度/新颖性执行清单                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: 数据确认（10分钟）                                        │
│  □ 1.1 确认 data/text/hs300_research_2022_2025.parquet 存在       │
│  □ 1.2 确认字段: symbol, date, title_embedding (BERT向量)         │
│  □ 1.3 抽样验证向量维度一致                                        │
│                                                                     │
│  Step 2: 代码实现（30分钟）                                        │
│  □ 2.1 实现 src/factors/text_factors.py                           │
│  □ 2.2 实现 calc_dispersion_factor                                 │
│  □ 2.3 实现 calc_novelty_factor                                    │
│  □ 2.4 单元测试: 向量计算、NaN处理                                 │
│                                                                     │
│  Step 3: 分歧度回测（10分钟，后台运行）                            │
│  □ 3.1 python -m src.cli backtest_text_factor --factor dispersion │
│  □ 3.2 扫描窗口: 10, 20, 30, 60                                   │
│  □ 3.3 记录最优 rank_ic                                            │
│                                                                     │
│  Step 4: 新颖性回测（10分钟，后台运行）                            │
│  □ 4.1 python -m src.cli backtest_text_factor --factor novelty    │
│  □ 4.2 扫描窗口: 90, 180, 360                                     │
│  □ 4.3 记录最优 rank_ic                                            │
│                                                                     │
│  Step 5: 门控判定                                                  │
│  □ 5.1 分歧度 rank_ic > 0.015 ? → 入池                           │
│  □ 5.2 新颖性 rank_ic > 0.015 ? → 入池                           │
│  □ 5.3 至少一个通过 → Phase 9.1 PASS                              │
│  □ 5.4 两个都 FAIL → Phase 9 全部关闭                             │
│                                                                     │
│  Step 6: 报告与提交                                                │
│  □ 6.1 生成 PHASE9_1_TEXT_REPORT.md                               │
│  □ 6.2 如果 PASS: 更新 master_config.yaml 启用因子                │
│  □ 6.3 如果 FAIL: 打 tag phase9-text-rejected                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


## 六、决策规则

### 6.1 门控条件

| 因子 | 门控 | 说明 |
|------|------|------|
| 分歧度 | rank_ic > 0.015 | 与研报情绪相同标准 |
| 新颖性 | rank_ic > 0.015 | 与研报情绪相同标准 |
| **至少一个通过** | Phase 9.1 PASS | 进入多因子组合 |
| **两个都 FAIL** | Phase 9.1 FAIL | Phase 9 全部关闭 |

### 6.2 最终决策矩阵

| 分歧度 | 新颖性 | 行动 |
|--------|--------|------|
| ✅ PASS | — | 分歧度入池，新颖性可选入 |
| ❌ FAIL | ✅ PASS | 新颖性入池 |
| ✅ PASS | ✅ PASS | 两者都入池，组合回测 |
| ❌ FAIL | ❌ FAIL | **Phase 9 全部关闭** |


## 七、里程碑

```bash
# 研报情绪失败（已打）
git tag phase9.1-sentiment-rejected

# 分歧度/新颖性回测完成
git tag phase9.1-text-backtest-complete

# 如果 PASS
git tag phase9.1-text-passed

# 如果 FAIL
git tag phase9.1-text-rejected
```

**最终 Phase 9 状态**：

| 阶段 | 状态 |
|------|------|
| Phase 9.2（PEAD） | ❌ FAIL |
| Phase 9.1（研报情绪） | ❌ FAIL |
| Phase 9.1（分歧度/新颖性） | ⏳ **执行中** |
| Phase 9.3（多因子融合） | ⏳ **取决于文本结果** |


## 八、Phase 10 预览（分歧度/新颖性通过后）

如果文本因子通过门控：

```
因子池最终构成:
  - Phase 8: 低波+低换手 (5个) - 价量
  - Phase 9.1: 分歧度/新颖性 (1-2个) - 文本
  
多因子组合 → 回测 2022-2025 → 目标 Sharpe > 1.5
```

**如果文本因子也 FAIL**：

```
因子池最终构成:
  - Phase 8: 低波+低换手 (5个) - 价量

进入 Phase 10: 模拟盘部署（仅价量）
```


## 九、最终 Prompt 给 Claude Code

> **Claude Code**, execute Phase 9.1 text factor blueprint:
>
> 1. **First**, confirm `data/text/hs300_research_2022_2025.parquet` exists with BERT embeddings.
> 2. **Second**, implement `src/factors/text_factors.py` — `calc_dispersion_factor` and `calc_novelty_factor` (deterministic, no LLM calls).
> 3. **Third**, run backtest for dispersion (windows: 10, 20, 30, 60).
> 4. **Fourth**, run backtest for novelty (windows: 90, 180, 360).
> 5. **Fifth**, apply gate: rank_ic > 0.015 for each.
> 6. **Sixth**, generate `PHASE9_1_TEXT_REPORT.md` with results.
> 7. **Seventh**, if at least one passes, update `master_config.yaml` and proceed to Phase 9.3 (multi-factor combination).
> 8. **Eighth**, if both fail, tag `phase9-text-rejected` and report Phase 9 closure.
>
> **Critical**: Do NOT call LLM for this phase. All computations are deterministic using cached BERT embeddings.
> **Critical**: Gate threshold is 0.015 (same as Phase 9 standards). Do not lower.

---

**Blueprint version**: 1.0
**Created**: 2026-08-12
**Based on**: Phase 9.1 sentiment failure + text dispersion/novelty pivot
**Status**: 🔴 **Awaiting your approval** — once approved, Claude Code executes the blueprint