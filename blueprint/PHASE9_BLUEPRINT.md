# Implementation Blueprint: Phase 9 — 文本信号接入与多因子融合

> **Target**: Claude Code (or AI Agent)
> **Current State**: Phase 8 补救验证完成，5/20 = 25% < 30%，门控未达标。价量 Alpha 方向已穷尽（唯一有效家族为低波+低换手），继续在同一信息源挖掘边际收益递减。验证清单 4/4 全绿，系统能力已充分验证。
> **Goal**: 转向**不同信息源**——接入新闻情感、PEAD（盈余公告后漂移）等文本/基本面信号，与 Phase 8 已验证价量因子融合，构建多因子组合，目标 Sharpe > 1.5。
> **Prerequisite**: Phase 7 完成（12.48M bars, B1-B5 green）。Phase 8 代码修复已合入（物理屏蔽、模板重定向、去重机制、跨轮污染修复）。209 tests passed。

---

## 一、Phase 8 状态与 Phase 9 启动依据

### 1.1 Phase 8 最终状态

| 维度 | 状态 |
|------|------|
| 通过率 | 5/20 = 25%（< 30% 门控） |
| 验证清单 | ✅ pit/fincad/diversity/cost 全部 PASS |
| 有效 Alpha 方向 | 唯一：低波 + 低换手 |
| 结论 | 价量 Alpha 方向已穷尽 |

### 1.2 为什么 Phase 9 是正确方向

| 选项 | 评估 |
|------|------|
| 继续挖价量 | Alpha 方向已穷尽，再跑 50 轮只会重复低波变体 |
| 降阈值 | 蓝图明令禁止自行调参；且 5/20 不是因为阈值过严，而是可用 Alpha 太少 |
| **转向文本信号** | ✅ 不同信息源（新闻/财报），与价量因子低相关，组合平滑回撤，最符合机构级多因子实践 |

**Phase 8 产出留用**：低波+低换手家族 5 个因子全部保留，作为 Phase 9 价量基石。


## 二、Phase 9 总体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 9: 多源因子融合                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ 价量因子      │  │ 文本因子      │  │ 基本面因子    │             │
│  │ (Phase 8)    │  │ (Phase 9.1)  │  │ (Phase 9.2)  │             │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤             │
│  │ 低波+低换手   │  │ 新闻情感      │  │ PEAD         │             │
│  │ 5个因子       │  │ TriAgent     │  │ EPS意外      │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│         │                  │                  │                    │
│         └──────────────────┼──────────────────┘                    │
│                            ▼                                       │
│              ┌─────────────────────────┐                          │
│              │  多因子组合优化          │                          │
│              │  (ICIR加权 + PCA中性)   │                          │
│              └─────────────────────────┘                          │
│                            │                                       │
│                            ▼                                       │
│              回测验证 (2022-2025 样本外)                           │
│              目标: Sharpe > 1.5                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


## 三、Phase 9.1 — 新闻情感因子（TriAgent 分层框架）

### 3.1 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TriAgent 分层情感框架                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  新闻文本 → 词级 → 句子级 → 跨句推理 → 情感分数                    │
│              │         │          │                                │
│              ▼         ▼          ▼                                │
│            VADER    FinBERT    Qwen2.5-7B                          │
│           (规则)    (微调)      (LLM)                              │
│                                                                     │
│  分层策略：                                                         │
│  - 词级 VADER：高通量初筛（~1000 条/秒）                           │
│  - 句子级 FinBERT：中等精度验证（~50 条/秒）                       │
│  - 跨句推理 Qwen：深度上下文推理（仅对分歧样本）                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 数据源

| 数据源 | 接口 | 覆盖范围 | 更新频率 |
|--------|------|---------|---------|
| 东方财富新闻 | `akshare.stock_news_em` | A 股全市场 | 实时 |
| 东方财富研报 | `akshare.stock_research_report_em` | 券商研报 | 日更 |
| 公司公告 | `akshare.stock_notice_report` | 上交所/深交所 | 日更 |

### 3.3 情感因子生成流程

```python
# src/sentiment/triagent.py

class TriAgentSentiment:
    def __init__(self):
        self.vader = VADER()  # 词级规则
        self.finbert = FinBERT()  # 句子级微调模型
        self.llm = QwenClient()  # 跨句推理（仅分歧样本）
    
    def compute_daily_sentiment(self, symbol: str, date: str) -> float:
        """
        计算单只股票在特定日期的情感分数。
        返回: -1 (极度负面) ~ +1 (极度正面)
        """
        # 1. 获取当日新闻
        news = self._fetch_news(symbol, date)
        if not news:
            return 0.0
        
        # 2. 词级初筛（VADER）
        vader_scores = [self.vader.polarity_scores(t)['compound'] for t in news]
        
        # 3. 句子级验证（FinBERT）——仅对极端样本
        if abs(mean(vader_scores)) > 0.5:
            finbert_scores = [self.finbert.predict(t) for t in news]
        else:
            finbert_scores = vader_scores
        
        # 4. 跨句推理（LLM）——仅对高分歧样本
        if std(finbert_scores) > 0.3:
            llm_score = self.llm.analyze_context(news)
            # 加权融合：LLM权重0.5，FinBERT权重0.3，VADER权重0.2
            final_score = 0.5 * llm_score + 0.3 * mean(finbert_scores) + 0.2 * mean(vader_scores)
        else:
            final_score = mean(finbert_scores)
        
        return np.clip(final_score, -1, 1)
    
    def compute_sentiment_factor(self, symbols: List[str], as_of_date: str) -> pd.Series:
        """
        截面情感因子：对所有股票计算当日情感分数，返回截面排序。
        """
        scores = {}
        for symbol in symbols:
            scores[symbol] = self.compute_daily_sentiment(symbol, as_of_date)
        return pd.Series(scores).rank(pct=True)
```

### 3.4 成本控制

| 层级 | 调用频率 | 成本 |
|------|---------|------|
| VADER | 全部新闻（~10万条/日） | 免费 |
| FinBERT | 极端情感样本（~20%） | 免费（本地） |
| Qwen LLM | 高分歧样本（~5%） | ~$0.001/日 |

**总成本 < $1/月**（Qwen 仅对极少数分歧样本推理）


## 四、Phase 9.2 — PEAD 因子（盈余公告后漂移）

### 4.1 数据源

```bash
# Baostock 季报数据（已有适配器）
python -m src.data.ingestion.baostock_adapter fetch_financials \
    --symbols all \
    --years 2010-2025 \
    --fields eps,revenue,roe
```

### 4.2 PEAD 因子实现

```python
# src/factors/pead.py

class PEADFactor:
    """
    盈余公告后漂移（Post-Earnings Announcement Drift）
    核心逻辑：EPS 超预期 → 做多；低于预期 → 做空
    """
    
    def compute_eps_surprise(self, symbol: str, report_date: str) -> float:
        """
        计算 EPS 意外 = (实际 EPS - 预期 EPS) / 预期 EPS
        预期 EPS = TTM 平滑（或分析师一致预期，当前用历史平均替代）
        """
        actual_eps = self._get_actual_eps(symbol, report_date)
        expected_eps = self._get_expected_eps(symbol, report_date)
        if expected_eps == 0:
            return 0.0
        return (actual_eps - expected_eps) / abs(expected_eps)
    
    def compute_pead_factor(self, symbols: List[str], as_of_date: str) -> pd.Series:
        """
        截面 PEAD 因子：按最近公告的 EPS 意外排序。
        仅当公告日距今 ≤ 60 个交易日才有效。
        """
        surprises = {}
        for symbol in symbols:
            latest_report = self._get_latest_report(symbol, as_of_date)
            if not latest_report:
                continue
            days_since = (pd.Timestamp(as_of_date) - latest_report.date).days
            if days_since > 60:
                continue  # 信号过期
            surprises[symbol] = latest_report.surprise
        
        return pd.Series(surprises).rank(pct=True)
```

### 4.3 PEAD 与价量因子的正交性

| 因子类型 | 与 PEAD 相关性（预期） |
|----------|----------------------|
| 低波+低换手（Phase 8） | < 0.1（事件驱动 vs 价量结构） |
| 动量因子 | ~0.3（PEAD 后趋势跟踪） |
| 反转因子 | ~-0.2（PEAD 后漂移 ≠ 反转） |

**结论**：PEAD 与 Phase 8 因子低相关，组合平滑效果显著。


## 五、Phase 9.3 — 多因子组合与回测

### 5.1 因子池构成（Phase 9 最终）

| 来源 | 因子 | 数量 |
|------|------|------|
| Phase 8 | 低波+低换手家族 | 5 |
| Phase 9.1 | 新闻情感因子（TriAgent） | 1 |
| Phase 9.2 | PEAD 因子 | 1 |
| **总计** | | **7** |

### 5.2 组合优化

```python
# src/portfolio/optimizer.py

class MultiFactorOptimizer:
    def __init__(self, factors: List[Factor], weights: str = "icir"):
        self.factors = factors
        self.weights_method = weights
    
    def compute_composite(self, symbols: List[str], as_of_date: str) -> pd.Series:
        """
        多因子加权合成。
        支持：equal / icir_weighted / dynamic
        """
        # 1. 计算各因子值
        factor_values = {}
        for factor in self.factors:
            factor_values[factor.name] = factor.compute(symbols, as_of_date)
        
        # 2. 因子中性化（行业 + 市值）
        for name, values in factor_values.items():
            factor_values[name] = self._neutralize(values, as_of_date)
        
        # 3. 加权合成
        weights = self._get_weights(as_of_date)
        composite = sum(weights[name] * factor_values[name] for name in weights)
        
        return composite
    
    def _neutralize(self, values: pd.Series, as_of_date: str) -> pd.Series:
        """PCA 中性化：移除市值、行业、风格暴露"""
        # 使用 sklearn.decomposition.PCA
        # 移除前 K 个主成分（对应市场/风格因子）
        pass
```

### 5.3 回测验证

```bash
# 回测多因子组合（2022-2025 样本外）
python -m src.cli backtest \
    --factor-pool ./outputs/phase9_factor_pool.json \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --weights icir_weighted \
    --neutralize industry,size \
    --output ./outputs/phase9_backtest.html
```

### 5.4 成功标准

| 指标 | 目标 | 最低可接受 |
|------|------|-----------|
| Sharpe 比率 | > 1.5 | > 1.0 |
| 年化收益 | > 15% | > 10% |
| 最大回撤 | < 15% | < 20% |
| 与 Phase 8 相关性 | < 0.5 | < 0.6 |
| 因子 IC 衰减 | 6 个月 ICIR > 0.3 | > 0.2 |


## 六、配置更新

### 6.1 master_config.yaml（Phase 9 版本）

```yaml
# configs/master_config.yaml

project:
  name: "FQA_Phase9"
  start_date: "2010-01-01"
  end_date: "2025-12-31"

data:
  real_data: true
  pit_database_url: "${DATABASE_URL}"

# === Phase 9 新增：情感配置 ===
sentiment:
  enabled: true
  triagent:
    vader_enabled: true
    finbert_model: "yiyanghkust/finbert-tone"
    llm_model: "qwen2.5-7b"
    llm_trigger_threshold: 0.3  # 分歧标准差超过此值触发 LLM
    cost_budget_monthly_usd: 1.0

# === Phase 9 新增：PEAD 配置 ===
pead:
  enabled: true
  signal_expiry_days: 60
  min_eps_history: 8  # 至少 8 个季度

# === 因子组合配置 ===
factor_combination:
  method: "icir_weighted"
  lookback_days: 252
  rebalance_frequency: "daily"
  max_single_factor_weight: 0.30
  min_ic: 0.02
  min_icir: 0.30

  neutralization:
    - factor: "market_cap"
      method: "pca"
    - factor: "industry"
      method: "dummy"

  # Phase 9 因子池
  factors:
    - source: "phase8"
      family: "low_vol_low_turnover"
      count: 5
    - source: "phase9"
      family: "sentiment"
      count: 1
    - source: "phase9"
      family: "pead"
      count: 1
```


## 七、执行清单

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 9 执行清单                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 9.1 情感信号（Week 1）                                      │
│  □ 1.1 实现 src/sentiment/vader.py（词级规则）                    │
│  □ 1.2 实现 src/sentiment/finbert.py（HuggingFace 微调模型）      │
│  □ 1.3 实现 src/sentiment/triagent.py（分层框架）                 │
│  □ 1.4 实现 src/sentiment/ingestion.py（新闻数据接入）            │
│  □ 1.5 单元测试：情感分数范围 [-1, 1]，VADER/FinBERT 一致性      │
│  □ 1.6 回测情感因子（单因子，2022-2025）                          │
│                                                                     │
│  Phase 9.2 PEAD 信号（Week 1）                                    │
│  □ 2.1 实现 src/factors/pead.py（EPS 意外计算）                   │
│  □ 2.2 接入 Baostock 季报数据（已有 adapter）                     │
│  □ 2.3 单元测试：EPS 意外计算与预期差逻辑                         │
│  □ 2.4 回测 PEAD 因子（单因子，2022-2025）                        │
│                                                                     │
│  Phase 9.3 多因子组合（Week 2）                                   │
│  □ 3.1 实现 src/portfolio/optimizer.py（组合加权 + 中性化）      │
│  □ 3.2 合并 Phase 8 价量因子 + Phase 9.1 情感 + Phase 9.2 PEAD  │
│  □ 3.3 回测多因子组合（2022-2025 样本外）                         │
│  □ 3.4 验证 Sharpe > 1.5                                          │
│  □ 3.5 生成 Phase 9 报告                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


## 八、门控与里程碑

### 8.1 门控条件

| 阶段 | 门控 | 失败处理 |
|------|------|---------|
| 9.1 完成 | 情感单因子 IC > 0.015 | 检查新闻数据覆盖度，调整 FinBERT 阈值 |
| 9.2 完成 | PEAD 单因子 IC > 0.015 | 检查 EPS 数据质量，调整信号过期窗口 |
| 9.3 完成 | 组合 Sharpe > 1.0 | 检查因子相关性，调整中性化方法 |
| **Phase 9 最终** | **组合 Sharpe > 1.5** | 如果 > 1.0 但 < 1.5，评估是否进入 Phase 10 |

### 8.2 里程碑标签

```bash
# Phase 9.1 完成
git tag phase9-sentiment

# Phase 9.2 完成
git tag phase9-pead

# Phase 9 最终完成
git tag phase9-complete  # 条件：Sharpe > 1.5
```


## 九、Phase 10 预览（Phase 9 成功后）

当 Phase 9 完成且组合 Sharpe > 1.5 后，进入 Phase 10（模拟盘部署）：

| 任务 | 说明 |
|------|------|
| 策略蒸馏 | 将多因子组合固化为确定性 Python 代码 |
| 模拟盘运行 | 3-6 个月模拟盘验证（无实盘资金） |
| 成本建模 | 真实滑点/佣金校准 |
| 券商接入 | 实盘 API 对接（可选） |


## 十、风险与应对

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| AKShare 新闻接口失效 | 中 | 高 | 备选：Tushare 新闻接口 / 商业新闻 API |
| FinBERT 模型精度不足 | 低 | 中 | 可微调 FinBERT on 东方财富新闻语料 |
| PEAD 数据质量差 | 中 | 中 | 使用 Baostock 财务数据 + 人工抽查 |
| 情感因子与价量因子相关性 > 0.5 | 低 | 低 | 预期低相关，若发生则重新评估文本信号质量 |
| LLM 成本超预算 | 低 | 低 | 默认只对 5% 分歧样本调用 Qwen |


## 十一、最终 Prompt 给 Claude Code

> **Claude Code**, execute Phase 9 blueprint in full. Follow this order strictly:
>
> 1. **Week 1**: Implement TriAgent sentiment pipeline (VADER + FinBERT + Qwen). Use AKShare for news data. Write unit tests for emotion score range and consistency.
> 2. **Week 1**: Implement PEAD factor using Baostock EPS data. Calculate EPS surprise as (actual - expected) / expected.
> 3. **Week 2**: Merge Phase 8 factors + sentiment + PEAD into a multi-factor optimizer with PCA neutralization.
> 4. **Run backtest** on 2022-2025 out-of-sample period with ICIR-weighted combination.
> 5. **Gate**: If Sharpe > 1.5, tag `phase9-complete` and proceed to Phase 10 simulation.
> 6. **If Sharpe < 1.0**, stop and report diagnostics. Do not adjust thresholds.
> 7. **If Sharpe between 1.0 and 1.5**, report and await decision on whether to proceed or refine.
>
> **Critical**: Do not continue Phase 8 mining. Phase 8 is archived. All new Alpha must come from textual/fundamental signals.

---

**Blueprint version**: 1.0
**Created**: 2026-08-10
**Based on**: PHASE8_REMEDY_REPORT.md + Phase 9 evaluation
**Status**: 🔴 **Awaiting your approval** — once approved, Claude Code executes Phase 9.