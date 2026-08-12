# Implementation Blueprint: Phase 9.1 — 新闻情感因子（中文 TriAgent）

> **Target**: Claude Code (or AI Agent)  
> **Current State**: Phase 9.2（PEAD）已判定 FAIL 并关闭。rank_ic=0.0016，A 股 2022–2025 窗口呈负漂移（盈余反转），方向反转后仍不通过门控。基础设施已提交，`cmd_pead --direction drift|reversal` 保留为诊断工具。  
> **Next**: Phase 9.1 新闻情感因子 — 中文词库 + 中文 BERT + DeepSeek 批判者，实时向前采集 + 历史回测。  
> **Prerequisite**: Phase 7 完成（12.48M bars, B1-B5 green）。Phase 8 价量因子（低波+低换手家族，5个）已确认有效。Phase 9.2 基建已合入。

---

## 一、Phase 9.1 定位与设计原则

### 1.1 为什么是新闻情感

| 维度 | 说明 |
|------|------|
| **与价量低相关** | 新闻情感信号与价量因子（低波+低换手）相关性预计 < 0.15 |
| **独立 Alpha 源** | 非价量信息，可补充 Phase 8 单一 Alpha 方向 |
| **A 股适用** | 中文新闻语料丰富，散户情绪驱动明显 |
| **Phase 8 教训** | 单一信息源（价量）已穷尽，必须转向不同信息源 |

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **中文原生** | 中文词库 + 中文 BERT（非英文 FinBERT 迁移） |
| **分层框架** | 词级规则 → 句子级 BERT → 跨句 DeepSeek 批判者（成本可控） |
| **实时向前采集** | 逐日增量采集，支持断点续传 |
| **历史回测** | 2022–2025 历史新闻回填，验证情感因子 IC |
| **接口统一** | 所有 LLM 调用使用 DeepSeek（已有 key，成本可控） |
| **Phase 8 教训** | 先事件研究验证，再编码因子；如果 IC < 0.01，停止并报告 |


## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 9.1: 中文 TriAgent 情感框架              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    数据采集层                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │   │
│  │  │ 东方财富新闻  │  │ 东方财富研报  │  │ 公司公告         │ │   │
│  │  │ (AKShare)   │  │ (AKShare)   │  │ (AKShare)        │ │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘ │   │
│  │         │                  │                  │            │   │
│  │         └──────────────────┼──────────────────┘            │   │
│  │                            ▼                               │   │
│  │              ┌─────────────────────────┐                   │   │
│  │              │   NewsStore (Parquet)   │                   │   │
│  │              │   (symbol, date, text)  │                   │   │
│  │              └─────────────────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    情感计算层                               │   │
│  │                                                             │   │
│  │  新闻文本 → 词级 → 句子级 → 跨句推理 → 情感分数            │   │
│  │              │         │          │                        │   │
│  │              ▼         ▼          ▼                        │   │
│  │         中文词库    Chinese     DeepSeek                   │   │
│  │         规则匹配    BERT        批判者                     │   │
│  │                    (微调)      (推理)                      │   │
│  │                                                             │   │
│  │  分层策略：                                                 │   │
│  │  - 词级规则：中文金融词典——高通量初筛（~1000 条/秒）       │   │
│  │  - 中文 BERT：哈工大 FinBERT_zh——中等精度验证（~50 条/秒） │   │
│  │  - DeepSeek：仅对分歧样本——深度推理（~1 条/秒）            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            │                                       │
│                            ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    因子输出层                               │   │
│  │                                                             │   │
│  │  每日情感因子：截面排序 [0,1] → 回测 IC/Sharpe             │   │
│  │  与 Phase 8 价量因子合并 → 多因子组合回测                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


## 三、组件实现

### 3.1 数据采集层

```python
# src/sentiment/ingestion.py

import akshare as ak
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

class NewsIngestor:
    """
    新闻采集器：实时向前采集 + 历史回填 + 断点续传
    采集状态记录在 data/news/crawl_state.json
    """
    
    def __init__(self, data_dir: str = "data/news"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "crawl_state.json"
        self.state = self._load_state()
    
    def _load_state(self) -> dict:
        """加载采集状态"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {"last_crawled_date": None, "symbols_done": []}
    
    def _save_state(self):
        """保存采集状态（断点续传）"""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def fetch_news_by_date(self, date: str) -> pd.DataFrame:
        """
        抓取单日所有新闻
        使用 AKShare stock_news_em（东方财富）
        """
        try:
            # 东方财富全市场新闻
            df = ak.stock_news_em(date=date)
            if df.empty:
                return pd.DataFrame()
            
            # 标准化字段
            df = df.rename(columns={
                "标题": "title",
                "发布时间": "publish_time",
                "内容": "content",
                "股票代码": "symbol"  # 注意：可能包含多个代码
            })
            # 处理多股票代码的新闻（拆分）
            return self._explode_symbols(df)
        except Exception as e:
            print(f"[NewsIngestor] 抓取 {date} 失败: {e}")
            return pd.DataFrame()
    
    def _explode_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理一条新闻对应多个股票代码的情况"""
        # 假设 "symbol" 字段可能包含 "600519,000858" 格式
        # 拆分为多行
        # 实现略
        pass
    
    def fetch_research_by_symbol(self, symbol: str) -> pd.DataFrame:
        """抓取单只股票的研报"""
        try:
            df = ak.stock_research_report_em(symbol=symbol)
            return df
        except Exception as e:
            print(f"[NewsIngestor] 抓取 {symbol} 研报失败: {e}")
            return pd.DataFrame()
    
    def backfill(self, start_date: str, end_date: str):
        """
        历史回填（2022–2025）
        逐日抓取，受 AKShare 频率限制（加 sleep）
        """
        date_range = pd.date_range(start_date, end_date, freq="D")
        
        for date in date_range:
            date_str = date.strftime("%Y-%m-%d")
            
            # 跳过已采集的日期
            if self.state.get("last_crawled_date") and date_str <= self.state["last_crawled_date"]:
                continue
            
            df = self.fetch_news_by_date(date_str)
            if not df.empty:
                self._save_news(date_str, df)
            
            self.state["last_crawled_date"] = date_str
            self._save_state()
            
            # 礼貌限速：AKShare 免费接口，不宜高频
            time.sleep(0.5)
        
        print(f"[NewsIngestor] 回填完成: {start_date} → {end_date}")
    
    def _save_news(self, date: str, df: pd.DataFrame):
        """保存单日新闻到 Parquet"""
        file_path = self.data_dir / f"news_{date}.parquet"
        df.to_parquet(file_path, index=False)
    
    def get_news_for_symbol(self, symbol: str, date: str) -> list:
        """查询某股票在特定日期的新闻"""
        # 从 Parquet 文件中读取
        # 实现略
        pass
```

### 3.2 中文词库（词级规则）

```python
# src/sentiment/lexicon.py

class ChineseFinancialLexicon:
    """
    中文金融情感词库
    包含：正面词、负面词、程度词、否定词
    """
    
    def __init__(self):
        self.positive_words = self._load_positive()
        self.negative_words = self._load_negative()
        self.intensifiers = self._load_intensifiers()
        self.negators = self._load_negators()
    
    def _load_positive(self) -> set:
        """正面金融词汇"""
        return {
            "增长", "上涨", "盈利", "利好", "超预期", "买入", "推荐",
            "新高", "突破", "反弹", "反转", "改善", "提升", "加速",
            "放量", "活跃", "强劲", "乐观", "看好", "加仓", "增持",
            "跑赢", "领先", "龙头", "稀缺", "溢价", "低估", "价值",
            "分红", "回购", "业绩", "确定性", "高景气",
        }
    
    def _load_negative(self) -> set:
        """负面金融词汇"""
        return {
            "下跌", "亏损", "利空", "低于预期", "减持", "卖出", "警惕",
            "新低", "破位", "回调", "恶化", "下滑", "放缓", "萎缩",
            "缩量", "低迷", "悲观", "看空", "减仓", "回避", "跑输",
            "滞后", "风险", "泡沫", "高估", "踩踏", "恐慌", "崩盘",
            "停牌", "问询", "处罚", "诉讼", "违约", "退市", "暴雷",
        }
    
    def _load_intensifiers(self) -> dict:
        """程度副词（权重系数）"""
        return {
            "非常": 1.5, "极度": 2.0, "轻微": 0.5,
            "大幅": 1.8, "小幅": 0.6, "持续": 1.2,
            "显著": 1.6, "温和": 0.7, "突然": 1.3,
        }
    
    def _load_negators(self) -> set:
        """否定词"""
        return {"不", "未", "无", "非", "不是", "不会", "尚未", "并未"}
    
    def score(self, text: str) -> float:
        """
        基于词库的情感评分
        返回：-1（极度负面）~ +1（极度正面）
        """
        # 简单实现：统计正面/负面词频 + 程度/否定调整
        # 具体实现略
        pass
```

### 3.3 中文 BERT（句子级）

```python
# src/sentiment/bert.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ChineseBertSentiment:
    """
    中文 BERT 情感分类
    使用哈工大 FinBERT_zh 或 bert-base-chinese 微调
    """
    
    def __init__(self, model_name: str = "bert-base-chinese"):
        # 推荐：使用 FinBERT_zh（哈工大）
        # 或自微调 bert-base-chinese on 金融语料
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # 正面/中性/负面
        )
        self.model.eval()
    
    def predict(self, text: str) -> float:
        """
        预测情感分数
        返回：0（负面）~ 1（正面），0.5 为中性
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            # 假设 logits 顺序: [negative, neutral, positive]
            # 计算 [-1, 1] 加权分数
            score = -1 * probs[0][0] + 0 * probs[0][1] + 1 * probs[0][2]
            # 转换为 [0, 1]
            return (score + 1) / 2
    
    def predict_batch(self, texts: list) -> list:
        """批量预测"""
        return [self.predict(t) for t in texts]
```

### 3.4 DeepSeek 批判者（跨句推理）

```python
# src/sentiment/critic.py

import os
from openai import OpenAI

class DeepSeekCritic:
    """
    DeepSeek 批判者：对分歧/复杂样本做深度推理
    替换原 Qwen 方案，接口统一为 DeepSeek
    """
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        self.model = "deepseek-v3"  # 或 deepseek-v4-flash
    
    def analyze(self, symbol: str, articles: list, bert_scores: list) -> float:
        """
        对一组新闻做跨句推理
        返回：0（负面）~ 1（正面）
        """
        # 构建上下文
        context = self._format_context(symbol, articles, bert_scores)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """你是金融情感分析专家。分析以下新闻组，给出整体情感倾向。

                    输出格式：只输出一个数字（0~1），不包含任何解释。
                    - 0：极度负面（重大利空）
                    - 0.5：中性/无明显倾向
                    - 1：极度正面（重大利好）"""
                },
                {"role": "user", "content": context}
            ],
            temperature=0.1,  # 低温度，保证一致性
            max_tokens=10
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0, min(1, score))
        except:
            # 解析失败，返回中性
            return 0.5
    
    def _format_context(self, symbol: str, articles: list, scores: list) -> str:
        """格式化上下文"""
        text = f"股票：{symbol}\n\n新闻及初步分析：\n"
        for i, (article, score) in enumerate(zip(articles, scores)):
            text += f"[{i+1}] {article[:200]}... (初步情感: {score:.2f})\n"
        text += "\n请给出整体情感裁定："
        return text
```

### 3.5 TriAgent 主框架

```python
# src/sentiment/triagent.py

import numpy as np
import pandas as pd
from src.sentiment.lexicon import ChineseFinancialLexicon
from src.sentiment.bert import ChineseBertSentiment
from src.sentiment.critic import DeepSeekCritic
from src.sentiment.ingestion import NewsIngestor

class TriAgentSentiment:
    """
    TriAgent 分层情感计算框架
    词级规则 → 中文 BERT → DeepSeek 批判者
    """
    
    def __init__(self, data_dir: str = "data/news"):
        self.ingestor = NewsIngestor(data_dir)
        self.lexicon = ChineseFinancialLexicon()
        self.bert = ChineseBertSentiment()
        self.critic = DeepSeekCritic()
        
        # 分层阈值
        self.lexicon_threshold = 0.3  # 超出此阈值触发 BERT
        self.bert_threshold = 0.25    # 分歧标准差超过此值触发批判者
        self.critic_articles_min = 3   # 至少 3 篇新闻才触发批判者
    
    def compute_emotion(self, symbol: str, date: str) -> float:
        """
        计算单只股票在特定日期的情感分数
        返回：0（极度负面）~ 1（极度正面）
        """
        # 1. 获取当日新闻
        articles = self.ingestor.get_news_for_symbol(symbol, date)
        if not articles:
            return 0.5  # 无新闻，中性
        
        # 2. 词级初筛（高通量）
        lexicon_scores = [self.lexicon.score(a) for a in articles]
        mean_lexicon = np.mean(lexicon_scores)
        
        # 3. 句子级验证（仅对极端样本）
        if abs(mean_lexicon) > self.lexicon_threshold:
            bert_scores = self.bert.predict_batch(articles)
        else:
            bert_scores = lexicon_scores
        
        # 4. 跨句推理（仅对高分歧样本）
        if (np.std(bert_scores) > self.bert_threshold and 
            len(articles) >= self.critic_articles_min):
            critic_score = self.critic.analyze(symbol, articles, bert_scores)
            # 加权融合
            final = 0.5 * critic_score + 0.3 * np.mean(bert_scores) + 0.2 * np.mean(lexicon_scores)
        else:
            final = np.mean(bert_scores)
        
        return np.clip(final, 0, 1)
    
    def compute_factor(self, symbols: List[str], as_of_date: str) -> pd.Series:
        """
        截面情感因子
        返回：所有股票在 as_of_date 的情感分数（0~1）
        """
        scores = {}
        for symbol in symbols:
            scores[symbol] = self.compute_emotion(symbol, as_of_date)
        return pd.Series(scores)
```

### 3.6 CLI 集成

```python
# src/cli.py — 新增情感相关命令

@cli.command()
@click.option('--start', default="2022-01-01")
@click.option('--end', default="2025-12-31")
def ingest_news(start, end):
    """历史回填新闻数据"""
    ingestor = NewsIngestor()
    ingestor.backfill(start, end)
    print(f"新闻回填完成: {start} → {end}")

@cli.command()
@click.option('--date', required=True)
@click.option('--symbols', default=None)
def sentiment_factor(date, symbols):
    """计算单日情感因子"""
    if symbols is None:
        symbols = load_universe("hs300")
    
    agent = TriAgentSentiment()
    factor = agent.compute_factor(symbols, date)
    print(f"情感因子计算完成，{len(factor)} 只股票")
    
    # 保存到 outputs/
    factor.to_csv(f"outputs/sentiment_{date}.csv")

@cli.command()
@click.option('--start', default="2022-01-01")
@click.option('--end', default="2025-12-31")
def backtest_sentiment(start, end):
    """情感因子回测"""
    # 逐日计算情感因子 + 回测
    # 复用 Phase 8 回测框架
    pass
```


## 四、配置更新

```yaml
# configs/master_config.yaml

# === Phase 9.1 新增：情感配置 ===
sentiment:
  enabled: true
  
  # 数据采集
  data_dir: "data/news"
  backfill_start: "2022-01-01"
  backfill_end: "2025-12-31"
  
  # 分层阈值
  lexicon_threshold: 0.3      # 词级 → BERT 触发阈值
  bert_threshold: 0.25        # BERT → 批判者触发阈值
  critic_min_articles: 3      # 触发批判者最少新闻数
  
  # 模型
  bert_model: "bert-base-chinese"  # 推荐：FinBERT_zh
  critic_model: "deepseek-v3"
  
  # 成本控制
  cost_budget_monthly_usd: 1.0

# === Phase 9.1 因子组合 ===
factor_combination:
  factors:
    - source: "phase8"
      family: "low_vol_low_turnover"
      count: 5
      weight_cap: 0.25
    
    - source: "phase9.1"
      family: "sentiment"
      count: 1
      weight_cap: 0.30
    
    # 注：PEAD (9.2) 已关闭
```


## 五、执行清单

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 9.1 执行清单                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Week 1: 数据采集与词库（3天）                                     │
│  □ 1.1 实现 src/sentiment/ingestion.py（新闻抓取 + 状态管理）     │
│  □ 1.2 实现 src/sentiment/lexicon.py（中文金融词库）              │
│  □ 1.3 单元测试：抓取、去重、状态持久化                           │
│  □ 1.4 历史回填 2022–2025（后台运行，约 4-6 小时）               │
│  □ 1.5 验证数据完整性（日覆盖 ≥ 80%）                            │
│                                                                     │
│  Week 2: 情感计算（2天）                                           │
│  □ 2.1 实现 src/sentiment/bert.py（中文 BERT）                    │
│  □ 2.2 实现 src/sentiment/critic.py（DeepSeek 批判者）            │
│  □ 2.3 实现 src/sentiment/triagent.py（分层主框架）               │
│  □ 2.4 单元测试：分数范围 [0,1]，分层触发逻辑                     │
│  □ 2.5 人工抽样验证情感标签质量（约 100 条样本）                  │
│                                                                     │
│  Week 3: 回测验证（2天）                                           │
│  □ 3.1 实现 src/cli.py cmd_sentiment + cmd_backtest_sentiment     │
│  □ 3.2 情感单因子回测（2022-2025，HS300）                         │
│  □ 3.3 与 Phase 8 价量因子组合回测                                │
│  □ 3.4 生成 Phase 9.1 报告                                        │
│  □ 3.5 门控判定：IC > 0.01                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


## 六、门控与里程碑

### 6.1 门控条件

| 阶段 | 门控 | 失败处理 |
|------|------|---------|
| 数据采集 | 日覆盖 ≥ 70%（2022–2025） | 检查 AKShare 接口，延长回填时间 |
| 情感计算 | 非全 0.5（有正负区分） | 检查词库 + BERT 精度 |
| **回测验证** | 情感单因子 IC > 0.01 | 如果 < 0.01，停止并报告（Phase 9 全关） |
| **多因子组合** | 组合 Sharpe 提升 ≥ 0.2 | 报告结果，进入 Phase 10 或结束 |

### 6.2 里程碑标签

```bash
# 数据采集完成
git tag phase9.1-news-ingested

# 情感计算完成
git tag phase9.1-sentiment-ready

# 回测验证完成
git tag phase9.1-complete  # 条件：IC > 0.01
```


## 七、如果 Phase 9.1 也 FAIL

如果 9.1 情感因子 IC < 0.01，Phase 9 全部关闭，系统最终状态：

| 模块 | 状态 | 产出 |
|------|------|------|
| Phase 7（数据） | ✅ | 12.48M bars，B1-B5 全绿 |
| Phase 8（价量） | ⚠️ 门控未达标 | 5 个低波+低换手因子（有效） |
| Phase 9.1（情感） | ❌ 如果 FAIL | 基础设施提交，情感不可用 |
| Phase 9.2（PEAD） | ❌ FAIL | 基础设施提交，PEAD 不可用 |

**最终结论**：系统仅保留价量因子（低波+低换手家族），进入模拟盘部署或结束项目。


## 八、Phase 8 教训到 Phase 9.1 的改进

| Phase 8 教训 | Phase 9.1 改进 |
|-------------|----------------|
| 信文献结论在 A 股不成立 | 先用事件研究验证，再编码因子 |
| LLM 无视 Prompt 硬约束 | DeepSeek 批判者只做推理，不做生成 |
| 模板池反复抽中同一公式 | 所有已测公式进入 blocked 集合 |
| 单一 Alpha 方向 | 明确探索不同信息源（非价量） |
| 门控 FAIL 后继续挖 | 门控 FAIL 立即停止并报告 |


## 九、风险与应对

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| AKShare 新闻接口限速/失效 | 中 | 高 | 增加重试 + 缓存；备选 Tushare |
| 中文 BERT 精度不足 | 中 | 中 | 先用规则 + 批判者兜底；可微调 |
| DeepSeek 批判者成本超支 | 低 | 低 | 仅对 < 5% 分歧样本调用；$1/月预算 |
| 历史回填时间过长 | 中 | 低 | 逐日抓取约 4-6 小时，后台运行 |
| 情感因子与价量因子相关性 > 0.5 | 低 | 中 | 预期低相关；若发生则重新评估 |


## 十、最终 Prompt 给 Claude Code

> **Claude Code**, execute Phase 9.1 blueprint in full. Follow this order strictly:
>
> 1. **Week 1**: Implement `NewsIngestor` with state management and backfill 2022-2025 news data. Implement Chinese financial lexicon.
> 2. **Week 2**: Implement Chinese BERT sentiment (`bert-base-chinese` or `FinBERT_zh`). Implement DeepSeek critic (replace Qwen from original plan). Integrate TriAgent framework.
> 3. **Week 3**: Run sentiment factor backtest on 2022-2025 HS300. Merge with Phase 8 low-vol+low-turnover factors.
> 4. **Gate**: If sentiment single-factor IC > 0.01, tag `phase9.1-complete` and proceed to Phase 10 simulation.
> 5. **If IC < 0.01**: Stop, report, and do not proceed. Phase 9 closes fully.
> 6. **Critical**: Do not assume English FinBERT works on Chinese text — use Chinese-native models.
> 7. **Critical**: All LLM calls must use DeepSeek (not Qwen). The DeepSeek API key is already configured.

---

**Blueprint version**: 1.0
**Created**: 2026-08-11
**Based on**: PHASE9_2_PEAD_REPORT.md + Phase 9 evaluation
**Status**: 🔴 **Awaiting your approval** — once approved, Claude Code executes Phase 9.1