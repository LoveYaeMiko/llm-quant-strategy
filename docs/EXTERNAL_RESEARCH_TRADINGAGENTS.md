# 外部参考项目调研 —— TradingAgents 系列 + AI 交易方法（2026-08-18）

> 状态：**已调研，暂不落地**（用户决策 2026-08-18：先存档，后续排期）。
> 来源：两个 GitHub 仓库 + 一篇 arXiv 论文 + 一个视频教程代码库。全部通读源码/全文。

## 一、调研对象与一句话价值

| 资源 | 定位 | 给 FQA 的核心价值 |
|------|------|------------------|
| [ai-trading-videos](https://github.com/frank-quant/ai-trading-videos)（EP004 四大 LLM 量化基准） | 四家 LLM 同题写加密策略，样本外揭盲评测 | ⭐ **最对症**：一套「因子门控纪律」——Deflated Sharpe 多重检验校正、全量/截断防未来函数、train/valid 落差惩罚、拒绝 argmax、block bootstrap |
| [TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN) | 中文版多智能体交易框架（LangGraph 编排） | LLM 工程层：Provider 抽象工厂、快/慢双模型分层、结构化信号多级降级、反思记忆 |
| [TradingAgents-CN-studio](https://github.com/frank-quant/TradingAgents-CN-studio) | 上述框架的零侵入运维增强层 | 运维层：统一事件流 + 单文件 HTML 回放、结构化日报提炼、多渠道通知 + cron 调度 |
| [arXiv:2412.20138](https://arxiv.org/abs/2412.20138)（TradingAgents 论文） | 原始框架论文（Tauric Research） | 方法论：Bull/Bear 辩论 + 结构化文档通信；但评测薄弱、成本高 |

## 二、总判断

**FQA 在数据严谨性与验证体系上已碾压 TradingAgents**：

- TradingAgents-CN 数据层是 **MongoDB 最新值覆盖式快照**，无 `as_of`/`point_in_time` 语义 → 必然前视偏差；FQA 的 PointInTimeStore 杜绝前视。
- TradingAgents-CN **没有回测引擎**（只有前向信号 + 纸面交易 + 反思日志）；FQA 有 IC/ICIR 门控 + 单因子 PIT 回测 + 三层组合仿真（Sharpe 1.70 / maxDD 9.2%）。

**结论：只抄它的「LLM 工程 + 编排」这层皮，坚决不抄它的「数据 + 回测」这层。**

真正值得借鉴的三块（按对症程度）：① 因子门控的多重检验校正（FQA 最真实短板）；② LLM 工程抽象；③ 运维/可观测性。

## 三、分类改进建议

### A. 门控严谨性（最优先，直接对症因子挖掘）

FQA 现状：IC 阈值 + 组合 Sharpe + 危机回撤 + Bonferroni 校正（`src/backtest/metrics.py::significance_threshold_sharpe`）。LLM 大规模生成因子是多重检验重灾区，缺更精细一环。

**A1. Deflated Sharpe（N 校正）—— 最该优先落地**
- 现状 Bonferroni 假设收益正态、对 trial 一视同仁。
- DSR 增量：回答「试了 N 次，纯运气能刷出的最高夏普是多少，你的夏普有没有显著超过」。偏度/峰度修非正态。
  - `SR₀ = √(各轮成绩方差) × [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]`，γ=0.5772。
  - `DSR = Φ( (SR̂−SR₀)·√(T−1) / √(1 − g₃·SR̂ + (g₄−1)/4·SR̂²) )`。
  - 判定：DSR ≥ 0.95 显著；< 0.9「大概率搜出来的运气」。
- 血泪教训（EP004 REPORT）：Kimi 搜 400 轮、验证夏普 1.57，但运气基线 2.55，DSR=0.094——**「看起来能用」统计上完全不成立**。FQA 若不对 N 惩罚，几千公式里挑 IC 最高那个几乎必是运气。
- 落地：`metrics.py` 加 `deflated_sharpe`；门控处把「家族阻塞后的有效独立试验数」记为 N（**用有效独立数而非全部生成数**，后者会过度惩罚误杀真实因子）。改动量中。

**A2. 全量 vs 截断因果检验（防未来函数）**
- 对同一时点 t，用「全量数据」与「截断到 t 的数据」算因子值，必须一致否则用了未来。抓 `shift(-1)`、中心化窗口（rolling 后全样本均值归一）、横截面含未来三类泄漏。
- FQA 已有 `src/bias_control/look_ahead_detector.py`，可增强为过门控因子的**硬门槛**。改动量小。

**A3. train/valid 落差惩罚 + 拒绝 argmax（邻域高原选参）**
- 四大 LLM 评测算出 train/valid 夏普相关性 r≈0.21——验证集排名基本是噪音，取 argmax = 取最大噪声。
- 落地：因子不只看「IC 最高那个」，还要看邻近公式变体是否也高（高原而非尖峰）；记录 in/out-of-sample IC 相关性 r 作为过拟合早期信号。改动量中。

**A4. 尾部多空价差 而非 rank IC 作为因子可交易性判据**
- EP004 硬结论：**横截面动量（10-45d）是唯一 train+valid 两端都强正的因素**；短期反转（1-5d）方向不稳定；**低波动 IC 为正但尾部价差为负（level 效应，不可交易）**——与 FQA 采用的 low-vol 因子方向直接相关，值得警惕。
- 通用规律：IC 会被 mid-book 噪音抵消，尾部价差才是真可交易信号。FQA `factor_eval` 已有 top/bottom 10% 多空组合，可升格为**主判据**。改动量中。

### B. LLM 工程抽象

**B1. LLM Provider 抽象层（改动小，价值高）**
- FQA 深度绑定 DeepSeek。TradingAgents-CN `llm_clients/` 包：provider 归一化工厂 + env key 映射 + 默认网关 + 12+ 家 OpenAI 兼容。换模型 = 改一行配置。
- 落地：新建 `fqa/llm_clients/`，抽掉现有裸客户端。改动量小。

**B2. Quick/Deep 双模型分层（改动小，直接省钱）**
- 中间推理节点用便宜快模型，综合裁决节点用贵慢模型。FQA 因子挖掘 + 组合仿真大量中间推理不需要最强模型。

**B3. 结构化信号多级降级（改动中）**
- `SignalProcessor` 的「LLM 抽 JSON → 13+ 种中文正则兜底 → 智能推算 → 保底默认值」四层容错，移植到因子公式解析与信号归一处。

### C. 运维 / 可观测性（FQA 完全空白）

**C1. 统一事件流 + 单文件 HTML 回放（改动小）**
- studio `core/events.py` 的 `TimelineEvent`（ts/phase/agent/content/kind）+ 单文件自包含 HTML。
- FQA TriAgent 三档输出（lexicon/BERT/critic）目前只有拒绝理由，无贯穿审计轨迹。统一成事件流后可一键导出「某股票某天舆情决策全过程」。

**C2. 结构化日报 + 通知调度（改动中）**
- FQA 因子挖掘/回测/组合仿真都是长任务，跑完不吭声。加「每日因子挖掘简报（LLM 压缩四段式）+ webhook/飞书推送 + cron 调度」。

## 四、明确不抄（避免盲目照搬）

- **TradingAgents 数据层**（MongoDB 快照无 PIT）—— FQA 已碾压
- **TradingAgents 回测**（根本没有）
- **单票交易执行层**（FQA 是因子挖掘，不是交易员角色）
- **加密专属因子**（funding rate、Choppiness 等）
- **论文评测方法**（3 个月 / 3 只美股 / Sharpe 异常高 5.6-8.2 / 缺 single-agent 消融 / 每次 11 LLM 调用 + 20+ 工具调用成本爆炸）
- **照搬交易层 vs 参考 QuantAgent**：对 FQA，论文 Related Work 的 **QuantAgent「writer(写因子)→judge(评审)→回测反馈 judge 闭环」** 才是比交易执行层更贴合的范式。

## 五、关键证据要点（供后续落地引用）

**四大 LLM 评测（EP004，加密永续，样本外 2025-07~2026-07 单边熊市）**：
- 排名 Fable 5 > DeepSeek V4 Flash ≈ Opus 5 > Kimi K3；四个策略样本外**全部由正转负**（验证 1.24/1.34/1.87/1.93 → 样本外 −3.71/−1.42/−0.62/−0.62）。
- 只有 Fable 有真 alpha（+7.9%）但仓位错（34% 净多头被熊市 beta 拖 14 点）；「各做对一半，没有两样都对」。
- 成本与产出无单调关系：最贵 Opus（¥363，5906 万 token）alpha 最差，最便宜 DeepSeek（¥1.04）成绩最好——成本由 token 总量驱动，非单价。

**TradingAgents 论文实验**：AAPL CR +26.62%（vs Buy&Hold −5.23%）、GOOGL +24.36%、AMZN +23.21%；MDD 0.9%-2.1%。作者自标 Sharpe >3 属「超经验区间」，归因于 3 个月回撤极少。

## 六、优先级建议与当前决策

推荐落地顺序（价值 × 改动量）：
1. **A1 Deflated Sharpe + A4 尾部价差判据**（门控严谨性，最对症）
2. **A2 全量/截断因果检验**（改动小收益大）
3. **B1 LLM Provider 抽象 + B2 双模型分层**（解绑 + 省钱，改动小）
4. **C1 事件流 + C2 日报/通知**（填运维空白）
5. 其余（记忆回路、辩论评审、邻域选参）中长期

**当前决策（2026-08-18）**：先存档不落地，上述建议待后续排期。落地时注意 DSR 的 N 需用「家族阻塞后的有效独立试验数」。
