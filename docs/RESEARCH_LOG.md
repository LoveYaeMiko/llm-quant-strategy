# 研究日志（Research Log）

按阶段记录因子研究的关键结论、门禁判定与资产重定位。每条记录指向对应蓝图与报告。

| 日期 | 阶段 | 结论 | 门禁 | 关键指标 | 去向 |
|------|------|------|------|----------|------|
| 2026-08-18 | 外部调研 | TradingAgents 系列/AI 交易方法：FQA 数据+验证已碾压，值得抄 LLM 工程+门控纪律 | 暂不落地 | 见 `docs/EXTERNAL_RESEARCH_TRADINGAGENTS.md` | 存档待排期 |
| 2026-08-12 | Phase 9 关闭 | 三次失败统一根因：HS300 已披露信息被充分定价 | PEAD/情绪/文本均 < 0.015 | 情绪 rank_ic=0.0094，文本 dispersion@20=0.0136 | Phase 10 三层融合 |
| 2026-08-12 | Phase 9.1b 文本因子 | 分歧度/新颖性 FAIL | rank_ic > 0.015 | dispersion@20=0.0136，novelty 全弱 | 关闭，向量缓存保留 |
| 2026-08-11 | Phase 9.2 PEAD | 负漂移，反转仍不过门 | rank_ic > 0.015 | Q4-Q0 fwd20 = −2.6% | 重定位为战术倾斜 |
| 2026-08-11 | Phase 9.1a 研报情绪 | 情绪 IC 不足 | rank_ic > 0.015 | rank_ic=0.0094，BERT 提升 6 倍 | 重定位为风控熔断 |
| 2026-08-10 | Phase 8 价量因子 | 低波+低换手有效 | IC/ICIR 双门槛 | 5 因子 rank_ic 0.0205–0.0225，Sharpe 1.58–1.90 | Phase 10 Alpha 核心 |

> 详见：`PHASE9_CLOSURE.md`、`docs/PHASE9_2_PEAD_REPORT.md`、`docs/PHASE8_REPORT.md`、`blueprint/PHASE10_BLUEPRINT.md`、`docs/EXTERNAL_RESEARCH_TRADINGAGENTS.md`。
