# 需求清单（框架已完成后的下一步）

基础框架已完成：**107 个测试全绿，`verify` 四检通过，5 个子命令离线可跑，
DeepSeek LLM 路径已接通并记账**。以下是从"可演示框架"走到"可实盘/可研究"所需
补充的需求清单，按优先级排列。

---

## A. 需要你提供（凭据 / 数据）

| # | 项 | 用途 | 现状 |
|---|---|---|---|
| A1 | **行情数据接入**（日线 OHLCV，建议 ≥ 5 年，含上市/退市事件） | 替换合成数据 `make_synthetic_market` | 框架用合成数据兜底，未接真实数据 |
| A2 | **Point-in-time 事实表**（财报、盈利修订、指数成分调整、事件时间戳） | PIT loader 的反向测试与防未来函数的真实输入 | 仅合成记录 |
| A3 | 新闻/另类数据 API key（如 NewsAPI、彭博、万得） | TriAgent 分层情感的真实数据源 | `.env` 中 `NEWS_API_KEY=` 为空 |
| A4 | DeepSeek key 已就位 ✅（存于 `.env`，勿入库） | 全部 LLM 角色 | 已配置 |
| A5 | 若要接入非 DeepSeek 模型（gpt-4o/claude） | 角色级模型路由（`routing.*.model`） | 框架预留设计意图，未配凭据 |
| A6 | 生产数据库（PostgreSQL 建议） | 替换 `sqlite:///data/pit_data.db` | `configs/master_config.yaml` 已预留 url |

## B. 数据层（最高优先）

- [ ] **B1 真实行情导入器**：CSV/Parquet → `PointInTimeStore.upsert`，校验
      `valid_from/valid_to` 无重叠、无未来事实（`has_future_leak`）。
- [ ] **B2 事件日历**：盈利公告日、指数调整日、财报日，用于 SchemaPlan 的
      Event 维度真实化。
- [ ] **B3 拆分/股息调整**：`adj_close` 的一致性处理，否则 `TS_Return` 在除权日
      产生假跳变。
- [ ] **B4 生存者偏差治理**：universe 必须包含已退市标的（PIT 语义已支持，
      需真实退市名单）。
- [ ] **B5 数据新鲜度监控**：行情截至日期、缺失率、异常点检测，作为审计记录字段。

## C. 因子挖掘（研究质量）

- [ ] **C1 更长的回看**：`--iterations` 现在默认 3×4，正式研究建议 ≥ 50×20，
      配合 Bonferroni 校正已有。
- [ ] **C2 walk-forward 参数化**：训练/验证/样本外三段切分，EvoQuant 候选只在
      训练段选择，验证段复验。
- [ ] **C3 因子衰减监控**：上线因子的 rolling rank_ic 与 ICIR 跟踪，跌破阈值
      自动下线（调度任务）。
- [ ] **C4 行业/市值中性化**：组合优化有 PCA 中性化，但因子本身未做行业中性，
      建议按 GICS 一/二级行业分组中性。
- [ ] **C5 更细的市场状态分桶**：现为 bull/bear/sideways，可加
      high_vol_low_ret 等四态/五态，`classify_market_state` 已参数化。

## D. 执行与风控（实盘前）

- [ ] **D1 券商/交易所 API 接入**：`OrderExecutor` 已内聚下单逻辑，需真实路由
      （撮合 API、回执、对账）。
- [ ] **D2 交易成本模型校准**：滑点/佣金现为固定 bps 假设，需按标的流动性、
      交易时段校准。
- [ ] **D3 资金约束**：组合优化未建模最小交易单位（手/股）、最小下单金额、
      现金余额硬约束。
- [ ] **D4 实时停牌/涨跌停处理**：现为黑名单 + 确定性撮合，实盘需处理停牌、
      集合竞价。
- [ ] **D5 券商当日风控**：单日最大成交额、单票最大亏损、账户净值阈值。
- [ ] **D6 回测与实盘一致性**：Slippage 假设、成交时点（收盘价 vs 次日开盘价）
      需与实盘路径一致，否则回测会过度乐观。

## E. LLM / 成本

- [ ] **E1 提示词缓存**：DeepSeek context caching 命中率统计，降低重复调用成本。
- [ ] **E2 失败重试与退避**：`llm_client` 现为单次调用，需指数退避 + 降级到
      离线规则。
- [ ] **E3 预算告警**：`CostTracker` 超过月度预算 80% 时输出告警（邮件/webhook）。
- [ ] **E4 结构化输出校验**：LLM 返回的公式需二次 parse + 66 算子白名单校验
      （已有，建议加自动重试一次）。
- [ ] **E5 可观测性**：每次 LLM 调用的 prompt/response 落盘（审计已记录用量，
      未记录内容——敏感数据脱敏策略待定）。

## F. 工程化

- [ ] **F1 持续集成**：GitHub Actions 跑 `pytest tests/ -q` + `verify`，密钥走
      secrets 不进仓库。
- [ ] **F2 实验追踪**：AuditRecord 已落 JSON，建议接 MLflow/W&B 或自建
      `runs/` 目录规范（run_id、config hash、因子池 diff）。
- [ ] **F3 调度**：日更数据 → 挖掘/进化 → 导出 → 回测 → 生成今日目标组合，
      用 cron/airflow 串起。
- [ ] **F4 打包与版本**：`pyproject.toml` 已可 `pip install -e .`，建议 lock 依赖
      版本、固定 Python ≥ 3.10。
- [ ] **F5 文档补全**：`docs/agents/*.md` 已有领域说明，补充 CLI 手册与
      config 字段字典。

## G. 明确不做（范围护栏）

- **不**做实时盘口 tick 级高频——本系统是日频/低频因子系统。
- **不**在在线层引入 torch/LLM——确定性是硬约束（`test_online.py` 守护）。
- **不**自动下单——D1 未完成前，`export` 产物仅用于模拟盘。

---

## 建议顺序

**阶段 1（数据地基）**：A1+A2+B1-B4 → 用真实 PIT 数据重跑 `verify`。
**阶段 2（研究可信）**：C1-C4 → 建立 walk-forward 与衰减监控。
**阶段 3（可模拟盘）**：D2+D4+D6+F3 → 校准成本、接入模拟盘路由。
**阶段 4（实盘前置）**：D1+D3+D5+E2-E3 → 券商接入 + 硬风控 + 成本告警。

每完成一个阶段，`verify` 四检都应保持全绿。
