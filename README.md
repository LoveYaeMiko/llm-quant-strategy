# LLM Quant Strategy 2026

LLM 驱动的量化交易系统 —— **离线研发（LLM 重）与在线执行（纯确定性）严格分离**的
TiMi 架构。因子挖掘、回测、进化、审计全部发生在离线层；真正上线的信号计算、组合
优化与下单执行是一段不依赖 torch/transformers/LLM 的确定性代码。

蓝图在 [PROJECT_BLUEPRINT.md](PROJECT_BLUEPRINT.md)，2026 SOTA 论文与方法学在
`paper/` 与 `resource/review.md`。**实现同时参考了两者**——蓝图给出骨架，review.md
中蓝图未覆盖的偏误控制、成本审计与多假设检验等内容被逐一并入（对应关系见下方
"论文 → 组件"映射）。

---

## 1. 快速开始

```bash
# 1. 安装依赖（numpy / pandas / PyYAML / pytest 为最小集）
python -m pip install -e ".[dev]"          # 离线即可跑

# 2. 配置 LLM（可选——缺省时系统完全离线、确定性运行）
cp .env.example .env                        # 填入 DEEPSEEK_API_KEY
#    .env 被 .gitignore 忽略，密钥永不入库

# 3. 跑测试（107 个用例，全离线、全确定性）
python -m pytest tests/ -q

# 4. 蓝图自检清单
python cli.py verify

# 5. 常用命令
python cli.py mine     --iterations 3 --hypotheses 4 --seed 1   # 因子挖掘
python cli.py backtest --seed 1                                 # 回测池内因子
python cli.py evolve   --formula "Neg(TS_ZScore(Close, 10))"    # EvoQuant 进化
python cli.py export   --formula "Rank_Mul(Rank(Close), Rank(TS_Return(Close,10)))" \
                       --name mom_10                            # 导出到在线层
```

`python cli.py <cmd>` 与 `python -m src.cli <cmd>` 等价。

---

## 2. 架构总览

```
┌───────────────────────────── OFFLINE 研发层（LLM 重）─────────────────────────────┐
│  SignalAgent ──> CodeAgent ──> EvalAgent ──> RiskAgent          多智能体流水线       │
│      ▲              │              ▲              │              │                 │
│  语义空间     公式生成      IC/Sharpe 评估    五段门控          DynamicRouter       │
│  SchemaPlan    66+算子库    Bonferroni 校正    过拟合/稳健/      按市场状态路由      │
│  (AlphaSchema) (EvoQuant)   (FINSABER)      市场状态/多假设/回撤 (Sleipnir)        │
│                                                                                    │
│  FinCADWrapper —— 上下文感知解码：未来日期 logits 抑制 + prompt 清洗 + LookAheadAudit│
│  MemoryManager —— 结构化记忆（AlphaMemo 频繁子树回避）                             │
│  EvoQuant      —— 自我进化：诊断 → 候选 → 门控 → 蒸馏                          │
│  CostTracker   —— 月度 $500 预算闸门，逐 token 记账；ExperimentAuditor 全量留痕   │
└────────────────────────────────────────────────────────────────────────────────────┘
        │ 通过全部门控的因子被 compile_factor() 编译为 JSON 工件（无 Python eval）
        ▼
┌───────────────────────────── ONLINE 执行层（纯确定性）───────────────────────────┐
│  signal_calculator  CompiledFactor 求值（仅算数 + ts/cs 算子，无动态代码）        │
│  portfolio_optimizer  逐日 PCA 中性化 → 排名 → 多空 → 相关性聚类限仓 → 仓位上限   │
│  order_executor     现价撮合、滑点/手续费、每股持仓上限、黑名单、确定性强随机      │
└────────────────────────────────────────────────────────────────────────────────────┘
```

- **离线层**：依赖 LLM，产出因子池、审计记录、编译工件。慢、贵、可复现。
- **在线层**：只读编译工件 + 行情，微秒级确定性执行。不 import torch / openai。
- **PIT 承诺**：一切历史查询走 `PointInTimeStore`（`valid_from/valid_to`）；因子求值
  只消费 `as_of` 之前可见的事实；回测仅用因子值 + 当期之后实现的收益。

---

## 3. 目录结构

```
FQA/
├── cli.py                  # 薄启动器（无需 pip install -e）
├── pyproject.toml
├── configs/                # 单点配置源（改 YAML 即改全部闸门）
│   ├── master_config.yaml      # 项目/数据/LLM 路由/风控/在线执行
│   ├── factor_thresholds.yaml  # IC 阈值、lookback 上下界
│   └── llm_routing.yaml        # 模型映射 + 成本 + AgenticAITA 触发器
├── src/
│   ├── cli.py                  # argparse 入口（mine/backtest/evolve/export/verify）
│   ├── config.py               # YAML 加载 + ${ENV} 插值 + 极简 .env（stdlib）
│   ├── cost_tracker.py         # LLM 成本逐 token 记账 + $500 月预算闸门
│   ├── llm_client.py           # OpenAI 兼容后端（DeepSeek），懒加载 openai
│   ├── audit.py                # 审计记录：config 快照/路由快照/PIT 窗口/checklist
│   ├── checklist.py            # 蓝图自检：pit/fincad/diversity/cost
│   ├── evolver.py              # EvoQuant 自进化
│   ├── agents/                 # signal / code / eval / risk + 门控 + 路由
│   ├── bias_control/           # FinCAD 上下文解码 + LookAheadAudit
│   ├── backtest/               # PIT 感知回测引擎 + 指标（numpy-only spearman）
│   ├── factors/                # 语义空间 / 66+ 算子库 / 记忆 / 探索
│   ├── data/                   # PIT loader / 合成市场 / 文本数据源
│   └── online/                 # 确定性信号计算 / 组合优化 / 下单执行
├── tests/                 # 107 个用例，全离线
├── docs/                   # 智能体领域说明与问题跟踪
├── resource/review.md      # 2026 SOTA 综述（实现依据之一）
└── paper/                  # 论文 PDF 与参考仓库
```

---

## 4. 论文 → 组件映射（蓝图之外的 SOTA 并入点）

| review.md 论文 | 并入的机制 | 位置 |
|---|---|---|
| FINSABER | Bias Traps：PIT 可见性窗口 + 未来事实不可见；Bonferroni 多假设 Sharpe 校正 | `data/point_in_time_loader.py`, `backtest/metrics.py` |
| FinCAD | 上下文感知解码：未来日期 token 的 logits 惩罚、prompt 清洗、`LookAheadAudit`（要求 >50% IC 下降） | `bias_control/` |
| AlphaSchema | 语义空间五元组 Event/Context/Qualities/Direction/Output | `factors/semantic_space.py` |
| AlphaJungle | SchemaPlan 邻域探索（EvoQuant 候选生成用 `space.neighbors`） | `factors/schema_explorer.py`, `evolver.py` |
| AlphaMemo | 结构化记忆 + 频繁子树回避（避免重复挖掘同构因子） | `factors/memory_manager.py` |
| EvoQuant | 验证器引导的自进化：诊断 → 候选 → 门控 → 蒸馏 | `evolver.py`, `agents/risk_agent.py` |
| AgenticAITA | Adaptive Z-Score Trigger + Inference Gating（只在统计异常时调重模型） | `configs/llm_routing.yaml`, `agents/inference_gate.py` |
| Sleipnir | 动态路由：按 bull/bear/sideways 排序智能体执行次序 | `agents/dynamic_router.py` |
| TriAgent | 分层情感（sentence → document → market），浅层用廉价模型 | `configs/llm_routing.yaml`, `data/text_feeds.py` |
| Beyond Agent Arch | 成本/延迟纳入设计：$500 月预算闸门 + 代码缓存 + 确定性在线层 | `cost_tracker.py`, `online/` |

---

## 5. 验证

```bash
python cli.py verify
```

四个蓝图自检全部通过时输出：

```
[PASS] pit        query(2019-06-28) returned 40 facts, 0 born after 2019-07-01
[PASS] fincad     output scrubbed=True; cheating-factor IC 1.000 -> suppressed 0.000, reduction 100% (required > 50%)
[PASS] diversity  min pairwise AST distance = 1.00 (floor 0.4)
[PASS] cost       projected monthly cost $0.53 <= $500.00
```

---

## 6. LLM 配置与成本

- 后端：DeepSeek `https://api.deepseek.com/v1`，模型 `deepseek-v4-flash`
  （OpenAI 兼容，`src/llm_client.py` 懒加载 `openai`）。
- 密钥：`.env` 中 `DEEPSEEK_API_KEY`（`.gitignore` 已忽略）；`load_config` 用 stdlib
  读取 `.env` 并做 `${VAR}` 插值。
- 未配置密钥 / 未安装 openai 时，系统**完全离线确定性运行**（智能体走内置规则），
  测试与 `verify` 均不依赖 LLM。
- 每一次调用按 prompt/completion token 记账到 `CostTracker`；超过
  `budget.monthly_llm_cost_usd`（默认 $500）即拒绝继续调用。`deepseek-v4-flash`
  单价已内置（in $0.27 / out $1.10 每百万 token）。

---

## 7. 在线层确定性保证

1. `export` 把公式编译为 `CompiledFactor`（含算子树 JSON、lookback、字段、算子清单），
   在线求值器**只做白名单算子 + 字段访问**，无 `eval`/动态代码路径。
2. 组合优化逐日独立：PCA 中性化（SVD 残差化）→ 排名 → 多空 → 相关性聚类限仓 →
   单名上限，全部 pandas/numpy，无随机数。
3. 下单执行用固定种子随机分配成交价，结果可重放；`test_online.py` 断言两次执行
   输出字典完全一致。
4. 在线模块 `import` 链不含 torch/transformers/openai，物理上不可能接触 LLM。

---

## 8. 运行清单（自查）

- [x] 107 个单元测试全绿（`pytest tests/ -q`）
- [x] `verify` 四检通过（PIT / FinCAD / 多样性 / 成本）
- [x] `mine` / `backtest` / `evolve` / `export` 离线可跑
- [x] 配置了 `DEEPSEEK_API_KEY` 时，LLM 路径端到端生效且成本记账
- [x] `.env` 被忽略，密钥不入库

下一步工作项与待补充数据，见 [docs/requirements.md](docs/requirements.md)。
