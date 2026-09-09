# 前向期协议（Forward Protocol）

> 本文件是「前向期（observe 模式）到底能证明什么」的权威说明，配合
> `configs/forward_policy.yaml`（阈值/候选/切换规则，预注册口径）与
> `src/forward/{prereg,risk_gate}.py`（实现）使用。
> 2026-09-09 审计项 1.1–1.4、2、3 的落地文档。

## 0. 一句话结论

**前向期不是 alpha 检验，而是管线可信度检验。** 前向窗口在统计上无法证明 alpha
（§1.1）；它能高信噪比地回答三件事：管线是否忠实执行冻结规格、成本模型是否与真实
成交一致、运行是否可靠（§1.2）。因此**能力闸门（Sharpe 门限）被风险闸门取代**（§1.3），
所有结论必须先预注册、后评估（§1.4）。候选参数只做配对跟踪、只记录、不自动切换（§2）。
在数据地基的偏差压力测试通过之前，**禁止开启任何 alpha 层自进化**（§3）。

---

## 1. 前向期能证明什么

### 1.1 功效分析：前向窗口永远证明不了 alpha

年化 Sharpe 的抽样标准误为

```
SE(SR) = sqrt(252 / N)            N = 窗口内交易日数
t      = SR / SE = SR × sqrt(N / 252)
```

反解达到给定 `t` 所需年数（`N = 252 (t/SR)²`）：

| 真实 Sharpe | t=2.0 需要 | t=2.5 需要 |
| --- | --- | --- |
| 1.0 | 4.0 年 | **6.3 年** |
| 0.7 | 8.2 年 | 12.8 年 |
| 0.5 | 16.0 年 | **25.0 年** |

而策略参数与市场制度的匹配有效期只有 **1–3 年**（`regime_matching_horizon_years`）。
等窗口攒够功效时，参数已不属于那个制度——所以「用前向 Sharpe 判断策略好坏」测的是
噪声，不是证据。**前向期不得用于宣称 alpha 显著**；任何「前向 X 个月 Sharpe=…」的
引用都必须在旁边标注 `SE` 与 `t`（`src/forward/risk_gate.py::years_for_t` 可复算）。

### 1.2 前向期唯一能回答的三类高信噪比问题

| 问题 | 指标 | 为什么前向期有功效 |
| --- | --- | --- |
| `pipeline_tracking_error` | 逐日「实录 vs 重放」差值（pp/日）+ 符号偏差检验 | 实现差异是确定性的，不需要 alpha 的信噪比 |
| `cost_model_calibration` | 计费 vs 预注册成本规格；成交价 vs 市场参考价 | 每笔成交都是一次观测，与收益无关 |
| `operational_reliability` | 可用率、数据新鲜度、违规计数、符号级覆盖率 | 二值事件，样本量以「日/笔」计而非以「年」计 |

---

## 1.3 风险闸门（替代能力闸门）

`configs/forward_policy.yaml → forward.risk_gate`；实现 `src/forward/risk_gate.py::evaluate_gate`；
运行 `python scripts/forward_health.py`（退出码 0 = 全部通过）。

### 硬闸门（任一失败 ⇒ 前向期判定失败，停机排查）

硬闸门失败的含义是**管线不可信**，不是「策略不行」。

| 闸门 | 阈值 | 测量方式 | 未测量时 |
| --- | --- | --- | --- |
| 跟踪误差 | ≤ 0.2 pp/日 | 复制生产账本、截断到窗口开始前的状态，用**当前数据**重放同一段；逐日 `|实录 − 重放|` | 判失败 |
| 跟踪误差符号偏差 | 二项检验 p ≥ 0.05 | 差值为正/负的日数做精确双侧二项检验 | 判失败 |
| 费用偏差 | ≤ 20% | 账本计费 vs **预注册成本规格**（按每笔自身 notional/方向重算） | 判失败 |
| 成交价完整性 | ≤ 2 bp | 成交价 vs 市场参考价（收盘/拍卖单用当日调整收盘价，实时单用同分钟成交价），仅 `live/auction/close` 来源 | 判失败 |
| 违规计数 | = 0 | T+1 / 整手 / 最小变动价位 / 涨跌停（方向感知）/ 停牌，五项之和 | 判失败 |
| 实时层可用率 | ≥ 99% | `outputs/live_<acct>.jsonl` 心跳在决策窗口内、间隔 ≤5 分钟的覆盖率 | 判失败（窗口内无心跳记录） |
| 数据新鲜度 | ≤ 1 自然日 | `pit_records.max(valid_from)` vs 账本最后一个交易日 | 判失败 |
| 符号级分钟覆盖 | ≥ 95% | 按**每个符号自身的可交易日**为分母（停牌/上市前不算洞） | 判失败 |

**「未测量」一律判失败**（`evaluate_gate` 对缺项返回 `ok=false`）。这条规则是刻意的：
把「没测」读成「没问题」正是审计发现的那类错误。

### 软指标（只记录，永不参与判定）

`sharpe / max_drawdown / excess_return / hit_rate / n_fills / turnover`
—— 前向窗口对它们没有功效（§1.1），写进工件是为了**留档**，不是为了判分。

### 为什么旧红线是同义反复

旧 `cost_deviation` 红线把账本 `commission` 与同一套成本模型重算的值相比：两边来自
同一个模型，永远接近 0。新实现把可测与不可测分开：

* **可测**：计费 vs 预注册规格（能发现配置漂移）；成交价 vs 市场参考价（能发现陈旧/
  错误/凭空的成交价）。
* **不可测**：市场冲击成本（`market_impact_bps`）。纸面账户没有券商成交单，**无法**
  观测冲击成本。因此该分量明确列入 `unmeasured`，并作为**实盘前置条件**：
  `deployment.real_money_enabled` 打开前，必须用真实成交回填这一项。

---

## 1.4 预注册（pre-registration）

模板六字段（`src/forward/prereg.py::PREREG_FIELDS`）：

| 字段 | 含义 |
| --- | --- |
| `rule_id` | 规则稳定标识，永不复用 |
| `frozen_at` | 冻结时间戳（校验时必须在过去） |
| `scope` | 适用范围：账户 / 股票池 / 窗口 / 参数 / 本次唯一改动 |
| `decision` | 通过逻辑，细到可机械执行（hard / soft / verdict） |
| `stopping` | 何时停止评估：窗口长度、kill 条件 |
| `trials` | 多重比较记账：家族、既往次数、本次序号 |

另附 provenance：`code_commit`（HEAD）、`config_sha256`（合并后配置）、
`record_sha256`（自身规范 JSON 哈希）。记录**只追加**：

```bash
python scripts/prereg.py template --rule-id d_forward_2026h2 > rule.json   # 填好六字段
python scripts/prereg.py new --file rule.json        # 冻结 → outputs/forward/prereg/
python scripts/prereg.py verify --require-commit      # 校验：字段/哈希/时间/代码
python scripts/prereg.py list
```

* 改规则必须**升版本 + 写 `supersedes`**（等价于承认这是新的一次试验）；
* `write_preregistration` 拒绝覆盖内容不同的同名记录；
* 任何人在看结果后手改 JSON，`record_sha256` 会立刻对不上（`verify` 退出码 1）。

---

## 2. 前向候选：`atr_1p0_25_40`

```bash
python scripts/forward_candidate.py run --date 2026-09-10   # 推进候选影子账本
python scripts/forward_candidate.py report                  # 配对比较
python scripts/forward_candidate.py daily                   # run + report（PAICC 每日任务）
```

* **隔离**：候选有自己的账本与状态文件
  （`outputs/forward/candidate_atr_1p0_25_40/{ledger.sqlite,status.json}`），
  生产账本只读、绝不写入。
* **同一制度、唯一差异**：数据切片、股票池、代码路径、入场/退出规则、14:50 订单列表、
  15:00 拍卖成交全部与生产一致（`pb_preclose_account` 让候选继承**生产**的订单列表）；
  唯一差异是止损宽度 `pb_atr_mult=1.0, pb_stop_lo=0.025, pb_stop_hi=0.040`。
* **只记录、不自动切换**：`deployment.mode = observe`，候选没有晋升通道；
  切换必须人工执行并重新预注册。
* **配对比较**（而不是比较两个 Sharpe）：两本书日收益相关性实测 **0.884**，
  共同因子占绝大部分方差，必须用**同日差值序列**做检验。

### 预注册切换规则

```
前向窗口 120 个配对交易日：
    配对日差均值 > 0  且  配对 t > 1.5   →  切换（记为新的一次试验）
    否则                                →  保持现役
```

### 干净窗口实测（2025-09-01→12-31，回补后）

| 口径 | 累计 | Sharpe | maxDD | 成交笔数 |
| --- | --- | --- | --- | --- |
| 现役 flat 3.5% | +1.64% | 0.40 | 5.00% | 106 |
| 候选 atr 1.0/[2.5%,4.0%] | +3.38% | 0.65 | 7.25% | 114 |

配对结果：`corr=0.884`、日差均值 `+0.0226 pp/日`（年化 +5.68pp）、`t=0.376`、
`hit_rate=0.494`。**按观测到的效应量，达到 t>1.5 需要约 1291 个配对交易日（≈5.1 年）**
（`days_needed_for_t`）。结论：**保持现役（HOLD）**，继续前向跟踪。

### 「可能永不分离」是合法结果

候选与现役的差异只有 0.85pp 的止损宽度，且相关性 0.884；即使候选真有优势，
分离所需样本也超过制度有效期。因此规则明确接受 `may_never_separate: true`：
**不设时间压力、不为了「有结论」而降低阈值、不因候选暂时领先就切换。**

---

## 3. 数据地基：自进化的前置条件

三条阻断项（审计 C1/C2/C3）未解决前，**alpha 层自进化没有意义**——进化循环会把数据
缺陷当成 alpha 学进去。

### C1 幸存者偏差（最严重）

`pit_records` 中约 **272 个 `universe` 符号没有任何价格记录**（退市/从未交易），
而任何基于「当前成分股」的回测都会静默丢掉这些名字；动量类策略尤其受益（幸存者里
的赢家被保留、输家被剔除）。

* 测量：`scripts/bias_stress_test.py` → `outputs/bias_stress_<label>.json`；
* 闸门：`偏差 > 0.3 × 目标 alpha`（默认 0.3 × 8pp = **2.4pp/年**）⇒
  `bias_blocking_evolution = true`，禁止开启自进化；
* 做法：断点（break-even）分析 + 经验退市率上界 + **去偏子集对照回测**（基线 vs
  剔除高风险段后的股票池），详见 `docs/BIAS_STRESS.md`。

**实测（2025H2 干净窗口，2026-09-09）**：

| 口径 | 结果 |
| --- | --- |
| 市场级年化退市率 | 0.798%/年（230/2594 消失，其中 131 含「退」、72 含「ST」） |
| D 池 800 名 | 11.6 年零消失 → rule-of-three 上界 0.0528%/年 |
| 上界构造 `x_upper_bound` | 0.798% × (40/252) × 1 = **0.1267%** |
| 拖累（L=−50%，166 笔入场/年） | **1.75pp/年 = 0.22 × α < 2.4pp ⇒ 不阻断** |
| 盈亏平衡退市率 | 1.09%/年 —— 实测退市率已达其 **73%**（余量很窄） |
| 去偏子集对照（剔除低价+低流动性 12.1%） | 年化 **+5.20% → −4.40%（−9.60pp）** |

**读法**：退市通道的上界低于阈值，但**去偏子集对照显示 D 轨的样本外收益恰恰集中在
风险段**（差值 −9.6pp 远大于 2.4pp 阈值），且 L=−80% 情形下拖累为 2.80pp > 2.4pp。
因此结论是**「阻断项仍未解除」**：在补齐历史成分股名单、或明确接受 L=−80% 情形之前，
**不得开启 alpha 层自进化**。该判定由 `scripts/bias_stress_test.py` 的
`bias_blocking_evolution` 与 `docs/BIAS_STRESS.md` 共同给出。

### C2 存储层混合基准

`pit_records` 里 `close` 是**复权后**价格，而 `open/high/low` 是**原始**价格；
唯一处理过它的消费者是 pullback book 内部的 ATR 修复。契约落在 `src/data/basis.py`：
读取层显式标注每一列的基准、可断言、可转换（`to_adjusted` / `to_raw`），
**不静默改动数值**（否则所有历史证据都会变）。详见 `docs/BASIS_CONTRACT.md`。

### C3 复权锚点无版本

`close = raw_close × adjust_factor`，因子锚定在**最新一根 bar**，且没有任何版本记录。
每次重新入库都会整体重基（re-base）历史：昨天算出的 close 序列不等于今天的 close 序列，
于是「同参数重跑」也不可复现，前向跟踪误差会被基准漂移污染。

* 记录：`scripts/check_adjust_anchor.py --baseline/--capture/--compare` →
  `outputs/data/adjust_anchor*.json`；
* 规则：开长窗前先冻结基线；`--compare` 出现漂移即退出码 1，必须显式重新冻结并记录原因；
* 前向跟踪误差（§1.3）正是这条缺陷的运行时探测器：数据一漂移，重放立刻偏离实录。
  详见 `docs/ADJUST_ANCHOR.md`。

---

## 4. 运行手册

| 频率 | 命令 | 作用 |
| --- | --- | --- |
| 每日（收盘后） | `python scripts/forward_candidate.py daily` | 推进候选影子账本 + 配对报告 |
| 每周 | `python scripts/forward_health.py` | 风险闸门全量评估（含重放跟踪误差） |
| 每周 | `python scripts/check_adjust_anchor.py --compare` | 复权锚点漂移检测 |
| 每次发布证据前 | `python scripts/check_provenance.py` | 工件 provenance 完整性（五项必填） |
| 每月 | `python scripts/bias_stress_test.py` | 偏差压力测试（自进化闸门） |

* `forward_health.py --no-replay` 只用于排障：重放被跳过后跟踪误差与符号覆盖率
  均为「未测量」，闸门必然失败（这是设计，不是 bug）。
* PAICC 面板：`quant_scheduler` 已包含 `quant_live_watchdog`（实时层存活），
  `quant_d_cycle_enabled=false` 保持不变（3 个挑战者中 2 个 OOS 为负）。

---

## 5. 局限与未测量项（必须随结论一起引用）

1. **市场冲击成本不可测**：纸面账户没有券商成交单；`slippage_bps=2.0` 是假设，
   不是观测。实盘前必须用真实成交回填（§1.3）。
2. **跟踪误差测的是「一致性」不是「正确性」**：重放与实录一致只能证明管线可复现；
   如果冻结的规格本身错了，两边会一起错。
3. **可用率只能从心跳机制上线之日起测**（2026-09-09 起，
   `outputs/live_D_5W.jsonl`）；更早的窗口判为「未测量」。
4. **前向窗口对 Sharpe / maxDD / 超额没有功效**（§1.1），软指标仅供留档。
5. **C1 的偏差是估计上界**：退市名没有价格数据，无法直接回测；压力测试给出的是
   断点与上界，不是精确值。结论的强度取决于「上界 < 0.3α」是否成立。

## 6. 相关文件

| 文件 | 作用 |
| --- | --- |
| `configs/forward_policy.yaml` | 阈值、候选、切换规则、偏差闸门（唯一权威来源） |
| `src/forward/prereg.py` | 预注册记录：构造/校验/追加写/多重比较计数 |
| `src/forward/risk_gate.py` | 功效分析、跟踪误差、成本、违规、可用率、配对比较（纯函数） |
| `scripts/prereg.py` | 预注册 CLI（template / new / verify / list / hash） |
| `scripts/forward_health.py` | 风险闸门评估 → `outputs/forward/forward_health.json` |
| `scripts/forward_candidate.py` | 候选影子账本 + 配对报告 |
| `src/provenance.py` + `scripts/check_provenance.py` | 工件 provenance 五项必填校验 |
| `docs/D_TRACK_EVIDENCE.md` | D 轨证据总表（窗口/口径/样本量/出处） |
