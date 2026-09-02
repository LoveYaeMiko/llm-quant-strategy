# Factor Zoo — GitHub 公式动物园提取管道

把外部开源公式库翻译进 FQA 闭式算子库，并全量 walk-forward 扫描。**只提取假设
（公式），不提取结论**——一切候选都经 FQA 自己的 PIT 数据与三窗门禁验证。

## 管道

```
① 提取  scripts/extract_factor_zoo.py / extract_alpha158.py
         → inventory.json（原文/AQML/元数据）
② 翻译  src/exploration/translate.py（AST 递归下降，三方方言）
         → translated.json（FQA 文法公式 + 状态/原因）
③ 求值  合成市场逐条 eval_expression 校验（全部可求值）
④ 扫描  scripts/scan_factor_zoo.py（train/val/test 三窗 rank_ic/ICIR）
         → scan_results_*.json（"可靠" = test |rank_ic|≥0.02 且 ICIR≥0.30 且与 train 同号）
⑤ 门禁  幸存者走生产硬门（deflated Sharpe / 涨跌停可交易性 / 危机窗口）后才可 promote
```

## 来源与规模（2026-08-31 更新）

| 家族 | 来源（许可证） | 原始公式 | 可译 | 可求值 |
|---|---|---|---|---|
| alpha101 | WorldQuant 101（via aurumq-rl, MIT） | 107 文档/101 原文 | 84 | 84 |
| gtja191 | 国泰君安 191（aurumq-rl docstring + Daic115/alpha191 第二来源） | 191 | 73 | 73 |
| alpha158 | microsoft/qlib（Apache-2.0） | 159 | 159 | 159 |
| **合计** | | **451** | **316** | **316** |

2026-08-31 增量：
* **amount 解锁**：PIT 库 12.53M 条价格记录本就含成交额，市场面板恢复该列后，
  `VWAP → Div(Amount, Volume)` 映射解锁 46 处 GTJA VWAP 公式 + alpha158 VWAP0；
* **GTJA 第二来源**：Daic115/alpha191.py（存档 paper/repos/gtja191-reference/）
  补齐 88 条缺失公式（191/191）；
* 翻译器支持 Daic115 方言：单字母字段 C/H/L/O/V/A/AMT、RET 字段、
  `if/elif/else` 链、`where` 命名子式、`REGBETA(x, SEQUENCE(n))`→斜率、
  `MAX(0, x)` 标量在前交换。

## 扫描结果（74 条价量域 walk-forward，2026-08-31）

8 条"可靠"（test 2022–2025）：

| rank_ic | ICIR | 公式（节选） |
|---|---|---|
| 0.0352 | 3.40 | `Mul(Neg(1), Rank(Ts_Std(High, 10))) × Ts_Corr(High, Volume, 10)` |
| 0.0306 | 2.89 | `Neg((Low-Close)·Open^5) / ((Low-High)·Close^5)` |
| 0.0259 | 3.06 | `Neg(Rank(Ts_Rank(Close, 10))) × Rank(Close/Open)` |
| 0.0254 | 2.70 | `Neg(Rank(1 − Open/Close))` |
| 0.0242 | 2.33 | `Min2(Rank(decay_linear(…)), …)` |
| 0.0232 | 1.87 | `Ts_Mean(Close, 12)/Close` |
| 0.0223 | 2.10 | `Neg(Ts_Zscore(Close, 5))` |
| 0.0205 | 2.13 | `Neg(Sign(Δclose7 + close−delay7)) × (1+Rank(…))` |

**注意**：幸存者几乎全是 5–12 日短周期反转/均值回归类。这与
LIMIT_DOWN 蓝图的"最小 lookback 60"生产纪律冲突——短周期反转在
涨跌停可交易序列上的组合回撤曾在 Phase 8.1 系统性翻车。IC 门通过只是
必要条件；**promote 前必须过组合级可交易性门**（见 scripts/ml_portfolio_eval.py
同款路径）。alpha158 的 158 条扫描尚未完成（内存受限，待跑）。

## 相关文件

- `inventory.json` / `translated.json` — 提取与翻译产物
- `scan_results_*.json` — 扫描判定
- `src/exploration/translate.py` — 翻译器（含三方方言映射）
- `tests/test_translate.py` — 13 个翻译器测试
