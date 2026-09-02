# FQA 提高收益三轨报告（GitHub 提取 / 新数据域 / ML 训练）

> 目标：在影子盘（虚拟盘）阶段优先寻找收益来源，而不是优化成本。
> 三条渠道并行推进，所有候选**必须**经 PIT 数据 + walk-forward 门禁后才可进入影子盘。
> 生成日期：2026-08-31（进行中，未填数字见下轮更新）。

## 轨道 A：GitHub 公式动物园（= ML 特征源）

### 管道

```
① 提取  scripts/extract_factor_zoo.py / extract_alpha158.py
         → paper/factor_zoo/inventory.json
② 翻译  src/exploration/translate.py（AST 递归下降，WorldQuant/GTJA/AQML/qlib 四方方言）
         → paper/factor_zoo/translated.json
③ 求值  合成市场逐条 eval_expression 校验
④ 扫描  scripts/scan_factor_zoo.py — train/val/test 三窗 rank_ic/ICIR
```

### 规模（截至本轮）

| 家族 | 来源/许可证 | 可译 | 可求值 |
|---|---|---|---|
| alpha101 | WorldQuant 101（aurumq-rl, MIT） | 55 | 55 |
| gtja191 | 国泰君安 191（aurumq-rl, MIT） | 24 | 24 |
| alpha158 | microsoft/qlib（Apache-2.0） | 158 | 158 |
| **合计** | | **237** | **237** |

新增算子：`ts_sma` / `ts_count` / `ts_vwap` / `ts_resi`（纯增量，生产挖掘路径不受影响）。
翻译器测试 13 个；顺带修复 FQA `_cond` 的标量分支 ndarray 退化 bug。

### 扫描结果

- **价量域 74 条**：8 条"可靠"（test rank_ic 0.0205–0.0352 / ICIR 1.87–3.40），几乎全是
  5–12 日短周期反转类 → 与生产 min_lookback 60 纪律冲突，**promote 前必须过组合级可交易性门**
  （Phase 8.1 教训：短反转在涨跌停序列上崩溃延续）。
- **alpha158 158 条**：扫描结果待补（串行链第 2 阶段）。

## 轨道 B：新数据域 PIT 管道（2 个已落地）

| 数据域 | 管道 | 记录 | 门禁结果（发现/验证分段） |
|---|---|---|---|
| 融资融券 | `src/data/margin.py` + ingest/gate | 788,304 条（2010-03→2026-08，HS300） | **全历史（发现 2010–2023 / 验证 2024–2026）**：fin/sl_growth 反向 ~0.9–1.5% rank_ic 方向一致但低于 0.015 单因子门；sl_mix 不稳定（发现≈0、验证 -0.016）。结论：弱但方向稳定的拥挤度信号，适合作 ML 特征而非独立因子 |
| 龙虎榜 | `src/data/lhb.py` + ingest/gate | 72,876 条（2022-01→2026-08） | net_buy 样本太薄；net_buy_5d 2026 符号翻转（失败）；HS300 上榜稀疏、信号不稳定 |

关键纪律落实：
- 融资融券：余额盘后发布 → `valid_from = 次日`；快照闭区间 + 跨批关闭（修正 upsert 语义）。
- 龙虎榜：**上游内嵌"上榜后 1/2/5/10 日"未来收益列，归一化按构造剥离**（测试锁定）。
- 生产库断言级验证：无前视列、区间闭合、单标的单活跃记录 —— 全部通过。
- 门禁协议：发现窗（2024–2025）选方向 → 验证窗（2026）纯 OOS 确认。sl_mix 的失败证明
  该协议有效（合并窗口的 +0.0264 部分为小样本噪声）。

## 轨道 C：ML 训练（LightGBM，Purged-KFold + embargo + walk-forward）

### 管线（src/ml/）

- `labels.py` 多周期前向标签（涨跌停不可交易掩码 + 逐日截面标准化）；
- `cv.py` PurgedKFold + embargo（标签窗口重叠清除，AFML 标准）；
- `train.py` 训练器 + 冻结工件（文本树 + 元数据 JSON，**含 feature_formulas**，
  单线程位级可复现评分）；4 个首轮 bug 已修（CRLF 模型文件致 C 解析器 abort 等）。
- `promote.py` 生产硬门纯函数：绝对地板（Sharpe≥1.0 / maxDD≤15% / |rank_ic|≥0.02）
  + **双轴击败在任池**。

### 结果

| 实验 | 特征 | test rank_ic | test ICIR | test LS 净年化 | 备注 |
|---|---|---|---|---|---|
| 首轮 LightGBM | 34 | 0.0455 | 6.0 | +0.13% | 尾部信号弱 |
| 扩特征 LightGBM | 108 | 0.0493 | 5.47 | -0.50% | 中部截面改善、尾部未改善 |
| **GPU MLP（torch）** | 108 | **0.0558** | **8.28** | **+0.81%** | LS Sharpe 2.17 / maxDD -1.28%，RTX 4060 训练 ~20 分钟 |

### 对决协议（scripts/ml_portfolio_eval.py）

模型打分 → 动量+beta 中性化 → 多空十分位 + regime 空腿 → PaperRunner 真实成本
（2022–2025）→ 与在任 5 因子池**同路径**对比 → `src/ml/promote.py` 硬门判定
（绝对地板 + 双轴击败）。

**结果（v2，详见 docs/SHOWDOWN_REPORT.md）**：GPU MLP 年化 +19.5% / Sharpe 2.35 /
maxDD 6.8% **双轴击败在任池**（+11.6% / 1.63 / 7.7%）→ **promote 门 PASS**。
LightGBM 0.97 未过门。v1→v2 修了两条方法论错误（事后中性化毁已拟合模型；
100k 账户微尘成本伪信号）。

## 结论与下一步

- 已产出 237 条可复用公式特征 + 2 个新数据域 + ML 双模型管线（LightGBM + GPU MLP）
  与首个 OOS 组合级对决。
- **GPU MLP（torch）是当前最佳候选**：test rank_ic 0.0558 / ICIR 8.28 / 尾部净收益转正。
- 若对决中 MLP 过 promote 硬门（双轴击败在任池 + 绝对地板），下一步接线设计：

### 影子盘接线设计（promote 通过后）

1. **新工件类型** `ml_mlp_torch`：`outputs/models/mlp_*.pt/.json` 已在工件层支持
   （特征公式 + 归一化参数 + 拟合窗口）；
2. **离线打分包装** `src/paper/ml_book.py`：MLBookPortfolio = 工件打分 →
   动量+beta 中性化 → regime 空腿簿（复用 showdown 的 `_ml_books` 逻辑并入 src）；
3. **替换门**：autopilot 增加 `ml_promote` 周期任务——新模型在 test 窗过门后，
   影子盘下一轮起用 ML 簿替代 AlphaCore 簿，`factor_decayed` 语义由"因子池衰减"
   扩展为"当前信号源（因子池或模型）衰减"（DecayTracker 对模型打分同样适用）；
4. **回滚**：模型衰减 → remine 因子池或重训模型，保留旧工件一键回退。

- 尚未过门的候选（8 条短周期公式 / 融资融券因子）记录负结果，不进入影子盘。
- 数据域扩展候选：股东户数 / 一致预期（端点已验证可用）。
