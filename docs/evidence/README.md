# 证据快照（docs/evidence）

`outputs/` 被 `.gitignore` 忽略（运行期产物），但下列网格/验证结果是文档与配置注释
直接引用的数字来源，因此复制一份进入版本库，保证引用可追溯。

| 文件 | 用途 | 被引用处 |
| --- | --- | --- |
| `d_track_grid3.json` | `pb_k=6` 的 2026 段出处（`D_t21_k6`，门槛关闭口径） | `configs/master_config.yaml` |
| `d_track_grid4.json` | 满仓化对比（`D_base` vs `D_full_cap25`） | `configs/master_config.yaml` |
| `d_track_intraday.json` | 尾盘量比门槛（`D_tail50` vs `D_base`） | `configs/master_config.yaml` |
| `d_track_grid7.json` | 触发约定敏感性（**2026-09-09 修复后重跑**） | `configs/master_config.yaml`、`docs/D_TRACK_EVIDENCE.md` §二 |
| `d_atr_impact.json` | A/B：固定 2.5% vs ATR 自适应 vs 无开盘豁免 | `configs/master_config.yaml`、`docs/D_TRACK_EVIDENCE.md` §三 |
| `d_stop_grid.json` | **D-8c** 止损宽度网格首发版（8 变体，IS 2026，**只有 301 只标的**的截面）——**已被 800 只池复跑取代，数字作废**（仅存决策轨迹） | `docs/D_TRACK_EVIDENCE.md` §三（历史） |
| `d_oos_is_2026.json` | 2026 段（IS）OOS harness，**旧版** 11 项断言 | `docs/D_TRACK_EVIDENCE.md` §六 |
| `d_oos_oos_2025h2.json` | 2025-09→12（OOS）旧部署口径 2.5% | 同上 |
| `d_oos_oos_2025h2_flat3p5.json` | 2025-09→12（OOS）新部署口径 3.5% | 同上 |
| `d_oos_oos_2025h2_atr1p0_25_35.json` | 2025-09→12（OOS）ATR 1.0/[2.5%,3.5%] 备选 | 同上 |
| `d_oos_oos_2025h2_atr1p0_25_40.json` | 2025-09→12（OOS）ATR 1.0/[2.5%,4%] 备选 | 同上 |
| `d_oos_oos_2025h2_v2.json` | 2025-09→12（OOS）**符号级覆盖探针**首次运行 | 同上 |
| `d_oos_oos_2025h2_v3.json` | 2025-09→12（OOS）**回补后、可引用**：13/13 通过、`citable=true` | `docs/D_TRACK_EVIDENCE.md` §六/§九 |
| `d_oos_oos_2025h2_v3_flat2p5.json` | 同上窗口的旧宽度 2.5%（候选，`citable=false`、`data_ok`） | §六「干净窗口重评」 |
| `d_oos_oos_2025h2_v3_atr1p0_25_35.json` | 同上窗口的 ATR 1.0/[2.5%,3.5%]（候选） | 同上 |
| `d_oos_oos_2025h2_v3_atr1p0_25_40.json` | 同上窗口的 ATR 1.0/[2.5%,4%]（候选，OOS 最优） | 同上 |
| `d_oos_is_2026_v4.json` | IS 2026（**14 项断言**、符号级窗口覆盖探针、含 provenance） | `docs/D_TRACK_EVIDENCE.md` §九 |
| `d_oos_oos_2025h2_v5.json` | OOS 2025H2（回补后、可引用、含 provenance） | 同上 |
| `bias_stress_oos_2025h2_v1.json` | 幸存者偏差压力测试（自进化闭环准入判据） | `docs/BIAS_STRESS.md`、`docs/FORWARD_PROTOCOL.md` §3 |
| `adjust_anchor_baseline.json` | 复权锚点基线（漂移检测参考） | `docs/ADJUST_ANCHOR.md` |
| `forward_prereg_d_forward_2026h2_v1.json` | 前向期风险闸门预注册记录（v1，**已被 v3/v8 取代**） | `docs/FORWARD_PROTOCOL.md` §1.4 |
| `forward_prereg_d_forward_atr_candidate_2026h2_v1.json` | 候选切换规则预注册记录（v1，**已被 v3/v8 取代**） | `docs/FORWARD_PROTOCOL.md` §2 |
| `prereg_d_forward_2026h2_v3.json` | 前向期风险闸门预注册记录（v3，历史：首次绑定 policy_sha256） | 同上 |
| `prereg_d_forward_atr_candidate_2026h2_v3.json` | 候选两臂比较预注册记录（v3，历史） | 同上 |
| `prereg_d_forward_2026h2_v7.json` | 前向期风险闸门预注册记录（v7，历史：绑定代码指纹 `0542e5cc…`） | 同上 |
| `prereg_d_forward_atr_candidate_2026h2_v7.json` | 候选两臂比较预注册记录（v7，历史） | 同上 |
| `prereg_d_forward_2026h2_v8.json` | 前向期风险闸门预注册记录（**现行**，trial 5 / amendments 3；绑定 policy `302489534115…` + 代码指纹 `4a7e500f…`） | 同上 |
| `prereg_d_forward_atr_candidate_2026h2_v8.json` | 候选两臂比较预注册记录（**现行**，trial 14 / amendments 3，窗口 2026-09-11 起） | 同上 |
| `forward_health_20260910.json` | 风险闸门评估（开窗前最后一版，绑定 v8；窗口未开始 → 未测量即失败，`prereg_binding` 通过）；`metrics.universe.as_of = 2026-09-10`（实际测量的 bar）而 `requested_as_of = 2027-03-11`（窗口末端） | `docs/FORWARD_PROTOCOL.md` §1.3 |
| `forward_health_shakedown_20260909.json` | 风险闸门全链路试运行（2026-09-01→09-09，**非**前向窗口） | 同上 |
| `shadow_series_latest.json` | 影子盘历史的**口径分段** + 当前口径重放序列（①） | `docs/D_TRACK_EVIDENCE.md` §一之二 |
| `d_stop_grid_is_2026_800.json` | 止损宽度网格，IS 2026-01→08-28（159 日），**修正后的 800 只池**（②，取代旧 `d_stop_grid.json`）；含 provenance 五项 + `panel` 健康度（`effective_ratio=1.00`）、逐变体 `params_hash`/`n_symbols`/`n_bars`；`flat_3p5` 与 `d_oos_is_2026_v5.json` **逐位一致** | `docs/D_TRACK_EVIDENCE.md` §三 |
| `d_stop_grid_oos_2025h2_800.json` | 止损宽度网格，OOS 2025-09→12-31（82 日），同一 800 只池/同一市场切片（②）；该窗口的池**未被 2026 年价格回补影响**（`flat_3p5` 与回补前的 `d_oos_oos_2025h2_v5.json` 逐位一致） | 同上 |

> **v1 → v3 的取代原因**（2026-09-10 独立审计）：v1 时代的闸门「契约是真的、锁没装上」——
> 评估不绑定预注册、`min_days` 可被配置置零绕过、失败闸门在任务卡显示成功、面板渲染试跑
> 工件、代码漂移不报警；候选臂还因为继承生产订单清单而**结构上无法与现役分离**。修复改变了
> 测量本身，按协议计为新试验，窗口自 2026-09-11 重新起算（2026-09-10 那天在旧设计下推进，
> 作废）。v2 是开窗前的中间版本，记录一并保留以示轨迹。
>
> **v3 → v8 的取代原因**（同为开窗前、记账性）：v4 把冻结从 commit 改为绑定**代码内容指纹**；
> v5 因**政策面变化**（候选由 25_40 换成 25_35）重签；v6 收窄指纹覆盖到 src/scripts/tests；
> v7 因提交 `1ef754c`（止损宽度证据写入注释/文档 + 前向候选的「无事可做就不建行情」守卫）
> 与 `6d41d4b`（`write_preregistration` 拒绝无 policy 绑定的记录）改变了代码内容指纹而重签；
> v8 因 `7ed4da7`（`panel_universe_health` 记的 `as_of` 改为**实际测量的那根 bar**，不再把
> 请求日当成测量日——首次前向评估因此写出过 `as_of = 2027-03-11`）再次重签。
> 五次的 `this_trial` 均未变（5 / 14），只有 `amendments` 递增。**v7 的第一次写入作废**：
> 当时用脚本直接调 `new_record` 绕过了 CLI，写出的记录 `policy_sha256=null` —— 它能通过
> `prereg verify`（只查六字段与自哈希）却永远无法通过 `prereg_gate` 的绑定检查；两份文件已删除，
> 真正的 v7 经 `scripts/prereg.py new` 重新冻结，该陷阱现已由 `write_preregistration` 拒绝写入。

## 口径提醒

- **provenance 五项必填**（2026-09-09 起）：对外 JSON 必须带
  `window / convention / data_as_of / artifact_sha256 / code_commit`，
  由 `scripts/check_provenance.py` 校验（缺失或哈希不符 → 退出码 1）。
  本目录中 2026-09-09 之前生成的旧工件为 **legacy**（无 provenance），
  仅作历史记录，`--strict` 下会判失败。
- **`citable` 字段是唯一的引用许可**：只有 `citable=true` 的工件数字可以作为"健全口径"
  引用。当前 `citable=true` 的只有 `d_oos_is_2026_v4.json`（IS 2026）与
  `d_oos_oos_2025h2_v5.json`（OOS 2025H2，回补后）。其余 `d_oos_*` 为 `false`
  ——旧版断言/日期级探针/窗口数据洞，或候选参数（`is_candidate_run`）。
- **`data_ok`** 与 `citable` 分开：候选跑（`--set`）也可以是 `data_ok=true`，
  用于在健全数据上比较参数，但不能当作部署配置的数字。
- `d_track_grid3.json`、`d_track_grid4.json`、`d_track_intraday.json` 是 **D-8 修复前**
  （TR 被压成一维 → 止损恒为 2.5% 地板）的产物，且部分脚本当时未加载分钟特征包
  （尾盘量比门槛未生效）；只能用于"当时的决策背景"，不能当作当前口径的数字。
- `d_track_grid7.json` 已于 2026-09-09 用修复后代码重跑，但其 BASE 是 **ATR 自适应**
  止损口径；**部署口径的绝对数字**以 `d_stop_grid_is_2026_800.json`（IS）与
  `d_stop_grid_oos_2025h2_800.json`（OOS）的 `flat_3p5` 为准 —— 这两份是 2026-09-10
  在**真实 800 只池**上的复跑，各自带 `panel` 区块（`n_columns / n_with_price /
  n_warm_20 / n_warm_60 / effective_ratio`）与 provenance，可与
  `d_oos_is_2026_v5.json` / `d_oos_oos_2025h2_v5.json` 逐位对照。
- **止损宽度网格的排序在两个窗口之间不一致**（`d_stop_grid_*_800.json`：IS 冠军
  `atr_1p0_25_35`、OOS 冠军 `atr_1p5_25_40`，Spearman ≈ −0.19）。因此**不得**从这两份
  工件里挑单个窗口的最优变体当作改进依据；部署 `flat_3p5` 的依据是**先定好的 max-min
  规则**（跨窗口最差 Sharpe 0.40，为 8 个变体中的最大值），见 `D_TRACK_EVIDENCE.md` §三。
- 所有数字引用时必须同时给出：级别（IS/OOS/SHADOW/LIVE-EXEC）· 窗口 · 触发约定 ·
  止损宽度 · 样本量 · Sharpe 标准误 · 数据出处。
