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
| `d_stop_grid.json` | **D-8c** 止损宽度网格（8 变体，IS 2026；`flat_3p5` 为现金约束修复后重跑） | `configs/master_config.yaml`、`docs/D_TRACK_EVIDENCE.md` §三 |
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
| `forward_prereg_d_forward_2026h2_v1.json` | 前向期风险闸门预注册记录（v1，**已被 v3 取代**） | `docs/FORWARD_PROTOCOL.md` §1.4 |
| `forward_prereg_d_forward_atr_candidate_2026h2_v1.json` | 候选切换规则预注册记录（v1，**已被 v3 取代**） | `docs/FORWARD_PROTOCOL.md` §2 |
| `prereg_d_forward_2026h2_v3.json` | 前向期风险闸门预注册记录（**现行**，绑定 policy_sha256 + commit） | 同上 |
| `prereg_d_forward_atr_candidate_2026h2_v3.json` | 候选两臂比较预注册记录（**现行**，窗口 2026-09-11 起） | 同上 |
| `forward_health_20260910.json` | 风险闸门首次实评（窗口未开始 → 未测量即失败；`prereg_binding` 通过） | `docs/FORWARD_PROTOCOL.md` §1.3 |
| `forward_health_shakedown_20260909.json` | 风险闸门全链路试运行（2026-09-01→09-09，**非**前向窗口） | 同上 |

> **v1 → v3 的取代原因**（2026-09-10 独立审计）：v1 时代的闸门「契约是真的、锁没装上」——
> 评估不绑定预注册、`min_days` 可被配置置零绕过、失败闸门在任务卡显示成功、面板渲染试跑
> 工件、代码漂移不报警；候选臂还因为继承生产订单清单而**结构上无法与现役分离**。修复改变了
> 测量本身，按协议计为新试验，窗口自 2026-09-11 重新起算（2026-09-10 那天在旧设计下推进，
> 作废）。v2 是开窗前的中间版本，记录一并保留以示轨迹。

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
  止损口径；**部署口径的绝对数字**以 `d_stop_grid.json`（`flat_3p5`）为准。
- 所有数字引用时必须同时给出：级别（IS/OOS/SHADOW/LIVE-EXEC）· 窗口 · 触发约定 ·
  止损宽度 · 样本量 · Sharpe 标准误 · 数据出处。
