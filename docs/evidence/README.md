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
| `d_stop_grid.json` | **D-8c** 止损宽度网格（8 变体，IS 2026） | `configs/master_config.yaml`、`docs/D_TRACK_EVIDENCE.md` §三 |
| `d_oos_is_2026.json` | 2026 段（IS）OOS harness，11 项断言全通过 | `docs/D_TRACK_EVIDENCE.md` §六 |
| `d_oos_oos_2025h2.json` | 2025-09→12（OOS）旧部署口径 2.5% | 同上 |
| `d_oos_oos_2025h2_flat3p5.json` | 2025-09→12（OOS）新部署口径 3.5% | 同上 |
| `d_oos_oos_2025h2_atr1p0_25_35.json` | 2025-09→12（OOS）ATR 1.0/[2.5%,3.5%] 备选 | 同上 |
| `d_oos_oos_2025h2_atr1p0_25_40.json` | 2025-09→12（OOS）ATR 1.0/[2.5%,4%] 备选 | 同上 |

## 口径提醒

- `d_track_grid3.json`、`d_track_grid4.json`、`d_track_intraday.json` 是 **D-8 修复前**
  （TR 被压成一维 → 止损恒为 2.5% 地板）的产物，且部分脚本当时未加载分钟特征包
  （尾盘量比门槛未生效）；引用它们只能用于"当时的决策背景"，不能当作当前口径的数字。
- `d_track_grid7.json` 已于 2026-09-09 用修复后代码重跑，但其 BASE 是 **ATR 自适应**
  止损口径；**部署口径的绝对数字**以 `d_stop_grid.json`（flat_3p5）与
  `d_oos_is_2026.json` 为准。
- 所有数字引用时必须同时给出：级别（IS/OOS/SHADOW/LIVE-EXEC）· 窗口 · 触发约定 ·
  止损宽度 · 样本量 · Sharpe 标准误 · 数据出处。
