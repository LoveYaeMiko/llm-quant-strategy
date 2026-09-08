# 证据快照（docs/evidence）

`outputs/` 被 `.gitignore` 忽略（运行期产物），但下列网格结果是文档与配置注释直接
引用的数字来源，因此复制一份进入版本库，保证引用可追溯。

| 文件 | 用途 | 被引用处 |
| --- | --- | --- |
| `d_track_grid3.json` | `pb_k=6` 的 2026 段出处（`D_t21_k6`） | `configs/master_config.yaml` |
| `d_track_grid4.json` | 满仓化对比（`D_base` vs `D_full_cap25`） | `configs/master_config.yaml` |
| `d_track_intraday.json` | 尾盘量比门槛（`D_tail50` vs `D_base`） | `configs/master_config.yaml` |
| `d_track_grid7.json` | 触发约定敏感性 + 部署口径 `D_close_skip30` | `configs/master_config.yaml`、`docs/D_TRACK_EVIDENCE.md` |
| `d_atr_impact.json` | 修复后口径的 A/B（固定 2.5% vs ATR 自适应） | `configs/master_config.yaml`、`docs/D_TRACK_EVIDENCE.md` §三 |
| `d_oos_is_2026.json` | 2026 段（IS）11 项断言全通过 | `docs/D_TRACK_EVIDENCE.md` §六 |
| `d_oos_oos_2025h2.json` | 2025-09→12 段（OOS）11 项断言全通过 | `docs/D_TRACK_EVIDENCE.md` §六/§九 |

注意：`d_track_grid7.json` 等是 **D-8 修复前**（TR 被压成一维 → 止损恒为 2.5% 地板）
的口径，用于说明触发约定的相对敏感性；部署口径的绝对数字以
`outputs/d_atr_impact.json`（生产同构 A/B）为准。
