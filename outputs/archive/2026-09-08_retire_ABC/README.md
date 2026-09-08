# A/B/C 轨退役归档（2026-09-08）

## 结论

自 2026-09-08 起，`configs/master_config.yaml` 的 `shadow.accounts` **仅注册 D_5W 单轨**
（回撤买入 + 实时日内止损）。A_200W / B_10W / C_5W 三条 ML 截面轨退役。

退役口径：

- 不再参与日度闭环（`cli.py shadow` / `cli.py autopilot` 按 `shadow.accounts` 迭代）。
- 不再计入红线监控、日报、邮件与 PAICC 量化面板。
- 本目录文件为**只读历史证据**，不参与任何计算；如需复现，把对应条目加回
  `shadow.accounts` 并把账本/状态文件移回 `outputs/` 即可。

## 退役时点账本快照（最后一日 = 2026-09-08）

| 轨 | 账本 | 累计收益 | Sharpe | 最大回撤 | 成交笔数 |
| --- | --- | --- | --- | --- | --- |
| A_200W | `shadow_ledger_A_200W.sqlite` | +26.98% | 1.81 | — | — |
| B_10W | `shadow_ledger_B_10W.sqlite` | +21.61% | 1.90 | — | — |
| C_5W | `shadow_ledger_C_5W.sqlite` | +11.65% | 1.35 | — | — |

（数值出处：各 `shadow_report_*.md` / `shadow_status_*.json`，退役当日 16:00 前最后一轮闭环。）

## 归档内容

- 逐轨账本 / 状态 / 日报 / 成交明细：`shadow_ledger_*.sqlite`、`shadow_status_*.json`、
  `shadow_report_*.md`、`trades_*.csv`
- 逐轨自动闭环档位与报告：`autopilot_state_*.json(.bak)`、`autopilot_report_*.md`
- 逐轨挑战者账本：`_dcycle_CH_A.sqlite`、`_dcycle_CH_A_inc.sqlite`、`_dcycle_CH_B.sqlite`、`_dcycle_CH_C.sqlite`
- 多轨改造之前的默认单轨残留：`shadow_ledger.sqlite`、`shadow_status.json`、`shadow_report.md`、
  `autopilot_state.json(.bak)`、`autopilot_report.md`、`autopilot_alerts.jsonl`

## 保留在 `outputs/` 的研究证据（不归档）

`a_track_grid.json`、`b_track_grid.json`、`c_track_grid.json`、`compliant_grid.json`、
`dual_track_tune.json` 等网格证据仍在 `outputs/`，供 `docs/DUAL_TRACK_REPORT.md` 引用。
