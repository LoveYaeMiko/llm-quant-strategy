"""Summarise every d_oos_*.json artifact in outputs/ (one line each)."""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "outputs" / "d_oos_*.json"))):
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except ValueError:
            continue
        m = d.get("metrics", {})
        rows.append(
            {
                "label": d.get("label"),
                "cum": m.get("total_return"),
                "ann": m.get("annualized_return"),
                "sharpe": m.get("sharpe"),
                "t": d.get("sharpe_t_stat"),
                "dd": m.get("max_drawdown"),
                "fills": m.get("n_fills"),
                "mincov": d.get("minute_min_symbol_coverage"),
                "hole": d.get("fills_in_low_coverage_days"),
                "citable": d.get("citable"),
                "candidate": d.get("is_candidate_run"),
            }
        )
    head = f"{'label':32s} {'cum':>9s} {'ann':>9s} {'sharpe':>7s} {'t':>6s} {'dd':>7s} {'fills':>6s} {'mincov':>7s} {'hole':>5s} {'citable':>8s}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{str(r['label'])[:32]:32s} "
            f"{(r['cum'] or 0):+9.2%} {(r['ann'] or 0):+9.2%} "
            f"{(r['sharpe'] or 0):+7.2f} {(r['t'] or 0):+6.2f} "
            f"{(r['dd'] or 0):7.2%} {int(r['fills'] or 0):6d} "
            f"{(r['mincov'] or 0):7.2%} {int(r['hole'] or 0):5d} {str(r['citable']):>8s}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
