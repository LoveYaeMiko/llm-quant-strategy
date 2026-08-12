"""Phase 10 — momentum-exposure probe for the alpha composite (paper-backed).

MLMultiFactorBiasCorrection (arXiv:2507.07107) shows raw factors embed
unintended systematic exposures (market cap / industry / momentum) and that
cross-sectional neutralization raises mean IC (0.023 -> 0.041) and IR
(0.147 -> 0.461) on 2010-2024 A-shares. Before proposing neutralization for the
Phase 10 market-neutral book (2025 short-leg squeeze), measure how much of the
composite's cross-section IS momentum:

  corr_t = spearman(composite_z[d, :], momentum_z[d, :])   per date

If the composite's short leg (high-vol / high-turnover names) is strongly
negatively correlated with recent-momentum winners, the 2025 drawdown is a
momentum exposure and neutralization is the direct fix.

Prints a console report; no files written.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from src.cli import _cached_universe_json
    from src.config import load_config
    from src.factors.code_generator import FactorContext
    from src.portfolio.alpha_core import AlphaCore
    from src.portfolio.backtest_runner import load_hs300_market

    cfg = load_config()
    symbols = _cached_universe_json("hs300", cfg)
    t0 = time.time()
    market = load_hs300_market(cfg, symbols, start="2009-01-06", end="2025-12-31")
    formulas = [x.get("factor", {}).get("formula", x.get("formula"))
                for x in json.loads((ROOT / "outputs" / "factors.json").read_text(encoding="utf-8"))]
    alpha = AlphaCore(FactorContext(market.long), formulas,
                      long_pct=0.10, short_pct=0.10, max_position_pct=0.05)
    print(f"probe: loaded market + composite in {time.time() - t0:.1f}s")

    # --- momentum cross-section per date from the close panel (PIT: past only) ---
    close = market.price_panel
    mom = {}
    for lbl, n in (("mom20", 20), ("mom60", 60), ("mom120", 120), ("mom252", 252)):
        # close_t / close_{t-n} - 1; shift leaves NaN for the first n bars
        m = close / close.shift(n) - 1.0
        mom[lbl] = m.stack().rename(lbl)
    mom_frame = pd.concat(mom.values(), axis=1)  # (date, symbol) x lookback

    comp = alpha.composite.rename("comp")
    merged = mom_frame.join(comp, how="inner").dropna(subset=["comp"])

    # per-date spearman between composite z and each momentum z
    rows = []
    for d, day in merged.groupby(level=0):
        if day.shape[0] < 30:
            continue
        for lbl in mom:
            x = day[lbl].rank()
            y = day["comp"].rank()
            r = np.corrcoef(x, y)[0, 1]
            rows.append((d, lbl, r))
    corr = pd.DataFrame(rows, columns=["date", "mom", "corr"])
    print("\n=== composite vs momentum cross-sectional correlation (2010-2025) ===")
    for lbl in mom:
        sub = corr[corr["mom"] == lbl]
        pos = (sub["corr"] > 0.3).mean()
        neg = (sub["corr"] < -0.3).mean()
        print(f"  {lbl:<6} mean={sub['corr'].mean():+.3f} med={sub['corr'].median():+.3f} "
              f"|corr|>0.3: {pos + neg:.1%}")

    # 2025 specifically — the squeeze year
    print("\n=== 2025 only ===")
    for lbl in mom:
        sub = corr[(corr["mom"] == lbl) & (corr["date"] >= "2025-01-01")]
        if len(sub):
            print(f"  {lbl:<6} mean={sub['corr'].mean():+.3f} med={sub['corr'].median():+.3f} n={len(sub)}")

    # who is shorted in 2025? composite bottom-decile momentum profile
    print("\n=== short-leg (bottom-decile composite) vs market momentum, 2025 ===")
    m252 = merged["mom252"]
    for y in (2011, 2015, 2018, 2021, 2025):
        yday = merged[merged.index.get_level_values(0).year == y]
        if not len(yday):
            continue
        bottom = yday[yday["comp"] <= yday["comp"].quantile(0.10)]
        top = yday[yday["comp"] >= yday["comp"].quantile(0.90)]
        print(f"  {y}: short-leg mom252={bottom['mom252'].mean():+.3f} "
              f"long-leg mom252={top['mom252'].mean():+.3f} "
              f"gap={bottom['mom252'].mean() - top['mom252'].mean():+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
