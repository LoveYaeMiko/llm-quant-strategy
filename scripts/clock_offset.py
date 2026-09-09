"""Clock-offset probe: exchange minute-bar label vs the local system clock.

If the machine clock drifts behind the exchange, every session window and every
scheduled job fires late: the 14:50 preclose decision would land after the 15:00
auction, and the live trader would keep polling past the close. This prints the
offset over a few samples so the drift can be seen (negative = bar label behind
the clock, positive = bar ahead of the clock).

Usage: python scripts/clock_offset.py [symbol] [samples]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


def main() -> int:
    from src.config import load_config
    from src.data.ingestion.alphafeed_adapter import AlphaFeedAdapter, normalize_bar_timestamps

    symbol = sys.argv[1] if len(sys.argv) > 1 else "601872.SH"
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    cfg = load_config()
    adapter = AlphaFeedAdapter(api_key=str(cfg.get("data.alphafeed.api_key", "")))
    offsets: list[float] = []
    for i in range(samples):
        try:
            got = adapter.fetch_minute_klines([symbol], period="1m", count=1)
        except Exception as exc:  # noqa: BLE001
            print(f"[clock] fetch failed: {type(exc).__name__}: {exc}")
            return 1
        df = normalize_bar_timestamps((got or {}).get(symbol))
        if df is None or len(df) == 0:
            print("[clock] no bars returned")
            return 1
        sys_now = pd.Timestamp.now()
        bar = pd.Timestamp(df["timestamp"].iloc[-1])
        offset = (bar - sys_now).total_seconds()
        offsets.append(offset)
        print(f"[clock] system={sys_now:%H:%M:%S} bar={bar:%H:%M:%S} "
              f"offset={offset:+.0f}s close={float(df['close'].iloc[-1]):.2f}", flush=True)
        if i < samples - 1:
            time.sleep(65)
    worst = max(abs(o) for o in offsets)
    print(f"[clock] max |offset| = {worst:.0f}s "
          f"({'OK (<60s)' if worst < 60 else 'WARNING — fix the system clock'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
