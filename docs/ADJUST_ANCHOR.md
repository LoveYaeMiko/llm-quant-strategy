# The price-adjustment anchor (defect C3)

> **One line:** the PIT price panel is backward-adjusted against an *implicit*
> anchor (the newest ingested bar), that anchor is recorded nowhere, and every
> re-ingest that sees a new corporate action silently re-bases the whole
> history — so an identical backtest re-run is not reproducible and forward
> tracking error is confounded by basis drift.
>
> `src/data/adjust_anchor.py` makes the anchor **explicit, versioned and
> drift-detectable**; `scripts/check_adjust_anchor.py` freezes a reference,
> captures the current state and fails loudly when the basis moved.

---

## 1. The defect

The `record_type='price'` payload is

```json
{"open": .., "high": .., "low": ..,
 "close": <ADJUSTED>, "raw_close": <RAW>,
 "adjust_factor": <float>, "volume": .., "amount": .., "name": ..}
```

with `close = round(raw_close * adjust_factor, 4)` (ADR-0002). The factor is
built at ingest time in
`src/data/ingestion/alphafeed_adapter.py::to_price_records`:

```python
events["suffix"] = (1.0 / events["ex_factor"]).groupby(events["symbol"]) \
                     .transform(lambda s: s[::-1].cumprod()[::-1])
matched = pd.merge_asof(bars, events[["symbol", "date", "suffix"]],
                        on="date", by="symbol", direction="forward",
                        allow_exact_matches=False)
out["adjust_factor"] = matched["suffix"].fillna(1.0)
out["close"] = (out["raw_close"] * out["adjust_factor"]).round(4)
```

`adjust_factor` at bar *t* is therefore the product of `1/ex_factor` over every
corporate-action event **strictly after** *t*. Three consequences follow
directly from that definition:

1. **The anchor is implicit.** The newest bar of the ingest batch has no event
   after it, so its factor is exactly `1.0`. That newest bar *is* the anchor —
   but nothing stores which bar it was, and nothing stores a `data_as_of`.
2. **Re-ingesting re-bases history.** When a later ingest sees a new ex-event
   (dividend, bonus issue, split), `suffix` for *every earlier bar* is
   multiplied by `1/ex_factor`. The whole stored `close` series shifts, with no
   code, config or parameter change.
3. **Therefore results are not comparable across ingests.** A re-run of the same
   backtest over the same window on the same strategy produces different numbers;
   and a forward tracking-error measurement against a historical run mixes basis
   drift into what is meant to measure strategy behaviour.

There is no `data_as_of` in any artifact, so a reader cannot even tell *which*
basis a stored number is on.

### Measured evidence from the live store (2026-09-09)

* 5,164 price symbols, 12,535,534 price rows, newest bar `2026-09-09`.
* 2,651 / 5,164 symbols carry `adjust_factor != 1` at their **last** stored bar
  (all of them `< 1`; minimum `0.456273903049096`).
* Ad-hoc probe (same store):

  ```sql
  SELECT (payload->>'adjust_factor')::float8 AS f, max(valid_from) AS last_bar, count(*)
  FROM pit_records WHERE payload->>'record_type'='price' GROUP BY symbol;
  ```

  | last stored bar | symbols | of which `factor != 1` |
  |---|---|---|
  | `2026-09-09` | 800 | 0 (every factor is exactly `1.0`) |
  | `2025-12-31` | 4,355 | 2,644 |
  | `2025-12-24 … 2025-12-30` | 9 | 6 |

  The 800 symbols whose newest bar is the store's newest bar all sit at
  `factor == 1.0` — exactly the "anchor is the newest bar" signature. The other
  4,364 symbols are adjusted against an anchor **outside their own stored
  range**.
* `000793.SZ` is the clearest case: its factor steps only at its ex-event dates
  (`2010-08-23, 2011-07-11, …, 2018-07-17`) and is then **constant
  `0.456273903049096` for every bar from 2018-07-17 to its last bar
  (2025-12-31)**. Under the ingest rule above, a constant tail factor below 1.0
  can only come from a corporate-action event dated *after* the symbol's last
  stored bar. The stored history of that symbol is therefore adjusted against an
  anchor that its own bars never reach — which corporate action that is has not
  been verified here.

**Verification status.** Points 1–3 are read off the ingest code
(`to_price_records`) and are reproducible. The last-bar-date breakdown and the
`000793.SZ` factor series are measured directly against `pit_data`. The claim
that a *future* re-ingest will change a stored factor is an inference from the
code path, not something this task verified by mutating the store (the rules
forbid it); §5 shows it is detected when it happens, and
`tests/test_adjust_anchor.py::test_reingest_rebase_is_detected_offline`
demonstrates the arithmetic offline.

---

## 2. The mechanism in one table

Assume a symbol with one ex-event on `2026-06-10` and `1/ex_factor = 0.98`:

| bar date | factor before the event is ingested | factor after |
|---|---|---|
| 2026-01-05 | 1.0 | 0.98 |
| 2026-06-09 | 1.0 | 0.98 |
| 2026-06-10 (ex-date) | 1.0 | 1.0 |

Every bar *before* the event is scaled by 0.98; the anchor moves forward to the
newest bar. `close` (adjusted) changes for the whole history; `raw_close` does
not. Nothing in the payload says this happened.

`capture_anchor` fingerprints the observable of that event: the per-symbol
`adjust_factor` **at each symbol's last bar**. That vector is `1.0` for symbols
sitting on the anchor and `< 1` for symbols whose history is behind one or more
events, so any re-basing shows up as a change in it.

---

## 3. Commands

```powershell
# freeze the reference anchor (refuses to overwrite without --force)
python scripts/check_adjust_anchor.py --baseline

# snapshot the current anchor: outputs/data/adjust_anchor.json
# + a timestamped copy in outputs/data/adjust_anchor_history/
python scripts/check_adjust_anchor.py --capture --json

# is the store still on the baseline's price basis?  exit 1 when drifted
python scripts/check_adjust_anchor.py --compare

# machine-readable drift report
python scripts/check_adjust_anchor.py --compare --json

# compare two arbitrary anchor files (e.g. pre-/post-re-ingest)
python scripts/check_adjust_anchor.py --compare --prev a.json --curr b.json

# capture a subset only
python scripts/check_adjust_anchor.py --capture --symbols 600519.SH,000001.SZ
```

Exit codes: `0` stable / write succeeded, `1` **drift detected**, `2` usage,
IO or a refused overwrite. `--json` puts the machine-readable object on stdout
and progress on stderr, so the output can be piped.

Library API (`src/data/adjust_anchor.py`):

```python
from src.config import load_config
from src.data.adjust_anchor import capture_anchor, compare_anchors, write_anchor, load_anchor

cfg = load_config()
anchor = capture_anchor(cfg)              # or capture_anchor(url)
write_anchor("outputs/data/adjust_anchor.json", anchor, history_dir="outputs/data/adjust_anchor_history")
report = compare_anchors(load_anchor("outputs/data/adjust_anchor_baseline.json"), anchor)
if report["drifted"]:
    ...                                    # do not compare metrics across this boundary
```

Anchor files are canonical JSON (sorted keys, no whitespace,
`ensure_ascii=False`) with a `record_sha256` over the payload excluding that
field; `load_anchor` verifies it and refuses to return a hand-edited anchor.

---

## 4. Current measured anchor

`python scripts/check_adjust_anchor.py --capture --json` against the live
`pit_data` store (Docker `fqa-pit-db`):

| field | value |
|---|---|
| `captured_at` | `2026-09-09T17:28:35+08:00` |
| `data_as_of` | **2026-09-09** |
| `anchor_policy` | `newest_bar_backward_adjust` |
| `n_symbols` | **5164** |
| `n_rows` | **12,535,534** |
| `factor_stats.n_symbols_factor_ne_1` | **2651** |
| `factor_stats.n_symbols_factor_lt_1` | 2651 |
| `factor_stats.n_symbols_factor_gt_1` | 0 |
| `factor_stats.min` | `0.456273903049096` |
| `factor_stats.max` | `1.0` |
| `factors_sha256` | `9011534279896df17a44294a2b14d144a8fdd851d7d17c15e429a60d85fed8db` |

Top 5 by `|factor - 1|`:

| symbol | factor at last bar |
|---|---|
| `000793.SZ` | `0.456273903049096` |
| `603409.SH` | `0.6646691144214587` |
| `300501.SZ` | `0.6654028448633229` |
| `688332.SH` | `0.6654232124902598` |
| `605128.SH` | `0.6654666085492495` |

The baseline frozen one minute later (`--baseline --force`, `captured_at`
`2026-09-09T17:30:16+08:00`) produced the **same** `factors_sha256` and the same
factor statistics, and

```
$ python scripts/check_adjust_anchor.py --compare
prev: outputs/data/adjust_anchor_baseline.json
curr: outputs/data/adjust_anchor.json
data_as_of: 2026-09-09 -> 2026-09-09 (moved=False)
symbols: common=5164 new=0 lost=0
factors_sha256: 90115342…fed8db -> 90115342…fed8db (changed=False)
drifted: 0/5164 common symbols (tol=1e-09, max_rel_change=0.000e+00)
top drifted symbols: (none)
VERDICT: stable
```

exits `0`. Hash stability across two independent captures on unchanged data is
the property that makes the baseline usable as a reference. A third independent
capture (`captured_at` `2026-09-09T17:38:13+08:00`, a clean `--capture --json`
run with exit code `0`) produced the identical `factors_sha256` and identical
factor statistics.

Drift detection itself is demonstrated offline (no store mutation) by re-basing
a captured factor map and comparing:

```
drifted: 2/2 common symbols (tol=1e-09, max_rel_change=2.000e-02)
top drifted symbols (of 2):
  000001.SZ    1.0 -> 0.98  rel=-2.000e-02
  600519.SH    1.0 -> 0.98  rel=-2.000e-02
VERDICT: DRIFTED          # exit code 1
```

---

## 5. Operating rule

1. **Freeze a baseline before any long forward window opens.** The baseline is
   the statement "these numbers were produced on this price basis". Without it,
   no forward measurement is attributable.
   ```powershell
   python scripts/check_adjust_anchor.py --baseline
   ```
2. **Check for drift before comparing anything across time.** Run `--compare`
   after every ingest and before re-running a backtest or computing tracking
   error. `drifted=True` (exit 1) means metrics from before and after the
   boundary are on different price bases and must not be compared: re-run the
   earlier measurement, or re-state it with its `data_as_of`.
3. **Re-freeze only deliberately, and record why.** `--force` overwrites the
   reference; the old baseline survives only in
   `outputs/data/adjust_anchor_history/`. Append an entry to the log below
   whenever you use it.
4. **Read `data_as_of` on every artifact.** A moved `data_as_of` with unchanged
   factors is harmless (new bars arrived, no re-basing); a moved
   `factors_sha256` is not.
5. **Drift is a data event, not a strategy event.** Never "fix" a drifted
   comparison by re-tuning the strategy.

### Baseline re-freeze log

| date | actor | reason |
|---|---|---|
| 2026-09-09 | C3 remediation (`--baseline --force`) | initial reference, frozen after `--capture` confirmed hash stability |

---

## 6. Limits and non-goals

* **The fingerprint is a proxy for the anchor, not the anchor date.** It detects
  that the basis moved; it does not recover the anchor bar. A drift report names
  *which symbols* moved and by how much, not *which corporate action* caused it.
* **Drift is only visible for symbols present in both captures.** A symbol that
  disappeared is reported as `lost_symbols` (and counts as drift); a symbol that
  is new is reported but is not drift by itself.
* **Capture reads one aggregate query per symbol, not `store.snapshot("price")`.**
  The snapshot materializes all ~12.5M bars and is documented in
  `src/portfolio/backtest_runner.py` as having blown past 32 GB RAM; the
  aggregate returns ~5k rows in ~70 s and is the single read pass per capture.
  Non-Postgres backends still go through exactly one
  `store.snapshot("price")` call.
* **This does not re-base anything.** Stored prices are untouched; the module
  only makes the basis explicit and its changes detectable (same decision as
  `src/data/basis.py` for C2).
* **Anchoring the series at a fixed date is out of scope.** Doing that would
  rewrite every historical artifact; the versioned anchor is the prerequisite
  for ever changing the policy safely.

## 7. Tests

```powershell
python -m pytest tests/test_adjust_anchor.py -q     # 22 offline tests, no DB
```

Covers: identical / drifted / lost-symbol / moved-`data_as_of` comparisons,
tolerance behaviour, JSON-serializability and ordering of the report,
`write_anchor` refusing overwrite and honouring `--force`, history copies (arg
and env var), canonical-JSON determinism and key-order invariance, hash
stability, round-trip through `load_anchor` including tamper detection, and an
offline re-ingest re-basing scenario.
