# Price-basis contract (defect C2)

**Status:** implemented 2026-09-09 · **Code:** `src/data/basis.py` · **Tests:** `tests/test_data_basis.py`
**Related:** [ADR-0002](adr/0002-dual-column-adjustment.md) (dual-column storage), defect D-8 (`tests/test_pullback_atr_basis.py`)

---

## 1. The finding

The PIT price payload is **not on one price basis**. For `record_type='price'` the stored payload is

```json
{"open": .., "high": .., "low": ..,
 "close": <ADJUSTED>, "raw_close": <RAW>,
 "adjust_factor": <float>, "volume": .., "amount": .., "name": ..}
```

ADR-0002 chose this deliberately: `close` is **backward-adjusted**
(`close = raw_close × adjust_factor`, the factor anchored at the newest bar) so
return factors have no ex-dividend gaps, while `raw_close` + `adjust_factor`
preserve the unadjusted tape for consistency checks. ADR-0002 does **not** state
— and the storage layer never enforced — that `open` / `high` / `low` are written
**raw**.

`src/cli.py::_market_from_records` builds `market.long` and `market.price_panel`
straight from those columns:

```python
long = rec.set_index(["date", "symbol"])[["open", "high", "low", "close", "volume", "amount"]]
close_wide = long["close"].unstack()
```

so every consumer inherits a panel whose `close` column is on a **different
basis** than its `open` / `high` / `low` columns. Any true range, intraday gap or
high/low breakout computed off that panel is arithmetically wrong on every
corporate-action bar — a 10:1 split makes `high / close ≈ 10`.

### Empirical verification (production store)

Counts, measured with SQL against the running container `fqa-pit-db`
(`SELECT count(*) … WHERE payload->>'record_type'='price'`):

| quantity | value |
| --- | --- |
| price rows | 12,535,534 |
| distinct symbols | 5,164 |
| coverage | 2010-01-04 … 2026-09-09 |
| rows with `|adjust_factor − 1| > 0.02` ("action rows") | 8,708,807 |

Per-column inference, measured on a 6-symbol / 22,928-bar slice
(21,901 action rows, `adjust_factor` 0.041…1.0) with `basis_report`:

| column | inferred basis | median rel. error vs `raw_close` | median rel. error vs `raw_close × factor` | verdict |
| --- | --- | --- | --- | --- |
| `open` | **raw** | 9.32e-3 | 3.06e-1 | raw is 33× closer |
| `high` | **raw** | 1.04e-2 | 3.23e-1 | raw is 31× closer |
| `low` | **raw** | 9.59e-3 | 2.88e-1 | raw is 30× closer |
| `close` | **adjusted** | 2.35e-1 | 1.26e-6 | adjusted is 187,000× closer |

The ~1e-2 residual on the raw columns is **not noise** — it is the day's real
intraday move from the previous close. `max_rel_mismatch` on that slice is
1.69e-1, i.e. the widest single-day intraday move in the sample, again expected.

Spot table (real rows, ratios to `raw_close`):

| symbol | date | open | high | low | close | raw_close | adjust_factor | open/raw_close | high/raw_close | low/raw_close | close/raw_close |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 002415.SZ | 2010-05-28 | 78.00 | 83.00 | 76.88 | 3.3627 | 81.93 | 0.041043 | 0.952032 | 1.013060 | 0.938362 | 0.041044 |
| 002415.SZ | 2010-05-31 | 85.88 | 87.77 | 83.51 | 3.4887 | 85.00 | 0.041043 | 1.010353 | 1.032588 | 0.982471 | 0.041044 |
| 002415.SZ | 2026-09-09 | 33.63 | 33.87 | 33.36 | 33.7500 | 33.75 | 1.000000 | 0.996444 | 1.003556 | 0.988444 | 1.000000 |
| 000333.SZ | 2013-09-18 | 40.50 | 46.40 | 39.88 | 7.0965 | 42.24 | 0.168004 | 0.958807 | 1.098485 | 0.944129 | 0.168004 |
| 000333.SZ | 2013-09-23 | 41.90 | 46.46 | 40.67 | 7.7618 | 46.20 | 0.168004 | 0.906926 | 1.005628 | 0.880303 | 0.168004 |
| 000333.SZ | 2026-09-09 | 85.00 | 86.04 | 84.44 | 86.0200 | 86.02 | 1.000000 | 0.988142 | 1.000233 | 0.981632 | 1.000000 |
| 600000.SH | 2010-01-04 | 21.83 | 21.87 | 21.16 | 4.7078 | 21.19 | 0.222170 | 1.030203 | 1.032091 | 0.998584 | 0.222171 |
| 600000.SH | 2010-01-05 | 21.41 | 21.58 | 20.79 | 4.7433 | 21.35 | 0.222170 | 1.002810 | 1.010773 | 0.973770 | 0.222169 |
| 600000.SH | 2026-09-09 | 9.25 | 9.29 | 9.21 | 9.2300 | 9.23 | 1.000000 | 1.002167 | 1.006501 | 0.997833 | 1.000000 |

Reading the table: `open` / `high` / `low` are **within a few percent of
`raw_close`** (the intraday move) — they are on the **raw** basis. `close` is
**exactly `raw_close × adjust_factor`** to the last digit — it is on the
**adjusted** basis. The two bases differ by 2–24× on these rows.

---

## 2. The contract

`src/data/basis.py` (numpy/pandas only) makes the basis **explicit, measurable
and assertable**:

| symbol | meaning |
| --- | --- |
| `BASIS_ADJUSTED = "adjusted"`, `BASIS_RAW = "raw"`, `BASIS_UNKNOWN = "unknown"` | the vocabulary; `PRICE_COLUMNS = ("open","high","low","close")` |
| `infer_basis(records) -> dict[str, str]` | per price column, the basis it is **actually** on |
| `basis_report(records) -> dict` | JSON-serializable evidence snapshot |
| `to_adjusted(records)` / `to_raw(records)` | the **sanctioned** conversions (copy in, copy out, never mutate) |
| `assert_single_basis(records, target)` | hard failure when a column is not on `target` |

### Inference rule (empirical, never assumed)

Only rows with `|adjust_factor − 1| > 0.02` can distinguish the two bases —
everywhere else `raw_close == close` and the question is unanswerable. On those
*action rows* each column is compared against its two **constructed** twins:

```
raw twin      = raw_close
adjusted twin = raw_close × adjust_factor
```

The twins are built from the tape, never read from the column under test —
comparing `close` against the stored `close` would be circular and would hide a
payload whose stored `close` was written raw. The basis whose median
`|column / twin − 1|` is smaller wins, provided that error is at most
`MATCH_TOL` (1e-9) **or** at most `_DECISIVE_RATIO` (25%) of the alternative's.
Everything else — a tie, no action rows, a missing `raw_close` and no
`adjust_factor` — returns `"unknown"`. The function never guesses.

### Report contents

```python
{
  "columns": {"open": "raw", "high": "raw", "low": "raw", "close": "adjusted"},
  "n_rows": 22928, "n_symbols": 6,
  "n_action_rows": 21901, "n_action_symbols": 6,
  "factor_min": 0.04104337492888055, "factor_max": 1.0,
  "median_rel_error": {"open": {"raw": 0.00932, "adjusted": 0.30584}, ...},
  "max_rel_mismatch": 0.16908077994428972,
  "consistent": false, "mixed_columns": ["close"],
  "dominant_basis": "raw", "threshold": 0.02, "basis_column": "_basis",
}
```

`consistent` is true only when every *classified* column sits on the same basis
(a column that could not be classified does not make a panel "consistent").
Every value is a builtin, so the dict survives `json.dumps` — it can be stored in
an audit record or evidence artifact as-is.

---

## 3. The decision: do **not** silently re-base the panel

**Rejected:** converting `market.long` / `market.price_panel` to a single basis
inside the read layer.

Re-basing would rewrite the input of every historical backtest, every stored
evidence artifact (`docs/D_TRACK_EVIDENCE.md`, the shadow ledger
`outputs/shadow_ledger_D_5W.sqlite`, the ML artifacts' cached training frames)
and every already-published metric. A defect fix would then be indistinguishable
from a strategy change: no previously reported number could be reproduced, and
every comparison against past runs would be invalid. The mixed basis is a real
defect, but it is *bounded and known* (see §4) — silently moving the numbers
would trade a visible, documented defect for an invisible, undocumented
discontinuity in the research record.

**Adopted:**

1. **The panel keeps its current numbers.** Not a single stored value changes;
   every historical result stays reproducible byte-for-byte.
2. **The basis of every column is now explicit and assertable.**
   `_market_from_records` attaches `market.basis = basis_report(rec)` and
   `market.basis_target = "adjusted"` (the basis of `market.price_panel`), so any
   artifact can record what its numbers are on, and any consumer can call
   `assert_single_basis(...)` before mixing columns.
3. **`to_adjusted` / `to_raw` are the sanctioned conversion functions.** A
   consumer that genuinely needs one basis converts explicitly, with a per-symbol
   forward-filled `adjust_factor` (1.0 fallback for a missing factor — exact for
   the newest bars, where the factor is anchored). Both are pure: they return a
   copy, mark it `_basis`, and never mutate the input.

`SyntheticMarket` gained an optional `basis: dict | None = None` field (appended
last with a default, so every positional and keyword construction keeps working;
`None` for the synthetic fixture, whose OHLC is all on one basis).

---

## 4. Blast radius: the pullback book is the only exposed consumer

A repo-wide grep for other uses of `market.long` / `["high"]` / `["low"]` /
`["open"]`:

| location | what it reads | basis exposure |
| --- | --- | --- |
| `src/paper/pullback_book.py:130-131` | `long_["high"]`, `long_["low"]` | **exposed** — and already handled (below) |
| `src/paper/ml_book.py:64` | `market.long` tail panel | feature formulas; every formula in the deployed artifacts is built from `close`/`volume`, no raw OHLC — **not exposed** |
| `src/preclose.py:91-93, 114, 176` | minute-bar `open`/`high`/`low` | minute bars come from AlphaFeed **already 前复权** (verified in `tests/test_pullback_atr_basis.py::test_minute_prints_are_compared_as_is`) — **not exposed** |
| `src/data/intraday.py:108, 118, 131-132` | minute-bar OHLC | same, minute bars are adjusted — **not exposed** |
| `src/cli.py` (`FactorContext(market.long)`, `_market_trend(market.price_panel)`) | `close`/`volume` columns | **not exposed** |
| `src/portfolio/backtest_runner.py:88` | builds `long` from an in-memory frame | not the PIT read path |

So exactly one consumer multiplies raw OHLC into an adjusted close, and it
already does so explicitly (`_adjust_factor_frame`, defect D-8, 2026-09-08).

### The book's factor multiplication == `to_adjusted` (verified numerically)

For `002415.SZ` (factor `0.04104337492888055`), on the action date
`2010-05-28`:

```
book high = 3.4066001190970856    to_adjusted high = 3.4066001190970856
book low  = 3.1554146645323367    to_adjusted low  = 3.1554146645323367
```

and over **all** 3,902 bars of that symbol:

```
max |book_high - to_adjusted(high)| = 0.000e+00
max |book_low  - to_adjusted(low) | = 0.000e+00
```

Bit-identical. The book's existing correction and the sanctioned conversion agree
exactly, so the contract documents and verifies behaviour that is already correct
rather than changing it.

---

## 5. Usage

```python
from src.data.basis import basis_report, assert_single_basis, to_adjusted

market.basis                    # {"columns": {"open": "raw", ..., "close": "adjusted"}, ...}
market.basis_target             # "adjusted" — the basis of market.price_panel

assert_single_basis(recs, "adjusted", columns=["close"])   # passes
assert_single_basis(recs, "adjusted", columns=["high"])    # ValueError, names the column

panel = to_adjusted(recs)       # every price column on the adjusted basis
panel.attrs["basis"]            # "adjusted"; panel["_basis"] is the same marker
raw = to_raw(panel)             # back to the raw tape
```

`assert_single_basis` fails on `"unknown"` as well as on a mismatched basis:
unverifiable is not verified.

---

## 6. Reproducing the verification

```powershell
# unit tests (offline, hand-built frames)
python -m pytest tests/test_data_basis.py -q

# the pre-existing consumer contract
python -m pytest tests/test_pullback_atr_basis.py -q

# full suite (known pre-existing failure: tests/test_phase8_remedy.py)
python -m pytest tests/ --ignore=tests/test_phase8_remedy.py -q
```

The production numbers in §1 come from `fqa-pit-db`:

```sql
SELECT count(*), count(DISTINCT symbol), min(valid_from), max(valid_from)
FROM pit_records WHERE payload->>'record_type' = 'price';

SELECT count(*) FROM pit_records
WHERE payload->>'record_type' = 'price'
  AND abs((payload->>'adjust_factor')::float8 - 1) > 0.02;
```

Per-column error figures were computed by `basis_report` on a 6-symbol slice
(`000001.SZ`, `000002.SZ`, `000333.SZ`, `002415.SZ`, `600000.SH`, `600519.SH`).
The full 12.5M-row frame cannot be hydrated in memory on this machine
(`_rows_to_frame` needs ~669 MiB for one 7-column block and the snapshot path
OOMs), so the per-column inference was measured on the slice while the row counts
came from SQL. That limitation is itself worth noting: the read layer cannot
currently materialize the whole price store at once.
