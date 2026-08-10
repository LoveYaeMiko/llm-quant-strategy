# ADR-0003: Postgres JSONB PIT backend

The production PIT store is a `PostgresPointInTimeLoader` in `src/data/postgres_loader.py`, driven by raw psycopg2 SQL (no SQLAlchemy) over a single `pit_records(symbol, valid_from, valid_to, record_type, payload JSONB)` table with `ON CONFLICT (symbol, valid_from, record_type) DO UPDATE`. JSONB mirrors the existing SQLite payload-blob contract so all three backends share one hydration path (`_rows_to_frame`) and a future fundamentals/news schema never needs a migration. SQLAlchemy was rejected (one fixed-contract table), click rejected (the CLI is argparse), python-dotenv rejected (stdlib `.env` loader exists). Local dev runs Postgres via `docker-compose.yml`; SQLite remains the offline test backend.

## Update (2026-08-10): `record_type` joins the primary key

The original PK was `(symbol, valid_from)` with `record_type` only a payload discriminator. During the full A-share ingest (5,162 symbols × 2010–2025, 12.48M bars), the price phase re-upserted bars on dates that the Baostock universe snapshots also used (e.g. `(X, 2015-01-05)` as both a price bar and a universe membership record). `ON CONFLICT (symbol, valid_from) DO UPDATE` silently **clobbered the universe payload with the price payload**: the 2015 cohort shrank from 2,594 to 444 names, and B4's survivorship count under-reported.

The fix makes `record_type` a real column and part of the key — `(symbol, valid_from, record_type)` — across **all three** backends:

- Postgres: `record_type TEXT NOT NULL DEFAULT ''`, composite `PRIMARY KEY (symbol, valid_from, record_type)`.
- SQLite: same column + composite PK (`INSERT OR REPLACE`).
- In-memory `PointInTimeStore`: `drop_duplicates(subset=[symbol, valid_from, record_type], keep="last")`; frames without a discriminator normalize to `""`.

`record_type` remains in the payload too (the `_rows_to_frame` hydration and `payload->>'record_type'` read filters are unchanged), so an existing DB can be migrated in place:

```sql
ALTER TABLE pit_records ADD COLUMN record_type TEXT NOT NULL DEFAULT '';
UPDATE pit_records SET record_type = payload->>'record_type' WHERE record_type = '';
ALTER TABLE pit_records DROP CONSTRAINT pit_records_pkey;
ALTER TABLE pit_records ADD PRIMARY KEY (symbol, valid_from, record_type);
```

A price bar and a universe snapshot on the same `(symbol, valid_from)` now coexist; a restatement of the *same* `record_type` on the same key still supersedes in place.
