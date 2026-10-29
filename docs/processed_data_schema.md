# Processed Data Schema — NarrativeCast

**Version:** 1.0 — verified against full processed Parquet
**Dataset:** `data/processed/reddit_technology_clean.parquet`
**Rows:** 11,598,268
**Source:** MADOC Reddit r/technology, cleaned by `scripts/preprocessing/clean_data.py`.
**Unit of observation:** one Reddit interaction (POST or COMMENT).

## 1. Data dictionary

| Column | Actual Arrow type | Null policy | Meaning |
|---|---|---|---|
| `post_id` | `string` | Required | Opaque ID for each interaction; preserve as a string. |
| `publish_date` | `int64` | Required | Unix epoch in seconds, not milliseconds. |
| `publish_datetime_utc` | `timestamp[ns, tz=UTC]` | Required | Canonical UTC field for temporal windows and chronological splitting. |
| `user_id` | `string` | Nullable | Anonymized author ID; preserve nulls. |
| `interaction_type` | `dictionary<values=string, indices=int8, ordered=0>` | Required | Interaction category (e.g., POST or COMMENT); preserve categories. |
| `parent_id` | `string` | Nullable | Parent interaction; null often expected for root posts. |
| `parent_user_id` | `string` | Nullable | Parent author; null allowed. |
| `content` | `string` | Nullable | Cleaned text for embedding if `is_usable_text = true`. |
| `content_hash` | `uint64` | Required | Content hash for deduplication; not a post ID. |
| `word_count` | `int32` | Required | Word count after cleaning. |
| `is_usable_text` | `bool` | Required | Eligibility flag for embedding. |
| `sentiment_vader` | `float` | Nullable | VADER score from input; null allowed. |

## 2. Contract and downstream usage

- Keep the exact column names and types shown above; do not independently rename fields in downstream modules.
- Use `publish_datetime_utc` (UTC) for time windows and chronological splits; all windows use `[start, end)` semantics.
- Keep `publish_date` as Unix seconds, and verify it equals the UTC timestamp.
- Treat IDs as opaque strings, never numeric identifiers. `content_hash` is a content fingerprint, not a record ID.
- Preserve `POST` and `COMMENT` separately for volume and narrative analytics.
- For embeddings, select rows with `is_usable_text == True` and nonblank `content`.
- Keep unusable-text records for auditing/temporal-volume statistics unless another documented filtering rule applies.
- Raw data and processed Parquet remain in Drive/local storage; `data/processed/` is ignored by Git.
- Narrative labels, embeddings and future labels should be saved as derived tables, not added to the canonical cleaned data.

## 3. Full-data validation

- Status: **PASS**
- Data start: `2014-01-01T00:00:16+00:00`
- Data end: `2020-12-31T23:59:25+00:00`
- Usable texts: `10844912`
- Duplicate `post_id` rows: `0`
- Epoch/timestamp mismatches: `0`
- Extra columns: `[]`
- Issues: `[]`
- Full counts: `reports/results/processed_schema_validation.json`.

## 4. Reproduce

From the repository root, after installing cleaning dependencies:
```bash
python scripts/preprocessing/clean_data.py --input data/raw/reddit_technology_madoc-001.parquet --output data/processed/reddit_technology_clean.parquet
python scripts/validate_processed_schema.py --input data/processed/reddit_technology_clean.parquet --report reports/results/processed_schema_validation.json --doc docs/processed_data_schema.md
```
Cleaning parameters and software versions should be recorded when regenerating the data.

## 5. Limitations / sign-off

- This contract applies to Reddit technology data, not necessarily other social platforms.
- Statistical validation alone does not prove the cleaning rules are scientifically optimal.
- Cleaning, embedding, and tracking owners should agree on the input/output conventions.
