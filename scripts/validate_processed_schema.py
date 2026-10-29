"""NarrativeCast Task 7: validate large Parquet without pandas full-file load.

Usage:
 python validate_processed_schema.py --input data/processed/reddit_technology_clean.parquet --report reports/results/processed_schema_validation.json --doc docs/processed_data_schema.md
Dependencies: pyarrow, duckdb
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb

EXPECTED = [
    ("post_id", "string", "Required", "Opaque ID for each interaction; preserve as a string."),
    ("publish_date", "int64", "Required", "Unix epoch in seconds, not milliseconds."),
    ("publish_datetime_utc", "timestamp[ns, tz=UTC]", "Required", "Canonical UTC field for temporal windows and chronological splitting."),
    ("user_id", "string", "Nullable", "Anonymized author ID; preserve nulls."),
    ("interaction_type", "dictionary<string, int8>", "Required", "Interaction category (e.g., POST or COMMENT); preserve categories."),
    ("parent_id", "string", "Nullable", "Parent interaction; null often expected for root posts."),
    ("parent_user_id", "string", "Nullable", "Parent author; null allowed."),
    ("content", "string", "Nullable", "Cleaned text for embedding if `is_usable_text = true`."),
    ("content_hash", "uint64", "Required", "Content hash for deduplication; not a post ID."),
    ("word_count", "int32", "Required", "Word count after cleaning."),
    ("is_usable_text", "bool", "Required", "Eligibility flag for embedding."),
    ("sentiment_vader", "float32", "Nullable", "VADER score from input; null allowed."),
]

def actual_dtype(field):
    return str(field.type)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--report', default='reports/results/processed_schema_validation.json')
    ap.add_argument('--doc', default='docs/processed_data_schema.md')
    ap.add_argument('--memory-limit', default='2GB')
    args = ap.parse_args()
    path = Path(args.input)
    if not path.is_file():
        raise SystemExit(f'ERROR: file not found: {path}')
    pf = pq.ParquetFile(path)
    arrow = pf.schema_arrow
    cols = arrow.names
    expected_names = [x[0] for x in EXPECTED]
    missing = [x for x in expected_names if x not in cols]
    extra = [x for x in cols if x not in expected_names]
    types = {f.name: actual_dtype(f) for f in arrow}
    issues = []
    warnings = []
    if missing: issues.append(f'Missing mandatory schema columns: {missing}')
    if extra: warnings.append(f'Additional columns present: {extra} (record them before freezing contract)')
    expect_type = {
        'post_id': pa.string(), 'publish_date': pa.int64(),
        'publish_datetime_utc': pa.timestamp('ns', tz='UTC'),
        'user_id': pa.string(), 'parent_id': pa.string(), 'parent_user_id': pa.string(),
        'content': pa.string(), 'content_hash': pa.uint64(),
        'word_count': pa.int32(), 'is_usable_text': pa.bool_(),
        'sentiment_vader': pa.float32(),
    }
    for k, expected in expect_type.items():
        if k in cols and arrow.field(k).type != expected:
            issues.append(f'{k}: expected {expected}, got {arrow.field(k).type}')
    if 'interaction_type' in cols:
        fieldtype = arrow.field('interaction_type').type
        if not ((pa.types.is_dictionary(fieldtype) and pa.types.is_string(fieldtype.value_type)) or pa.types.is_string(fieldtype)):
            issues.append(f'interaction_type: string/dictionary(string) expected, got {fieldtype}')
    # Query only when the mandatory schema exists, so SQL cannot reference missing fields.
    stats = {}
    if not missing:
        print('Scanning data with DuckDB (out-of-core, no pandas read_parquet)...', flush=True)
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{args.memory_limit}'")
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET threads=2")
        query = '''SELECT
          COUNT(*) AS rows,
          COUNT(*) FILTER (WHERE post_id IS NULL) AS null_post_id,
          COUNT(*) FILTER (WHERE publish_date IS NULL) AS null_publish_date,
          COUNT(*) FILTER (WHERE publish_datetime_utc IS NULL) AS null_timestamp,
          COUNT(*) FILTER (WHERE interaction_type IS NULL) AS null_interaction_type,
          COUNT(*) FILTER (WHERE content_hash IS NULL) AS null_content_hash,
          COUNT(*) FILTER (WHERE word_count IS NULL) AS null_word_count,
          COUNT(*) FILTER (WHERE is_usable_text IS NULL) AS null_is_usable_text,
          COUNT(*) FILTER (WHERE user_id IS NULL) AS null_user_id,
          COUNT(*) FILTER (WHERE parent_id IS NULL) AS null_parent_id,
          COUNT(*) FILTER (WHERE parent_user_id IS NULL) AS null_parent_user_id,
          COUNT(*) FILTER (WHERE content IS NULL) AS null_content,
          COUNT(*) FILTER (WHERE sentiment_vader IS NULL) AS null_sentiment_vader,
          COUNT(*) FILTER (WHERE is_usable_text = TRUE) AS usable_text_rows,
          COUNT(*) FILTER (WHERE is_usable_text = TRUE AND (content IS NULL OR TRIM(content) = '')) AS usable_but_blank,
          COUNT(*) FILTER (WHERE publish_date IS NOT NULL AND publish_datetime_utc IS NOT NULL AND epoch(publish_datetime_utc) <> publish_date) AS epoch_mismatch,
          COUNT(*) FILTER (WHERE publish_datetime_utc < TIMESTAMPTZ '2014-01-01 00:00:00+00' OR publish_datetime_utc >= TIMESTAMPTZ '2021-01-01 00:00:00+00') AS outside_2014_2020,
          MIN(publish_datetime_utc) AS date_min_utc,
          MAX(publish_datetime_utc) AS date_max_utc
        FROM read_parquet(?)'''
        rel = con.execute(query, [str(path)])
        names = [c[0] for c in rel.description]
        stats = dict(zip(names, rel.fetchone()))
        for k in ('date_min_utc', 'date_max_utc'):
            if stats[k] is not None: stats[k] = stats[k].isoformat()
        # Exact duplicate IDs without storing all IDs in Python memory.
        # Memory pressure is managed by DuckDB with its local spill directory.
        print('Checking duplicate post_id...', flush=True)
        duplicate_rows = con.execute('''
            SELECT COALESCE(SUM(n-1), 0) FROM (
              SELECT COUNT(*) AS n FROM read_parquet(?)
              WHERE post_id IS NOT NULL GROUP BY post_id HAVING COUNT(*) > 1
            ) t
        ''', [str(path)]).fetchone()[0]
        stats['duplicate_post_id_rows'] = int(duplicate_rows)
        stats['interaction_type_counts'] = {
            str(k): int(v) for k,v in con.execute('''
               SELECT CAST(interaction_type AS VARCHAR), COUNT(*)
               FROM read_parquet(?) GROUP BY 1 ORDER BY 2 DESC
            ''', [str(path)]).fetchall()
        }
        con.close()
        for col in ['null_post_id', 'null_publish_date', 'null_timestamp',
                    'null_interaction_type', 'null_content_hash', 'null_word_count', 'null_is_usable_text',
                    'usable_but_blank', 'epoch_mismatch', 'outside_2014_2020', 'duplicate_post_id_rows']:
            if stats[col]: issues.append(f'{col}: {stats[col]}')
        if stats['rows'] != pf.metadata.num_rows:
            issues.append('Row count differs from Parquet metadata')

    result = {
        'dataset': path.name, 'generated_utc': datetime.now(timezone.utc).isoformat(),
        'row_count_from_metadata': pf.metadata.num_rows,
        'row_groups': pf.num_row_groups,
        'schema': types, 'missing_columns': missing, 'extra_columns': extra,
        'stats': stats, 'issues': issues, 'warnings': warnings,
        'status': 'PASS' if not issues else 'CHECK_REQUIRED',
    }
    report = Path(args.report); report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    # Build definitive schema doc based on observed Arrow schema and validated statistics.
    doc = Path(args.doc); doc.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Processed Data Schema — NarrativeCast', '',
        '**Version:** 1.0 — verified against full processed Parquet' if not issues else '**Version:** draft — issues require review',
        f'**Dataset:** `data/processed/{path.name}`',
        f'**Rows:** {pf.metadata.num_rows:,}',
        '**Source:** MADOC Reddit r/technology, cleaned by `scripts/preprocessing/clean_data.py`.',
        '**Unit of observation:** one Reddit interaction (POST or COMMENT).', '',
        '## 1. Data dictionary', '',
        '| Column | Actual Arrow type | Null policy | Meaning |',
        '|---|---|---|---|',
    ]
    for name, fallback, policy, meaning in EXPECTED:
        dtype = types.get(name, 'MISSING')
        lines.append(f'| `{name}` | `{dtype}` | {policy} | {meaning} |')
    lines += ['', '## 2. Contract and downstream usage', '',
        '- Keep the exact column names and types shown above; do not independently rename fields in downstream modules.',
        '- Use `publish_datetime_utc` (UTC) for time windows and chronological splits; all windows use `[start, end)` semantics.',
        '- Keep `publish_date` as Unix seconds, and verify it equals the UTC timestamp.',
        '- Treat IDs as opaque strings, never numeric identifiers. `content_hash` is a content fingerprint, not a record ID.',
        '- Preserve `POST` and `COMMENT` separately for volume and narrative analytics.',
        '- For embeddings, select rows with `is_usable_text == True` and nonblank `content`.',
        '- Keep unusable-text records for auditing/temporal-volume statistics unless another documented filtering rule applies.',
        '- Raw data and processed Parquet remain in Drive/local storage; `data/processed/` is ignored by Git.',
        '- Narrative labels, embeddings and future labels should be saved as derived tables, not added to the canonical cleaned data.',
        '', '## 3. Full-data validation', '',
        f'- Status: **{result["status"]}**',
        f'- Data start: `{stats.get("date_min_utc", "not checked")}`',
        f'- Data end: `{stats.get("date_max_utc", "not checked")}`',
        f'- Usable texts: `{stats.get("usable_text_rows", "not checked")}`',
        f'- Duplicate `post_id` rows: `{stats.get("duplicate_post_id_rows", "not checked")}`',
        f'- Epoch/timestamp mismatches: `{stats.get("epoch_mismatch", "not checked")}`',
        f'- Extra columns: `{extra}`',
        f'- Issues: `{issues}`',
        '- Full counts: `reports/results/processed_schema_validation.json`.',
        '', '## 4. Reproduce', '',
        'From the repository root, after installing cleaning dependencies:',
        '```bash',
        'python scripts/preprocessing/clean_data.py --input data/raw/reddit_technology_madoc-001.parquet --output data/processed/reddit_technology_clean.parquet',
        'python scripts/validate_processed_schema.py --input data/processed/reddit_technology_clean.parquet --report reports/results/processed_schema_validation.json --doc docs/processed_data_schema.md',
        '```',
        'Cleaning parameters and software versions should be recorded when regenerating the data.',
        '', '## 5. Limitations / sign-off', '',
        '- This contract applies to Reddit technology data, not necessarily other social platforms.',
        '- Statistical validation alone does not prove the cleaning rules are scientifically optimal.',
        '- Cleaning, embedding, and tracking owners should agree on the input/output conventions.',
    ]
    doc.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\nRESULT:', result['status'])
    print('Rows:', pf.metadata.num_rows, '| Columns:', len(cols))
    print('Issues:', issues)
    print('Warnings:', warnings)
    print('Report:', report)
    print('Schema doc:', doc)
    if issues: raise SystemExit(1)

if __name__ == '__main__':
    main()
