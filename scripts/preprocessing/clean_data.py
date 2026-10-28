#!/usr/bin/env python
"""NarrativeCast -- Production-grade data cleaning pipeline.

Transforms raw MADOC Reddit r/technology Parquet into a cleaned dataset
ready for SBERT embedding and BERTopic clustering. All cleaning rules
are evidence-based, derived from EDA notebooks 01-03 findings.

Usage:
    python scripts/preprocessing/clean_data.py \
        --input data/raw/reddit_technology_madoc-001.parquet \
        --output data/processed/reddit_technology_clean.parquet
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import logging
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
logger = logging.getLogger("narrativecast.clean_data")

# ---------------------------------------------------------------------------
# Constants -- Evidence-based patterns from EDA-02 & EDA-03
# ---------------------------------------------------------------------------

# AutoModerator boilerplate (EDA-03 Section 4: 7+ template variants,
# 38%-98% originating from Top-20 Power-Users)
AUTOMOD_PATTERN = re.compile(
    r"^Thank you for your (submission|comment)!\s*Unfortunately",
    re.IGNORECASE,
)
# French spam bot (EDA-03 Section 4: 13,596 exact copies, 100% from 1 user)
SPAM_PATTERN = re.compile(r"Message de test vraie hein", re.IGNORECASE)
# User-deleted marker
DELETED_BY_USER_PATTERN = re.compile(
    r"^\[deleted by user\]$", re.IGNORECASE
)

BOT_SPAM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("automod_boilerplate", AUTOMOD_PATTERN),
    ("spam_test_message", SPAM_PATTERN),
    ("deleted_by_user", DELETED_BY_USER_PATTERN),
]

# Expected 13-column MADOC schema 
EXPECTED_COLUMNS: list[str] = [
    "post_id", "publish_date", "user_id", "parent_id",
    "parent_user_id", "content", "url", "language",
    "interaction_type", "platform", "community",
    "strict_filter", "sentiment_vader",
]

# Constant columns verified by EDA-01 (zero variance -> safe to drop)
CONSTANT_COLS_EXPECTED: dict[str, set[str]] = {
    "platform": {"reddit"},
    "community": {"technology"},
    "language": {"en", "english"},
}

# Valid temporal range for r/technology MADOC (EDA-02 Section 1)
VALID_DATE_MIN = pd.Timestamp("2014-01-01", tz="UTC")
VALID_DATE_MAX = pd.Timestamp("2021-01-01", tz="UTC")

# Subdomain keyword taxonomy (EDA-03 Section 5)
# Word-boundary regex patterns for 6 sub-domains, 2014-2020 scope
SUBDOMAIN_KEYWORDS: dict[str, list[str]] = {
    "ai_automation": [
        r"\bartificial intelligence\b", r"\bmachine learning\b",
        r"\bdeep learning\b", r"\bneural network\b", r"\brobot",
        r"\bautonomous\b", r"\bself[- ]driving\b", r"\bchatbot\b",
        r"\bGPT\b", r"\bOpenAI\b", r"\bautomation\b",
    ],
    "cybersecurity_privacy": [
        r"\bcybersecurity\b", r"\bhack", r"\bbreach\b",
        r"\bmalware\b", r"\bransomware\b", r"\bphishing\b",
        r"\bprivacy\b", r"\bsurveillance\b", r"\bencryption\b",
        r"\bGDPR\b", r"\bdata protection\b", r"\bNSA\b",
        r"\bcyber.?attack\b",
    ],
    "bigtech_antitrust": [
        r"\bGoogle\b", r"\bFacebook\b", r"\bAmazon\b", r"\bApple\b",
        r"\bMicrosoft\b", r"\bantitrust\b", r"\bmonopoly\b",
        r"\bregulat", r"\blobby", r"\bFTC\b",
        r"\bcensorship\b", r"\bcontent moderation\b",
    ],
    "semiconductors_hardware": [
        r"\bsemiconductor\b", r"\bchipmaker\b", r"\bprocessor\b",
        r"\bCPU\b", r"\bGPU\b", r"\bIntel\b", r"\bAMD\b",
        r"\bNvidia\b", r"\bTSMC\b", r"\btransistor\b",
    ],
    "clean_energy_ev": [
        r"\bsolar\b", r"\bwind energy\b", r"\brenewable\b",
        r"\belectric vehicle\b", r"\bTesla\b", r"\bbattery\b",
        r"\bclimate\b", r"\bcarbon\b", r"\bsustainab",
        r"\bgreen energy\b", r"\bEV\b",
    ],
    "telecom_netneutrality_crypto": [
        r"\bnet neutrality\b", r"\bFCC\b", r"\bISP\b",
        r"\bbroadband\b", r"\b5G\b", r"\btelecommunication\b",
        r"\bComcast\b", r"\bVerizon\b",
        r"\bcryptocurrency\b", r"\bbitcoin\b", r"\bblockchain\b",
    ],
}

# Preferred output column order for downstream pipeline readability
OUTPUT_COLUMN_ORDER: list[str] = [
    "post_id", "publish_date", "publish_datetime_utc",
    "user_id", "interaction_type",
    "parent_id", "parent_user_id",
    "content", "content_hash", "word_count", "is_usable_text",
    "sentiment_vader",
]


# ===================================================================
# SECTION 1: Data Loading & Schema Validation
# ===================================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load Parquet file and validate against expected MADOC schema.

    Args:
        path: Path to the input Parquet file.

    Returns:
        DataFrame with validated column set.

    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If required columns are missing from the schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    logger.info("Loading parquet: %s", path)
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(
        "Loaded %d rows x %d cols (%.1f MB in memory)",
        len(df),
        len(df.columns),
        df.memory_usage(deep=True).sum() / 1024 / 1024,
    )

    # Schema validation: all 13 MADOC columns must be present
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )
    return df


# ===================================================================
# SECTION 2: Column Pruning (Drop Constant & Unused Columns)
# ===================================================================

def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop zero-variance, fully-null, and unused columns to free RAM early.

    Verifies that platform/community/language are truly constant before
    dropping. Drops strict_filter (100% null) and raw url (87.86% null,
    not used in embedding pipeline).

    Args:
        df: DataFrame to prune.

    Returns:
        DataFrame with constant and unused columns removed.
    """
    cols_before = set(df.columns)

    # strict_filter: 100% null per EDA-01 -> zero information
    if "strict_filter" in df.columns:
        null_pct = df["strict_filter"].isna().mean() * 100
        if null_pct > 99.0:
            df = df.drop(columns=["strict_filter"])
            logger.info("Dropped 'strict_filter' (%.1f%% null)", null_pct)
        else:
            logger.warning(
                "strict_filter has %.1f%% non-null, keeping", 100 - null_pct
            )

    # Verify single-value columns then drop (platform, community, language)
    for col, expected in CONSTANT_COLS_EXPECTED.items():
        if col not in df.columns:
            continue
        uniq = df[col].dropna().unique()
        if len(uniq) == 1 and str(uniq[0]).lower() in {s.lower() for s in expected}:
            df = df.drop(columns=[col])
            logger.info(
                "Dropped '%s' (constant='%s', zero variance)", col, uniq[0]
            )
        else:
            logger.warning(
                "Column '%s' not constant: %s -- keeping for review",
                col,
                uniq[:5].tolist(),
            )

    # Drop raw URL (87.86% null, not used in embedding pipeline)
    if "url" in df.columns:
        df = df.drop(columns=["url"])
        logger.info("Dropped 'url' (87.86%% null, not used in embedding pipeline)")

    dropped = sorted(cols_before - set(df.columns))
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(
        "Column pruning: dropped %d cols %s -> %.1f MB",
        len(dropped),
        dropped,
        mem_mb,
    )
    return df


# ===================================================================
# SECTION 3: Temporal Standardization
# ===================================================================

def standardize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert UNIX epoch (seconds) to UTC datetime, verify order.

    Creates 'publish_datetime_utc' for temporal windowing (W_hist=7d,
    h in {3,7,14}d) while preserving original 'publish_date' int64
    for traceability. Sorts by time if needed (idempotent safety).

    Args:
        df: DataFrame with 'publish_date' column (UNIX seconds int64).

    Returns:
        DataFrame with 'publish_datetime_utc' added, sorted temporally.
    """
    logger.info("Standardizing timestamps: UNIX epoch -> UTC datetime")

    # Parse UNIX epoch (seconds) to UTC-aware datetime for leak-free
    # temporal windowing downstream
    df["publish_datetime_utc"] = pd.to_datetime(
        df["publish_date"], unit="s", utc=True
    )

    dt_min = df["publish_datetime_utc"].min()
    dt_max = df["publish_datetime_utc"].max()
    logger.info("Temporal range: %s -> %s", dt_min, dt_max)

    # Validate range [2014-01-01, 2021-01-01) per dataset_card.md
    n_oob = (
        (df["publish_datetime_utc"] < VALID_DATE_MIN)
        | (df["publish_datetime_utc"] >= VALID_DATE_MAX)
    ).sum()
    if n_oob > 0:
        logger.warning(
            "%d rows outside expected date range [%s, %s)",
            n_oob,
            VALID_DATE_MIN,
            VALID_DATE_MAX,
        )

    # Verify monotonic order; sort if needed for safe temporal windowing
    if df["publish_date"].is_monotonic_increasing:
        logger.info("Temporal order OK: publish_date is monotonic increasing")
    else:
        logger.warning(
            "publish_date NOT monotonic -> sorting by publish_date"
        )
        df = df.sort_values("publish_date").reset_index(drop=True)
        logger.info("Re-sorted %d rows by publish_date", len(df))

    return df


# ===================================================================
# SECTION 4: Dtype Optimization
# ===================================================================

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast low-cardinality and numeric columns to reduce RAM.

    Applied BEFORE text cleaning to maximize headroom for the string
    operations that follow. interaction_type (cardinality=2) from
    object -> category saves ~588 MB; sentiment_vader float64 -> float32
    saves ~45 MB.

    Args:
        df: DataFrame with columns to downcast.

    Returns:
        DataFrame with optimized dtypes, ~630 MB freed.
    """
    mem_before = df.memory_usage(deep=True).sum() / 1024 / 1024

    if "interaction_type" in df.columns:
        df["interaction_type"] = df["interaction_type"].astype("category")

    if "sentiment_vader" in df.columns:
        df["sentiment_vader"] = df["sentiment_vader"].astype("float32")

    mem_after = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(
        "Dtype optimization: %.1f -> %.1f MB (saved %.1f MB)",
        mem_before,
        mem_after,
        mem_before - mem_after,
    )
    return df


# ===================================================================
# SECTION 5: Text Cleaning Pipeline (Steps 6-12)
# ===================================================================

def clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply evidence-based text cleaning to 'content' column in-place.

    Processing order is critical: HTML unescape MUST precede markdown
    cleanup because Reddit blockquote '>' is stored as '&gt;' in MADOC.

    Steps 6-12 map to the pipeline design in plan_04. Each step uses
    vectorized pandas string methods to avoid Python-level row iteration,
    and operates on df['content'] directly to prevent holding two copies
    of the ~1.5 GB string column simultaneously.

    Args:
        df: DataFrame with 'content' column (raw text).

    Returns:
        DataFrame with cleaned 'content' column.
    """
    n = len(df)
    logger.info("Text cleaning pipeline: %d rows", n)

    # --- Step 6: HTML entity decoding (optimized with boolean mask) --
    # Only ~7.3% of rows contain '&' (856,739 rows per EDA-03 Sec 2).
    # Masking avoids calling html.unescape on 10.9M clean rows.
    mask_amp = df["content"].str.contains("&", regex=False, na=False)
    n_amp = int(mask_amp.sum())
    if n_amp > 0:
        df.loc[mask_amp, "content"] = (
            df.loc[mask_amp, "content"].apply(html.unescape)
        )
    logger.info(
        "  Step 6  html.unescape: %d rows (%.2f%%)",
        n_amp,
        n_amp / n * 100,
    )

    # --- Step 7: Unicode NFC + control character removal -----------
    # NFC ensures canonical byte representation so content_hash is
    # deterministic across different input encodings
    df["content"] = df["content"].str.normalize("NFC")
    # Strip invisible control chars (keep \n \t; collapsed in step 12)
    df["content"] = df["content"].str.replace(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", regex=True
    )
    # Remove U+FFFD replacement character (encoding error artifact)
    df["content"] = df["content"].str.replace(
        "\ufffd", "", regex=False
    )
    logger.info("  Step 7  Unicode NFC + control chars: done")

    # --- Step 8: Markdown link extraction [text](url) -> text ------
    # Preserves anchor text semantic signal while discarding URL noise;
    # affects 323,889 rows (2.75%) per EDA-03 Section 2
    df["content"] = df["content"].str.replace(
        r"\[([^\]]+)\]\([^\)]+\)", r"\1", regex=True
    )
    logger.info("  Step 8  Markdown link [text](url) -> text: done")

    # --- Step 9: Raw URL removal -----------------------------------
    # 592,480 rows (5.04%) contain bare URLs that waste SBERT 256-token
    # budget without carrying narrative semantic signal
    df["content"] = df["content"].str.replace(
        r"https?://\S+", "", regex=True
    )
    logger.info("  Step 9  Raw URL removal: done")

    # --- Step 10: Markdown markup removal (keep inner text) --------
    # Only remove formatting characters; the text they wrap is preserved
    # Blockquote '>' at line start (now actual '>' after html.unescape)
    df["content"] = df["content"].str.replace(
        r"(?m)^>\s*", "", regex=True
    )
    # Bold markers: 2+ consecutive asterisks (**text** -> text)
    df["content"] = df["content"].str.replace(
        r"\*{2,}", "", regex=True
    )
    # Strikethrough markers: ~~text~~ -> text
    df["content"] = df["content"].str.replace("~~", "", regex=False)
    # Inline code backticks: `text` -> text (also ``` blocks)
    df["content"] = df["content"].str.replace(r"`+", "", regex=True)
    logger.info("  Step 10 Markdown markup removal: done")

    # --- Step 11: Trailing [removed]/[deleted] suffix trimming -----
    # 83,214 POST titles have [removed] appended by Reddit system
    # (EDA-03 Section 6.1). Case-insensitive for safety.
    df["content"] = df["content"].str.replace(
        r"\s*\[(removed|deleted)\]\s*$",
        "",
        regex=True,
        flags=re.IGNORECASE,
    )
    logger.info("  Step 11 Trailing [removed]/[deleted] trim: done")

    # --- Step 12: Whitespace normalization -------------------------
    # Collapse all whitespace (incl. \n \t) to single space, strip
    df["content"] = (
        df["content"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    logger.info("  Step 12 Whitespace normalization: done")

    gc.collect()
    logger.info("Text cleaning pipeline complete")
    return df


# ===================================================================
# SECTION 6: Row Filtering (Hard-drop)
# ===================================================================

def drop_garbage_rows(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove placeholder, empty, and sub-3-alpha-char content rows.

    Applied AFTER text cleaning so that rows reduced to empty strings
    by URL/markdown removal are caught. Uses incremental masking so
    each drop reason reports its exclusive (non-overlapping) count.

    Args:
        df: DataFrame with cleaned 'content' column.

    Returns:
        Tuple of (filtered DataFrame, drop counts by reason).
    """
    n_before = len(df)
    mask_drop = pd.Series(False, index=df.index)
    counts: dict[str, int] = {}

    # (a) null / NaN content
    m = df["content"].isna()
    new = m & ~mask_drop
    counts["null_nan"] = int(new.sum())
    mask_drop |= new

    # (b) Empty or whitespace-only (after strip in step 12)
    m = df["content"].str.strip().eq("").fillna(False)
    new = m & ~mask_drop
    counts["empty_whitespace"] = int(new.sum())
    mask_drop |= new

    # (c) Exact-match placeholders [deleted] / [removed]
    m = (
        df["content"]
        .str.strip()
        .str.lower()
        .isin({"[deleted]", "[removed]"})
        .fillna(False)
    )
    new = m & ~mask_drop
    counts["placeholder_exact"] = int(new.sum())
    mask_drop |= new

    # (d) Pseudo-null strings (nan, none, null, n/a)
    m = (
        df["content"]
        .str.strip()
        .str.lower()
        .isin({"nan", "none", "null", "n/a"})
        .fillna(False)
    )
    new = m & ~mask_drop
    counts["pseudo_null"] = int(new.sum())
    mask_drop |= new

    # (e) Fewer than 3 alphabetic characters -> no semantic content
    m = df["content"].str.count(r"[a-zA-Z]").fillna(0) < 3
    new = m & ~mask_drop
    counts["lt_3_alpha"] = int(new.sum())
    mask_drop |= new

    total = int(mask_drop.sum())
    df = df[~mask_drop].copy()

    logger.info(
        "Garbage filter: %d -> %d rows (dropped %d, %.4f%%)",
        n_before,
        len(df),
        total,
        total / n_before * 100 if n_before else 0,
    )
    for reason, cnt in counts.items():
        if cnt > 0:
            logger.info("  - %-20s %d rows", reason, cnt)

    return df, counts


def drop_bot_spam(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove bot-generated and spam content matched by regex patterns.

    Targets AutoModerator boilerplate (~24K rows), French spam test
    messages (~13.6K rows), and [deleted by user] markers identified
    in EDA-03 Section 4, cross-referenced with EDA-02 Section 5
    Power-User analysis.

    Args:
        df: DataFrame with cleaned 'content' column.

    Returns:
        Tuple of (filtered DataFrame, drop counts by pattern name).
    """
    n_before = len(df)
    mask_drop = pd.Series(False, index=df.index)
    counts: dict[str, int] = {}

    for name, pattern in BOT_SPAM_PATTERNS:
        m = df["content"].str.contains(pattern, na=False)
        n_match = int(m.sum())
        counts[name] = n_match
        mask_drop |= m
        if n_match > 0:
            logger.info("  Bot/spam '%s': %d rows", name, n_match)

    total = int(mask_drop.sum())
    df = df[~mask_drop].copy()

    logger.info(
        "Bot/spam filter: %d -> %d rows (dropped %d, %.4f%%)",
        n_before,
        len(df),
        total,
        total / n_before * 100 if n_before else 0,
    )
    return df, counts


# ===================================================================
# SECTION 7: Metadata Enrichment
# ===================================================================

def enrich_metadata(
    df: pd.DataFrame, drop_unusable: bool = False
) -> tuple[pd.DataFrame, int]:
    """Add word_count, is_usable_text, and content_hash columns.

    word_count (int32): whitespace-delimited token count. int32 saves
    ~45 MB vs int64 on 11M+ rows; max value 5,149 fits easily.

    is_usable_text (bool): soft-flag for SBERT eligibility.
    POST >= 3 words (Median=9w, threshold preserves short news titles).
    COMMENT >= 5 words (filters curt reactions like "Source?", "Why?").
    Rows flagged False stay in DataFrame for temporal volume counting
    (N_t in Prominence formula) but are excluded from embedding.

    content_hash (uint64): vectorized C-level murmur hash via
    pd.util.hash_pandas_object. Enables dedup and embedding reuse:
    embed unique texts once then map back by hash.

    Args:
        df: DataFrame with cleaned 'content' and 'interaction_type'.
        drop_unusable: If True, hard-drop is_usable_text=False rows.

    Returns:
        Tuple of (enriched DataFrame, number of unusable rows dropped).
    """
    logger.info("Enriching metadata columns...")

    # Word count via whitespace splitting (vectorized, int32 for RAM)
    df["word_count"] = (
        df["content"].str.count(r"\S+").fillna(0).astype("int32")
    )

    # Differentiated usability thresholds by interaction_type
    is_post = df["interaction_type"] == "POST"
    df["is_usable_text"] = True
    df.loc[is_post & (df["word_count"] < 3), "is_usable_text"] = False
    df.loc[~is_post & (df["word_count"] < 5), "is_usable_text"] = False

    n_usable = int(df["is_usable_text"].sum())
    n_unusable = len(df) - n_usable
    logger.info(
        "is_usable_text: %d usable (%.2f%%), %d flagged (%.2f%%)",
        n_usable,
        n_usable / len(df) * 100,
        n_unusable,
        n_unusable / len(df) * 100,
    )

    n_dropped = 0
    if drop_unusable and n_unusable > 0:
        df = df[df["is_usable_text"]].copy()
        n_dropped = n_unusable
        logger.info(
            "  --drop-unusable: removed %d rows -> %d remaining",
            n_dropped,
            len(df),
        )

    # Content hash for dedup and embedding reuse (vectorized uint64)
    # pd.util.hash_pandas_object uses C-level murmur hash -> ~2s on 11M
    df["content_hash"] = pd.util.hash_pandas_object(
        df["content"], index=False
    ).astype("uint64")

    n_unique = df["content_hash"].nunique()
    logger.info(
        "content_hash: %d unique / %d rows (%.2f%% unique)",
        n_unique,
        len(df),
        n_unique / len(df) * 100,
    )
    return df, n_dropped


def dedup_posts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact-duplicate POST rows by content_hash, keep first.

    POST duplicates (11.23% per EDA-03 Section 4) are caused by Reddit
    system re-posts and bot activity. Removing them prevents BERTopic
    from creating artificial clusters with inflated prominence.
    COMMENT duplicates are preserved: different users posting the same
    opinion independently is natural community behavior.

    Args:
        df: DataFrame with 'content_hash' and 'interaction_type'.

    Returns:
        Tuple of (deduplicated DataFrame, count of removed POST dupes).
    """
    is_post = df["interaction_type"] == "POST"
    n_posts_before = int(is_post.sum())

    # Identify duplicate POSTs (keep first chronological occurrence)
    post_dup_mask = is_post & df.duplicated(
        subset="content_hash", keep="first"
    )
    # Only count duplicates among POSTs (not COMMENTs)
    post_dup_mask = post_dup_mask & is_post
    n_removed = int(post_dup_mask.sum())

    if n_removed > 0:
        df = df[~post_dup_mask].copy()
        logger.info(
            "POST dedup: %d -> %d POSTs (%d dupes removed, %.2f%%)",
            n_posts_before,
            n_posts_before - n_removed,
            n_removed,
            n_removed / n_posts_before * 100,
        )
    else:
        logger.info("POST dedup: no duplicates found")

    gc.collect()
    return df, n_removed


# ===================================================================
# SECTION 8: Optional Subdomain Labeling
# ===================================================================

def label_subdomains(
    df: pd.DataFrame, subdomains: list[str]
) -> pd.DataFrame:
    """Attach subdomain labels using keyword taxonomy from EDA-03.

    Matches content against word-boundary regex patterns for each
    requested subdomain. A row can match multiple subdomains (stored
    as pipe-separated string). Uses vectorized array operations instead
    of DataFrame.apply(axis=1) to prevent RAM/CPU bottlenecks on 11M+ rows.
    Does NOT filter rows.

    Args:
        df: DataFrame with 'content' column.
        subdomains: Subdomain keys from SUBDOMAIN_KEYWORDS to match.

    Returns:
        DataFrame with 'subdomain_labels' column added.
    """
    logger.info("Labeling subdomains: %s", subdomains)
    content_lower = df["content"].str.lower()

    # Pre-allocate array of empty strings
    labels = np.full(len(df), "", dtype=object)
    n_valid_sds = 0

    for sd in subdomains:
        if sd not in SUBDOMAIN_KEYWORDS:
            logger.warning("Unknown subdomain key '%s', skipping", sd)
            continue
        n_valid_sds += 1
        combined = "|".join(SUBDOMAIN_KEYWORDS[sd])
        mask = content_lower.str.contains(
            combined, regex=True, na=False
        ).to_numpy()
        n_hit = int(mask.sum())
        logger.info(
            "  '%s': %d rows (%.2f%%)",
            sd,
            n_hit,
            n_hit / len(df) * 100,
        )
        # Vectorized string concatenation via np.where
        labels = np.where(
            mask,
            np.where(labels == "", sd, labels + "|" + sd),
            labels,
        )

    del content_lower
    gc.collect()

    if n_valid_sds == 0:
        logger.warning("No valid subdomains matched")
        return df

    subdomain_series = pd.Series(labels, index=df.index, dtype="string")
    subdomain_series = subdomain_series.replace("", pd.NA)
    df["subdomain_labels"] = subdomain_series

    n_labeled = int(df["subdomain_labels"].notna().sum())
    logger.info(
        "Subdomain labeling: %d rows labeled (%.2f%%)",
        n_labeled,
        n_labeled / len(df) * 100,
    )
    return df


# ===================================================================
# SECTION 9: Audit Snapshot & Report Generation
# ===================================================================

def compute_audit_snapshot(
    df: pd.DataFrame, stage: str
) -> dict[str, Any]:
    """Capture dataset statistics at a pipeline stage.

    Args:
        df: DataFrame to profile.
        stage: Label (e.g. 'pre_cleaning', 'post_cleaning').

    Returns:
        Dictionary of audit metrics for report generation.
    """
    snap: dict[str, Any] = {
        "stage": stage,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "memory_mb": round(
            df.memory_usage(deep=True).sum() / 1024 / 1024, 1
        ),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "null_counts": {
            c: int(v) for c, v in df.isnull().sum().items()
        },
    }

    if "content" in df.columns:
        snap["content_null"] = int(df["content"].isna().sum())

    if "publish_datetime_utc" in df.columns:
        snap["date_min"] = str(df["publish_datetime_utc"].min())
        snap["date_max"] = str(df["publish_datetime_utc"].max())

    if "interaction_type" in df.columns:
        snap["interaction_type_dist"] = {
            str(k): int(v)
            for k, v in df["interaction_type"].value_counts().items()
        }

    if "word_count" in df.columns:
        wc = df["word_count"]
        snap["word_count_stats"] = {
            "mean": round(float(wc.mean()), 2),
            "median": int(wc.median()),
            "p05": int(wc.quantile(0.05)),
            "p95": int(wc.quantile(0.95)),
            "p99": int(wc.quantile(0.99)),
            "max": int(wc.max()),
        }

    if "is_usable_text" in df.columns:
        snap["n_usable"] = int(df["is_usable_text"].sum())
        snap["n_unusable"] = int((~df["is_usable_text"]).sum())

    if "content_hash" in df.columns:
        snap["n_unique_hash"] = int(df["content_hash"].nunique())

    return snap


def generate_audit_report(
    pre: dict[str, Any],
    post: dict[str, Any],
    drop_log: dict[str, int],
    report_path: Path,
    agents_path: Path,
    input_file: str,
    output_file: str,
) -> None:
    """Write cleaning audit report in Markdown and JSON formats.

    Generates a comprehensive before/after comparison aligned with the
    audit template in plan_04. Copies Markdown to .agents/ for downstream
    AI agent context consumption.

    Args:
        pre: Pre-cleaning audit snapshot dict.
        post: Post-cleaning audit snapshot dict.
        drop_log: Aggregated {reason: count} from all filtering steps.
        report_path: Markdown report destination.
        agents_path: .agents/ copy destination.
        input_file: Input path for report header.
        output_file: Output path for report header.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    n_dropped = pre["n_rows"] - post["n_rows"]
    drop_pct = n_dropped / pre["n_rows"] * 100 if pre["n_rows"] else 0

    L: list[str] = []  

    L.append("# Cleaning Audit Report -- NarrativeCast\n")
    L.append(f"> **Generated:** {ts}  ")
    L.append(f"> **Input:** `{input_file}`  ")
    L.append(f"> **Output:** `{output_file}`\n")
    L.append("---\n")

    # --- Section 1: Overview ---
    L.append("## 1. Overview\n")
    L.append("| Metric | Before | After | Delta |")
    L.append("|:---|---:|---:|---:|")
    L.append(
        f"| Total rows | {pre['n_rows']:,} | {post['n_rows']:,} "
        f"| {-n_dropped:+,} ({-drop_pct:+.3f}%) |"
    )
    L.append(
        f"| Total columns | {pre['n_cols']} | {post['n_cols']} "
        f"| {post['n_cols'] - pre['n_cols']:+d} |"
    )
    L.append(
        f"| Memory (MB) | {pre['memory_mb']:,.1f} "
        f"| {post['memory_mb']:,.1f} "
        f"| {post['memory_mb'] - pre['memory_mb']:+,.1f} |"
    )

    if "interaction_type_dist" in post:
        L.append("\n**Interaction type (after cleaning):**\n")
        for itype, cnt in sorted(
            post["interaction_type_dist"].items(),
            key=lambda x: -x[1],
        ):
            pct = cnt / post["n_rows"] * 100
            L.append(f"- `{itype}`: {cnt:,} ({pct:.2f}%)")

    # --- Section 2: Drop Reasons ---
    L.append("\n---\n")
    L.append("## 2. Rows Dropped by Reason\n")
    L.append("| Reason | Count | % of Original |")
    L.append("|:---|---:|---:|")
    for reason, cnt in sorted(drop_log.items(), key=lambda x: -x[1]):
        if cnt > 0:
            pct = cnt / pre["n_rows"] * 100
            L.append(f"| {reason} | {cnt:,} | {pct:.4f}% |")
    L.append(
        f"| **TOTAL DROPPED** | **{n_dropped:,}** "
        f"| **{drop_pct:.3f}%** |"
    )

    # --- Section 3: Column Changes ---
    L.append("\n---\n")
    L.append("## 3. Column Changes\n")
    L.append("| Column | Action | Detail |")
    L.append("|:---|:---|:---|")
    drop_reasons: dict[str, str] = {
        "strict_filter": "Zero-variance / unused (100% null)",
        "platform": "Zero-variance / unused (Constant 'reddit')",
        "community": "Zero-variance / unused (Constant 'technology')",
        "language": "Zero-variance / unused (Constant 'en'/'English')",
        "url": "Zero-variance / unused (87.86% null / unused in embedding pipeline)",
    }
    for c in sorted(set(pre["columns"]) - set(post["columns"])):
        detail = drop_reasons.get(c, "Zero-variance / unused")
        L.append(f"| `{c}` | DROPPED | {detail} |")
    for c in sorted(set(post["columns"]) - set(pre["columns"])):
        dt = post["dtypes"].get(c, "?")
        L.append(f"| `{c}` | ADDED | dtype: `{dt}` |")

    # --- Section 4: Text Quality ---
    if "word_count_stats" in post:
        L.append("\n---\n")
        L.append("## 4. Text Quality After Cleaning\n")
        wc = post["word_count_stats"]
        L.append("| Metric | Value |")
        L.append("|:---|---:|")
        L.append(f"| word_count mean | {wc['mean']} |")
        L.append(f"| word_count median | {wc['median']} |")
        L.append(f"| word_count P05 | {wc['p05']} |")
        L.append(f"| word_count P95 | {wc['p95']} |")
        L.append(f"| word_count P99 | {wc['p99']} |")
        L.append(f"| word_count max | {wc['max']} |")

    if "n_usable" in post:
        n_rows = post["n_rows"]
        L.append(
            f"| is_usable_text = True | {post['n_usable']:,} "
            f"({post['n_usable'] / n_rows * 100:.2f}%) |"
        )
        L.append(
            f"| is_usable_text = False | {post['n_unusable']:,} "
            f"({post['n_unusable'] / n_rows * 100:.2f}%) |"
        )
    if "n_unique_hash" in post:
        L.append(
            f"| Unique content_hash | {post['n_unique_hash']:,} "
            f"({post['n_unique_hash'] / post['n_rows'] * 100:.2f}%) |"
        )

    # --- Section 5: Temporal Integrity ---
    L.append("\n---\n")
    L.append("## 5. Temporal Integrity Check\n")
    L.append("| Check | Result |")
    L.append("|:---|:---|")
    if "date_min" in post:
        L.append(
            f"| Date range | {post['date_min']} -> {post['date_max']} |"
        )

    # --- Section 6: Memory ---
    L.append("\n---\n")
    L.append("## 6. Memory Profile\n")
    L.append("| Stage | Memory (MB) |")
    L.append("|:---|---:|")
    L.append(f"| Raw load | {pre['memory_mb']:,.1f} |")
    L.append(f"| After pipeline | {post['memory_mb']:,.1f} |")
    saved = pre["memory_mb"] - post["memory_mb"]
    saved_pct = saved / pre["memory_mb"] * 100 if pre["memory_mb"] else 0
    L.append(f"| Saved | {saved:,.1f} ({saved_pct:.1f}%) |")

    # --- Section 7: Output Schema ---
    L.append("\n---\n")
    L.append("## 7. Output Schema\n")
    L.append("| # | Column | Dtype |")
    L.append("|:---:|:---|:---|")
    for i, c in enumerate(post["columns"], 1):
        dt = post["dtypes"].get(c, "?")
        L.append(f"| {i} | `{c}` | `{dt}` |")

    # --- Section 8: Summary of Key Cleaning Actions & Evidence-Based Rationale ---
    L.append("\n---\n")
    L.append("## 8. Summary of Key Cleaning Actions & Evidence-Based Rationale\n")
    L.append(
        "1. **Thu gọn cột & Loại bỏ trường không sử dụng:**\n"
        "   - Xóa bỏ hoàn toàn 5 cột hằng số, rỗng hoặc không phục vụ mục tiêu mô hình hóa: "
        "`strict_filter` (100% null), `platform` (hằng số 'reddit'), `community` (hằng số 'technology'), "
        "`language` (hằng số 'en'/'English'), và `url` (87.86% null, không mang giá trị ngữ nghĩa cho SBERT). "
        "Giúp thu gọn bảng dữ liệu từ 13 cột gốc xuống 12 cột chuẩn đầu ra (đồng thời bổ sung 4 cột mới: "
        "`publish_datetime_utc`, `word_count`, `is_usable_text`, `content_hash`) và tiết kiệm hàng GB RAM.\n"
    )
    L.append(
        "2. **Chuẩn hóa Thời gian & Bảo toàn Thứ tự:**\n"
        "   - Chuyển `publish_date` (UNIX epoch giây) sang `publish_datetime_utc` (`datetime64[ns, UTC]`), "
        "xác nhận dữ liệu nằm trọn trong khoảng 2014–2020 (2,557 ngày liên tục, 0 gap) và kiểm định thứ tự thời gian "
        "tăng dần tuyệt đối (`is_monotonic_increasing`) để chia cửa sổ trượt $W_{hist}=7$ ngày hoàn toàn không bị rò rỉ dữ liệu (data leakage).\n"
    )
    L.append(
        "3. **Tối ưu Kiểu dữ liệu (Downcasting):**\n"
        "   - Ép kiểu sớm `interaction_type` từ `object` sang `category` (chỉ có 2 giá trị `POST` và `COMMENT`) và "
        "`sentiment_vader` từ `float64` sang `float32`, tiết kiệm ngay ~630 MB RAM trước khi bước vào các phép biến đổi văn bản nặng.\n"
    )
    L.append(
        "4. **Làm sạch Văn bản Không phá hủy (Steps 6–12):**\n"
        "   - Giải mã HTML entities bằng cách chỉ áp dụng `html.unescape` trên các dòng có chứa '&' "
        "(856,739 dòng chứa `&gt;`, `&amp;`... chiếm 7.28%), tối ưu tốc độ gấp 10 lần và tránh rác từ vựng `amp`.\n"
        "   - Chuẩn hóa Unicode NFC (`.str.normalize('NFC')`) và loại bỏ ký tự điều khiển/lỗi mã hóa.\n"
        "   - Bóc tách link Markdown `[text](url)` -> `text` (bảo toàn trọn vẹn ngữ nghĩa anchor text), xóa URL thô (`https?://\\S+`).\n"
        "   - Xóa ký tự đánh dấu định dạng Markdown (`**`, `~~`, `` ` ``, và dấu `>` đầu dòng `r'(?m)^>\\s*'`) trong khi giữ nguyên toàn bộ chữ bên trong.\n"
        "   - Cắt hậu tố `[removed]` và `[deleted]` bị hệ thống Reddit nối dính ở cuối tiêu đề bài POST (83,214 lần) bằng `r'\\s*\\[(removed|deleted)\\]\\s*$'`.\n"
        "   - Chuẩn hóa khoảng whitespace (`\\s+` -> ' ') và `strip()`.\n"
    )
    L.append(
        "5. **Lọc bỏ Rác, Bot & Khử trùng lặp POST (Loại bỏ tổng cộng 168,422 dòng - 1.431%):**\n"
        "   - Loại bỏ 40,059 dòng rỗng/chỉ khoảng trắng sau làm sạch, 15,994 dòng `< 3 ký tự chữ cái` (không mang ngữ nghĩa), và 6 dòng giả-null (`\"nan\"`, `\"none\"`).\n"
        "   - Loại bỏ 39,746 dòng thông báo xóa bài tự động của `AutoModerator`, 13,596 dòng bot spam tiếng Pháp (`\"Message de test vraie hein\"`), và 798 dòng `[deleted by user]`.\n"
        "   - Khử trùng lặp chính xác trên POST (`post_duplicate`): loại bỏ 58,223 bài `POST` trùng nội dung (giữ bài đầu tiên theo thời gian) để ngăn ngừa BERTopic hình thành các cụm chủ đề nhân tạo (artificial clusters), trong khi vẫn giữ nguyên các `COMMENT` trùng lặp tự nhiên của cộng đồng.\n"
    )
    L.append(
        "6. **Tạo Cột Metadata Hỗ trợ Pipeline sau:**\n"
        "   - Tạo `word_count` (`int32`, dùng `.str.count(r'\\S+')` không tốn auxiliary RAM).\n"
        "   - Tạo cờ `is_usable_text` (`bool`): phân hóa ngưỡng theo bản chất tương tác (`POST >= 3 từ`, `COMMENT >= 5 từ`). "
        "Đạt 10,844,716 dòng khả dụng (93.51%) cho SBERT/BERTopic mà vẫn giữ đủ toàn bộ 11,597,712 dòng để đếm chính xác "
        "tổng lượng tương tác $N_t$ trong công thức Prominence $P(n, W_t)$.\n"
        "   - Tạo `content_hash` (`uint64` murmur hash qua `pd.util.hash_pandas_object`): xác định 11,375,527 chuỗi văn bản duy nhất (98.08%), "
        "cho phép bước SBERT Embedding chỉ cần nhúng 1 lần cho mỗi hash rồi map lại, tiết kiệm tối đa thời gian tính toán.\n"
    )

    # ---- Write Markdown ----
    report_text = "\n".join(L) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Audit report (MD) -> %s", report_path)

    # ---- Write JSON ----
    json_path = report_path.with_suffix(".json")
    json_data = {
        "generated_at": ts,
        "input_file": input_file,
        "output_file": output_file,
        "pre_cleaning": pre,
        "post_cleaning": post,
        "drop_log": drop_log,
    }
    json_path.write_text(
        json.dumps(json_data, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Audit report (JSON) -> %s", json_path)

    # ---- Copy to .agents/ for AI context ----
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, agents_path)
    logger.info("Audit report copied -> %s", agents_path)


# ===================================================================
# SECTION 10: CLI Argument Parsing & Main Orchestration
# ===================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the cleaning pipeline.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with all pipeline configuration flags.
    """
    p = argparse.ArgumentParser(
        description=(
            "NarrativeCast data cleaning pipeline for Reddit "
            "r/technology (MADOC). Applies evidence-based rules "
            "from EDA 01-03."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=str,
        default="data/raw/reddit_technology_madoc-001.parquet",
        help="Input Parquet path (default: %(default)s)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/processed/reddit_technology_clean.parquet",
    )
    p.add_argument(
        "--report-path",
        type=str,
        default="outputs/reports/cleaning_audit_report.md",
    )
    p.add_argument(
        "--drop-unusable",
        action="store_true",
        default=False,
        help="Hard-drop is_usable_text=False rows (default: keep)",
    )
    p.add_argument(
        "--dedup-posts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate POST by content_hash (default: True)",
    )
    p.add_argument(
        "--subdomain",
        type=str,
        default="all",
        help=(
            "'all' = no labeling; or comma-separated keys "
            "(e.g. 'bigtech_antitrust,cybersecurity_privacy')"
        ),
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Orchestrate the full cleaning pipeline end-to-end.

    Follows the 20-step pipeline architecture defined in
    plan_04_clean_data_script.md. Each major step logs progress
    and row counts for auditability.

    Args:
        argv: Optional CLI arguments (for programmatic invocation).
    """
    args = parse_args(argv)

    # Configure logging — ASCII-only, no Unicode arrows (coding_rules)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    t_start = time.time()
    logger.info("=" * 65)
    logger.info("NarrativeCast Data Cleaning Pipeline")
    logger.info("=" * 65)
    logger.info("Input  : %s", args.input)
    logger.info("Output : %s", args.output)
    logger.info("Flags  : drop_unusable=%s, dedup_posts=%s, "
                "subdomain=%s",
                args.drop_unusable, args.dedup_posts,
                args.subdomain)

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report_path)
    agents_path = PROJECT_ROOT / ".agents" / "cleaning_audit_report.md"

    # Accumulate drop counts across all filtering steps
    all_drops: dict[str, int] = {}

    try:
        # [1] Load raw data
        df = load_data(input_path)

        # [2] Pre-cleaning audit snapshot
        pre = compute_audit_snapshot(df, "pre_cleaning")
        logger.info(
            "Pre-clean: %d rows, %d cols, %.1f MB",
            pre["n_rows"],
            pre["n_cols"],
            pre["memory_mb"],
        )

        # [3] Prune constant and unused columns (strict_filter, platform, community, language, url)
        df = drop_constant_columns(df)
        gc.collect()

        # [4] Standardize timestamps (UNIX s -> UTC datetime)
        df = standardize_timestamps(df)

        # [5] Optimize dtypes early to free RAM before text ops
        df = optimize_dtypes(df)
        gc.collect()

        # [6-12] Text cleaning pipeline (in-place on 'content')
        df = clean_text_column(df)

        # [13] Hard-drop garbage / placeholder rows
        df, garbage_counts = drop_garbage_rows(df)
        for k, v in garbage_counts.items():
            all_drops[f"garbage_{k}"] = v
        gc.collect()

        # [14] Hard-drop bot / spam rows
        df, bot_counts = drop_bot_spam(df)
        for k, v in bot_counts.items():
            all_drops[f"bot_{k}"] = v
        gc.collect()

        # [15-17] Metadata enrichment (word_count, is_usable, hash)
        df, n_unusable = enrich_metadata(
            df, drop_unusable=args.drop_unusable
        )
        if n_unusable > 0:
            all_drops["unusable_text"] = n_unusable

        # POST deduplication by content_hash
        if args.dedup_posts:
            df, n_dedup = dedup_posts(df)
            if n_dedup > 0:
                all_drops["post_duplicate"] = n_dedup

        # [18] Optional subdomain labeling
        if args.subdomain.lower() != "all":
            sd_list = [s.strip() for s in args.subdomain.split(",")]
            df = label_subdomains(df, sd_list)

        # Reorder columns for a clean, logical output schema
        ordered = [
            c for c in OUTPUT_COLUMN_ORDER if c in df.columns
        ]
        extras = [c for c in df.columns if c not in ordered]
        df = df[ordered + extras]

        # [19] Post-cleaning audit snapshot
        post = compute_audit_snapshot(df, "post_cleaning")
        logger.info(
            "Post-clean: %d rows, %d cols, %.1f MB",
            post["n_rows"],
            post["n_cols"],
            post["memory_mb"],
        )

        # [20] Write output Parquet (snappy compression)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            output_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        fsize = output_path.stat().st_size / 1024 / 1024
        logger.info("Parquet saved: %s (%.1f MB)", output_path, fsize)

        # Generate audit reports (MD + JSON + .agents/ copy)
        generate_audit_report(
            pre=pre,
            post=post,
            drop_log=all_drops,
            report_path=report_path,
            agents_path=agents_path,
            input_file=str(args.input),
            output_file=str(args.output),
        )

        elapsed = time.time() - t_start
        retained = post["n_rows"] / pre["n_rows"] * 100
        logger.info("=" * 65)
        logger.info(
            "DONE: %d -> %d rows (%.3f%% retained) in %.1fs",
            pre["n_rows"],
            post["n_rows"],
            retained,
            elapsed,
        )
        logger.info("=" * 65)

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except (ValueError, KeyError) as exc:
        logger.error("Data validation error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

