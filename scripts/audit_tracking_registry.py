"""Verification and audit script for the causal narrative tracking registry.

Validates 8 integrity criteria derived from configs/forecast.yaml,
data/interim/narrative_assignments.parquet, and docs/tracking_rules.md:
1. Calendar window coverage (362 active registry windows + 2 warmup weeks = 364 calendar
   weeks; 339 active non-empty + 2 warmup non-empty = 341; 23 empty weeks).
   Warmup weeks have no candidates and 0 ASSIGNED posts.
2. Total ASSIGNED post count match against narrative_assignments.parquet (6,411 / 7,331).
3. Mandatory non-null checks, NaN gating for share (NaN iff window_total_valid_posts == 0,
   i.e. 23 empty weeks), dispersion NaN iff current_window_posts == 0,
   similarity_to_previous_centroid NaN iff current_window_posts == 0, and metric bounds.
4. 1:1 row/order alignment of narrative_registry_centroids.npz (allow_pickle=False),
   finite values, and unit norms in [0.9999, 1.0001], plus source embeddings check.
5. Runtime verification of load_frozen_registry_at_cutoff() at Train and Validation cutoffs.
6. Unique (window_end, narrative_id) primary key, causal ordering
   (first_seen_utc <= last_active_utc < window_end), and silent-week consistency.
7. Stratified QA pairs sample CSV (>= 30 pairs, < 2019-01-01, all 3 strata, no NaN
   similarity, no duplicate (window_end_utc, narrative_id_1, narrative_id_2) across strata,
   including both ordered and unordered pair deduplication).
8. Qualitative audit printout of 10 consecutive-week linked cluster pairs with text excerpts.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.tracking.track_narratives import load_frozen_registry_at_cutoff  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
logger = logging.getLogger("NarrativeCast.AuditTrackingRegistry")

REGISTRY_PARQUET = Path("data/interim/narrative_registry.parquet")
SUMMARY_PARQUET = Path("data/interim/narrative_registry_summary.parquet")
CENTROIDS_NPZ = Path("data/interim/narrative_registry_centroids.npz")
ASSIGNMENTS_PARQUET = Path("data/interim/narrative_assignments.parquet")
SCOPE_CSV = Path("data/D4/net_neutrality_scope_expanded_v1.csv")
EMBEDDINGS_NPZ = Path("data/interim/scope_expanded_v1_embeddings.npz")
PAIRS_SAMPLE_CSV = Path("reports/results/tracking_pairs_sample.csv")

EXPECTED_TOTAL_CALENDAR_WINDOWS = 364
EXPECTED_WARMUP_WEEKS = 2
EXPECTED_ACTIVE_REGISTRY_WINDOWS = 362
EXPECTED_ASSIGNMENTS_NONEMPTY_WEEKS = 341
EXPECTED_ACTIVE_NONEMPTY_WEEKS = 339
EXPECTED_EMPTY_WEEKS = 23
EXPECTED_ASSIGNED_POSTS = 6411
EXPECTED_TOTAL_POSTS = 7331


def _status_label(passed: bool) -> str:
    """Format check status as PASS or FAIL."""
    return "PASS" if passed else "FAIL"


def _sanitize_ascii(text: str) -> str:
    """Normalize text excerpt to single-line ASCII for terminal output."""
    return (
        text.replace("\n", " ")
        .replace("\r", " ")
        .encode("ascii", errors="replace")
        .decode("ascii")
    )


def load_core_artifacts() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load registry, summary, assignments, and scope tables with error handling."""
    required_paths = [REGISTRY_PARQUET, SUMMARY_PARQUET, ASSIGNMENTS_PARQUET, SCOPE_CSV]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    try:
        registry_df = pd.read_parquet(REGISTRY_PARQUET)
        summary_df = pd.read_parquet(SUMMARY_PARQUET)
        assignments_df = pd.read_parquet(ASSIGNMENTS_PARQUET)
        scope_df = pd.read_csv(SCOPE_CSV)
    except Exception as exc:
        logger.error("Failed to read core tracking artifacts: %s", exc)
        raise

    for col in ("window_start", "window_end", "first_seen_utc", "last_active_utc"):
        if col in registry_df.columns:
            registry_df[col] = pd.to_datetime(registry_df[col], utc=True)

    for col in ("timestamp", "window_start", "window_end"):
        if col in assignments_df.columns:
            assignments_df[col] = pd.to_datetime(assignments_df[col], utc=True)

    return registry_df, summary_df, assignments_df, scope_df


def check_criterion_1_window_coverage(
    registry_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> bool:
    """Check (1): 362 active + 2 warmup = 364 calendar; 339 active non-empty + 2 warmup = 341; 23 empty.

    Warmup weeks are the two calendar weeks before the first registry window.
    They have posts in assignments but zero candidates (no ASSIGNED posts)
    because the model had insufficient data to fit.
    """
    unique_windows = int(registry_df["window_end"].nunique())
    win_valid = registry_df.groupby("window_end")["window_total_valid_posts"].first()
    reg_nonempty = int((win_valid > 0).sum())
    reg_empty = int((win_valid == 0).sum())
    reg_span_weeks = int((registry_df["window_end"].max() - registry_df["window_start"].min()).days // 7)

    first_reg_start = registry_df["window_start"].min()
    warmup_df = assignments_df.loc[assignments_df["window_start"] < first_reg_start]
    warmup_windows = int(
        warmup_df[["window_start", "window_end"]].drop_duplicates().shape[0]
    )
    warmup_nonempty = int(
        warmup_df.groupby(["window_start", "window_end"]).size().gt(0).sum()
    )
    warmup_assigned = int((warmup_df["assignment_status"] == "ASSIGNED").sum())

    # Warmup weeks must have zero ASSIGNED posts (no candidate narratives)
    warmup_has_no_candidates = warmup_assigned == 0

    asg_nonempty = int(
        assignments_df[["window_start", "window_end"]].drop_duplicates().shape[0]
    )

    passed = (
        unique_windows == EXPECTED_ACTIVE_REGISTRY_WINDOWS
        and reg_span_weeks == EXPECTED_ACTIVE_REGISTRY_WINDOWS
        and warmup_windows == EXPECTED_WARMUP_WEEKS
        and warmup_nonempty == EXPECTED_WARMUP_WEEKS
        and warmup_has_no_candidates
        and unique_windows + warmup_windows == EXPECTED_TOTAL_CALENDAR_WINDOWS
        and reg_nonempty == EXPECTED_ACTIVE_NONEMPTY_WEEKS
        and reg_nonempty + warmup_nonempty == EXPECTED_ASSIGNMENTS_NONEMPTY_WEEKS
        and asg_nonempty == EXPECTED_ASSIGNMENTS_NONEMPTY_WEEKS
        and reg_empty == EXPECTED_EMPTY_WEEKS
    )

    print(f"[Criterion 1 - {_status_label(passed)}] Calendar Window Coverage")
    print(
        f"  - Active registry windows      : {unique_windows} "
        f"(expected {EXPECTED_ACTIVE_REGISTRY_WINDOWS}; span={reg_span_weeks}w)"
    )
    print(
        f"  - Warmup weeks (< {first_reg_start.date()}) : {warmup_windows} "
        f"(expected {EXPECTED_WARMUP_WEEKS}; non-empty={warmup_nonempty}, "
        f"no_candidates={warmup_has_no_candidates})"
    )
    print(
        f"  - Combined calendar weeks      : {unique_windows + warmup_windows} "
        f"({unique_windows} active + {warmup_windows} warmup = {EXPECTED_TOTAL_CALENDAR_WINDOWS})"
    )
    print(
        f"  - Active non-empty / empty     : {reg_nonempty} / {reg_empty} "
        f"(expected {EXPECTED_ACTIVE_NONEMPTY_WEEKS} / {EXPECTED_EMPTY_WEEKS})"
    )
    print(
        f"  - Combined non-empty weeks     : {reg_nonempty + warmup_nonempty} vs assignments={asg_nonempty} "
        f"({reg_nonempty} active + {warmup_nonempty} warmup = {EXPECTED_ASSIGNMENTS_NONEMPTY_WEEKS})"
    )
    return passed


def check_criterion_2_assigned_totals(
    registry_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> bool:
    """Check (2): ASSIGNED post totals match 100% with assignments (6,411 / 7,331)."""
    total_posts = int(len(assignments_df))
    asg_assigned = int((assignments_df["assignment_status"] == "ASSIGNED").sum())
    reg_assigned = int(registry_df["current_window_posts"].sum())
    sum_assigned = int(summary_df["total_posts"].sum())

    asg_by_nid = (
        assignments_df.loc[assignments_df["assignment_status"] == "ASSIGNED"]
        .assign(narrative_id=lambda d: d["narrative_id"].astype(str))
        .groupby("narrative_id")
        .size()
        .to_dict()
    )
    reg_by_nid = (
        registry_df.assign(narrative_id=lambda d: d["narrative_id"].astype(str))
        .groupby("narrative_id")["current_window_posts"]
        .sum()
        .astype(int)
        .to_dict()
    )
    per_id_match = asg_by_nid == reg_by_nid

    passed = (
        total_posts == EXPECTED_TOTAL_POSTS
        and asg_assigned == EXPECTED_ASSIGNED_POSTS
        and reg_assigned == asg_assigned
        and sum_assigned == asg_assigned
        and per_id_match
    )

    print(f"[Criterion 2 - {_status_label(passed)}] ASSIGNED Post Count Reconciliation")
    print(
        f"  - Assignments total / ASSIGNED : {total_posts:,} / {asg_assigned:,} "
        f"(expected {EXPECTED_TOTAL_POSTS:,} / {EXPECTED_ASSIGNED_POSTS:,})"
    )
    print(f"  - Registry timeline sum        : {reg_assigned:,}")
    print(f"  - Registry summary sum         : {sum_assigned:,}")
    print(f"  - Per-narrative exact match    : {per_id_match}")
    return passed


def check_criterion_3_null_and_nan_invariants(registry_df: pd.DataFrame) -> bool:
    """Check (3): 0 nulls in mandatory cols; share NaN iff empty week; disp & sim NaN iff 0 posts.

    Specifically verifies:
    - current_window_share is NaN when and only when window_total_valid_posts == 0
    - intra_window_dispersion is NaN when and only when current_window_posts == 0
    - similarity_to_previous_centroid is NaN when and only when current_window_posts == 0
    """
    nullable_cols = {
        "current_window_share",
        "intra_window_dispersion",
        "similarity_to_previous_centroid",
    }
    mandatory_cols = [c for c in registry_df.columns if c not in nullable_cols]
    mandatory_nulls = int(registry_df[mandatory_cols].isna().sum().sum())

    is_empty_week = registry_df["window_total_valid_posts"] == 0
    share_col = registry_df["current_window_share"]
    share_nan_exact = bool((share_col.isna() == is_empty_week).all())

    win_valid = registry_df.groupby("window_end")["window_total_valid_posts"].first()
    empty_weeks_count = int((win_valid == 0).sum())
    nonempty_weeks_count = int((win_valid > 0).sum())

    nonempty_shares = share_col[~is_empty_week]
    share_bounds_ok = bool(
        np.all(np.isfinite(nonempty_shares.to_numpy()))
        and (nonempty_shares >= 0.0).all()
        and (nonempty_shares <= 1.0).all()
    )

    is_zero_posts = registry_df["current_window_posts"] == 0
    disp_col = registry_df["intra_window_dispersion"]
    disp_nan_exact = bool((disp_col.isna() == is_zero_posts).all())
    disp_finite_ok = bool(np.all(np.isfinite(disp_col[~is_zero_posts].to_numpy())))

    sim_col = registry_df["similarity_to_previous_centroid"]
    sim_nan_exact = bool((sim_col.isna() == is_zero_posts).all())

    # Detailed similarity NaN audit
    sim_nan_when_zero = bool(sim_col[is_zero_posts].isna().all())
    sim_finite_when_nonzero = bool(sim_col[~is_zero_posts].notna().all())

    nonempty_sims = sim_col[~is_zero_posts]
    sim_bounds_ok = bool(
        np.all(np.isfinite(nonempty_sims.to_numpy()))
        and (nonempty_sims >= -1.0001).all()
        and (nonempty_sims <= 1.0001).all()
    )

    passed = (
        mandatory_nulls == 0
        and empty_weeks_count == EXPECTED_EMPTY_WEEKS
        and nonempty_weeks_count == EXPECTED_ACTIVE_NONEMPTY_WEEKS
        and share_nan_exact
        and share_bounds_ok
        and disp_nan_exact
        and disp_finite_ok
        and sim_nan_exact
        and sim_nan_when_zero
        and sim_finite_when_nonzero
        and sim_bounds_ok
    )

    print(f"[Criterion 3 - {_status_label(passed)}] Null / NaN & Metric Bounds")
    print(f"  - Mandatory columns null count : {mandatory_nulls}")
    print(
        f"  - Empty weeks count            : {empty_weeks_count} "
        f"(expected {EXPECTED_EMPTY_WEEKS})"
    )
    print(
        f"  - Active non-empty weeks       : {nonempty_weeks_count} "
        f"(expected {EXPECTED_ACTIVE_NONEMPTY_WEEKS}; + {EXPECTED_WARMUP_WEEKS} warmup = {EXPECTED_ASSIGNMENTS_NONEMPTY_WEEKS})"
    )
    print(f"  - Share NaN iff empty week     : {share_nan_exact}")
    print(f"  - Non-empty share in [0, 1]    : {share_bounds_ok}")
    print(f"  - Dispersion NaN iff 0 posts   : {disp_nan_exact} (finite active={disp_finite_ok})")
    print(
        f"  - Similarity NaN iff 0 posts   : {sim_nan_exact} "
        f"(NaN@zero={sim_nan_when_zero}, finite@nonzero={sim_finite_when_nonzero}, "
        f"active in [-1, 1]={sim_bounds_ok})"
    )
    return passed


def check_criterion_4_centroids_and_embeddings(
    registry_df: pd.DataFrame,
    centroids_path: Path,
    embeddings_path: Path,
) -> bool:
    """Check (4): Centroids .npz 1:1 Parquet alignment, 0 NaN/Inf, norms in [0.9999, 1.0001]."""
    if not centroids_path.exists():
        raise FileNotFoundError(f"Centroids file not found: {centroids_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    try:
        with np.load(str(centroids_path), allow_pickle=False) as cent_npz:
            c_wends = cent_npz["window_end_utc"].astype(str)
            c_nids = cent_npz["narrative_id"].astype(str)
            c_vecs = cent_npz["centroids"].astype(np.float32)
        with np.load(str(embeddings_path), allow_pickle=False) as emb_npz:
            e_vecs = emb_npz["vectors"].astype(np.float32)
    except Exception as exc:
        logger.error("Failed to load npz archive with allow_pickle=False: %s", exc)
        raise

    reg_wends = np.array([ts.isoformat() for ts in registry_df["window_end"]], dtype=str)
    reg_nids = registry_df["narrative_id"].astype(str).to_numpy()

    row_count_match = len(c_vecs) == len(registry_df)
    order_match = bool(np.array_equal(c_wends, reg_wends) and np.array_equal(c_nids, reg_nids))
    c_nan_inf = int((~np.isfinite(c_vecs)).sum())
    c_norms = np.linalg.norm(c_vecs, axis=1)
    norms_ok = bool(np.all((c_norms >= 0.9999) & (c_norms <= 1.0001)))

    # Also verify source embeddings archive (allow_pickle=False)
    e_nan_inf = int((~np.isfinite(e_vecs)).sum())
    e_norms = np.linalg.norm(e_vecs, axis=1)
    e_zero_norms = int((e_norms <= 1e-12).sum())

    passed = (
        row_count_match
        and order_match
        and c_nan_inf == 0
        and norms_ok
        and e_nan_inf == 0
        and e_zero_norms == 0
    )

    print(f"[Criterion 4 - {_status_label(passed)}] Centroids & Embeddings Integrity (allow_pickle=False)")
    print(f"  - Centroids shape              : {c_vecs.shape} vs registry rows {len(registry_df)}")
    print(f"  - 1:1 order match with Parquet : {order_match}")
    print(f"  - Centroids NaN/Inf count      : {c_nan_inf}")
    print(
        f"  - Centroid norm range          : [{float(c_norms.min()):.6f}, {float(c_norms.max()):.6f}]"
    )
    print(
        f"  - Source embeddings ({embeddings_path.name}): shape={e_vecs.shape}, "
        f"NaN/Inf={e_nan_inf}, zero_norm={e_zero_norms}"
    )
    return passed


def check_criterion_5_cutoff_loader(
    registry_df: pd.DataFrame,
    registry_path: Path,
    centroids_path: Path,
) -> bool:
    """Check (5): Invoke load_frozen_registry_at_cutoff() at Train and Validation cutoffs."""
    train_windows = registry_df.loc[
        registry_df["window_end"] < pd.Timestamp("2018-01-01T00:00:00Z"),
        "window_end",
    ]
    val_windows = registry_df.loc[
        (registry_df["window_end"] >= pd.Timestamp("2018-01-01T00:00:00Z"))
        & (registry_df["window_end"] < pd.Timestamp("2019-01-01T00:00:00Z")),
        "window_end",
    ]

    train_cutoff = train_windows.max()
    val_cutoff = val_windows.max()

    train_df, train_cents = load_frozen_registry_at_cutoff(
        registry_path, centroids_path, train_cutoff
    )
    val_df, val_cents = load_frozen_registry_at_cutoff(
        registry_path, centroids_path, val_cutoff
    )
    # Also test non-UTC timezone conversion (+07:00 -> UTC)
    val_df_p7, val_cents_p7 = load_frozen_registry_at_cutoff(
        registry_path, centroids_path, val_cutoff.tz_convert("Asia/Ho_Chi_Minh")
    )

    train_ok = (
        not train_df.empty
        and len(train_df) == len(train_cents)
        and bool((train_df["first_seen_utc"] < train_cutoff).all())
        and bool((train_df["last_active_utc"] < train_cutoff).all())
    )
    val_ok = (
        not val_df.empty
        and len(val_df) == len(val_cents)
        and bool((val_df["first_seen_utc"] < val_cutoff).all())
        and bool((val_df["last_active_utc"] < val_cutoff).all())
        and val_df.equals(val_df_p7)
        and np.allclose(val_cents, val_cents_p7, atol=1e-7)
    )
    passed = train_ok and val_ok

    print(f"[Criterion 5 - {_status_label(passed)}] Frozen Cutoff Loader Verification")
    print(
        f"  - Train cutoff ({train_cutoff.isoformat()}): "
        f"{len(train_df)} candidates, centroids={train_cents.shape}"
    )
    print(
        f"  - Val cutoff   ({val_cutoff.isoformat()}): "
        f"{len(val_df)} candidates, centroids={val_cents.shape}, +07:00_equiv={val_ok}"
    )
    return passed


def check_criterion_6_keys_causal_and_silent(
    registry_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> bool:
    """Check (6): Unique PK, causal bounds, and max_consecutive_silent_weeks consistency."""
    dup_count = int(registry_df.duplicated(subset=["window_end", "narrative_id"]).sum())
    causal_lower_bad = int((registry_df["first_seen_utc"] > registry_df["last_active_utc"]).sum())
    causal_upper_bad = int((registry_df["last_active_utc"] >= registry_df["window_end"]).sum())

    merged = registry_df[["narrative_id", "consecutive_silent_weeks"]].merge(
        summary_df[["narrative_id", "max_consecutive_silent_weeks"]],
        on="narrative_id",
        how="left",
    )
    silent_bad = int(
        (merged["max_consecutive_silent_weeks"] < merged["consecutive_silent_weeks"]).sum()
    )
    peak_by_nid = (
        registry_df.groupby("narrative_id")["consecutive_silent_weeks"]
        .max()
        .astype(int)
        .sort_index()
    )
    sum_peak = (
        summary_df.set_index("narrative_id")["max_consecutive_silent_weeks"]
        .astype(int)
        .sort_index()
    )
    peak_exact_match = peak_by_nid.equals(sum_peak)

    passed = (
        dup_count == 0
        and causal_lower_bad == 0
        and causal_upper_bad == 0
        and silent_bad == 0
        and peak_exact_match
    )

    print(f"[Criterion 6 - {_status_label(passed)}] Primary Key, Causal Bounds & Silence Invariants")
    print(f"  - Duplicate (window_end, narrative_id)   : {dup_count}")
    print(f"  - first_seen_utc > last_active_utc       : {causal_lower_bad}")
    print(f"  - last_active_utc >= window_end          : {causal_upper_bad}")
    print(f"  - max_silent < consecutive_silent        : {silent_bad}")
    print(f"  - Summary max silent exact match         : {peak_exact_match}")
    return passed


def check_criterion_7_pairs_sample(pairs_csv_path: Path) -> bool:
    """Check (7): QA pairs CSV has >= 30 pairs, < 2019-01-01, all 3 strata, no NaN, no dup pairs."""
    if not pairs_csv_path.exists():
        raise FileNotFoundError(f"QA pairs sample CSV not found: {pairs_csv_path}")

    try:
        pairs_df = pd.read_csv(pairs_csv_path)
    except Exception as exc:
        logger.error("Failed to read QA pairs sample CSV: %s", exc)
        raise

    row_count = int(len(pairs_df))
    w_ends = pd.to_datetime(pairs_df["window_end_utc"], utc=True)
    all_before_2019 = bool((w_ends < pd.Timestamp("2019-01-01T00:00:00Z")).all())

    expected_types = {"linked_active", "potential_merge_or_split", "unlinked_near_miss"}
    observed_types = set(pairs_df["event_type"].astype(str).unique())
    has_all_strata = expected_types.issubset(observed_types)
    exact_strata_only = observed_types == expected_types
    sim_nan_count = int(pairs_df["similarity"].isna().sum())
    id_nan_count = int(
        pairs_df["narrative_id_1"].isna().sum() + pairs_df["narrative_id_2"].isna().sum()
    )

    pair_cols = ["window_end_utc", "narrative_id_1", "narrative_id_2"]
    dup_pair_count = int(pairs_df.duplicated(subset=pair_cols).sum())
    cross_strata_dup_count = int(
        pairs_df.groupby(pair_cols)["event_type"].nunique().gt(1).sum()
    )
    canonical_pairs = pairs_df.assign(
        _nid_lo=pairs_df[["narrative_id_1", "narrative_id_2"]].astype(str).min(axis=1),
        _nid_hi=pairs_df[["narrative_id_1", "narrative_id_2"]].astype(str).max(axis=1),
    )
    unordered_cross_strata_dup_count = int(
        canonical_pairs.groupby(["window_end_utc", "_nid_lo", "_nid_hi"])["event_type"]
        .nunique()
        .gt(1)
        .sum()
    )

    passed = (
        row_count >= 30
        and all_before_2019
        and has_all_strata
        and exact_strata_only
        and sim_nan_count == 0
        and id_nan_count == 0
        and dup_pair_count == 0
        and cross_strata_dup_count == 0
        and unordered_cross_strata_dup_count == 0
    )

    print(f"[Criterion 7 - {_status_label(passed)}] QA Pairs Sample CSV Integrity")
    print(f"  - Sample row count             : {row_count} (>= 30 required)")
    print(f"  - Max window_end_utc           : {w_ends.max()} (< 2019-01-01: {all_before_2019})")
    print(f"  - Event type counts            : {pairs_df['event_type'].value_counts().to_dict()}")
    print(f"  - NaN similarity / IDs         : {sim_nan_count} / {id_nan_count}")
    print(
        f"  - Dup (window_end, id1, id2)   : {dup_pair_count} "
        f"(cross-strata={cross_strata_dup_count}, unordered={unordered_cross_strata_dup_count})"
    )
    return passed


def extract_consecutive_cluster_pairs(
    registry_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    scope_df: pd.DataFrame,
    num_samples: int = 10,
) -> List[Dict[str, Any]]:
    """Extract 10 consecutive-week linked cluster examples with post text excerpts."""
    content_map: Dict[str, str] = {}
    if "content" in scope_df.columns:
        pids = scope_df["post_id"].astype(str).str.strip()
        texts = scope_df["content"].fillna("").astype(str)
        content_map = dict(zip(pids, texts))

    asg = assignments_df.copy()
    asg["post_id"] = asg["post_id"].astype(str).str.strip()

    active_rows = registry_df[
        (registry_df["lifecycle_status"] == "active")
        & (registry_df["current_window_posts"] > 0)
    ].sort_values(
        by=["cumulative_posts", "current_window_posts", "window_end", "narrative_id"],
        ascending=[False, False, True, True],
    )

    sampled_examples: List[Dict[str, Any]] = []
    seen_narratives = set()

    for _, row in active_rows.iterrows():
        nid = str(row["narrative_id"])
        if nid in seen_narratives:
            continue
        w_curr_start = row["window_start"]
        w_curr_end = row["window_end"]
        w_prev_start = w_curr_start - pd.Timedelta(days=7)
        w_prev_end = w_curr_start

        prev_posts = asg[
            (asg["narrative_id"] == nid)
            & (asg["window_start"] == w_prev_start)
            & (asg["assignment_status"] == "ASSIGNED")
        ]
        curr_posts = asg[
            (asg["narrative_id"] == nid)
            & (asg["window_start"] == w_curr_start)
            & (asg["assignment_status"] == "ASSIGNED")
        ]

        if len(prev_posts) > 0 and len(curr_posts) > 0:
            prev_cid = int(prev_posts["local_cluster_id"].iloc[0])
            curr_cid = int(curr_posts["local_cluster_id"].iloc[0])
            prev_pid = str(prev_posts["post_id"].iloc[0])
            curr_pid = str(curr_posts["post_id"].iloc[0])

            prev_text = _sanitize_ascii(content_map.get(prev_pid, "[Text unavailable]")[:120])
            curr_text = _sanitize_ascii(content_map.get(curr_pid, "[Text unavailable]")[:120])

            sampled_examples.append({
                "narrative_id": nid,
                "prev_window": f"{w_prev_start.strftime('%Y-%m-%d')} to {w_prev_end.strftime('%Y-%m-%d')}",
                "curr_window": f"{w_curr_start.strftime('%Y-%m-%d')} to {w_curr_end.strftime('%Y-%m-%d')}",
                "prev_cluster_id": prev_cid,
                "curr_cluster_id": curr_cid,
                "prev_posts_count": len(prev_posts),
                "curr_posts_count": len(curr_posts),
                "similarity_to_prev": float(row["similarity_to_previous_centroid"]),
                "centroid_drift": float(row["centroid_drift"]),
                "prev_sample_text": prev_text,
                "curr_sample_text": curr_text,
            })
            seen_narratives.add(nid)

            if len(sampled_examples) >= num_samples:
                break

    return sampled_examples


def check_criterion_8_consecutive_examples(
    registry_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    scope_df: pd.DataFrame,
) -> bool:
    """Check (8): Extract and print 10 consecutive-week cluster pairs with text excerpts."""
    examples = extract_consecutive_cluster_pairs(
        registry_df=registry_df,
        assignments_df=assignments_df,
        scope_df=scope_df,
        num_samples=10,
    )
    passed = len(examples) == 10
    print(f"[Criterion 8 - {_status_label(passed)}] 10 Consecutive-Week Linked Cluster Examples")
    print()
    for i, ex in enumerate(examples, 1):
        print(f"Example {i:02d} | Narrative ID: {ex['narrative_id']}")
        print(
            f"  - Week t-1 [{ex['prev_window']}]: Local Cluster ID = {ex['prev_cluster_id']}, "
            f"Posts = {ex['prev_posts_count']}"
        )
        print(f'    Sample post: "{ex["prev_sample_text"]}..."')
        print(
            f"  - Week t   [{ex['curr_window']}]: Local Cluster ID = {ex['curr_cluster_id']}, "
            f"Posts = {ex['curr_posts_count']}"
        )
        print(f'    Sample post: "{ex["curr_sample_text"]}..."')
        print(
            f"  - Continuity Metric: Cosine Similarity = {ex['similarity_to_prev']:.4f} "
            f"| Centroid Drift = {ex['centroid_drift']:.4f}"
        )
        print()
    return passed


def main() -> None:
    """Run all 8 registry checks on generated tracking artifacts."""
    registry_df, summary_df, assignments_df, scope_df = load_core_artifacts()

    print("Narrative Registry Audit Report (8 Criteria)")
    print()

    results = [
        check_criterion_1_window_coverage(registry_df, assignments_df),
        check_criterion_2_assigned_totals(registry_df, summary_df, assignments_df),
        check_criterion_3_null_and_nan_invariants(registry_df),
        check_criterion_4_centroids_and_embeddings(
            registry_df, CENTROIDS_NPZ, EMBEDDINGS_NPZ
        ),
        check_criterion_5_cutoff_loader(registry_df, REGISTRY_PARQUET, CENTROIDS_NPZ),
        check_criterion_6_keys_causal_and_silent(registry_df, summary_df),
        check_criterion_7_pairs_sample(PAIRS_SAMPLE_CSV),
        check_criterion_8_consecutive_examples(registry_df, assignments_df, scope_df),
    ]

    passed_count = sum(1 for r in results if r)
    print()
    print(f"AUDIT SUMMARY: {passed_count}/{len(results)} CRITERIA PASSED")


if __name__ == "__main__":
    main()
