"""Build a normalized, leakage-free temporal forecasting dataset.

Produces candidate-level forecasting instances across weekly rolling origins:
- Partitions chronologically into train (2014-2017), validation (2018), and test (2019-2020).
- Enforces purged split boundaries:
    * label_end(train) < validation_start
    * label_end(validation) < test_start
- Guarantees complete observation and future horizon at the end of the period
  ("không có dòng thiếu tương lai cuối kỳ").
- Forbids fitting transformers, scalers, UMAP, or clustering across the full dataset
  or future periods (raises TemporalLeakageError).
- Computes post-based prominence targets and historical momentum features per
  `configs/splits.yaml` and `docs/forecast_protocol.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

logger = logging.getLogger("NarrativeCast.ForecastDataset")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    )

DEFAULT_SPLITS_CONFIG = Path("configs/splits.yaml")
DEFAULT_ASSIGNMENTS_PATH = Path("data/interim/narrative_assignments.parquet")
DEFAULT_OUTPUT_PARQUET = Path("data/features/forecast_dataset.parquet")
DEFAULT_OUTPUT_MANIFEST = Path("data/features/forecast_dataset_manifest.json")

PARQUET_METADATA_KEY = b"narrativecast_split_metadata"


class TemporalLeakageError(RuntimeError):
    """Raised when an operation violates temporal ordering or attempts full-dataset fitting."""


@dataclass(frozen=True)
class SplitsConfig:
    """Holds configuration for temporal splits and anti-leakage rules."""

    timezone: str = "UTC"
    observation_days: int = 7
    horizon_days: int = 7
    first_complete_week: pd.Timestamp = pd.Timestamp("2014-01-06T00:00:00Z")
    last_complete_week: pd.Timestamp = pd.Timestamp("2020-12-21T00:00:00Z")
    dataset_end_utc: pd.Timestamp = pd.Timestamp("2020-12-28T00:00:00Z")

    train_start_utc: pd.Timestamp = pd.Timestamp("2014-01-06T00:00:00Z")
    validation_start_utc: pd.Timestamp = pd.Timestamp("2018-01-01T00:00:00Z")
    test_start_utc: pd.Timestamp = pd.Timestamp("2019-01-01T00:00:00Z")
    min_historical_posts: int = 5

    @classmethod
    def from_yaml(cls, path: Path) -> SplitsConfig:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        time_cfg = raw.get("time", {})
        splits_cfg = raw.get("splits", {})
        eval_cfg = raw.get("evaluation", {})

        tz = raw.get("timezone", "UTC")
        return cls(
            timezone=tz,
            observation_days=int(time_cfg.get("observation_days", 7)),
            horizon_days=int(time_cfg.get("horizon_days", 7)),
            first_complete_week=pd.Timestamp(time_cfg.get("first_complete_week", "2014-01-06T00:00:00Z"), tz=tz),
            last_complete_week=pd.Timestamp(time_cfg.get("last_complete_week", "2020-12-21T00:00:00Z"), tz=tz),
            dataset_end_utc=pd.Timestamp(time_cfg.get("dataset_end_utc", "2020-12-28T00:00:00Z"), tz=tz),
            train_start_utc=pd.Timestamp(splits_cfg.get("train", {}).get("start_utc", "2014-01-06T00:00:00Z"), tz=tz),
            validation_start_utc=pd.Timestamp(splits_cfg.get("validation", {}).get("start_utc", "2018-01-01T00:00:00Z"), tz=tz),
            test_start_utc=pd.Timestamp(splits_cfg.get("test", {}).get("start_utc", "2019-01-01T00:00:00Z"), tz=tz),
            min_historical_posts=int(eval_cfg.get("min_historical_posts_for_eval", 5)),
        )

    def determine_split(
        self,
        origin_utc: pd.Timestamp,
        label_end_utc: pd.Timestamp,
    ) -> Optional[str]:
        """Determine partition with strict purged boundary checks.

        Returns 'train', 'validation', 'test', or None if purged due to boundary collision
        or missing future.
        """
        if origin_utc.tzinfo is None or label_end_utc.tzinfo is None:
            raise ValueError(f"Timestamps must be timezone-aware (expected {self.timezone}).")

        # Future extends beyond dataset boundary -> exclude (no missing future)
        if label_end_utc > self.dataset_end_utc:
            return None

        # Train: must originate before validation_start and label_end must be strictly < validation_start
        if origin_utc < self.validation_start_utc:
            if label_end_utc < self.validation_start_utc:
                return "train"
            # Label touches or extends into validation -> purge!
            return None

        # Validation: originates in validation period and label_end must be strictly < test_start
        if self.validation_start_utc <= origin_utc < self.test_start_utc:
            if label_end_utc < self.test_start_utc:
                return "validation"
            # Label touches or extends into test -> purge!
            return None

        # Test: originates in test period and label_end within dataset_end
        if self.test_start_utc <= origin_utc:
            if label_end_utc <= self.dataset_end_utc:
                return "test"
            return None

        return None


def validate_no_full_dataset_fit(
    data: pd.DataFrame,
    allowed_split: Optional[str] = "train",
    caller_name: str = "fit_transformer",
) -> None:
    """Guard against fitting scalers, encoders, or models on the entire dataset.

    Raises `TemporalLeakageError` if the input data spans across test/val without
    explicitly restricting to train or past data.
    """
    if "split" not in data.columns:
        raise TemporalLeakageError(
            f"[{caller_name}] Cannot fit on unpartitioned data: missing 'split' column. "
            "Full-dataset fitting is strictly prohibited."
        )

    unique_splits = set(data["split"].dropna().unique())
    if "test" in unique_splits:
        raise TemporalLeakageError(
            f"[{caller_name}] Temporal leakage detected: attempted to fit on data containing 'test' split."
        )

    if allowed_split == "train" and "validation" in unique_splits:
        raise TemporalLeakageError(
            f"[{caller_name}] Temporal leakage detected: attempted to fit training model on 'validation' split."
        )


def build_forecast_dataset(
    assignments_df: pd.DataFrame,
    config: SplitsConfig = SplitsConfig(),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a complete rolling-origin forecast dataset for all candidate narratives.

    Guarantees:
    - `label_end(train) < validation_start`
    - `label_end(validation) < test_start`
    - No rows with missing future at end of period (`label_end <= dataset_end`)
    - All timestamps verified in UTC
    - Candidate prominence targets and historical momentum computed causally
    """
    req_cols = {"post_id", "timestamp", "narrative_id", "is_usable_text"}
    missing = req_cols - set(assignments_df.columns)
    if missing:
        raise ValueError(f"assignments_df missing required columns: {sorted(missing)}")

    posts = assignments_df.copy()
    posts["ts_utc"] = pd.to_datetime(posts["timestamp"], utc=True, errors="coerce")
    if posts["ts_utc"].isna().any():
        raise ValueError("Invalid/null timestamps found in assignments table; all timestamps must be UTC.")

    posts["post_id"] = posts["post_id"].astype(str).str.strip()
    if not posts["post_id"].is_unique:
        raise ValueError("Duplicate post_id found in assignments table.")

    # Only valid in-scope posts are used for candidate assignment; unusable/unassigned/noise are kept in denominator.
    # Exclude special tokens from candidate registry
    special_labels = {"NOISE", "UNASSIGNED", ""}
    cand_posts = posts[~posts["narrative_id"].isin(special_labels) & posts["is_usable_text"]].copy()

    # Pre-index candidate discovery times (first appearance strictly defines candidate birth)
    cand_first_seen = cand_posts.groupby("narrative_id")["ts_utc"].min().to_dict()

    # Generate complete weekly calendar index
    week_starts = pd.date_range(
        start=config.first_complete_week,
        end=config.last_complete_week,
        freq="W-MON",
        tz=config.timezone,
    )

    rows: List[Dict[str, Any]] = []
    purged_windows: List[Dict[str, Any]] = []
    included_origins_by_split: Dict[str, Set[pd.Timestamp]] = {"train": set(), "validation": set(), "test": set()}

    obs_delta = pd.Timedelta(days=config.observation_days)
    horiz_delta = pd.Timedelta(days=config.horizon_days)

    for origin in week_starts:
        hist_start = origin - obs_delta
        hist_end = origin
        label_start = origin
        label_end = origin + horiz_delta

        split = config.determine_split(origin, label_end)
        if split is None:
            # Purged / boundary window
            purged_windows.append(
                {
                    "forecast_origin_utc": origin.isoformat(),
                    "history_start_utc": hist_start.isoformat(),
                    "history_end_utc": hist_end.isoformat(),
                    "label_start_utc": label_start.isoformat(),
                    "label_end_utc": label_end.isoformat(),
                    "reason": "touches_next_split_or_exceeds_dataset_boundary",
                }
            )
            continue

        included_origins_by_split[split].add(origin)

        # Slice historical observation window: [hist_start, hist_end)
        hist_mask = (posts["ts_utc"] >= hist_start) & (posts["ts_utc"] < hist_end)
        hist_sub = posts[hist_mask]
        hist_total_posts = int(len(hist_sub))

        # Slice prior observation window for growth momentum: [hist_start - obs_delta, hist_start)
        prev_hist_mask = (posts["ts_utc"] >= (hist_start - obs_delta)) & (posts["ts_utc"] < hist_start)
        prev_hist_sub = posts[prev_hist_mask]

        # Slice future target window: [label_start, label_end)
        future_mask = (posts["ts_utc"] >= label_start) & (posts["ts_utc"] < label_end)
        future_sub = posts[future_mask]
        future_total_posts = int(len(future_sub))

        # Active candidate pool strictly observable up to origin t: first_seen < origin
        active_candidates = [cid for cid, f_seen in cand_first_seen.items() if f_seen < origin]

        hist_counts = hist_sub["narrative_id"].value_counts().to_dict()
        prev_hist_counts = prev_hist_sub["narrative_id"].value_counts().to_dict()
        future_counts = future_sub["narrative_id"].value_counts().to_dict()

        # Cumulative posts up to origin
        cum_posts_mask = (posts["ts_utc"] < origin) & (~posts["narrative_id"].isin(special_labels))
        cum_counts = posts[cum_posts_mask]["narrative_id"].value_counts().to_dict()

        if not active_candidates:
            # Warmup window: no candidates discovered before origin
            rows.append(
                {
                    "forecast_origin_utc": origin,
                    "history_start_utc": hist_start,
                    "history_end_utc": hist_end,
                    "label_start_utc": label_start,
                    "label_end_utc": label_end,
                    "split": split,
                    "candidate_id": "__WARMUP__",
                    "historical_post_count": 0,
                    "historical_total_posts": hist_total_posts,
                    "historical_prominence": 0.0,
                    "historical_prev_post_count": 0,
                    "historical_growth_momentum": 0.0,
                    "historical_cumulative_posts": 0,
                    "candidate_age_days": 0,
                    "future_post_count": 0,
                    "future_total_posts": future_total_posts,
                    "target_prominence": np.nan,
                    "target_rank": -1,
                    "target_top1_hit": False,
                    "target_top3_hit": False,
                    "is_target_defined": False,
                    "is_eligible_origin": hist_total_posts >= config.min_historical_posts,
                }
            )
            continue

        # Compute future prominence targets across all candidates in C_t
        cand_prominence_list = []
        for cid in active_candidates:
            n_c_fut = future_counts.get(cid, 0)
            if future_total_posts > 0:
                p_fut = float(n_c_fut) / float(future_total_posts)
            else:
                p_fut = np.nan
            cand_prominence_list.append((cid, p_fut, n_c_fut))

        # Rank candidates by future prominence (descending)
        if future_total_posts > 0:
            cand_prominence_list.sort(key=lambda x: (x[1], x[2]), reverse=True)
            ranks = {cid: rank + 1 for rank, (cid, _, _) in enumerate(cand_prominence_list)}
        else:
            ranks = {cid: -1 for cid in active_candidates}

        for cid in active_candidates:
            n_c_hist = int(hist_counts.get(cid, 0))
            p_hist = float(n_c_hist) / float(hist_total_posts) if hist_total_posts > 0 else 0.0
            n_c_prev = int(prev_hist_counts.get(cid, 0))
            growth = float(n_c_hist - n_c_prev) / float(n_c_prev + 1)
            cum_posts = int(cum_counts.get(cid, 0))
            age_days = max(0, (origin - cand_first_seen[cid]).days)

            n_c_fut = int(future_counts.get(cid, 0))
            is_def = future_total_posts > 0
            p_fut = float(n_c_fut) / float(future_total_posts) if is_def else np.nan

            rank = ranks[cid]
            top1 = bool(rank == 1 and n_c_fut > 0)
            top3 = bool(rank in (1, 2, 3) and n_c_fut > 0)

            rows.append(
                {
                    "forecast_origin_utc": origin,
                    "history_start_utc": hist_start,
                    "history_end_utc": hist_end,
                    "label_start_utc": label_start,
                    "label_end_utc": label_end,
                    "split": split,
                    "candidate_id": cid,
                    "historical_post_count": n_c_hist,
                    "historical_total_posts": hist_total_posts,
                    "historical_prominence": np.float32(p_hist),
                    "historical_prev_post_count": n_c_prev,
                    "historical_growth_momentum": np.float32(growth),
                    "historical_cumulative_posts": cum_posts,
                    "candidate_age_days": age_days,
                    "future_post_count": n_c_fut,
                    "future_total_posts": future_total_posts,
                    "target_prominence": np.float32(p_fut),
                    "target_rank": rank,
                    "target_top1_hit": top1,
                    "target_top3_hit": top3,
                    "is_target_defined": is_def,
                    "is_eligible_origin": hist_total_posts >= config.min_historical_posts,
                }
            )

    df_out = pd.DataFrame(rows)

    # Convert dtypes cleanly
    df_out["split"] = df_out["split"].astype("category")
    df_out["candidate_id"] = df_out["candidate_id"].astype("string")

    # Strict audit checks
    train_mask = df_out["split"] == "train"
    val_mask = df_out["split"] == "validation"
    test_mask = df_out["split"] == "test"

    train_label_max = df_out.loc[train_mask, "label_end_utc"].max() if train_mask.any() else None
    val_label_max = df_out.loc[val_mask, "label_end_utc"].max() if val_mask.any() else None
    test_label_max = df_out.loc[test_mask, "label_end_utc"].max() if test_mask.any() else None

    train_lt_val = bool(train_label_max < config.validation_start_utc) if train_label_max is not None else True
    val_lt_test = bool(val_label_max < config.test_start_utc) if val_label_max is not None else True
    no_missing_fut = bool(test_label_max <= config.dataset_end_utc) if test_label_max is not None else True

    manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "splits_config": {
            "timezone": config.timezone,
            "observation_days": config.observation_days,
            "horizon_days": config.horizon_days,
            "first_complete_week": config.first_complete_week.isoformat(),
            "last_complete_week": config.last_complete_week.isoformat(),
            "dataset_end_utc": config.dataset_end_utc.isoformat(),
            "train_start_utc": config.train_start_utc.isoformat(),
            "validation_start_utc": config.validation_start_utc.isoformat(),
            "test_start_utc": config.test_start_utc.isoformat(),
            "min_historical_posts": config.min_historical_posts,
        },
        "leakage_audit": {
            "all_train_label_end_lt_validation_start": train_lt_val,
            "all_validation_label_end_lt_test_start": val_lt_test,
            "no_missing_future_at_end_of_period": no_missing_fut,
            "train_label_end_max_utc": train_label_max.isoformat() if train_label_max is not None else None,
            "validation_start_utc": config.validation_start_utc.isoformat(),
            "validation_label_end_max_utc": val_label_max.isoformat() if val_label_max is not None else None,
            "test_start_utc": config.test_start_utc.isoformat(),
            "test_label_end_max_utc": test_label_max.isoformat() if test_label_max is not None else None,
            "dataset_end_utc": config.dataset_end_utc.isoformat(),
            "timestamps_all_utc": (
                str(df_out["forecast_origin_utc"].dt.tz) == "UTC"
                and str(df_out["label_end_utc"].dt.tz) == "UTC"
            ),
        },
        "summary": {
            "total_rows": int(len(df_out)),
            "total_purged_windows": int(len(purged_windows)),
            "unique_origins_total": int(df_out["forecast_origin_utc"].nunique()),
            "train_origins_count": int(len(included_origins_by_split["train"])),
            "validation_origins_count": int(len(included_origins_by_split["validation"])),
            "test_origins_count": int(len(included_origins_by_split["test"])),
            "train_rows_count": int(train_mask.sum()),
            "validation_rows_count": int(val_mask.sum()),
            "test_rows_count": int(test_mask.sum()),
            "candidates_evaluated_count": int(len(cand_first_seen)),
            "defined_target_rows": int(df_out["is_target_defined"].sum()),
            "undefined_target_rows": int((~df_out["is_target_defined"]).sum()),
            "eligible_origins_count": int(
                df_out[df_out["is_eligible_origin"]]["forecast_origin_utc"].nunique()
            ),
        },
        "purged_windows": purged_windows,
    }

    if not train_lt_val:
        raise TemporalLeakageError(
            f"Criterion failed: label_end train ({train_label_max}) must be < validation_start ({config.validation_start_utc})"
        )
    if not val_lt_test:
        raise TemporalLeakageError(
            f"Criterion failed: label_end validation ({val_label_max}) must be < test_start ({config.test_start_utc})"
        )
    if not no_missing_fut:
        raise TemporalLeakageError(
            f"Criterion failed: label_end test ({test_label_max}) must be <= dataset_end ({config.dataset_end_utc})"
        )

    return df_out, manifest


def save_forecast_dataset(
    df: pd.DataFrame,
    manifest: Dict[str, Any],
    out_parquet: Path,
    manifest_json: Optional[Path] = None,
) -> None:
    """Save forecast dataset to parquet with embedded JSON metadata and optional sidecar."""
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing_meta = dict(table.schema.metadata or {})
    existing_meta[PARQUET_METADATA_KEY] = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    table = table.replace_schema_metadata(existing_meta)
    pq.write_table(table, out_parquet, compression="snappy")

    if manifest_json is not None:
        manifest_json.parent.mkdir(parents=True, exist_ok=True)
        manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal forecast dataset with anti-leakage gates.")
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS_PATH)
    parser.add_argument("--splits-config", type=Path, default=DEFAULT_SPLITS_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logger.info("Loading split configuration from %s", args.splits_config)
    cfg = SplitsConfig.from_yaml(args.splits_config)

    logger.info("Loading narrative assignments from %s", args.assignments)
    assignments_df = pd.read_parquet(args.assignments)

    logger.info("Building forecast dataset across rolling origins (2014-2020)...")
    dataset_df, manifest = build_forecast_dataset(assignments_df, cfg)

    logger.info("Saving dataset to %s and manifest to %s", args.out, args.manifest_out)
    save_forecast_dataset(dataset_df, manifest, args.out, args.manifest_out)

    audit = manifest["leakage_audit"]
    summary = manifest["summary"]
    logger.info(
        "Build complete: %d rows across %d origins (train=%d, val=%d, test=%d). "
        "Leakage audit: train_lt_val=%s, val_lt_test=%s, no_missing_fut=%s",
        summary["total_rows"],
        summary["unique_origins_total"],
        summary["train_origins_count"],
        summary["validation_origins_count"],
        summary["test_origins_count"],
        audit["all_train_label_end_lt_validation_start"],
        audit["all_validation_label_end_lt_test_start"],
        audit["no_missing_future_at_end_of_period"],
    )


if __name__ == "__main__":
    main()
