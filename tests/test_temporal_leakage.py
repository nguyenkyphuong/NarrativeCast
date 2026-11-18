"""Comprehensive unit and integration test suite for temporal leakage prevention.

Verifies:
1. `label_end(train) < validation_start`
2. `label_end(validation) < test_start`
3. No rows with missing future at end of period (`label_end <= dataset_end`)
4. Edge timestamp tests:
   - Sunday 23:59:59.999999Z vs Monday 00:00:00.000000Z weekly interval assignment
   - Exact split cutoff boundaries (2018-01-01T00:00:00Z and 2019-01-01T00:00:00Z)
   - Timezone awareness (rejection of tz-naive timestamps)
   - Leap year continuity (2016-02-29, 2020-02-29)
5. Anti-leakage guards:
   - Prohibition of full-dataset fitting (raises TemporalLeakageError)
   - Future perturbation invariance (changing future data does not alter earlier features/targets)
6. Integrity of the generated `data/features/forecast_dataset.parquet` on disk.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.datasets.build_forecast_dataset import (
    PARQUET_METADATA_KEY,
    SplitsConfig,
    TemporalLeakageError,
    build_forecast_dataset,
    save_forecast_dataset,
    validate_no_full_dataset_fit,
)


def _make_synthetic_assignments(
    start_date: str = "2017-10-02T00:00:00Z",
    n_weeks: int = 24,
    posts_per_week: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a clean synthetic assignments table spanning train, val, and edge cases."""
    rng = np.random.default_rng(seed)
    base_ts = pd.Timestamp(start_date, tz="UTC")
    rows = []
    pid = 0

    cands = ["CAND_0001", "CAND_0002", "CAND_0003"]

    for w in range(n_weeks):
        w_start = base_ts + pd.Timedelta(days=7 * w)
        for i in range(posts_per_week):
            hours_offset = int((i * 15) % 160)
            post_ts = w_start + pd.Timedelta(hours=hours_offset)
            chosen_cand = cands[i % len(cands)]
            rows.append(
                {
                    "post_id": f"post_{pid:05d}",
                    "timestamp": post_ts.isoformat(),
                    "narrative_id": chosen_cand,
                    "is_usable_text": True,
                    "assignment_status": "ASSIGNED",
                }
            )
            pid += 1

    return pd.DataFrame(rows)


class TestTemporalLeakage(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = Path("configs/splits.yaml")
        if self.config_path.exists():
            self.cfg = SplitsConfig.from_yaml(self.config_path)
        else:
            self.cfg = SplitsConfig()

    def test_train_label_end_strictly_lt_validation_start(self) -> None:
        """Verify label_end(train) < validation_start strictly holds across all rows."""
        assignments = _make_synthetic_assignments()
        df, manifest = build_forecast_dataset(assignments, self.cfg)

        train_rows = df[df["split"] == "train"]
        self.assertFalse(train_rows.empty)

        max_label_end = train_rows["label_end_utc"].max()
        self.assertLess(
            max_label_end,
            self.cfg.validation_start_utc,
            f"Train label_end ({max_label_end}) must be strictly less than validation_start ({self.cfg.validation_start_utc})",
        )
        self.assertTrue(manifest["leakage_audit"]["all_train_label_end_lt_validation_start"])

    def test_validation_label_end_strictly_lt_test_start(self) -> None:
        """Verify label_end(validation) < test_start strictly holds across all rows."""
        assignments = _make_synthetic_assignments()
        df, manifest = build_forecast_dataset(assignments, self.cfg)

        val_rows = df[df["split"] == "validation"]
        self.assertFalse(val_rows.empty)

        max_label_end = val_rows["label_end_utc"].max()
        self.assertLess(
            max_label_end,
            self.cfg.test_start_utc,
            f"Validation label_end ({max_label_end}) must be strictly less than test_start ({self.cfg.test_start_utc})",
        )
        self.assertTrue(manifest["leakage_audit"]["all_validation_label_end_lt_test_start"])

    def test_no_missing_future_at_end_of_period(self) -> None:
        """Verify no row exceeds dataset_end_utc, ensuring full future horizon is present."""
        assignments = _make_synthetic_assignments(
            start_date="2020-06-01T00:00:00Z",
            n_weeks=30,
        )
        df, manifest = build_forecast_dataset(assignments, self.cfg)

        self.assertFalse(df.empty)
        max_label_end = df["label_end_utc"].max()
        self.assertLessEqual(
            max_label_end,
            self.cfg.dataset_end_utc,
            f"Max label_end ({max_label_end}) must not exceed dataset_end ({self.cfg.dataset_end_utc})",
        )
        self.assertTrue(manifest["leakage_audit"]["no_missing_future_at_end_of_period"])

    def test_boundary_timestamps(self) -> None:
        """Test timestamp boundary behavior: half-open intervals, split cutoffs, and leap years."""
        # 1. Sunday 23:59:59Z vs Monday 00:00:00Z
        sunday_night = pd.Timestamp("2017-12-31T23:59:59.999999Z")
        monday_morn = pd.Timestamp("2018-01-01T00:00:00.000000Z")

        # Sunday night belongs to train observation period
        self.assertLess(sunday_night, self.cfg.validation_start_utc)
        # Monday morning is exact validation_start
        self.assertEqual(monday_morn, self.cfg.validation_start_utc)

        # 2. Window determination for boundary origins
        # Origin 2017-12-25: label_end is 2018-01-01 -> touches validation -> purged (None)
        origin_touching_val = pd.Timestamp("2017-12-25T00:00:00Z")
        label_touching_val = origin_touching_val + pd.Timedelta(days=7)
        self.assertIsNone(self.cfg.determine_split(origin_touching_val, label_touching_val))

        # Origin 2017-12-18: label_end is 2017-12-25 < validation_start -> valid train
        origin_train_ok = pd.Timestamp("2017-12-18T00:00:00Z")
        label_train_ok = origin_train_ok + pd.Timedelta(days=7)
        self.assertEqual(self.cfg.determine_split(origin_train_ok, label_train_ok), "train")

        # Origin 2018-12-31: label_end is 2019-01-07 > test_start -> touches test -> purged (None)
        origin_touching_test = pd.Timestamp("2018-12-31T00:00:00Z")
        label_touching_test = origin_touching_test + pd.Timedelta(days=7)
        self.assertIsNone(self.cfg.determine_split(origin_touching_test, label_touching_test))

        # Origin 2018-12-24: label_end is 2018-12-31 < test_start -> valid validation
        origin_val_ok = pd.Timestamp("2018-12-24T00:00:00Z")
        label_val_ok = origin_val_ok + pd.Timedelta(days=7)
        self.assertEqual(self.cfg.determine_split(origin_val_ok, label_val_ok), "validation")

        # 3. Leap years: 2016-02-29 and 2020-02-29
        leap_2016 = pd.Timestamp("2016-02-29T12:00:00Z")
        leap_2020 = pd.Timestamp("2020-02-29T12:00:00Z")
        self.assertTrue(leap_2016.is_leap_year)
        self.assertTrue(leap_2020.is_leap_year)

        # 4. Timezone awareness check: naive timestamps must be rejected
        with self.assertRaises(ValueError):
            self.cfg.determine_split(
                pd.Timestamp("2017-12-18 00:00:00"),  # naive!
                pd.Timestamp("2017-12-25 00:00:00"),
            )

    def test_no_full_dataset_fitting_guards(self) -> None:
        """Verify guards prevent fitting transformers/scalers on full dataset or test data."""
        # 1. Fitting on data containing test split must raise TemporalLeakageError
        full_df = pd.DataFrame({"split": ["train", "validation", "test"], "val": [1, 2, 3]})
        with self.assertRaises(TemporalLeakageError):
            validate_no_full_dataset_fit(full_df, allowed_split="train", caller_name="StandardScaler")

        # 2. Fitting training model on validation split must raise TemporalLeakageError
        train_val_df = pd.DataFrame({"split": ["train", "validation"], "val": [1, 2]})
        with self.assertRaises(TemporalLeakageError):
            validate_no_full_dataset_fit(train_val_df, allowed_split="train", caller_name="UMAP")

        # 3. Fitting on dataframe missing 'split' column must raise TemporalLeakageError
        unpartitioned_df = pd.DataFrame({"val": [1, 2, 3]})
        with self.assertRaises(TemporalLeakageError):
            validate_no_full_dataset_fit(unpartitioned_df, caller_name="HDBSCAN")

        # 4. Fitting on train only must pass
        train_only_df = pd.DataFrame({"split": ["train", "train"], "val": [1, 2]})
        validate_no_full_dataset_fit(train_only_df, allowed_split="train", caller_name="SafeScaler")

    def test_future_perturbation_invariance(self) -> None:
        """Verify that altering future data has ZERO effect on historical features and training instances."""
        assignments_a = _make_synthetic_assignments(seed=101)
        assignments_b = assignments_a.copy()

        # Corrupt all assignments in 2018 (validation)
        val_mask = pd.to_datetime(assignments_b["timestamp"], utc=True) >= pd.Timestamp("2018-01-01T00:00:00Z")
        assignments_b.loc[val_mask, "narrative_id"] = "CORRUPTED_CANDIDATE"

        df_a, _ = build_forecast_dataset(assignments_a, self.cfg)
        df_b, _ = build_forecast_dataset(assignments_b, self.cfg)

        train_a = df_a[df_a["split"] == "train"].reset_index(drop=True)
        train_b = df_b[df_b["split"] == "train"].reset_index(drop=True)

        # Entire train partition must be completely identical
        pd.testing.assert_frame_equal(train_a, train_b)

    def test_generated_forecast_dataset_on_disk(self) -> None:
        """Verify the generated forecast_dataset.parquet on disk satisfies all criteria."""
        pq_path = Path("data/features/forecast_dataset.parquet")
        if not pq_path.exists():
            self.skipTest("Generated parquet not found; run build_forecast_dataset first.")

        df = pd.read_parquet(pq_path)
        required_cols = {
            "forecast_origin_utc",
            "history_start_utc",
            "history_end_utc",
            "label_start_utc",
            "label_end_utc",
            "split",
            "candidate_id",
            "historical_post_count",
            "historical_total_posts",
            "historical_prominence",
            "future_post_count",
            "future_total_posts",
            "target_prominence",
            "is_target_defined",
        }
        self.assertTrue(required_cols.issubset(df.columns))

        # Check UTC dtypes
        self.assertEqual(str(df["forecast_origin_utc"].dt.tz), "UTC")
        self.assertEqual(str(df["label_end_utc"].dt.tz), "UTC")

        # Criterion 1: label_end(train) < validation_start
        train_rows = df[df["split"] == "train"]
        self.assertEqual(train_rows["forecast_origin_utc"].nunique(), 207)
        self.assertLess(train_rows["label_end_utc"].max(), self.cfg.validation_start_utc)

        # Criterion 2: label_end(validation) < test_start
        val_rows = df[df["split"] == "validation"]
        self.assertEqual(val_rows["forecast_origin_utc"].nunique(), 52)
        self.assertLess(val_rows["label_end_utc"].max(), self.cfg.test_start_utc)

        # Criterion 3: No missing future at end of period
        test_rows = df[df["split"] == "test"]
        self.assertEqual(test_rows["forecast_origin_utc"].nunique(), 103)
        self.assertLessEqual(df["label_end_utc"].max(), self.cfg.dataset_end_utc)

        # Embedded metadata check
        table = pq.read_table(pq_path)
        raw_meta = table.schema.metadata.get(PARQUET_METADATA_KEY)
        self.assertIsNotNone(raw_meta)
        meta = json.loads(raw_meta.decode("utf-8"))
        audit = meta["leakage_audit"]
        self.assertTrue(audit["all_train_label_end_lt_validation_start"])
        self.assertTrue(audit["all_validation_label_end_lt_test_start"])
        self.assertTrue(audit["no_missing_future_at_end_of_period"])
        self.assertTrue(audit["timestamps_all_utc"])


if __name__ == "__main__":
    unittest.main()
