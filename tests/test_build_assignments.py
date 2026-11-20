"""Unit and integration tests for `src.tracking.build_assignments`."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.tracking import build_assignments as ba


def _make_synthetic_dataset(seed: int = 123) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create a 5-week synthetic dataset with clusters plus edge-case invalid rows."""
    rng = np.random.default_rng(seed)
    rows = []
    vecs = []
    present = []

    centers = rng.normal(size=(3, 16)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # Week 0 (warmup, 4 valid posts), Weeks 1..4 (12 valid posts each -> 52 valid total)
    week_counts = [4, 12, 12, 12, 12]
    base = pd.Timestamp("2020-01-06T00:00:00Z")  # Monday UTC
    pid_counter = 0

    for w_idx, n_w in enumerate(week_counts):
        w_start = base + pd.Timedelta(days=7 * w_idx)
        for j in range(n_w):
            ts = w_start + pd.Timedelta(hours=(j * 11) % 160)
            c = centers[j % 3] + 0.08 * rng.normal(size=16).astype(np.float32)
            c /= np.linalg.norm(c)
            rows.append(
                {
                    "post_id": f"p_{pid_counter:03d}",
                    "publish_datetime_utc": ts.isoformat(),
                    "is_usable_text": True,
                }
            )
            vecs.append(c)
            present.append(True)
            pid_counter += 1

    # Add 4 invalid edge-case rows across Week 2 and Week 3:
    # 1) unusable text
    rows.append(
        {
            "post_id": "p_unusable",
            "publish_datetime_utc": "2020-01-22T12:00:00+00:00",
            "is_usable_text": False,
        }
    )
    vecs.append(centers[0].copy())
    present.append(True)

    # 2) missing vector
    rows.append(
        {
            "post_id": "p_missing_vec",
            "publish_datetime_utc": "2020-01-23T12:00:00+00:00",
            "is_usable_text": True,
        }
    )
    vecs.append(np.full(16, np.nan, dtype=np.float32))
    present.append(False)

    # 3) NaN / Inf vector
    bad_vec = np.zeros(16, dtype=np.float32)
    bad_vec[0] = np.nan
    bad_vec[1] = np.inf
    rows.append(
        {
            "post_id": "p_nan_vec",
            "publish_datetime_utc": "2020-01-28T05:00:00+00:00",
            "is_usable_text": True,
        }
    )
    vecs.append(bad_vec)
    present.append(True)

    # 4) zero-norm vector
    rows.append(
        {
            "post_id": "p_zero_vec",
            "publish_datetime_utc": "2020-01-29T05:00:00+00:00",
            "is_usable_text": True,
        }
    )
    vecs.append(np.zeros(16, dtype=np.float32))
    present.append(True)

    # 5) outlier valid vector (distant from clusters -> should become NOISE when score < noise_threshold)
    outlier = -centers[0].copy()
    rows.append(
        {
            "post_id": "p_outlier",
            "publish_datetime_utc": "2020-02-04T05:00:00+00:00",
            "is_usable_text": True,
        }
    )
    vecs.append(outlier)
    present.append(True)

    df = pd.DataFrame(rows)
    return df, np.stack(vecs).astype(np.float32), np.asarray(present, dtype=bool)


class TestBuildAssignments(unittest.TestCase):
    def test_id_preservation_and_invalid_exclusion(self) -> None:
        scope_df, vecs, present = _make_synthetic_dataset()
        cfg = ba.AssignmentConfig(
            n_clusters=3,
            batch_size=32,
            min_fit_posts=10,
            noise_threshold=0.70,
            alignment_threshold=0.70,
            model_version="test_v1",
        )

        seen_fit_ids: list[str] = []
        seen_pred_ids: list[str] = []
        orig_discover = ba.discover_narratives
        orig_assign = ba.assign_existing

        def spy_discover(x, meta, dcfg):
            self.assertTrue(np.all(np.isfinite(x)))
            self.assertTrue(np.all(np.linalg.norm(x, axis=1) > 1e-9))
            seen_fit_ids.extend(meta["post_id"].tolist())
            return orig_discover(x, meta, dcfg)

        def spy_assign(x, meta, model, mv):
            self.assertTrue(np.all(np.isfinite(x)))
            self.assertTrue(np.all(np.linalg.norm(x, axis=1) > 1e-9))
            seen_pred_ids.extend(meta["post_id"].tolist())
            return orig_assign(x, meta, model, mv)

        with patch.object(ba, "discover_narratives", side_effect=spy_discover), patch.object(
            ba, "assign_existing", side_effect=spy_assign
        ):
            out_df, manifest = ba.build_temporal_assignments(
                scope_df=scope_df,
                vectors=vecs,
                is_vector_present=present,
                config=cfg,
            )

        # 1. Exact ID preservation (no lost/duplicated IDs)
        self.assertEqual(len(out_df), len(scope_df))
        self.assertTrue(out_df["post_id"].is_unique)
        self.assertEqual(set(out_df["post_id"]), set(scope_df["post_id"]))

        # 2. Invalid/unusable rows never entered fit or predict, yet remain UNASSIGNED in output
        forbidden = {"p_unusable", "p_missing_vec", "p_nan_vec", "p_zero_vec"}
        self.assertTrue(forbidden.isdisjoint(seen_fit_ids))
        self.assertTrue(forbidden.isdisjoint(seen_pred_ids))

        invalid_out = out_df.set_index("post_id").loc[list(forbidden)]
        self.assertTrue((invalid_out["narrative_id"] == "UNASSIGNED").all())
        self.assertTrue((invalid_out["local_cluster_id"] == -1).all())
        self.assertTrue(invalid_out["assignment_score"].isna().all())
        self.assertFalse(bool(invalid_out.loc["p_unusable", "is_usable_text"]))
        self.assertTrue(bool(invalid_out.loc["p_missing_vec", "is_usable_text"]))

        # 3. Week 0 warmup posts had <10 prior posts -> kept UNASSIGNED
        w0 = out_df[out_df["post_id"].isin([f"p_{i:03d}" for i in range(4)])]
        self.assertTrue((w0["narrative_id"] == "UNASSIGNED").all())
        self.assertTrue((w0["unassigned_reason"] == "insufficient_history").all())

        # 4. Outlier post preserved as NOISE
        outlier_row = out_df.set_index("post_id").loc["p_outlier"]
        self.assertEqual(outlier_row["narrative_id"], "NOISE")
        self.assertEqual(int(outlier_row["local_cluster_id"]), -1)
        self.assertTrue(np.isfinite(float(outlier_row["assignment_score"])))

        # 5. Summary integrity flags
        self.assertTrue(manifest["summary"]["no_lost_ids"])
        self.assertTrue(manifest["summary"]["no_duplicate_ids"])
        self.assertTrue(manifest["summary"]["all_timestamps_utc"])
        self.assertTrue(manifest["summary"]["zero_future_leakage"])

    def test_utc_weekly_half_open_boundaries(self) -> None:
        ts = pd.Series(
            [
                "2020-01-05T23:59:59+00:00",  # Sunday -> week 2019-12-30 .. 2020-01-06
                "2020-01-06T00:00:00+00:00",  # Monday -> week 2020-01-06 .. 2020-01-13
                "2020-01-05T20:00:00-05:00",  # 2020-01-06T01:00:00Z -> week 2020-01-06 .. 2020-01-13
            ]
        )
        ts_utc, win_start, win_end = ba.compute_utc_weekly_windows(ts)
        self.assertEqual(str(ts_utc.dt.tz), "UTC")
        self.assertEqual(win_start.iloc[0], pd.Timestamp("2019-12-30T00:00:00Z"))
        self.assertEqual(win_end.iloc[0], pd.Timestamp("2020-01-06T00:00:00Z"))
        self.assertEqual(win_start.iloc[1], pd.Timestamp("2020-01-06T00:00:00Z"))
        self.assertEqual(win_end.iloc[1], pd.Timestamp("2020-01-13T00:00:00Z"))
        self.assertEqual(win_start.iloc[2], pd.Timestamp("2020-01-06T00:00:00Z"))
        self.assertEqual(win_end.iloc[2], pd.Timestamp("2020-01-13T00:00:00Z"))

    def test_zero_future_leakage_invariance(self) -> None:
        scope_a, vecs_a, present_a = _make_synthetic_dataset(seed=7)
        scope_b = scope_a.copy()
        vecs_b = vecs_a.copy()

        # Corrupt all posts strictly in the last week (>= 2020-02-03 UTC)
        future_mask = pd.to_datetime(scope_b["publish_datetime_utc"], utc=True) >= pd.Timestamp("2020-02-03T00:00:00Z")
        rng = np.random.default_rng(999)
        vecs_b[future_mask.to_numpy()] = rng.normal(size=(int(future_mask.sum()), vecs_b.shape[1])).astype(np.float32)

        cfg = ba.AssignmentConfig(
            n_clusters=3,
            batch_size=32,
            min_fit_posts=10,
            noise_threshold=0.70,
            alignment_threshold=0.70,
        )
        out_a, _ = ba.build_temporal_assignments(scope_a, vecs_a, present_a, cfg)
        out_b, _ = ba.build_temporal_assignments(scope_b, vecs_b, present_a, cfg)

        past_a = out_a[~future_mask.to_numpy()].reset_index(drop=True)
        past_b = out_b[~future_mask.to_numpy()].reset_index(drop=True)
        pd.testing.assert_frame_equal(past_a, past_b)

    def test_parquet_metadata_roundtrip(self) -> None:
        import tempfile

        scope_df, vecs, present = _make_synthetic_dataset()
        cfg = ba.AssignmentConfig(n_clusters=3, batch_size=32, min_fit_posts=10)
        out_df, manifest = ba.build_temporal_assignments(scope_df, vecs, present, cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            pq_path = Path(tmpdir) / "assignments.parquet"
            json_path = Path(tmpdir) / "manifest.json"
            ba.save_assignments_with_metadata(out_df, manifest, pq_path, json_path)

            table = pq.read_table(pq_path)
            raw_meta = table.schema.metadata.get(ba.PARQUET_METADATA_KEY)
            self.assertIsNotNone(raw_meta)
            decoded = json.loads(raw_meta.decode("utf-8"))
            self.assertEqual(decoded["discovery_config"]["n_clusters"], 3)
            self.assertEqual(decoded["summary"]["output_posts"], len(scope_df))
            self.assertTrue(json_path.exists())

    def test_generated_artifact_on_disk_if_present(self) -> None:
        pq_path = Path("data/interim/narrative_assignments.parquet")
        if not pq_path.exists():
            self.skipTest("Generated parquet not found; run build_assignments CLI first.")

        df = pd.read_parquet(pq_path)
        required = {
            "post_id",
            "timestamp",
            "local_cluster_id",
            "narrative_id",
            "window_end",
            "model_version",
            "assignment_score",
            "is_usable_text",
        }
        self.assertTrue(required.issubset(df.columns))
        self.assertEqual(len(df), 7331)
        self.assertTrue(df["post_id"].is_unique)
        self.assertEqual(str(df["timestamp"].dt.tz), "UTC")
        self.assertEqual(str(df["window_end"].dt.tz), "UTC")
        self.assertTrue(((df["window_start"] <= df["timestamp"]) & (df["timestamp"] < df["window_end"])).all())

        table = pq.read_table(pq_path)
        meta = json.loads(table.schema.metadata[ba.PARQUET_METADATA_KEY].decode("utf-8"))
        self.assertTrue(meta["summary"]["no_lost_ids"])
        self.assertTrue(meta["summary"]["no_duplicate_ids"])
        self.assertTrue(meta["summary"]["zero_future_leakage"])


if __name__ == "__main__":
    unittest.main()
