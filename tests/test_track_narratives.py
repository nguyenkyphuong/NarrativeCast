"""Unit tests for src.tracking.track_narratives.

Covers centroid math, EMA drift, 0 vs NaN handling, prefix invariance,
lifecycle transitions, matching tie-breaks, cutoff loading, and pair sampling.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from src.tracking.track_narratives import (
    LifecycleStatus,
    NarrativeEntity,
    NarrativeTracker,
    SplitMergeFlag,
    TrackingConfig,
    compute_normalized_centroid,
    export_tracking_review_sample,
    load_frozen_registry_at_cutoff,
    match_clusters_to_registry,
    save_centroids_npz,
    save_registry_artifacts,
)

# Shared helpers

DIM = 384


def _make_unit_vector(seed_offset: int = 0) -> np.ndarray:
    """Return a deterministic L2-normalized 384-D vector."""
    rng = np.random.default_rng(42 + seed_offset)
    v = rng.standard_normal(DIM).astype(np.float64)
    v /= np.linalg.norm(v)
    return v


def _make_orthogonal_unit_pair(seed_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return two deterministic orthonormal 384-D vectors via Gram-Schmidt."""
    u = _make_unit_vector(seed_offset)
    raw = _make_unit_vector(seed_offset + 5000)
    v = raw - float(np.dot(raw, u)) * u
    v /= np.linalg.norm(v)
    return u, v


def _make_vector_at_cosine(
    base: np.ndarray, ortho: np.ndarray, target_sim: float
) -> np.ndarray:
    """Return a unit vector with exact cosine similarity target_sim to base."""
    vec = target_sim * base + math.sqrt(max(0.0, 1.0 - target_sim**2)) * ortho
    vec /= np.linalg.norm(vec)
    return vec


def _make_random_vectors(n: int, seed: int = 42) -> np.ndarray:
    """Return n random 384-D unit vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, DIM)).astype(np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_unnormalized_vectors(
    n: int, target_norm: float = 2.0, seed: int = 42
) -> np.ndarray:
    """Return n random 384-D vectors with L2 norm equal to target_norm."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, DIM)).astype(np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms * target_norm


def _iso(dt_str: str) -> pd.Timestamp:
    """Parse an ISO string to a tz-aware UTC Timestamp."""
    ts = pd.Timestamp(dt_str)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _week_range(start: str, n_weeks: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return a list of (window_start, window_end) for n_weeks consecutive weeks."""
    base = _iso(start)
    return [
        (base + pd.Timedelta(weeks=i), base + pd.Timedelta(weeks=i + 1))
        for i in range(n_weeks)
    ]


def _build_assignments_df(records: List[dict]) -> pd.DataFrame:
    """Build a minimal assignments DataFrame from a list of row dicts."""
    df = pd.DataFrame(records)
    for col in ("timestamp", "window_start", "window_end"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def _assert_timelines_identical(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Assert exact column-by-column equality for sorted timeline DataFrames."""
    a = df1.sort_values(["window_end", "narrative_id"]).reset_index(drop=True)
    b = df2.sort_values(["window_end", "narrative_id"]).reset_index(drop=True)
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]) and not pd.api.types.is_bool_dtype(a[col]):
            v1 = a[col].to_numpy(dtype=np.float64)
            v2 = b[col].to_numpy(dtype=np.float64)
            assert np.array_equal(v1, v2, equal_nan=True), (
                f"Column '{col}' differs between timelines"
            )
        elif isinstance(a[col].dtype, pd.CategoricalDtype) or isinstance(
            b[col].dtype, pd.CategoricalDtype
        ):
            assert a[col].astype(str).equals(b[col].astype(str)), (
                f"Categorical column '{col}' values differ"
            )
        else:
            assert a[col].equals(b[col]), f"Non-numeric column '{col}' differs"


def _assert_centroids_identical(
    cents1: List[dict], cents2: List[dict]
) -> None:
    """Assert exact 1:1 equality of centroid records keyed by (window_end_utc, narrative_id)."""
    map1 = {
        (pd.Timestamp(r["window_end_utc"]).tz_convert("UTC"), str(r["narrative_id"])): np.asarray(
            r["centroid"], dtype=np.float32
        )
        for r in cents1
    }
    map2 = {
        (pd.Timestamp(r["window_end_utc"]).tz_convert("UTC"), str(r["narrative_id"])): np.asarray(
            r["centroid"], dtype=np.float32
        )
        for r in cents2
    }
    assert set(map1.keys()) == set(map2.keys())
    for key in map1:
        assert np.array_equal(map1[key], map2[key]), (
            f"Centroid vector differs for {key}"
        )


# Centroid mathematics, EMA, and drift


class TestComputeNormalizedCentroid:
    """Public function compute_normalized_centroid(vectors)."""

    def test_single_vector_identity(self):
        v = _make_unit_vector(0)
        result = compute_normalized_centroid([v])
        expected = v / np.linalg.norm(v)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_centroid_of_two_similar_vectors(self):
        v1 = _make_unit_vector(10)
        v2 = _make_unit_vector(11)
        result = compute_normalized_centroid([v1, v2])
        mean_vec = (v1 + v2) / 2.0
        expected = mean_vec / np.linalg.norm(mean_vec)
        np.testing.assert_allclose(result, expected, atol=1e-6)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-6

    def test_centroid_of_many_vectors(self):
        vecs = _make_random_vectors(50, seed=42)
        result = compute_normalized_centroid(list(vecs))
        mean_vec = vecs.mean(axis=0)
        expected = mean_vec / np.linalg.norm(mean_vec)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_opposing_vectors_raises_valueerror(self):
        v = _make_unit_vector(0)
        with pytest.raises(ValueError):
            compute_normalized_centroid([v, -v])

    def test_nan_vector_raises_valueerror(self):
        v_good = _make_unit_vector(0)
        v_bad = np.full(DIM, np.nan)
        with pytest.raises(ValueError):
            compute_normalized_centroid([v_good, v_bad])

    def test_inf_vector_raises_valueerror(self):
        v_good = _make_unit_vector(0)
        v_bad = np.full(DIM, np.inf)
        with pytest.raises(ValueError):
            compute_normalized_centroid([v_good, v_bad])

    def test_zero_norm_vector_raises_valueerror(self):
        v_zero = np.zeros(DIM)
        with pytest.raises(ValueError):
            compute_normalized_centroid([v_zero])


class TestEmaDrift:
    """Verify EMA centroid, similarity, and drift calculations."""

    def test_ema_centroid_and_drift_two_weeks(self):
        """Over 2 weeks, verify raw week-1 centroid, week-2 similarity,
        EMA centroid = normalize(alpha * raw_w2 + (1 - alpha) * raw_w1),
        and centroid_drift = 1 - dot(raw_w1, ema_w2).
        """
        alpha = 0.30
        v_w1 = _make_random_vectors(5, seed=42)
        v_w2 = _make_random_vectors(5, seed=43)

        weeks = _week_range("2018-01-01T00:00:00Z", 2)
        records = []
        lookup = {}
        pid_counter = 0
        for week_idx, (ws, we) in enumerate(weeks):
            vecs = v_w1 if week_idx == 0 else v_w2
            for j, v in enumerate(vecs):
                pid = f"ema_{pid_counter:06d}"
                ts = ws + pd.Timedelta(hours=6 * (j + 1))
                records.append(
                    dict(
                        post_id=pid,
                        timestamp=str(ts),
                        window_start=str(ws),
                        window_end=str(we),
                        narrative_id="CAND_001",
                        assignment_status="ASSIGNED",
                        is_usable_text=True,
                        local_cluster_id=0,
                    )
                )
                lookup[pid] = v
                pid_counter += 1

        df = _build_assignments_df(records)
        cfg = TrackingConfig(embedding_dim=DIM, random_state=42, ema_alpha=alpha)
        tracker = NarrativeTracker(cfg)
        timeline, _, centroid_records = tracker.track_all_windows(df, lookup)

        raw_w1 = v_w1.mean(axis=0)
        raw_w1 /= np.linalg.norm(raw_w1)

        raw_w2 = v_w2.mean(axis=0)
        raw_w2 /= np.linalg.norm(raw_w2)
        ema_w2 = alpha * raw_w2 + (1.0 - alpha) * raw_w1
        ema_w2 /= np.linalg.norm(ema_w2)

        expected_sim = float(np.dot(raw_w1, raw_w2))
        expected_drift = 1.0 - float(np.dot(raw_w1, ema_w2))

        sorted_tl = timeline[timeline["narrative_id"] == "CAND_001"].sort_values("window_end")
        assert len(sorted_tl) == 2

        week1_row = sorted_tl.iloc[0]
        week2_row = sorted_tl.iloc[1]

        assert week1_row["centroid_drift"] == pytest.approx(0.0, abs=1e-6)
        assert week2_row["similarity_to_previous_centroid"] == pytest.approx(
            expected_sim, abs=1e-5
        )
        assert week2_row["centroid_drift"] == pytest.approx(expected_drift, abs=1e-5)

        sorted_cents = sorted(
            centroid_records, key=lambda r: pd.Timestamp(r["window_end_utc"])
        )
        assert len(sorted_cents) == 2
        np.testing.assert_allclose(sorted_cents[0]["centroid"], raw_w1, atol=1e-5)
        np.testing.assert_allclose(sorted_cents[1]["centroid"], ema_w2, atol=1e-5)


# Dispersion with un-normalized vectors


class TestDispersion:
    """Intra-window dispersion must be correct even for vectors with norm != 1."""

    def _run_single_window(
        self,
        post_vectors: List[np.ndarray],
        narrative_id: str = "CAND_001",
    ) -> pd.DataFrame:
        ws, we = "2018-01-01T00:00:00Z", "2018-01-08T00:00:00Z"
        records = []
        lookup: Dict[str, np.ndarray] = {}
        for i, vec in enumerate(post_vectors):
            pid = f"post_{i:04d}"
            ts = f"2018-01-0{min(i + 1, 7)}T12:00:00Z"
            records.append(
                dict(
                    post_id=pid,
                    timestamp=ts,
                    window_start=ws,
                    window_end=we,
                    narrative_id=narrative_id,
                    assignment_status="ASSIGNED",
                    is_usable_text=True,
                    local_cluster_id=0,
                )
            )
            lookup[pid] = vec
        df = _build_assignments_df(records)
        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        tracker = NarrativeTracker(cfg)
        timeline, _, _ = tracker.track_all_windows(df, lookup)
        return timeline

    def test_dispersion_unnormalized_vectors(self):
        vecs = _make_unnormalized_vectors(10, target_norm=2.0, seed=42)
        tl = self._run_single_window(list(vecs))
        disp = float(tl["intra_window_dispersion"].iloc[0])
        assert 0.0 <= disp <= 2.0, f"Dispersion {disp} out of [0, 2]"

        unit_vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        centroid = unit_vecs.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        expected_disp = float(np.mean(1.0 - unit_vecs @ centroid))
        assert disp == pytest.approx(expected_disp, abs=1e-5)

    def test_single_post_dispersion_zero(self):
        v = _make_unit_vector(50)
        tl = self._run_single_window([v])
        disp = float(tl["intra_window_dispersion"].iloc[0])
        assert disp == pytest.approx(0.0, abs=1e-9)

    def test_zero_posts_dispersion_nan(self):
        """A narrative absent in a window (0 posts) must have NaN dispersion."""
        ws, we = "2018-01-01T00:00:00Z", "2018-01-08T00:00:00Z"
        ws2, we2 = "2018-01-08T00:00:00Z", "2018-01-15T00:00:00Z"
        records = [
            dict(
                post_id="p1",
                timestamp="2018-01-02T12:00:00Z",
                window_start=ws,
                window_end=we,
                narrative_id="CAND_001",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=0,
            ),
            dict(
                post_id="p2",
                timestamp="2018-01-09T12:00:00Z",
                window_start=ws2,
                window_end=we2,
                narrative_id="CAND_002",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=1,
            ),
        ]
        lookup = {"p1": _make_unit_vector(60), "p2": _make_unit_vector(61)}
        df = _build_assignments_df(records)
        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        tracker = NarrativeTracker(cfg)
        timeline, _, _ = tracker.track_all_windows(df, lookup)
        row = timeline[
            (timeline["narrative_id"] == "CAND_001")
            & (timeline["window_end"] == _iso(we2))
        ]
        assert len(row) == 1
        assert math.isnan(float(row["intra_window_dispersion"].iloc[0]))


# Distinguishing 0 vs NaN: empty week vs absent narrative


class TestZeroVsNanSemantics:
    """Distinguish 0.0 vs NaN across empty weeks and absent-narrative weeks."""

    @pytest.fixture()
    def three_week_result(self):
        """Week 1: CAND_001 has 2 posts.
        Week 2: CAND_002 has 1 valid post, CAND_001 has 0 posts (absent in non-empty week).
        Week 3: 0 valid posts total (empty week via UNASSIGNED + is_usable_text=False).
        """
        weeks = _week_range("2018-01-01T00:00:00Z", 3)
        (ws1, we1), (ws2, we2), (ws3, we3) = weeks
        records = [
            dict(
                post_id="p1",
                timestamp="2018-01-02T12:00:00Z",
                window_start=ws1,
                window_end=we1,
                narrative_id="CAND_001",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=0,
            ),
            dict(
                post_id="p2",
                timestamp="2018-01-03T12:00:00Z",
                window_start=ws1,
                window_end=we1,
                narrative_id="CAND_001",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=0,
            ),
            dict(
                post_id="p3",
                timestamp="2018-01-09T12:00:00Z",
                window_start=ws2,
                window_end=we2,
                narrative_id="CAND_002",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=1,
            ),
            dict(
                post_id="p4",
                timestamp="2018-01-16T12:00:00Z",
                window_start=ws3,
                window_end=we3,
                narrative_id="CAND_001",
                assignment_status="UNASSIGNED",
                is_usable_text=False,
                local_cluster_id=-1,
            ),
        ]
        lookup = {
            "p1": _make_unit_vector(100),
            "p2": _make_unit_vector(101),
            "p3": _make_unit_vector(102),
        }
        df = _build_assignments_df(records)
        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        tracker = NarrativeTracker(cfg)
        timeline, _, centroid_records = tracker.track_all_windows(df, lookup)
        return timeline, centroid_records, weeks

    def test_absent_narrative_nonempty_week(
        self, three_week_result
    ):
        tl, _, weeks = three_week_result
        _, we2 = weeks[1]
        row = tl[(tl["narrative_id"] == "CAND_001") & (tl["window_end"] == we2)]
        assert len(row) == 1
        r = row.iloc[0]
        assert r["window_total_valid_posts"] == 1
        assert r["current_window_posts"] == 0
        assert float(r["current_window_share"]) == pytest.approx(0.0, abs=1e-9)
        assert math.isnan(float(r["intra_window_dispersion"]))
        assert math.isnan(float(r["similarity_to_previous_centroid"]))
        assert float(r["centroid_drift"]) == pytest.approx(0.0, abs=1e-9)

    def test_empty_week_nan_fields(
        self, three_week_result
    ):
        tl, _, weeks = three_week_result
        _, we3 = weeks[2]
        w3_rows = tl[tl["window_end"] == we3]
        assert not w3_rows.empty
        for _, r in w3_rows.iterrows():
            assert r["window_total_valid_posts"] == 0
            assert r["current_window_posts"] == 0
            assert math.isnan(float(r["current_window_share"]))
            assert math.isnan(float(r["intra_window_dispersion"]))
            assert math.isnan(float(r["similarity_to_previous_centroid"]))
            assert float(r["centroid_drift"]) == pytest.approx(0.0, abs=1e-9)

    def test_centroid_frozen_when_silent(self, three_week_result):
        _, centroid_records, weeks = three_week_result
        (_, we1), (_, we2), (_, we3) = weeks
        cand1_cents = {
            pd.Timestamp(r["window_end_utc"]).tz_convert("UTC"): np.asarray(
                r["centroid"], dtype=np.float32
            )
            for r in centroid_records
            if r["narrative_id"] == "CAND_001"
        }
        assert we1 in cand1_cents and we2 in cand1_cents and we3 in cand1_cents
        assert np.array_equal(cand1_cents[we1], cand1_cents[we2])
        assert np.array_equal(cand1_cents[we1], cand1_cents[we3])


# Prefix invariance


class TestPrefixInvariance:
    """Adding or modifying future windows must not change past timeline or centroids."""

    @staticmethod
    def _build_scenario(
        n_future_weeks: int, extra_future_noise: bool = False, shuffle: bool = False
    ):
        base_weeks = _week_range("2018-01-01T00:00:00Z", 4)
        records = []
        lookup = {}
        pid_counter = 0
        for ws, we in base_weeks:
            for j in range(5):
                pid = f"p_{pid_counter:06d}"
                ts = ws + pd.Timedelta(hours=12 * (j + 1))
                records.append(
                    dict(
                        post_id=pid,
                        timestamp=str(ts),
                        window_start=str(ws),
                        window_end=str(we),
                        narrative_id="CAND_001",
                        assignment_status="ASSIGNED",
                        is_usable_text=True,
                        local_cluster_id=0,
                    )
                )
                lookup[pid] = _make_unit_vector(pid_counter)
                pid_counter += 1

        if extra_future_noise:
            future_weeks = _week_range("2018-01-29T00:00:00Z", n_future_weeks)
            rng = np.random.default_rng(999)
            for ws, we in future_weeks:
                for j in range(25):
                    pid = f"future_{pid_counter:06d}"
                    ts = ws + pd.Timedelta(hours=5 * (j + 1))
                    nid = f"CAND_{int(rng.integers(1, 50)):03d}"
                    records.append(
                        dict(
                            post_id=pid,
                            timestamp=str(ts),
                            window_start=str(ws),
                            window_end=str(we),
                            narrative_id=nid,
                            assignment_status="ASSIGNED",
                            is_usable_text=True,
                            local_cluster_id=int(rng.integers(0, 24)),
                        )
                    )
                    v = rng.standard_normal(DIM).astype(np.float64)
                    v /= np.linalg.norm(v)
                    lookup[pid] = v
                    pid_counter += 1

        df = _build_assignments_df(records)
        if shuffle:
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        return df, lookup

    def test_future_noise_invariance(self):
        df_clean, lookup_clean = self._build_scenario(0, extra_future_noise=False)
        df_noisy, lookup_noisy = self._build_scenario(6, extra_future_noise=True)
        full_lookup = {**lookup_clean, **lookup_noisy}

        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        tl1, _, cents1 = NarrativeTracker(cfg).track_all_windows(df_clean, lookup_clean)
        tl2, _, cents2 = NarrativeTracker(cfg).track_all_windows(df_noisy, full_lookup)

        cutoff = _iso("2018-01-29T00:00:00Z")
        past1 = tl1[tl1["window_end"] <= cutoff]
        past2 = tl2[tl2["window_end"] <= cutoff]
        _assert_timelines_identical(past1, past2)

        past_cents1 = [
            c for c in cents1 if pd.Timestamp(c["window_end_utc"]).tz_convert("UTC") <= cutoff
        ]
        past_cents2 = [
            c for c in cents2 if pd.Timestamp(c["window_end_utc"]).tz_convert("UTC") <= cutoff
        ]
        _assert_centroids_identical(past_cents1, past_cents2)

    def test_row_shuffle_invariance(self):
        df_ordered, lookup = self._build_scenario(0, extra_future_noise=False, shuffle=False)
        df_shuffled, _ = self._build_scenario(0, extra_future_noise=False, shuffle=True)

        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        tl1, _, cents1 = NarrativeTracker(cfg).track_all_windows(df_ordered, lookup)
        tl2, _, cents2 = NarrativeTracker(cfg).track_all_windows(df_shuffled, lookup)

        _assert_centroids_identical(cents1, cents2)
        _assert_timelines_identical(tl1, tl2)

    def test_timestamp_at_window_end_raises(self):
        """A post with timestamp == window_end must raise ValueError."""
        records = [
            dict(
                post_id="bad_ts",
                timestamp="2018-01-08T00:00:00Z",
                window_start="2018-01-01T00:00:00Z",
                window_end="2018-01-08T00:00:00Z",
                narrative_id="CAND_001",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=0,
            )
        ]
        lookup = {"bad_ts": _make_unit_vector(0)}
        df = _build_assignments_df(records)
        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        with pytest.raises(ValueError):
            NarrativeTracker(cfg).track_all_windows(df, lookup)

    def test_timestamp_after_window_end_raises(self):
        """A post with timestamp strictly after window_end must raise ValueError."""
        records = [
            dict(
                post_id="bad_ts2",
                timestamp="2018-01-09T00:00:00Z",
                window_start="2018-01-01T00:00:00Z",
                window_end="2018-01-08T00:00:00Z",
                narrative_id="CAND_001",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=0,
            )
        ]
        lookup = {"bad_ts2": _make_unit_vector(0)}
        df = _build_assignments_df(records)
        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        with pytest.raises(ValueError):
            NarrativeTracker(cfg).track_all_windows(df, lookup)


# Denominator: window_total_valid_posts


class TestWindowTotalValidPosts:
    """window_total_valid_posts must include ASSIGNED + NOISE + UNASSIGNED
    where is_usable_text=True, and exclude is_usable_text=False.
    """

    def test_denominator_includes_all_usable_statuses(self):
        ws, we = "2018-01-01T00:00:00Z", "2018-01-08T00:00:00Z"
        records = [
            dict(
                post_id="a1",
                timestamp="2018-01-02T12:00:00Z",
                window_start=ws,
                window_end=we,
                narrative_id="CAND_001",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=0,
            ),
            dict(
                post_id="n1",
                timestamp="2018-01-03T12:00:00Z",
                window_start=ws,
                window_end=we,
                narrative_id=None,
                assignment_status="NOISE",
                is_usable_text=True,
                local_cluster_id=-1,
            ),
            dict(
                post_id="u1",
                timestamp="2018-01-04T12:00:00Z",
                window_start=ws,
                window_end=we,
                narrative_id=None,
                assignment_status="UNASSIGNED",
                is_usable_text=True,
                local_cluster_id=-1,
            ),
            dict(
                post_id="x1",
                timestamp="2018-01-05T12:00:00Z",
                window_start=ws,
                window_end=we,
                narrative_id="CAND_001",
                assignment_status="ASSIGNED",
                is_usable_text=False,
                local_cluster_id=0,
            ),
            dict(
                post_id="x2",
                timestamp="2018-01-06T12:00:00Z",
                window_start=ws,
                window_end=we,
                narrative_id=None,
                assignment_status="UNASSIGNED",
                is_usable_text=False,
                local_cluster_id=-1,
            ),
        ]
        lookup = {
            "a1": _make_unit_vector(200),
            "n1": _make_unit_vector(201),
            "u1": _make_unit_vector(202),
            "x1": _make_unit_vector(203),
        }
        df = _build_assignments_df(records)
        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        timeline, _, _ = NarrativeTracker(cfg).track_all_windows(df, lookup)
        row = timeline[timeline["narrative_id"] == "CAND_001"].iloc[0]
        assert int(row["window_total_valid_posts"]) == 3
        assert int(row["current_window_posts"]) == 1
        assert float(row["current_window_share"]) == pytest.approx(1.0 / 3.0, abs=1e-6)


# Lifecycle transitions


class TestLifecycleTransitions:
    """BIRTH -> ACTIVE -> INACTIVE -> DORMANT -> REACTIVATED and max silent peak."""

    @pytest.fixture()
    def lifecycle_timeline(self):
        """Scenario spanning 20 weeks:
        - Weeks 0, 1: CAND_001 active (BIRTH, ACTIVE)
        - Weeks 2..14: 13 consecutive silent weeks (exceeds dormant_threshold_weeks=12)
        - Week 15: CAND_001 active -> REACTIVATED
        - Weeks 16..17: 2 more silent weeks (max_consecutive_silent_weeks must stay 13)
        - Week 18..19: CAND_001 active again -> ACTIVE
        """
        records = []
        lookup = {}
        pid_counter = 0
        weeks = _week_range("2018-01-01T00:00:00Z", 20)

        active_weeks = {0, 1, 15, 18, 19}
        for i, (ws, we) in enumerate(weeks):
            if i in active_weeks:
                for j in range(3):
                    pid = f"p_{pid_counter:06d}"
                    ts = ws + pd.Timedelta(hours=12 * (j + 1))
                    records.append(
                        dict(
                            post_id=pid,
                            timestamp=str(ts),
                            window_start=str(ws),
                            window_end=str(we),
                            narrative_id="CAND_001",
                            assignment_status="ASSIGNED",
                            is_usable_text=True,
                            local_cluster_id=0,
                        )
                    )
                    lookup[pid] = _make_unit_vector(pid_counter)
                    pid_counter += 1

            pid = f"filler_{pid_counter:06d}"
            ts = ws + pd.Timedelta(hours=6)
            records.append(
                dict(
                    post_id=pid,
                    timestamp=str(ts),
                    window_start=str(ws),
                    window_end=str(we),
                    narrative_id="CAND_099",
                    assignment_status="ASSIGNED",
                    is_usable_text=True,
                    local_cluster_id=5,
                )
            )
            lookup[pid] = _make_unit_vector(pid_counter + 1000)
            pid_counter += 1

        df = _build_assignments_df(records)
        cfg = TrackingConfig(
            embedding_dim=DIM, random_state=42, dormant_threshold_weeks=12
        )
        timeline, summary, _ = NarrativeTracker(cfg).track_all_windows(df, lookup)
        return timeline, summary

    def test_full_lifecycle_transitions(
        self, lifecycle_timeline
    ):
        timeline, _ = lifecycle_timeline
        cand = timeline[timeline["narrative_id"] == "CAND_001"].sort_values("window_end")
        statuses = cand["lifecycle_status"].tolist()
        assert statuses[0] == LifecycleStatus.BIRTH.value
        assert statuses[1] == LifecycleStatus.ACTIVE.value
        assert LifecycleStatus.INACTIVE.value in statuses[2:14]
        assert statuses[14] == LifecycleStatus.DORMANT.value
        assert statuses[15] == LifecycleStatus.REACTIVATED.value
        assert statuses[16] == LifecycleStatus.INACTIVE.value
        assert statuses[18] in (
            LifecycleStatus.ACTIVE.value,
            LifecycleStatus.REACTIVATED.value,
        )
        assert statuses[19] == LifecycleStatus.ACTIVE.value

    def test_max_silent_weeks_peak(
        self, lifecycle_timeline
    ):
        timeline, summary = lifecycle_timeline
        cand_summary = summary[summary["narrative_id"] == "CAND_001"]
        assert len(cand_summary) == 1
        assert int(cand_summary["max_consecutive_silent_weeks"].iloc[0]) == 13

        for nid in timeline["narrative_id"].unique():
            nid_tl = timeline[timeline["narrative_id"] == nid]
            running_max = int(nid_tl["consecutive_silent_weeks"].max())
            nid_sum = summary[summary["narrative_id"] == nid]
            assert len(nid_sum) == 1
            stored_max = int(nid_sum["max_consecutive_silent_weeks"].iloc[0])
            assert stored_max == running_max


# Matching, 4-tier tie-breaking, split/merge flags, and relink mode


class TestMatchClustersToRegistry:
    """Test 4-tier tie-breaking, split/merge flags, and [0.75, 0.85) grey zone."""

    @staticmethod
    def _make_entity(
        nid: str,
        centroid: np.ndarray,
        first_seen: str = "2018-01-01T00:00:00Z",
        post_count: int = 10,
    ) -> NarrativeEntity:
        return NarrativeEntity(
            narrative_id=nid,
            birth_window_start=_iso("2018-01-01T00:00:00Z"),
            birth_window_end=_iso("2018-01-08T00:00:00Z"),
            first_seen_utc=_iso(first_seen),
            last_active_utc=_iso("2018-02-01T00:00:00Z"),
            current_centroid=centroid,
            lifetime_post_count=post_count,
            active_weeks_count=4,
            status=LifecycleStatus.ACTIVE,
        )

    def test_tier1_highest_similarity_wins(self):
        base, ortho = _make_orthogonal_unit_pair(300)
        q = _make_vector_at_cosine(base, ortho, 0.96)
        other = _make_vector_at_cosine(base, -ortho, 0.98)
        assert float(np.dot(q, base)) > float(np.dot(q, other)) >= 0.85

        entities = [
            self._make_entity("CAND_001", base, first_seen="2018-02-01T00:00:00Z"),
            self._make_entity("CAND_002", other, first_seen="2018-01-01T00:00:00Z"),
        ]
        c2n, _, _, _ = match_clusters_to_registry(
            current_centroids=np.array([q]),
            current_cluster_ids=[0],
            registry_entities=entities,
            threshold=0.85,
            split_merge_threshold=0.75,
        )
        assert c2n[0] == "CAND_001"

    def test_tier2_tiebreak_first_seen_asc(self):
        centroid = _make_unit_vector(310)
        e_later = self._make_entity(
            "CAND_A", centroid, first_seen="2018-02-01T00:00:00Z", post_count=999
        )
        e_earlier = self._make_entity(
            "CAND_B", centroid, first_seen="2018-01-01T00:00:00Z", post_count=1
        )
        c2n, _, _, _ = match_clusters_to_registry(
            current_centroids=np.array([centroid]),
            current_cluster_ids=[0],
            registry_entities=[e_later, e_earlier],
            threshold=0.85,
        )
        assert c2n[0] == "CAND_B"

    def test_tier3_tiebreak_post_count_desc(self):
        centroid = _make_unit_vector(311)
        e_low = self._make_entity(
            "CAND_A", centroid, first_seen="2018-01-01T00:00:00Z", post_count=10
        )
        e_high = self._make_entity(
            "CAND_B", centroid, first_seen="2018-01-01T00:00:00Z", post_count=50
        )
        c2n, _, _, _ = match_clusters_to_registry(
            current_centroids=np.array([centroid]),
            current_cluster_ids=[0],
            registry_entities=[e_low, e_high],
            threshold=0.85,
        )
        assert c2n[0] == "CAND_B"

    def test_tier4_tiebreak_narrative_id_asc(self):
        centroid = _make_unit_vector(312)
        e_z = self._make_entity(
            "CAND_002", centroid, first_seen="2018-01-01T00:00:00Z", post_count=20
        )
        e_a = self._make_entity(
            "CAND_001", centroid, first_seen="2018-01-01T00:00:00Z", post_count=20
        )
        c2n, _, _, _ = match_clusters_to_registry(
            current_centroids=np.array([centroid]),
            current_cluster_ids=[0],
            registry_entities=[e_z, e_a],
            threshold=0.85,
        )
        assert c2n[0] == "CAND_001"

    def test_match_invariant_to_entity_order(self):
        base, ortho = _make_orthogonal_unit_pair(330)
        e1 = self._make_entity("CAND_X", base, first_seen="2018-01-01T00:00:00Z", post_count=50)
        e2 = self._make_entity(
            "CAND_Y",
            _make_vector_at_cosine(base, ortho, 0.90),
            first_seen="2018-02-01T00:00:00Z",
            post_count=20,
        )
        query = _make_vector_at_cosine(base, ortho, 0.98)

        c2n_ab, _, flags_ab, _ = match_clusters_to_registry(
            current_centroids=np.array([query]),
            current_cluster_ids=[0],
            registry_entities=[e1, e2],
            threshold=0.85,
            split_merge_threshold=0.75,
        )
        c2n_ba, _, flags_ba, _ = match_clusters_to_registry(
            current_centroids=np.array([query]),
            current_cluster_ids=[0],
            registry_entities=[e2, e1],
            threshold=0.85,
            split_merge_threshold=0.75,
        )
        assert c2n_ab == c2n_ba
        assert flags_ab == flags_ba

    def test_split_candidate_flag(self):
        """Two current clusters >= split_merge_threshold to the same entity trigger SPLIT_CANDIDATE."""
        base, ortho = _make_orthogonal_unit_pair(340)
        c0 = _make_vector_at_cosine(base, ortho, 0.95)
        c1 = _make_vector_at_cosine(base, -ortho, 0.80)
        entity = self._make_entity("CAND_001", base)

        c2n, _, flags, _ = match_clusters_to_registry(
            current_centroids=np.array([c0, c1]),
            current_cluster_ids=[0, 1],
            registry_entities=[entity],
            threshold=0.85,
            split_merge_threshold=0.75,
        )
        assert c2n[0] == "CAND_001"
        assert 1 not in c2n
        assert flags[0] in (SplitMergeFlag.SPLIT_CANDIDATE, SplitMergeFlag.SPLIT_AND_MERGE)
        assert flags[1] in (SplitMergeFlag.SPLIT_CANDIDATE, SplitMergeFlag.SPLIT_AND_MERGE)

    def test_merge_candidate_flag(self):
        """One cluster >= split_merge_threshold to two entities triggers MERGE_CANDIDATE."""
        base, ortho = _make_orthogonal_unit_pair(350)
        e1_vec = base
        e2_vec = _make_vector_at_cosine(base, ortho, 0.82)
        query = _make_vector_at_cosine(base, ortho, 0.96)

        # Verify query has >= 0.75 similarity to e2_vec as well
        assert float(np.dot(query, e2_vec)) >= 0.75

        e1 = self._make_entity("CAND_001", e1_vec, first_seen="2018-01-01T00:00:00Z")
        e2 = self._make_entity("CAND_002", e2_vec, first_seen="2018-01-08T00:00:00Z")

        c2n, _, flags, _ = match_clusters_to_registry(
            current_centroids=np.array([query]),
            current_cluster_ids=[0],
            registry_entities=[e1, e2],
            threshold=0.85,
            split_merge_threshold=0.75,
        )
        assert c2n[0] == "CAND_001"
        assert flags[0] in (SplitMergeFlag.MERGE_CANDIDATE, SplitMergeFlag.SPLIT_AND_MERGE)

    def test_grey_zone_near_miss_logged_once(self):
        """A cluster in [0.75, 0.85) is not linked and logs one near-miss event."""
        base, ortho = _make_orthogonal_unit_pair(360)
        grey_vec = _make_vector_at_cosine(base, ortho, 0.80)
        sim = float(np.dot(base, grey_vec))
        assert 0.75 <= sim < 0.85

        entity = self._make_entity("CAND_001", base)
        c2n, _, flags, events = match_clusters_to_registry(
            current_centroids=np.array([grey_vec]),
            current_cluster_ids=[0],
            registry_entities=[entity],
            threshold=0.85,
            split_merge_threshold=0.75,
        )
        assert 0 not in c2n
        assert len(events) == 1, f"Expected exactly 1 audit event (no blind spot / double-log), got {events}"


class TestRelinkLocalClustersMode:
    """Tests for tracking_mode='relink_local_clusters'."""

    def test_relink_continuity_and_shuffle(self):
        u0, ortho = _make_orthogonal_unit_pair(400)
        u1 = _make_unit_vector(401)
        # Ensure u0 and u1 are distinct (< 0.75)
        assert abs(float(np.dot(u0, u1))) < 0.75

        weeks = _week_range("2018-01-01T00:00:00Z", 2)
        (ws1, we1), (ws2, we2) = weeks

        records = [
            # Week 1: local cluster 0 and local cluster 1
            dict(
                post_id="r1",
                timestamp="2018-01-02T10:00:00Z",
                window_start=ws1,
                window_end=we1,
                narrative_id="RAW_0",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=0,
            ),
            dict(
                post_id="r2",
                timestamp="2018-01-03T10:00:00Z",
                window_start=ws1,
                window_end=we1,
                narrative_id="RAW_1",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=1,
            ),
            # Week 2: local cluster 5 continues cluster 0 (sim 0.95)
            dict(
                post_id="r3",
                timestamp="2018-01-09T10:00:00Z",
                window_start=ws2,
                window_end=we2,
                narrative_id="RAW_5",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=5,
            ),
            # Week 2: local cluster 8 is in [0.75, 0.85) relative to u0 -> births new narrative
            dict(
                post_id="r4",
                timestamp="2018-01-10T10:00:00Z",
                window_start=ws2,
                window_end=we2,
                narrative_id="RAW_8",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=8,
            ),
        ]
        c8_vec = _make_vector_at_cosine(u0, ortho, 0.80)
        lookup = {
            "r1": u0,
            "r2": u1,
            "r3": _make_vector_at_cosine(u0, -ortho, 0.95),
            "r4": c8_vec,
        }
        df = _build_assignments_df(records)
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        cfg = TrackingConfig(
            embedding_dim=DIM,
            random_state=42,
            tracking_mode="relink_local_clusters",
            alignment_threshold=0.85,
            split_merge_threshold=0.75,
        )
        tl1, sum1, cents1 = NarrativeTracker(cfg).track_all_windows(df, lookup)
        tl2, sum2, cents2 = NarrativeTracker(cfg).track_all_windows(df_shuffled, lookup)

        _assert_timelines_identical(tl1, tl2)
        _assert_centroids_identical(cents1, cents2)
        assert len(sum1) == 3, f"Expected 3 sequential narratives, got {len(sum1)}"

        # Week 2 should have one ACTIVE (relinked from week 1), one INACTIVE (u1), and one BIRTH (r4)
        w2_rows = tl1[tl1["window_end"] == we2]
        w2_statuses = sorted(w2_rows["lifecycle_status"].tolist())
        assert w2_statuses == sorted([
            LifecycleStatus.ACTIVE.value,
            LifecycleStatus.INACTIVE.value,
            LifecycleStatus.BIRTH.value,
        ])


# Persistence: save/load centroids, Parquet, cutoff loader


class TestPersistenceAndCutoff:
    """save_centroids_npz, save_registry_artifacts, load_frozen_registry_at_cutoff."""

    def test_save_centroids_npz_allow_pickle_false(self):
        vecs = _make_random_vectors(5, seed=42)
        window_ends = [f"2018-01-{8 + i * 7:02d}T00:00:00Z" for i in range(5)]
        nids = [f"CAND_{i:03d}" for i in range(5)]
        centroid_records = [
            {"window_end_utc": we, "narrative_id": nid, "centroid": vec}
            for we, nid, vec in zip(window_ends, nids, vecs)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_centroids.npz"
            save_centroids_npz(centroid_records, out_path)
            with np.load(str(out_path), allow_pickle=False) as npz:
                loaded_vecs = npz["centroids"]
                loaded_wends = npz["window_end_utc"]
                loaded_nids = npz["narrative_id"]
            assert loaded_vecs.shape == (5, DIM)
            assert len(loaded_wends) == 5
            assert len(loaded_nids) == 5

    def test_save_centroids_npz_rejects_zero_norm(self):
        centroid_records = [
            {
                "window_end_utc": "2018-01-08T00:00:00Z",
                "narrative_id": "CAND_000",
                "centroid": np.zeros(DIM),
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_bad.npz"
            with pytest.raises(ValueError):
                save_centroids_npz(centroid_records, out_path)

    def test_save_centroids_npz_rejects_nan(self):
        centroid_records = [
            {
                "window_end_utc": "2018-01-08T00:00:00Z",
                "narrative_id": "CAND_000",
                "centroid": np.full(DIM, np.nan),
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_nan.npz"
            with pytest.raises(ValueError):
                save_centroids_npz(centroid_records, out_path)

    def test_parquet_npz_row_alignment(self):
        weeks = _week_range("2018-01-01T00:00:00Z", 4)
        records = []
        lookup = {}
        pid_counter = 0
        for ws, we in weeks:
            for k in range(3):
                pid = f"ord_{pid_counter:06d}"
                records.append(
                    dict(
                        post_id=pid,
                        timestamp=str(ws + pd.Timedelta(hours=8)),
                        window_start=str(ws),
                        window_end=str(we),
                        narrative_id=f"CAND_{k:03d}",
                        assignment_status="ASSIGNED",
                        is_usable_text=True,
                        local_cluster_id=k,
                    )
                )
                lookup[pid] = _make_unit_vector(pid_counter + 450)
                pid_counter += 1

        df = _build_assignments_df(records)
        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        timeline, summary, centroids = NarrativeTracker(cfg).track_all_windows(df, lookup)

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "reg.parquet"
            sum_path = Path(tmpdir) / "sum.parquet"
            man_path = Path(tmpdir) / "manifest.json"
            cent_path = Path(tmpdir) / "cent.npz"

            save_registry_artifacts(
                timeline,
                summary,
                {"status": "test_ok"},
                registry_parquet_path=reg_path,
                summary_parquet_path=sum_path,
                manifest_json_path=man_path,
            )
            save_centroids_npz(centroids, cent_path)

            reg_disk = pd.read_parquet(reg_path)
            with np.load(str(cent_path), allow_pickle=False) as npz:
                c_wends = npz["window_end_utc"].astype(str)
                c_nids = npz["narrative_id"].astype(str)
                c_vecs = npz["centroids"]

            reg_wends = np.array(
                [pd.Timestamp(ts).tz_convert("UTC").isoformat() for ts in reg_disk["window_end"]],
                dtype=str,
            )
            reg_nids = reg_disk["narrative_id"].astype(str).to_numpy()
            assert len(c_vecs) == len(reg_disk)
            assert np.array_equal(c_wends, reg_wends)
            assert np.array_equal(c_nids, reg_nids)
            assert json.loads(man_path.read_text(encoding="utf-8"))["status"] == "test_ok"

    def test_cutoff_timezone_conversion(self):
        weeks = _week_range("2018-01-01T00:00:00Z", 6)
        records = []
        lookup = {}
        pid_counter = 0
        for ws, we in weeks:
            for j in range(3):
                pid = f"tz_{pid_counter:06d}"
                ts = ws + pd.Timedelta(hours=12 * (j + 1))
                records.append(
                    dict(
                        post_id=pid,
                        timestamp=str(ts),
                        window_start=str(ws),
                        window_end=str(we),
                        narrative_id="CAND_001",
                        assignment_status="ASSIGNED",
                        is_usable_text=True,
                        local_cluster_id=0,
                    )
                )
                lookup[pid] = _make_unit_vector(pid_counter + 500)
                pid_counter += 1

        df = _build_assignments_df(records)
        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        timeline, summary, centroids = NarrativeTracker(cfg).track_all_windows(df, lookup)

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "reg.parquet"
            cent_path = Path(tmpdir) / "cent.npz"
            save_registry_artifacts(timeline, summary, None, registry_parquet_path=reg_path)
            save_centroids_npz(centroids, cent_path)

            cutoff_utc = _iso("2018-02-05T00:00:00Z")
            cutoff_p7 = cutoff_utc.tz_convert("Asia/Ho_Chi_Minh")

            df_utc, cents_utc = load_frozen_registry_at_cutoff(reg_path, cent_path, cutoff_utc)
            df_p7, cents_p7 = load_frozen_registry_at_cutoff(reg_path, cent_path, cutoff_p7)

            assert len(df_utc) == len(df_p7) > 0
            pd.testing.assert_frame_equal(
                df_utc.reset_index(drop=True), df_p7.reset_index(drop=True)
            )
            np.testing.assert_allclose(cents_utc, cents_p7, atol=1e-7)

    def test_cutoff_excludes_narratives_born_after(self):
        weeks = _week_range("2018-01-01T00:00:00Z", 8)
        records = []
        lookup = {}
        pid_counter = 0
        for i, (ws, we) in enumerate(weeks):
            pid = f"cut_{pid_counter:06d}"
            records.append(
                dict(
                    post_id=pid,
                    timestamp=str(ws + pd.Timedelta(hours=12)),
                    window_start=str(ws),
                    window_end=str(we),
                    narrative_id="CAND_001",
                    assignment_status="ASSIGNED",
                    is_usable_text=True,
                    local_cluster_id=0,
                )
            )
            lookup[pid] = _make_unit_vector(pid_counter + 600)
            pid_counter += 1
            if i >= 4:
                pid2 = f"cut_{pid_counter:06d}"
                records.append(
                    dict(
                        post_id=pid2,
                        timestamp=str(ws + pd.Timedelta(hours=18)),
                        window_start=str(ws),
                        window_end=str(we),
                        narrative_id="CAND_002",
                        assignment_status="ASSIGNED",
                        is_usable_text=True,
                        local_cluster_id=1,
                    )
                )
                lookup[pid2] = _make_unit_vector(pid_counter + 700)
                pid_counter += 1

        df = _build_assignments_df(records)
        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        timeline, summary, centroids = NarrativeTracker(cfg).track_all_windows(df, lookup)

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "reg.parquet"
            cent_path = Path(tmpdir) / "cent.npz"
            save_registry_artifacts(timeline, summary, None, registry_parquet_path=reg_path)
            save_centroids_npz(centroids, cent_path)

            cutoff = weeks[2][1]
            filtered_df, filtered_cents = load_frozen_registry_at_cutoff(
                reg_path, cent_path, cutoff
            )
            nids_returned = set(filtered_df["narrative_id"].astype(str).unique())
            assert "CAND_001" in nids_returned
            assert "CAND_002" not in nids_returned
            assert len(filtered_df) == len(filtered_cents)
            assert (pd.to_datetime(filtered_df["first_seen_utc"], utc=True) < cutoff).all()
            assert (pd.to_datetime(filtered_df["last_active_utc"], utc=True) < cutoff).all()


# Review pair CSV export


class TestExportTrackingReviewSample:
    """Tests for export_tracking_review_sample."""

    def test_cutoff_and_strata_dedup(self):
        audit_events = []
        strata = [
            "linked_active",
            "potential_merge_or_split",
            "unlinked_near_miss",
        ]
        # Shared pair appearing in multiple strata before cutoff to test cross-strata dedup
        for s in strata:
            audit_events.append(
                {
                    "window_end_utc": "2018-03-05T00:00:00Z",
                    "narrative_id_1": "CAND_DUP_1",
                    "narrative_id_2": "CAND_DUP_2",
                    "similarity": 0.80,
                    "event_type": s,
                }
            )
        # Distinct pairs across all 3 strata before cutoff
        base_date = pd.Timestamp("2018-01-08T00:00:00Z")
        for idx in range(20):
            we = (base_date + pd.Timedelta(weeks=idx)).isoformat()
            for s_idx, s in enumerate(strata):
                audit_events.append(
                    {
                        "window_end_utc": we,
                        "narrative_id_1": f"CAND_{s_idx}_{idx:02d}_A",
                        "narrative_id_2": f"CAND_{s_idx}_{idx:02d}_B",
                        "similarity": 0.78 + 0.01 * s_idx,
                        "event_type": s,
                    }
                )
        # Events after dev_cutoff_utc (must be excluded)
        for idx in range(15):
            we = (pd.Timestamp("2019-06-03T00:00:00Z") + pd.Timedelta(weeks=idx)).isoformat()
            audit_events.append(
                {
                    "window_end_utc": we,
                    "narrative_id_1": f"FUTURE_{idx:02d}_A",
                    "narrative_id_2": f"FUTURE_{idx:02d}_B",
                    "similarity": 0.90,
                    "event_type": "linked_active",
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            export_tracking_review_sample(
                audit_events,
                csv_path,
                target_rows=48,
                min_rows=30,
                random_state=42,
                dev_cutoff_utc="2019-01-01T00:00:00Z",
            )
            result_df = pd.read_csv(csv_path)
            assert len(result_df) >= 30
            w_ends = pd.to_datetime(result_df["window_end_utc"], utc=True)
            assert (w_ends <= pd.Timestamp("2019-01-01T00:00:00Z")).all()
            assert (
                result_df.duplicated(
                    subset=["window_end_utc", "narrative_id_1", "narrative_id_2"]
                ).sum()
                == 0
            )
            assert set(result_df["event_type"].unique()) == set(strata)

    def test_no_birth_renamed_to_linked_active(self):
        """birth events must not be coerced into linked_active."""
        audit_events = []
        # Include 20 birth events with recognizable IDs
        for i in range(20):
            audit_events.append(
                {
                    "window_end_utc": "2018-06-04T00:00:00Z",
                    "narrative_id_1": f"BIRTH_ONLY_{i:02d}",
                    "narrative_id_2": f"BIRTH_PAIR_{i:02d}",
                    "similarity": 0.90,
                    "event_type": "birth",
                }
            )
        # Also include valid 3-strata events so export succeeds and writes CSV
        strata = [
            "linked_active",
            "potential_merge_or_split",
            "unlinked_near_miss",
        ]
        for idx in range(15):
            we = (pd.Timestamp("2018-01-08T00:00:00Z") + pd.Timedelta(weeks=idx)).isoformat()
            for s_idx, s in enumerate(strata):
                audit_events.append(
                    {
                        "window_end_utc": we,
                        "narrative_id_1": f"VALID_{s_idx}_{idx:02d}_A",
                        "narrative_id_2": f"VALID_{s_idx}_{idx:02d}_B",
                        "similarity": 0.80,
                        "event_type": s,
                    }
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample_birth.csv"
            export_tracking_review_sample(
                audit_events,
                csv_path,
                target_rows=36,
                min_rows=30,
                random_state=42,
                dev_cutoff_utc="2019-01-01T00:00:00Z",
            )
            result_df = pd.read_csv(csv_path)
            assert not result_df["narrative_id_1"].astype(str).str.startswith("BIRTH_ONLY_").any()
            assert not result_df["narrative_id_2"].astype(str).str.startswith("BIRTH_PAIR_").any()


# Timeline schema


class TestTimelineSchema:
    """Schema and primary-key checks for track_all_windows output."""

    EXPECTED_COLUMNS = {
        "window_start",
        "window_end",
        "narrative_id",
        "lifecycle_status",
        "current_window_posts",
        "window_total_valid_posts",
        "current_window_share",
        "cumulative_posts",
        "age_weeks",
        "consecutive_silent_weeks",
        "first_seen_utc",
        "last_active_utc",
        "centroid_drift",
        "similarity_to_previous_centroid",
        "intra_window_dispersion",
        "is_in_candidate_set",
        "split_merge_flag",
    }

    def test_timeline_has_all_17_columns(self):
        ws, we = "2018-01-01T00:00:00Z", "2018-01-08T00:00:00Z"
        records = [
            dict(
                post_id="schema_p1",
                timestamp="2018-01-02T12:00:00Z",
                window_start=ws,
                window_end=we,
                narrative_id="CAND_001",
                assignment_status="ASSIGNED",
                is_usable_text=True,
                local_cluster_id=0,
            ),
        ]
        lookup = {"schema_p1": _make_unit_vector(900)}
        df = _build_assignments_df(records)
        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        timeline, _, _ = NarrativeTracker(cfg).track_all_windows(df, lookup)
        actual_cols = set(timeline.columns.tolist())
        assert actual_cols == self.EXPECTED_COLUMNS

    def test_timeline_primary_key_uniqueness(self):
        weeks = _week_range("2018-01-01T00:00:00Z", 4)
        records = []
        lookup = {}
        pid_counter = 0
        for ws, we in weeks:
            for nid_num in range(3):
                pid = f"pk_{pid_counter:06d}"
                records.append(
                    dict(
                        post_id=pid,
                        timestamp=str(ws + pd.Timedelta(hours=12)),
                        window_start=str(ws),
                        window_end=str(we),
                        narrative_id=f"CAND_{nid_num:03d}",
                        assignment_status="ASSIGNED",
                        is_usable_text=True,
                        local_cluster_id=nid_num,
                    )
                )
                lookup[pid] = _make_unit_vector(pid_counter + 950)
                pid_counter += 1

        df = _build_assignments_df(records)
        cfg = TrackingConfig(embedding_dim=DIM, random_state=42)
        timeline, _, _ = NarrativeTracker(cfg).track_all_windows(df, lookup)
        dups = timeline.duplicated(subset=["window_end", "narrative_id"]).sum()
        assert dups == 0
