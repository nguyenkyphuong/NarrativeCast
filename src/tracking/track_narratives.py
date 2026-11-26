"""Causal temporal narrative tracking and stable ID registry engine.

Tracks candidate narratives chronologically across continuous UTC weekly windows
without future leakage and freezes L2-normalized centroids at each cutoff.

Outputs:
    - data/interim/narrative_registry.parquet: 17-column weekly timeline.
    - data/interim/narrative_registry_centroids.npz: 384-D unit centroids (allow_pickle=False).
    - data/interim/narrative_registry_manifest.json: Run metadata and audit counts.
    - data/interim/narrative_registry_summary.parquet: Full-span summary (report only).
    - reports/results/tracking_pairs_sample.csv: Dev-period pair sample (<= 2019-01-01).

Notes for downstream feature and label pipelines:
    1. Do not join `narrative_registry_summary.parquet` into training features;
       it aggregates full 2014-2020 stats. Filter `narrative_registry.parquet`
       at `window_end <= cutoff_utc` for features.
    2. Distinguish NaN from 0.0 instead of calling `df.fillna(0)` across the table:
       - `current_window_share` is NaN iff the week has 0 valid posts (empty week),
         and 0.0 when the narrative is absent in a non-empty week.
       - `intra_window_dispersion` and `similarity_to_previous_centroid` are NaN
         iff `current_window_posts == 0`.
    3. At `lifecycle_status == "birth"` (`age_weeks == 0`),
       `similarity_to_previous_centroid` is initialized to 1.0 by convention;
       handle birth rows explicitly so models do not treat new narratives as
       mature stability.
    4. Always load cutoff snapshots via `load_frozen_registry_at_cutoff()` rather
       than manual `.npz` row slicing.

Uses calendar and scope settings aligned with configs/forecast.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("NarrativeCast.NarrativeTracking")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s"
    )

DEFAULT_ASSIGNMENTS_PARQUET = Path("data/interim/narrative_assignments.parquet")
DEFAULT_EMBEDDINGS_NPZ = Path("data/interim/scope_expanded_v1_embeddings.npz")
DEFAULT_REGISTRY_PARQUET = Path("data/interim/narrative_registry.parquet")
DEFAULT_SUMMARY_PARQUET = Path("data/interim/narrative_registry_summary.parquet")
DEFAULT_CENTROIDS_NPZ = Path("data/interim/narrative_registry_centroids.npz")
DEFAULT_MANIFEST_JSON = Path("data/interim/narrative_registry_manifest.json")
DEFAULT_PAIRS_SAMPLE_CSV = Path("reports/results/tracking_pairs_sample.csv")
REGISTRY_METADATA_KEY = b"narrativecast_registry_metadata"
_VECTOR_KEYS = ("vectors", "embeddings")
_ID_KEYS = ("post_ids", "ids")


class LifecycleStatus(str, Enum):
    """Lifecycle states of a tracked narrative across temporal windows."""

    BIRTH = "birth"
    ACTIVE = "active"
    INACTIVE = "inactive"
    REACTIVATED = "reactivated"
    DORMANT = "dormant"


class SplitMergeFlag(str, Enum):
    """Heuristic diagnostic flags for multi-cluster overlap."""

    NORMAL = "NORMAL"
    SPLIT_CANDIDATE = "SPLIT_CANDIDATE"
    MERGE_CANDIDATE = "MERGE_CANDIDATE"
    SPLIT_AND_MERGE = "SPLIT_AND_MERGE"


@dataclass(frozen=True)
class TrackingConfig:
    """Hyperparameters and calendar settings for narrative tracking."""

    alignment_threshold: float = 0.85
    ema_alpha: float = 0.30
    dormant_threshold_weeks: int = 12
    split_merge_threshold: float = 0.75
    tie_break_strategy: str = "highest_sim_then_oldest_then_id"
    id_prefix: str = "CAND"
    embedding_dim: int = 384
    random_state: int = 42
    calendar_start_utc: Optional[str] = None
    calendar_end_utc: Optional[str] = None
    tracking_mode: str = "registry_from_assignments"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NarrativeEntity:
    """In-memory state container for a single tracked narrative."""

    narrative_id: str
    birth_window_start: pd.Timestamp
    birth_window_end: pd.Timestamp
    first_seen_utc: pd.Timestamp
    last_active_utc: pd.Timestamp
    current_centroid: np.ndarray
    previous_centroid: Optional[np.ndarray] = None
    lifetime_post_count: int = 0
    active_weeks_count: int = 0
    consecutive_silent_weeks: int = 0
    max_consecutive_silent_weeks: int = 0
    status: LifecycleStatus = LifecycleStatus.BIRTH
    latest_drift: float = 0.0
    latest_similarity: float = 1.0


def _to_utc(ts: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(ts)
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


def _l2_normalize_rows(matrix: np.ndarray, context: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{context}: expected finite non-empty 2D array, got {arr.shape}.")
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{context}: matrix contains zero-norm row(s) (norm <= 1e-12).")
    return (arr / norms[:, np.newaxis]).astype(np.float32)


def load_embeddings_lookup(
    embeddings_npz_path: Path, required_assigned_ids: Optional[Set[str]] = None
) -> Dict[str, np.ndarray]:
    """Load and L2-normalize post embeddings using allow_pickle=False."""
    if not embeddings_npz_path.exists():
        raise FileNotFoundError(f"Embeddings archive not found: {embeddings_npz_path}")
    with np.load(str(embeddings_npz_path), allow_pickle=False) as data:
        vec_key = next((k for k in _VECTOR_KEYS if k in data.files), None)
        id_key = next((k for k in _ID_KEYS if k in data.files), None)
        if vec_key is None or id_key is None:
            raise ValueError(
                f"Archive {embeddings_npz_path} missing {_VECTOR_KEYS}/{_ID_KEYS}; found {data.files}"
            )
        vectors = data[vec_key].astype(np.float32)
        post_ids_raw = data[id_key]
        is_valid = (
            data["is_valid"].astype(bool)
            if "is_valid" in data.files
            else np.ones(len(post_ids_raw), dtype=bool)
        )
    norms = np.linalg.norm(vectors, axis=1)
    valid = is_valid & np.all(np.isfinite(vectors), axis=1) & (norms > 1e-12)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        raise ValueError(f"Embeddings lookup is empty after validating {embeddings_npz_path}.")
    normed = (vectors[valid_idx] / norms[valid_idx, np.newaxis]).astype(np.float32)
    lookup = {str(post_ids_raw[idx]).strip(): normed[pos] for pos, idx in enumerate(valid_idx)}
    if required_assigned_ids and (missing := required_assigned_ids - set(lookup.keys())):
        raise ValueError(
            f"{len(missing)} ASSIGNED post(s) lack valid embeddings (examples: {sorted(missing)[:5]})."
        )
    logger.info(
        "Loaded embeddings lookup: %d valid vectors from %s", len(lookup), embeddings_npz_path
    )
    return lookup


def compute_normalized_centroid(vectors: np.ndarray) -> np.ndarray:
    """Compute L2-normalized mean centroid; raise ValueError on degenerate/invalid input."""
    arr = np.asarray(vectors)
    if arr.ndim != 2 or arr.shape[0] == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("vectors must be a finite non-empty 2D array.")
    mean_vec = np.mean(arr, axis=0, dtype=np.float64)
    norm = float(np.linalg.norm(mean_vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"Centroid norm ({norm:.2e}) is <= 1e-12 or non-finite.")
    return (mean_vec / norm).astype(np.float32)


def _compute_intra_window_dispersion(
    post_ids: List[str], embeddings_lookup: Dict[str, np.ndarray], batch_centroid: np.ndarray
) -> np.float32:
    vecs = [embeddings_lookup[pid] for pid in post_ids if pid in embeddings_lookup]
    if not vecs:
        return np.float32(np.nan)
    norm_vecs = _l2_normalize_rows(
        np.stack(vecs, axis=0), "_compute_intra_window_dispersion"
    )
    centroid_arr = np.asarray(batch_centroid, dtype=np.float64)
    if centroid_arr.ndim != 1 or centroid_arr.shape[0] != norm_vecs.shape[1]:
        raise ValueError(
            "_compute_intra_window_dispersion: batch_centroid must be a 1D vector "
            f"of length {norm_vecs.shape[1]}, got shape {centroid_arr.shape}."
        )
    centroid_norm = float(np.linalg.norm(centroid_arr))
    if not np.isfinite(centroid_norm) or centroid_norm <= 1e-12:
        raise ValueError(
            "_compute_intra_window_dispersion: batch_centroid has zero/non-finite norm "
            f"({centroid_norm:.2e})."
        )
    if len(vecs) == 1:
        return np.float32(0.0)
    norm_centroid = (centroid_arr / centroid_norm).astype(np.float32)
    cosines = np.clip(norm_vecs @ norm_centroid, -1.0, 1.0)
    return np.float32(1.0 - float(np.mean(cosines, dtype=np.float64)))


def _combine_split_merge_flags(a: SplitMergeFlag, b: SplitMergeFlag) -> SplitMergeFlag:
    if a == SplitMergeFlag.NORMAL:
        return b
    return a if (b == SplitMergeFlag.NORMAL or a == b) else SplitMergeFlag.SPLIT_AND_MERGE


def match_clusters_to_registry(
    current_centroids: np.ndarray,
    current_cluster_ids: Sequence[str],
    registry_entities: Sequence[NarrativeEntity],
    threshold: float = 0.85,
    split_merge_threshold: float = 0.75,
) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, SplitMergeFlag], List[Dict[str, Any]]]:
    """Deterministic greedy 1-to-1 matching with 4-tier tie-breaking (sim desc, first_seen asc, count desc, id asc)."""
    n_curr = len(current_cluster_ids)
    n_reg = len(registry_entities)
    c2n: Dict[str, str] = {}
    match_sims: Dict[str, float] = {}
    sm_flags: Dict[str, SplitMergeFlag] = {
        cid: SplitMergeFlag.NORMAL for cid in current_cluster_ids
    }
    events: List[Dict[str, Any]] = []
    if n_curr == 0 or n_reg == 0:
        return c2n, match_sims, sm_flags, events

    sim_mat = (
        _l2_normalize_rows(current_centroids, "match.current")
        @ _l2_normalize_rows(
            np.stack([e.current_centroid for e in registry_entities], axis=0), "match.registry"
        ).T
    )
    flagged_pairs: Set[Tuple[int, int]] = set()
    for i, cid in enumerate(current_cluster_ids):
        high = [j for j in range(n_reg) if float(sim_mat[i, j]) >= split_merge_threshold]
        if len(high) > 1:
            sm_flags[cid] = SplitMergeFlag.MERGE_CANDIDATE
            flagged_pairs.update((i, j) for j in high)
            events.append(
                {
                    "event_type": "potential_merge",
                    "current_cluster_id": cid,
                    "candidate_narrative_ids": [registry_entities[j].narrative_id for j in high],
                    "similarities": [float(sim_mat[i, j]) for j in high],
                }
            )
    for j, ent in enumerate(registry_entities):
        high = [i for i in range(n_curr) if float(sim_mat[i, j]) >= split_merge_threshold]
        if len(high) > 1:
            flagged_pairs.update((i, j) for i in high)
            for i in high:
                sm_flags[current_cluster_ids[i]] = _combine_split_merge_flags(
                    sm_flags[current_cluster_ids[i]], SplitMergeFlag.SPLIT_CANDIDATE
                )
            events.append(
                {
                    "event_type": "potential_split",
                    "registry_narrative_id": ent.narrative_id,
                    "current_cluster_ids": [current_cluster_ids[i] for i in high],
                    "similarities": [float(sim_mat[i, j]) for i in high],
                }
            )

    candidates: List[Tuple[int, int, float]] = []
    for r in range(n_curr):
        for c in range(n_reg):
            sim = float(sim_mat[r, c])
            if sim >= threshold:
                candidates.append((r, c, sim))
            elif sim >= 0.60 and (r, c) not in flagged_pairs:
                events.append(
                    {
                        "event_type": "unlinked_near_miss",
                        "current_cluster_id": current_cluster_ids[r],
                        "registry_narrative_id": registry_entities[c].narrative_id,
                        "similarity": sim,
                        "threshold": threshold,
                    }
                )

    candidates.sort(
        key=lambda p: (
            -p[2],
            registry_entities[p[1]].first_seen_utc.timestamp()
            if registry_entities[p[1]].first_seen_utc is not None
            else 0.0,
            -registry_entities[p[1]].lifetime_post_count,
            registry_entities[p[1]].narrative_id,
            p[0],
        )
    )
    used_r: Set[int] = set()
    used_c: Set[int] = set()
    for r, c, sim in candidates:
        if r not in used_r and c not in used_c:
            cid = current_cluster_ids[r]
            c2n[cid] = registry_entities[c].narrative_id
            match_sims[cid] = sim
            used_r.add(r)
            used_c.add(c)
    return c2n, match_sims, sm_flags, events


def _build_calendar_week_grid(
    assignments_df: pd.DataFrame, calendar_start_utc: Optional[str], calendar_end_utc: Optional[str]
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if calendar_start_utc and calendar_end_utc:
        start = _to_utc(calendar_start_utc)
        end = _to_utc(calendar_end_utc)
    else:
        start = assignments_df["window_start"].min()
        end = assignments_df["window_end"].max()
    step = pd.Timedelta(days=7)
    weeks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        weeks.append((cursor, cursor + step))
        cursor += step
    return weeks


class NarrativeTracker:
    """Causal, leak-free temporal narrative tracker with stable registry."""

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        self.config = config or TrackingConfig()
        self.registry: Dict[str, NarrativeEntity] = {}
        self.next_id_num: int = 1
        self.audit_log: List[Dict[str, Any]] = []

    def track_all_windows(
        self, assignments_df: pd.DataFrame, embeddings_lookup: Dict[str, np.ndarray]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
        """Process all calendar weeks chronologically and build the narrative registry."""
        logger.info("Starting chronological narrative tracking")
        missing = {
            "post_id",
            "timestamp",
            "window_start",
            "window_end",
            "assignment_status",
            "narrative_id",
        } - set(assignments_df.columns)
        if missing:
            raise KeyError(f"assignments_df missing mandatory columns: {missing}")

        df = assignments_df.copy()
        df["post_id"] = df["post_id"].astype(str).str.strip()
        for col in ("window_start", "window_end", "timestamp"):
            df[col] = pd.to_datetime(df[col], utc=True)
        if not embeddings_lookup:
            raise ValueError("embeddings_lookup must not be empty.")
        df["is_usable_text"] = (
            df["is_usable_text"].fillna(False).astype(bool)
            if "is_usable_text" in df.columns
            else True
        )
        usable_assigned = (df["assignment_status"] == "ASSIGNED") & df["is_usable_text"]
        invalid = [
            pid
            for pid in df.loc[usable_assigned, "post_id"].unique().tolist()
            if pid not in embeddings_lookup
            or np.asarray(embeddings_lookup[pid]).ndim != 1
            or len(embeddings_lookup[pid]) == 0
            or not np.all(np.isfinite(embeddings_lookup[pid]))
            or float(np.linalg.norm(embeddings_lookup[pid])) <= 1e-12
        ]
        if invalid:
            raise ValueError(
                f"{len(invalid)} ASSIGNED post(s) lack valid embeddings (examples: {invalid[:5]})."
            )

        # Pre-group valid denominators and per-window batches once before iterating
        valid_counts = {
            (ws, we): int(g["is_usable_text"].sum())
            for (ws, we), g in df.groupby(["window_start", "window_end"], sort=False)
        }
        pregrouped = self._pregroup_assigned_batches(df)
        week_grid = _build_calendar_week_grid(
            df, self.config.calendar_start_utc, self.config.calendar_end_utc
        )
        if week_grid:
            logger.info(
                "Processing %d calendar weeks (%s to %s)",
                len(week_grid),
                week_grid[0][0].date(),
                week_grid[-1][1].date(),
            )

        timeline_records: List[Dict[str, Any]] = []
        centroid_records: List[Dict[str, Any]] = []
        for w_start, w_end in week_grid:
            key = (w_start, w_end)
            win_valid = valid_counts.get(key, 0)
            active_ids, curr_cents, nid_pids, m_flags = self._process_window_batches(
                pregrouped.get(key, []), embeddings_lookup, w_start, w_end
            )
            diag_flags = self._diagnose_split_merge(active_ids, curr_cents, w_start, w_end)
            self._update_silent_entities(active_ids)

            for nid, ent in self.registry.items():
                p_curr = len(nid_pids[nid]) if nid in active_ids else 0
                age_w = max(0, int((w_end - ent.birth_window_end).total_seconds() // (7 * 86400)))
                share = np.float32(p_curr / win_valid) if win_valid > 0 else np.float32(np.nan)
                disp = (
                    _compute_intra_window_dispersion(
                        nid_pids[nid], embeddings_lookup, curr_cents[nid]
                    )
                    if nid in active_ids
                    else np.float32(np.nan)
                )
                sm_flag = _combine_split_merge_flags(
                    m_flags.get(nid, SplitMergeFlag.NORMAL),
                    diag_flags.get(nid, SplitMergeFlag.NORMAL),
                )
                timeline_records.append(
                    {
                        "window_start": w_start,
                        "window_end": w_end,
                        "narrative_id": ent.narrative_id,
                        "lifecycle_status": ent.status.value,
                        "current_window_posts": np.int32(p_curr),
                        "window_total_valid_posts": np.int32(win_valid),
                        "current_window_share": share,
                        "cumulative_posts": np.int64(ent.lifetime_post_count),
                        "age_weeks": np.int32(age_w),
                        "consecutive_silent_weeks": np.int32(ent.consecutive_silent_weeks),
                        "first_seen_utc": ent.first_seen_utc,
                        "last_active_utc": ent.last_active_utc,
                        "centroid_drift": np.float32(ent.latest_drift),
                        "similarity_to_previous_centroid": np.float32(ent.latest_similarity),
                        "intra_window_dispersion": disp,
                        "is_in_candidate_set": True,
                        "split_merge_flag": sm_flag.value,
                    }
                )
                centroid_records.append(
                    {
                        "window_end_utc": w_end.isoformat(),
                        "narrative_id": ent.narrative_id,
                        "centroid": ent.current_centroid.copy(),
                    }
                )

        timeline_df, centroid_records = self._finalize_timeline(timeline_records, centroid_records)
        summary_df = self._build_summary()
        logger.info(
            "Tracking complete: %d narratives, %d weeks, %d timeline rows",
            len(self.registry),
            len(week_grid),
            len(timeline_df),
        )
        return timeline_df, summary_df, centroid_records

    def _pregroup_assigned_batches(
        self, df: pd.DataFrame
    ) -> Dict[
        Tuple[pd.Timestamp, pd.Timestamp],
        List[Tuple[str, int, pd.Timestamp, pd.Timestamp, List[str]]],
    ]:
        assigned = df[(df["assignment_status"] == "ASSIGNED") & df["is_usable_text"]]
        batches: Dict[
            Tuple[pd.Timestamp, pd.Timestamp],
            List[Tuple[str, int, pd.Timestamp, pd.Timestamp, List[str]]],
        ] = defaultdict(list)
        if assigned.empty:
            return batches
        if self.config.tracking_mode == "relink_local_clusters":
            if "local_cluster_id" not in assigned.columns:
                raise KeyError("relink_local_clusters mode requires 'local_cluster_id' column")
            filtered = assigned[assigned["local_cluster_id"].astype(int) != -1]
            group_col = "local_cluster_id"
        else:
            filtered = assigned[~assigned["narrative_id"].isin(("NOISE", "UNASSIGNED"))]
            group_col = "narrative_id"
        for (ws, we, key_val), grp in filtered.groupby(
            ["window_start", "window_end", group_col], sort=True
        ):
            batches[(ws, we)].append(
                (
                    str(key_val),
                    len(grp),
                    grp["timestamp"].min(),
                    grp["timestamp"].max(),
                    sorted(grp["post_id"].tolist()),
                )
            )
        return batches

    def _process_window_batches(
        self,
        batches: List[Tuple[str, int, pd.Timestamp, pd.Timestamp, List[str]]],
        embeddings_lookup: Dict[str, np.ndarray],
        w_start: pd.Timestamp,
        w_end: pd.Timestamp,
    ) -> Tuple[Set[str], Dict[str, np.ndarray], Dict[str, List[str]], Dict[str, SplitMergeFlag]]:
        active_ids: Set[str] = set()
        curr_centroids: Dict[str, np.ndarray] = {}
        nid_post_ids: Dict[str, List[str]] = {}
        matcher_flags: Dict[str, SplitMergeFlag] = {}
        if not batches:
            return active_ids, curr_centroids, nid_post_ids, matcher_flags

        centroids = [
            compute_normalized_centroid(np.stack([embeddings_lookup[p] for p in b[4]], axis=0))
            for b in batches
        ]
        c2n: Dict[str, str] = {}
        match_sims: Dict[str, float] = {}
        sm_flags: Dict[str, SplitMergeFlag] = {}
        if self.config.tracking_mode == "relink_local_clusters":
            c2n, match_sims, sm_flags, audit_evts = match_clusters_to_registry(
                np.stack(centroids, axis=0),
                [b[0] for b in batches],
                list(self.registry.values()),
                self.config.alignment_threshold,
                self.config.split_merge_threshold,
            )
            for evt in audit_evts:
                evt["window_start_utc"] = w_start.isoformat()
                evt["window_end_utc"] = w_end.isoformat()
            self.audit_log.extend(audit_evts)

        for (raw_id, cnt, f_ts, l_ts, pids), centroid in zip(batches, centroids):
            if self.config.tracking_mode == "relink_local_clusters":
                nid = c2n.get(raw_id) or f"{self.config.id_prefix}_{self.next_id_num:04d}"
                if raw_id not in c2n:
                    self.next_id_num += 1
                matcher_flags[nid] = sm_flags.get(raw_id, SplitMergeFlag.NORMAL)
            else:
                nid = raw_id
            self._upsert_entity(
                nid, centroid, cnt, f_ts, l_ts, w_start, w_end, match_sims.get(raw_id)
            )
            active_ids.add(nid)
            curr_centroids[nid] = centroid
            nid_post_ids[nid] = pids
        return active_ids, curr_centroids, nid_post_ids, matcher_flags

    def _upsert_entity(
        self,
        nid: str,
        batch_centroid: np.ndarray,
        p_count: int,
        first_ts: pd.Timestamp,
        last_ts: pd.Timestamp,
        w_start: pd.Timestamp,
        w_end: pd.Timestamp,
        precomputed_sim: Optional[float] = None,
    ) -> None:
        base_evt: Dict[str, Any] = {
            "narrative_id": nid,
            "window_start_utc": w_start.isoformat(),
            "window_end_utc": w_end.isoformat(),
            "post_count": p_count,
        }
        if nid not in self.registry:
            self.registry[nid] = NarrativeEntity(
                narrative_id=nid,
                birth_window_start=w_start,
                birth_window_end=w_end,
                first_seen_utc=first_ts,
                last_active_utc=last_ts,
                current_centroid=batch_centroid.copy(),
                previous_centroid=batch_centroid.copy(),
                lifetime_post_count=p_count,
                active_weeks_count=1,
            )
            self.audit_log.append({"event_type": "birth", **base_evt})
            return

        entity = self.registry[nid]
        prev_c = entity.current_centroid.copy()
        updated = (1.0 - self.config.ema_alpha) * prev_c + self.config.ema_alpha * batch_centroid
        norm = float(np.linalg.norm(updated))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f"EMA update for {nid} produced degenerate norm ({norm:.2e}).")
        new_c = (updated / norm).astype(np.float32)
        sim = float(
            np.clip(
                precomputed_sim if precomputed_sim is not None else np.dot(prev_c, batch_centroid),
                -1.0,
                1.0,
            )
        )
        drift = float(1.0 - np.clip(np.dot(prev_c, new_c), -1.0, 1.0))

        is_react = entity.status in (LifecycleStatus.INACTIVE, LifecycleStatus.DORMANT)
        status = LifecycleStatus.REACTIVATED if is_react else LifecycleStatus.ACTIVE
        extra = (
            {"silent_weeks_prior": entity.consecutive_silent_weeks}
            if is_react
            else {"narrative_id_1": nid, "narrative_id_2": nid}
        )
        self.audit_log.append(
            {
                "event_type": "reactivation" if is_react else "linked_active",
                "similarity": sim,
                **extra,
                **base_evt,
            }
        )

        entity.previous_centroid = prev_c
        entity.current_centroid = new_c
        entity.lifetime_post_count += p_count
        entity.active_weeks_count += 1
        entity.consecutive_silent_weeks = 0
        entity.last_active_utc = max(entity.last_active_utc, last_ts)
        entity.status = status
        entity.latest_drift = drift
        entity.latest_similarity = sim

    def _diagnose_split_merge(
        self,
        active_in_window: Set[str],
        current_centroids: Dict[str, np.ndarray],
        w_start: pd.Timestamp,
        w_end: pd.Timestamp,
    ) -> Dict[str, SplitMergeFlag]:
        flags: Dict[str, SplitMergeFlag] = {nid: SplitMergeFlag.NORMAL for nid in active_in_window}
        active_list = sorted(active_in_window)
        if len(active_list) <= 1:
            return flags
        norm_mat = _l2_normalize_rows(
            np.stack([current_centroids[k] for k in active_list], axis=0), "_diagnose_split_merge"
        )
        sim_mat = norm_mat @ norm_mat.T
        w_s_iso = w_start.isoformat()
        w_e_iso = w_end.isoformat()
        for i, j in zip(*np.triu_indices(len(active_list), k=1)):
            sim_val = float(sim_mat[i, j])
            id_a = active_list[i]
            id_b = active_list[j]
            if sim_val >= self.config.split_merge_threshold:
                flags[id_a] = flags[id_b] = SplitMergeFlag.MERGE_CANDIDATE
                etype = "potential_merge_or_split"
            elif 0.60 <= sim_val < self.config.split_merge_threshold:
                etype = "unlinked_near_miss"
            else:
                continue
            self.audit_log.append(
                {
                    "event_type": etype,
                    "window_start_utc": w_s_iso,
                    "window_end_utc": w_e_iso,
                    "narrative_id_1": id_a,
                    "narrative_id_2": id_b,
                    "cosine_similarity": sim_val,
                }
            )
        return flags

    def _update_silent_entities(self, active_in_window: Set[str]) -> None:
        for nid, ent in self.registry.items():
            if nid not in active_in_window:
                ent.consecutive_silent_weeks += 1
                ent.max_consecutive_silent_weeks = max(
                    ent.max_consecutive_silent_weeks, ent.consecutive_silent_weeks
                )
                ent.latest_drift = 0.0
                ent.latest_similarity = float("nan")
                ent.status = (
                    LifecycleStatus.DORMANT
                    if ent.consecutive_silent_weeks >= self.config.dormant_threshold_weeks
                    else LifecycleStatus.INACTIVE
                )

    def _finalize_timeline(
        self, timeline_records: List[Dict[str, Any]], centroid_records: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        timeline_df = pd.DataFrame(timeline_records)
        if timeline_df.empty:
            return timeline_df, centroid_records
        for col in ("lifecycle_status", "split_merge_flag"):
            timeline_df[col] = timeline_df[col].astype("category")
        timeline_df = timeline_df.sort_values(["window_end", "narrative_id"], kind="mergesort")
        centroid_records = [centroid_records[int(idx)] for idx in timeline_df.index.to_numpy()]
        timeline_df = timeline_df.reset_index(drop=True)
        if timeline_df[["window_end", "narrative_id"]].duplicated().any():
            raise ValueError("Duplicate primary key (window_end, narrative_id) detected.")
        for tc in ("window_start", "window_end", "first_seen_utc", "last_active_utc"):
            timeline_df[tc] = pd.to_datetime(timeline_df[tc], utc=True)
        if (timeline_df["first_seen_utc"] > timeline_df["last_active_utc"]).any() or (
            timeline_df["last_active_utc"] >= timeline_df["window_end"]
        ).any():
            raise ValueError(
                "Causal timestamp invariant violated: require first_seen_utc <= last_active_utc < window_end."
            )
        return timeline_df, centroid_records

    def _build_summary(self) -> pd.DataFrame:
        cols = [
            "narrative_id",
            "first_seen_utc",
            "last_active_utc",
            "birth_window_start",
            "birth_window_end",
            "total_posts",
            "active_weeks",
            "final_lifecycle_status",
            "mean_weekly_posts",
            "max_consecutive_silent_weeks",
        ]
        if not self.registry:
            return pd.DataFrame(columns=cols)
        summary_df = (
            pd.DataFrame(
                [
                    {
                        "narrative_id": e.narrative_id,
                        "first_seen_utc": e.first_seen_utc,
                        "last_active_utc": e.last_active_utc,
                        "birth_window_start": e.birth_window_start,
                        "birth_window_end": e.birth_window_end,
                        "total_posts": e.lifetime_post_count,
                        "active_weeks": e.active_weeks_count,
                        "final_lifecycle_status": e.status.value,
                        "mean_weekly_posts": float(
                            e.lifetime_post_count / max(1, e.active_weeks_count)
                        ),
                        "max_consecutive_silent_weeks": e.max_consecutive_silent_weeks,
                    }
                    for e in self.registry.values()
                ]
            )
            .sort_values(by="narrative_id")
            .reset_index(drop=True)
        )
        summary_df["final_lifecycle_status"] = summary_df["final_lifecycle_status"].astype(
            "category"
        )
        return summary_df


def save_centroids_npz(centroid_records: List[Dict[str, Any]], out_path: Path) -> None:
    """Save L2-normalized centroid snapshots with allow_pickle=False; raise on NaN/zero-norm."""
    if not centroid_records:
        logger.warning("No centroid records to save; skipping %s", out_path)
        return
    centroids = _l2_normalize_rows(
        np.stack([r["centroid"].astype(np.float32) for r in centroid_records], axis=0),
        "save_centroids_npz",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        window_end_utc=np.array([r["window_end_utc"] for r in centroid_records], dtype="<U32"),
        narrative_id=np.array([r["narrative_id"] for r in centroid_records], dtype="<U32"),
        centroids=centroids,
    )
    logger.info(
        "Saved %d centroid snapshots (%d-D) to %s",
        len(centroid_records),
        centroids.shape[1],
        out_path,
    )


def load_frozen_registry_at_cutoff(
    registry_path: Path, centroids_path: Path, cutoff_utc: pd.Timestamp
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load the registry and aligned centroid matrix frozen at ``window_end == cutoff_utc``."""
    if not registry_path.exists() or not centroids_path.exists():
        raise FileNotFoundError(f"Missing artifact(s): {registry_path}, {centroids_path}")
    reg_df = pd.read_parquet(registry_path)
    reg_df["window_end"] = pd.to_datetime(reg_df["window_end"], utc=True)
    cutoff = _to_utc(cutoff_utc)
    filtered = (
        reg_df[reg_df["window_end"] == cutoff].sort_values("narrative_id").reset_index(drop=True)
    )
    if filtered.empty:
        raise ValueError(f"No registry rows found at window_end == {cutoff.isoformat()}.")
    with np.load(str(centroids_path), allow_pickle=False) as npz:
        mask = pd.to_datetime(npz["window_end_utc"], utc=True) == cutoff
        matched_nids = npz["narrative_id"][mask]
        matched_centroids = npz["centroids"][mask]
    order = np.argsort(matched_nids)
    if not np.array_equal(filtered["narrative_id"].to_numpy(dtype=str), matched_nids[order]):
        raise ValueError(f"Centroid-registry alignment mismatch at cutoff {cutoff.isoformat()}.")
    return filtered, matched_centroids[order].astype(np.float32)


def export_tracking_review_sample(
    audit_events: List[Dict[str, Any]],
    out_csv_path: Path,
    target_rows: int = 50,
    min_rows: int = 30,
    random_state: int = 42,
    dev_cutoff_utc: str = "2019-01-01T00:00:00Z",
) -> None:
    """Export a stratified, reproducible QA sample of tracking pairs from the dev period."""
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not audit_events:
        return
    dev_cutoff = _to_utc(dev_cutoff_utc)
    strata: Dict[str, List[Dict[str, Any]]] = {
        "linked_active": [],
        "potential_merge_or_split": [],
        "unlinked_near_miss": [],
    }
    supported_events = set(strata) | {"potential_merge", "potential_split"}
    for evt in audit_events:
        w_end_str = evt.get("window_end_utc", "")
        etype = evt.get("event_type", "")
        if not w_end_str or etype not in supported_events:
            continue
        try:
            w_end_ts = _to_utc(w_end_str)
            if pd.isna(w_end_ts) or w_end_ts > dev_cutoff:
                continue
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping audit event with invalid timestamp %r: %s", w_end_str, exc)
            continue

        w_start_str = evt.get("window_start_utc", "")
        if etype == "potential_merge":
            cid = str(evt.get("current_cluster_id", ""))
            pairs = [
                (cid, str(cand_id), sim)
                for cand_id, sim in zip(
                    evt.get("candidate_narrative_ids") or [], evt.get("similarities") or []
                )
            ]
            target_stratum = "potential_merge_or_split"
        elif etype == "potential_split":
            reg_id = str(evt.get("registry_narrative_id", ""))
            pairs = [
                (str(curr_id), reg_id, sim)
                for curr_id, sim in zip(
                    evt.get("current_cluster_ids") or [], evt.get("similarities") or []
                )
            ]
            target_stratum = "potential_merge_or_split"
        else:
            nid1 = evt.get(
                "narrative_id", evt.get("narrative_id_1", evt.get("current_cluster_id", ""))
            )
            nid2 = evt.get("narrative_id_2", evt.get("registry_narrative_id", ""))
            if etype == "linked_active" and not nid2:
                nid2 = nid1
            pairs = [(nid1, nid2, evt.get("cosine_similarity", evt.get("similarity", np.nan)))]
            target_stratum = etype

        for nid1, nid2, sim_val in pairs:
            strata[target_stratum].append(
                {
                    "event_type": target_stratum,
                    "window_start_utc": w_start_str,
                    "window_end_utc": w_end_str,
                    "narrative_id_1": nid1,
                    "narrative_id_2": nid2,
                    "similarity": sim_val,
                }
            )

    seen_pairs: Set[Tuple[str, str, str]] = set()
    for key in ("linked_active", "potential_merge_or_split", "unlinked_near_miss"):
        deduped_pool: List[Dict[str, Any]] = []
        for row in strata[key]:
            utc_iso = _to_utc(row["window_end_utc"]).isoformat()
            id1 = str(row["narrative_id_1"])
            id2 = str(row["narrative_id_2"])
            pair_key = (utc_iso, min(id1, id2), max(id1, id2))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                deduped_pool.append(row)
        strata[key] = deduped_pool

    rng = np.random.RandomState(random_state)
    per_stratum = max(target_rows // 3, 10)
    sampled: List[Dict[str, Any]] = []
    for key in strata:
        pool = strata[key]
        if len(pool) <= per_stratum:
            sampled.extend(pool)
        else:
            chosen_idx = sorted(rng.choice(len(pool), size=per_stratum, replace=False))
            for i in chosen_idx:
                sampled.append(pool[i])
    if len(sampled) < min_rows:
        sampled_ids = {id(r) for r in sampled}
        remaining = [r for k in strata for r in strata[k] if id(r) not in sampled_ids]
        if remaining:
            sampled.extend(
                remaining[i]
                for i in rng.choice(
                    len(remaining), size=min(min_rows - len(sampled), len(remaining)), replace=False
                )
            )
    if sampled:
        pd.DataFrame(sampled).to_csv(out_csv_path, index=False, encoding="utf-8")
        logger.info(
            "Exported %d QA tracking pairs to %s (dev period only)", len(sampled), out_csv_path
        )


def save_registry_artifacts(
    timeline_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    manifest: Dict[str, Any],
    registry_parquet_path: Path,
    summary_parquet_path: Optional[Path] = None,
    manifest_json_path: Optional[Path] = None,
) -> None:
    """Persist narrative registry timeline, entity summary, and manifest JSON."""

    def _write_pq(df: pd.DataFrame, path: Path, meta: Optional[Dict[bytes, bytes]] = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(
            tbl.replace_schema_metadata({**dict(tbl.schema.metadata or {}), **meta})
            if meta
            else tbl,
            path,
            compression="snappy",
        )

    try:
        _write_pq(
            timeline_df,
            registry_parquet_path,
            {REGISTRY_METADATA_KEY: json.dumps(manifest, ensure_ascii=False).encode("utf-8")},
        )
        if summary_parquet_path is not None:
            _write_pq(summary_df, summary_parquet_path)
        if manifest_json_path is not None:
            manifest_json_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_json_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        logger.info("Saved tracking artifacts to %s", registry_parquet_path)
    except Exception as exc:
        logger.error("Failed to persist narrative registry artifacts: %s", exc)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for narrative tracking."""
    parser = argparse.ArgumentParser(description="Build causal narrative registry with stable IDs.")
    for flag, typ, default in (
        ("--assignments", Path, DEFAULT_ASSIGNMENTS_PARQUET),
        ("--embeddings", Path, DEFAULT_EMBEDDINGS_NPZ),
        ("--out", Path, DEFAULT_REGISTRY_PARQUET),
        ("--out-summary", Path, DEFAULT_SUMMARY_PARQUET),
        ("--out-centroids", Path, DEFAULT_CENTROIDS_NPZ),
        ("--out-manifest", Path, DEFAULT_MANIFEST_JSON),
        ("--out-pairs-sample", Path, DEFAULT_PAIRS_SAMPLE_CSV),
        ("--alignment-threshold", float, 0.85),
        ("--ema-alpha", float, 0.30),
        ("--dormant-threshold-weeks", int, 12),
        ("--split-merge-threshold", float, 0.75),
        ("--calendar-start-utc", str, "2014-01-06"),
        ("--calendar-end-utc", str, "2020-12-21"),
    ):
        parser.add_argument(flag, type=typ, default=default)
    parser.add_argument(
        "--tracking-mode",
        type=str,
        default="registry_from_assignments",
        choices=["registry_from_assignments", "relink_local_clusters"],
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint for narrative tracking pipeline."""
    args = parse_args(argv)
    if not args.assignments.exists():
        raise FileNotFoundError(f"Assignments file not found: {args.assignments}")
    logger.info("Reading assignments from %s", args.assignments)
    assignments_df = pd.read_parquet(args.assignments)
    assigned_ids = set(
        assignments_df.loc[assignments_df["assignment_status"] == "ASSIGNED", "post_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )
    embeddings_lookup = load_embeddings_lookup(args.embeddings, required_assigned_ids=assigned_ids)
    calendar_end = pd.Timestamp(args.calendar_end_utc, tz="UTC") + pd.Timedelta(days=7)
    cfg = TrackingConfig(
        alignment_threshold=args.alignment_threshold,
        ema_alpha=args.ema_alpha,
        dormant_threshold_weeks=args.dormant_threshold_weeks,
        split_merge_threshold=args.split_merge_threshold,
        tracking_mode=args.tracking_mode,
        calendar_start_utc=args.calendar_start_utc,
        calendar_end_utc=calendar_end.isoformat(),
    )
    tracker = NarrativeTracker(config=cfg)
    timeline_df, summary_df, centroid_records = tracker.track_all_windows(
        assignments_df, embeddings_lookup
    )
    cal_total = len(
        _build_calendar_week_grid(assignments_df, args.calendar_start_utc, calendar_end.isoformat())
    )
    active_win = (
        int(timeline_df[["window_start", "window_end"]].drop_duplicates().shape[0])
        if not timeline_df.empty
        else 0
    )
    manifest: Dict[str, Any] = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").replace(microsecond=0).isoformat(),
        "tracking_config": cfg.to_dict(),
        "summary": {
            "calendar_weeks_total": cal_total,
            "warmup_weeks_without_candidates": cal_total - active_win,
            "active_registry_windows": active_win,
            "total_tracked_narratives": len(tracker.registry),
            "timeline_records_count": len(timeline_df),
            "total_posts_tracked": int(summary_df["total_posts"].sum())
            if not summary_df.empty
            else 0,
            "lifecycle_status_distribution": {
                str(k): int(v)
                for k, v in summary_df["final_lifecycle_status"].value_counts().items()
            }
            if not summary_df.empty
            else {},
            "audit_events_count": len(tracker.audit_log),
            "zero_future_leakage_guaranteed": True,
        },
        "audit_events_by_type": dict(
            sorted(Counter(str(e.get("event_type", "unknown")) for e in tracker.audit_log).items())
        ),
        "audit_events_sample": tracker.audit_log[:50],
    }
    save_registry_artifacts(
        timeline_df, summary_df, manifest, args.out, args.out_summary, args.out_manifest
    )
    save_centroids_npz(centroid_records=centroid_records, out_path=args.out_centroids)
    export_tracking_review_sample(tracker.audit_log, args.out_pairs_sample)
    s = manifest["summary"]
    logger.info(
        "Registry complete: %d narratives, %d timeline rows, %d audit events -> %s",
        s["total_tracked_narratives"],
        s["timeline_records_count"],
        s["audit_events_count"],
        args.out,
    )


if __name__ == "__main__":
    main()
