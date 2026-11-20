"""Build a normalized, leakage-free temporal narrative assignment table.

Produces per-post weekly narrative assignments aligned with the ND4 temporal
protocol (`configs/forecast.yaml`):
- Calendar weeks in UTC using Monday-Sunday half-open `[window_start, window_end)`.
- Strictly past-only rolling-origin fitting (`fit_end <= window_start`) via
  `src.narrative.discover_narratives`.
- Causal past-only centroid alignment across origins (`CAND_XXXX`) without using
  any retrospective full-period ND3 cluster IDs as historical labels.
- Invalid or unusable rows (`is_usable_text != True`, missing/NaN/Inf/zero-norm
  vectors, or insufficient warmup history) excluded from clustering fit/predict
  but retained in the output table with status `UNASSIGNED` (alongside `NOISE`).
- Full discovery configuration and per-window fit scope persisted in both the
  output Parquet schema metadata and a sidecar JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

from src.embeddings.embedding_store import load_embeddings
from src.narrative.discover_narratives import (
    DiscoveryConfig,
    assign_existing,
    discover_narratives,
)

logger = logging.getLogger("NarrativeCast.TemporalAssignments")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    )

DEFAULT_SCOPE_PATH = Path("data/D4/net_neutrality_scope_expanded_v1.csv")
DEFAULT_CORPUS_PATH = Path("data/processed/reddit_technology_clean.parquet")
DEFAULT_EMBEDDINGS_NPZ = Path("data/interim/scope_expanded_v1_embeddings.npz")
DEFAULT_OUTPUT_PARQUET = Path("data/interim/narrative_assignments.parquet")
DEFAULT_OUTPUT_MANIFEST = Path("data/interim/narrative_assignments_manifest.json")

PARQUET_METADATA_KEY = b"narrativecast_discovery_metadata"
VECTOR_KEYS = ("embeddings", "vectors", "X")
ID_KEYS = ("post_ids", "post_id", "ids")


@dataclass(frozen=True)
class AssignmentConfig:
    """Configuration for causal weekly narrative discovery and assignment."""

    n_clusters: int = 24
    random_state: int = 42
    batch_size: int = 2048
    n_init: int = 5
    max_iter: int = 200
    model_version: str = "nd4_causal_k24_v1"
    min_fit_posts: int = 120
    refit_every_weeks: int = 1
    max_fit_window_days: Optional[int] = None
    noise_threshold: float = 0.65
    alignment_threshold: float = 0.85
    ema_alpha: float = 0.30
    timezone: str = "UTC"
    week_start: str = "MONDAY"
    observation_days: int = 7

    def to_discovery_config(self) -> DiscoveryConfig:
        """Convert to `DiscoveryConfig` consumed by ND3 clustering functions."""
        return DiscoveryConfig(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=self.batch_size,
            n_init=self.n_init,
            max_iter=self.max_iter,
            model_version=self.model_version,
        )


def compute_utc_weekly_windows(timestamps: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Normalize timestamps to UTC and map each post to its half-open Monday week.

    Each post with timestamp `t` falls into `[window_start, window_end)` where
    `window_start` is Monday `00:00:00+00:00` and `window_end = window_start + 7d`.
    """
    ts_utc = pd.to_datetime(timestamps, utc=True, errors="coerce")
    if ts_utc.isna().any():
        bad_count = int(ts_utc.isna().sum())
        raise ValueError(f"Found {bad_count} null/unparseable timestamps; all rows require UTC timestamps.")

    window_start = ts_utc.dt.floor("D") - pd.to_timedelta(ts_utc.dt.dayofweek, unit="D")
    window_end = window_start + pd.Timedelta(days=7)

    in_bounds = (window_start <= ts_utc) & (ts_utc < window_end)
    if not bool(in_bounds.all()):
        raise AssertionError("Half-open window invariant failed: window_start <= timestamp < window_end.")
    return ts_utc, window_start, window_end


def assess_vector_validity(
    vectors: np.ndarray,
    is_usable_text: np.ndarray,
    is_vector_present: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify rows eligible for clustering vs rows that must be `UNASSIGNED`.

    Returns `(eligible_mask, unassigned_reason)` of length N. Invalid or unusable
    rows are excluded from both clustering `fit` and `predict`.
    """
    n_rows = len(is_usable_text)
    if vectors.ndim != 2 or vectors.shape[0] != n_rows:
        raise ValueError(f"Expected 2D vectors with {n_rows} rows, got shape {vectors.shape}.")

    usable = np.asarray(is_usable_text, dtype=bool)
    present = np.ones(n_rows, dtype=bool) if is_vector_present is None else np.asarray(is_vector_present, dtype=bool)
    finite = np.all(np.isfinite(vectors), axis=1)
    norms = np.linalg.norm(np.where(np.isfinite(vectors), vectors, 0.0), axis=1)
    nonzero = norms > 1e-12

    eligible = usable & present & finite & nonzero
    reasons = np.full(n_rows, "", dtype=object)
    reasons[~nonzero & finite & present & usable] = "zero_norm_vector"
    reasons[~finite & present & usable] = "invalid_vector"
    reasons[~present & usable] = "missing_vector"
    reasons[~usable] = "unusable_text"
    return eligible, reasons


def align_clusters_past_only(
    centroids: np.ndarray,
    registry: Dict[str, np.ndarray],
    next_id_num: int,
    alignment_threshold: float = 0.85,
    ema_alpha: float = 0.30,
) -> Tuple[Dict[int, str], int]:
    """Match current local cluster centroids against strictly past candidate centroids.

    Uses Hungarian assignment on cosine similarity. Matched clusters reuse the past
    `CAND_XXXX` ID and update the registered centroid via EMA; unmatched clusters
    spawn new `CAND_XXXX` identifiers. `registry` is updated in-place strictly AFTER
    the origin's alignment step for future origins only.
    """
    normed = normalize(np.asarray(centroids, dtype=np.float32), norm="l2", axis=1)
    local_to_cand: Dict[int, str] = {}

    if not registry:
        for cid in range(normed.shape[0]):
            cand_id = f"CAND_{next_id_num:04d}"
            registry[cand_id] = normed[cid].copy()
            local_to_cand[cid] = cand_id
            next_id_num += 1
        return local_to_cand, next_id_num

    past_ids = list(registry.keys())
    past_matrix = np.stack([registry[k] for k in past_ids], axis=0)
    sim_matrix = normed @ past_matrix.T  # shape: (K, M)

    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    matched_local = set()
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if float(sim_matrix[r, c]) >= alignment_threshold:
            cand_id = past_ids[c]
            local_to_cand[r] = cand_id
            matched_local.add(r)
            updated = (1.0 - ema_alpha) * registry[cand_id] + ema_alpha * normed[r]
            updated_norm = float(np.linalg.norm(updated))
            if updated_norm > 1e-12:
                registry[cand_id] = (updated / updated_norm).astype(np.float32)

    for cid in range(normed.shape[0]):
        if cid not in matched_local:
            cand_id = f"CAND_{next_id_num:04d}"
            registry[cand_id] = normed[cid].copy()
            local_to_cand[cid] = cand_id
            next_id_num += 1

    return local_to_cand, next_id_num


def build_temporal_assignments(
    scope_df: pd.DataFrame,
    vectors: np.ndarray,
    is_vector_present: Optional[np.ndarray] = None,
    config: AssignmentConfig = AssignmentConfig(),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run causal rolling-origin weekly discovery & assignment across all input posts.

    Guarantees:
    - Exact 1:1 preservation of `post_id` (no dropped or duplicated posts).
    - All timestamps normalized to UTC (`timestamp`, `window_start`, `window_end`).
    - Invalid/unusable posts never enter clustering `fit` or `predict`, yet remain
      in output with `narrative_id='UNASSIGNED'`, `local_cluster_id=-1`.
    - Weekly fit sets satisfy `fit_end_utc <= window_start_utc` (zero future leakage).
    """
    required_cols = {"post_id", "publish_datetime_utc"}
    missing = required_cols - set(scope_df.columns)
    if missing:
        raise ValueError(f"Input scope_df missing required columns: {sorted(missing)}")

    work = scope_df.copy().reset_index(drop=True)
    work["post_id"] = work["post_id"].astype(str).str.strip()
    if work["post_id"].eq("").any():
        raise ValueError("Input contains empty post_id values.")
    if not work["post_id"].is_unique:
        raise ValueError("Input scope_df contains duplicate post_id values.")

    if "is_usable_text" in work.columns:
        usable_arr = work["is_usable_text"].fillna(False).astype(bool).to_numpy()
    else:
        usable_arr = np.ones(len(work), dtype=bool)

    ts_utc, win_start, win_end = compute_utc_weekly_windows(work["publish_datetime_utc"])
    eligible_mask, invalid_reasons = assess_vector_validity(
        vectors=vectors,
        is_usable_text=usable_arr,
        is_vector_present=is_vector_present,
    )

    n_rows = len(work)
    out_local_cluster = np.full(n_rows, -1, dtype=np.int32)
    out_narrative_id = np.full(n_rows, "UNASSIGNED", dtype=object)
    out_score = np.full(n_rows, np.nan, dtype=np.float32)
    out_status = np.full(n_rows, "UNASSIGNED", dtype=object)
    out_reason = invalid_reasons.copy()

    work["_ts_utc"] = ts_utc
    work["_win_start"] = win_start
    work["_win_end"] = win_end
    work["_eligible"] = eligible_mask

    discovery_cfg = config.to_discovery_config()
    eff_min_fit = max(config.min_fit_posts, config.n_clusters)

    unique_weeks = sorted(work["_win_start"].unique())
    registry: Dict[str, np.ndarray] = {}
    next_id_num = 1
    current_model = None
    current_local_map: Dict[int, str] = {}
    weeks_since_refit = config.refit_every_weeks

    fit_scope_records: List[Dict[str, Any]] = []
    meta_cols = ["post_id", "publish_datetime_utc"]

    for w_start in unique_weeks:
        w_end = w_start + pd.Timedelta(days=7)
        win_idx = work.index[work["_win_start"] == w_start].to_numpy()
        win_eligible_idx = work.index[(work["_win_start"] == w_start) & work["_eligible"]].to_numpy()

        # Strictly past-only training pool: timestamp < w_start and eligible.
        past_cond = (work["_ts_utc"] < w_start) & work["_eligible"]
        if config.max_fit_window_days is not None:
            lookback_start = w_start - pd.Timedelta(days=int(config.max_fit_window_days))
            past_cond = past_cond & (work["_ts_utc"] >= lookback_start)
        fit_idx = work.index[past_cond].to_numpy()

        refitted = False
        fit_start_iso: Optional[str] = None
        fit_end_iso: Optional[str] = None
        fit_n_posts = int(len(fit_idx))

        if fit_n_posts < eff_min_fit:
            # Warmup / insufficient past history: keep eligible posts UNASSIGNED without leaking future.
            for idx in win_eligible_idx:
                out_reason[idx] = "insufficient_history"
        else:
            if fit_n_posts > 0:
                fit_ts_series = work.loc[fit_idx, "_ts_utc"]
                fit_start_iso = fit_ts_series.min().isoformat()
                fit_max_ts = fit_ts_series.max()
                if fit_max_ts >= w_start:
                    raise AssertionError(
                        f"Future leakage detected at week {w_start.isoformat()}: "
                        f"max fit timestamp {fit_max_ts.isoformat()} >= {w_start.isoformat()}"
                    )
                fit_end_iso = w_start.isoformat()

            if current_model is None or weeks_since_refit >= max(1, config.refit_every_weeks):
                fit_vecs = np.asarray(vectors[fit_idx], dtype=np.float32)
                fit_meta = work.loc[fit_idx, meta_cols].copy()
                _, current_model = discover_narratives(fit_vecs, fit_meta, discovery_cfg)
                current_local_map, next_id_num = align_clusters_past_only(
                    centroids=current_model.cluster_centers_,
                    registry=registry,
                    next_id_num=next_id_num,
                    alignment_threshold=config.alignment_threshold,
                    ema_alpha=config.ema_alpha,
                )
                weeks_since_refit = 0
                refitted = True

            if len(win_eligible_idx) > 0:
                win_vecs = np.asarray(vectors[win_eligible_idx], dtype=np.float32)
                win_meta = work.loc[win_eligible_idx, meta_cols].copy()
                pred_df = assign_existing(win_vecs, win_meta, current_model, config.model_version)

                cids = pred_df["local_cluster_id"].to_numpy(dtype=np.int32)
                scores = pred_df["assignment_score"].to_numpy(dtype=np.float32)
                for row_pos, idx in enumerate(win_eligible_idx):
                    sc = float(scores[row_pos])
                    cid = int(cids[row_pos])
                    out_score[idx] = np.float32(sc)
                    out_reason[idx] = ""
                    if sc < config.noise_threshold:
                        out_local_cluster[idx] = -1
                        out_narrative_id[idx] = "NOISE"
                        out_status[idx] = "NOISE"
                    else:
                        out_local_cluster[idx] = cid
                        out_narrative_id[idx] = current_local_map[cid]
                        out_status[idx] = "ASSIGNED"

            weeks_since_refit += 1

        win_statuses = out_status[win_idx]
        fit_scope_records.append(
            {
                "window_start_utc": w_start.isoformat(),
                "window_end_utc": w_end.isoformat(),
                "fit_start_utc": fit_start_iso,
                "fit_end_utc": fit_end_iso,
                "fit_n_posts": fit_n_posts,
                "refitted": refitted,
                "n_total_posts": int(len(win_idx)),
                "n_eligible_posts": int(len(win_eligible_idx)),
                "n_assigned": int(np.sum(win_statuses == "ASSIGNED")),
                "n_noise": int(np.sum(win_statuses == "NOISE")),
                "n_unassigned": int(np.sum(win_statuses == "UNASSIGNED")),
            }
        )

    out_df = pd.DataFrame(
        {
            "post_id": work["post_id"].astype(str),
            "timestamp": ts_utc,
            "local_cluster_id": out_local_cluster,
            "narrative_id": pd.Series(out_narrative_id, dtype="string"),
            "window_start": win_start,
            "window_end": win_end,
            "model_version": pd.Series([config.model_version] * n_rows, dtype="string"),
            "assignment_score": pd.Series(out_score, dtype="float32"),
            "is_usable_text": pd.Series(usable_arr, dtype="bool"),
            "assignment_status": pd.Series(out_status, dtype="string"),
            "unassigned_reason": pd.Series(out_reason, dtype="string"),
        }
    )

    fitted_records = [r for r in fit_scope_records if r["fit_end_utc"] is not None]
    leakage_free = all(r["fit_end_utc"] <= r["window_start_utc"] for r in fitted_records)

    manifest: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "discovery_config": asdict(discovery_cfg),
        "assignment_config": asdict(config),
        "protocol": {
            "timezone": config.timezone,
            "unit": "week",
            "week_start": config.week_start,
            "interval_convention": "half_open",
            "observation_days": config.observation_days,
            "rolling_origin_past_only": True,
            "fit_future_data": False,
            "include_noise_in_denominator": True,
            "include_unassigned_in_denominator": True,
        },
        "summary": {
            "input_posts": int(len(work)),
            "output_posts": int(len(out_df)),
            "unique_output_post_ids": int(out_df["post_id"].nunique()),
            "no_lost_ids": len(out_df) == len(work),
            "no_duplicate_ids": bool(out_df["post_id"].is_unique),
            "all_timestamps_utc": str(out_df["timestamp"].dt.tz) == "UTC" and str(out_df["window_end"].dt.tz) == "UTC",
            "zero_future_leakage": leakage_free,
            "usable_text_true_count": int(usable_arr.sum()),
            "clustering_eligible_count": int(eligible_mask.sum()),
            "assigned_count": int((out_df["assignment_status"] == "ASSIGNED").sum()),
            "noise_count": int((out_df["assignment_status"] == "NOISE").sum()),
            "unassigned_count": int((out_df["assignment_status"] == "UNASSIGNED").sum()),
            "n_discovered_candidate_narratives": int(len(registry)),
            "total_windows": int(len(fit_scope_records)),
            "fitted_windows": int(len(fitted_records)),
            "refit_count": int(sum(1 for r in fit_scope_records if r["refitted"])),
            "global_fit_start_utc": min((r["fit_start_utc"] for r in fitted_records if r["fit_start_utc"]), default=None),
            "global_fit_end_utc": max((r["fit_end_utc"] for r in fitted_records if r["fit_end_utc"]), default=None),
        },
        "window_fit_scopes": fit_scope_records,
    }
    return out_df, manifest


def load_scope_metadata(scope_path: Path, corpus_path: Optional[Path] = None) -> pd.DataFrame:
    """Load scope dataset and ensure `is_usable_text` & UTC timestamps are populated."""
    if scope_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(scope_path)
    else:
        df = pd.read_csv(scope_path)

    if "post_id" not in df.columns or "publish_datetime_utc" not in df.columns:
        raise ValueError(f"Scope file {scope_path} must contain 'post_id' and 'publish_datetime_utc'.")

    df = df.copy()
    df["post_id"] = df["post_id"].astype(str).str.strip()
    if not df["post_id"].is_unique:
        raise ValueError(f"Scope file {scope_path} contains duplicate post_id rows.")

    if "is_usable_text" not in df.columns:
        if corpus_path is None or not corpus_path.exists():
            raise FileNotFoundError(
                f"`is_usable_text` missing in {scope_path} and corpus file not found: {corpus_path}"
            )
        target_ids = set(df["post_id"].tolist())
        pf = pq.ParquetFile(corpus_path)
        chunks = []
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["post_id", "is_usable_text"]).to_pandas()
            rg["post_id"] = rg["post_id"].astype(str).str.strip()
            matched = rg[rg["post_id"].isin(target_ids)]
            if not matched.empty:
                chunks.append(matched)
        usable_df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["post_id"]) if chunks else pd.DataFrame(columns=["post_id", "is_usable_text"])
        df = df.merge(usable_df, on="post_id", how="left")
        df["is_usable_text"] = df["is_usable_text"].fillna(False).astype(bool)

    return df


def _load_vectors_from_shards(
    post_ids: Sequence[str],
    embeddings_dir: Path,
    shard_index: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings aligned to `post_ids` from a sharded `.npz` directory."""
    target_set = set(post_ids)
    idx_df = pd.read_csv(shard_index)
    file_col = next((c for c in ("file", "shard_file", "filename", "path", "shard_path") if c in idx_df.columns), None)
    if file_col is None:
        raise ValueError(f"Shard index {shard_index} missing filename/path column.")

    lookup: Dict[str, np.ndarray] = {}
    dim: Optional[int] = None
    for raw_name in idx_df[file_col].astype(str):
        shard_path = embeddings_dir / Path(raw_name).name
        if not shard_path.exists():
            continue
        with np.load(shard_path, allow_pickle=False) as zf:
            vec_key = next((k for k in VECTOR_KEYS if k in zf.files), None)
            id_key = next((k for k in ID_KEYS if k in zf.files), None)
            if vec_key is None or id_key is None:
                continue
            vecs = zf[vec_key]
            ids = zf[id_key]
            if dim is None and vecs.ndim == 2:
                dim = vecs.shape[1]
            for pos, raw_pid in enumerate(ids):
                pid = str(raw_pid).strip()
                if pid in target_set:
                    lookup[pid] = np.asarray(vecs[pos], dtype=np.float32)

    if dim is None:
        raise RuntimeError(f"No readable shard vectors found under {embeddings_dir}.")
    return _align_lookup(post_ids, lookup, dim)


def _align_lookup(
    post_ids: Sequence[str],
    lookup: Dict[str, np.ndarray],
    dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align vector dictionary to `post_ids`, marking missing entries explicitly."""
    n = len(post_ids)
    out = np.full((n, dim), np.nan, dtype=np.float32)
    present = np.zeros(n, dtype=bool)
    for i, pid in enumerate(post_ids):
        vec = lookup.get(pid)
        if vec is not None:
            out[i] = vec
            present[i] = True
    return out, present


def load_vectors_for_posts(
    post_ids: Sequence[str],
    embeddings_npz: Optional[Path] = None,
    embeddings_dir: Optional[Path] = None,
    shard_index: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load post vectors from either an `EmbeddingStore` archive or Drive shard dir."""
    if embeddings_npz is not None and embeddings_npz.exists():
        store = load_embeddings(embeddings_npz, validate=True)
        dim = int(store.vectors.shape[1])
        lookup = {
            str(pid).strip(): np.asarray(store.vectors[i], dtype=np.float32)
            for i, pid in enumerate(store.post_ids)
            if bool(store.is_valid[i])
        }
        return _align_lookup(post_ids, lookup, dim)

    if embeddings_dir is not None and shard_index is not None and embeddings_dir.exists() and shard_index.exists():
        return _load_vectors_from_shards(post_ids, embeddings_dir, shard_index)

    raise FileNotFoundError(
        "No valid embedding source found. Provide --embeddings-npz or both --embeddings-dir and --shard-index."
    )


def save_assignments_with_metadata(
    df: pd.DataFrame,
    manifest: Dict[str, Any],
    out_parquet: Path,
    manifest_json: Optional[Path] = None,
) -> None:
    """Write `narrative_assignments.parquet` with embedded discovery/fit-scope metadata."""
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
    parser = argparse.ArgumentParser(description="Build causal weekly narrative assignment table.")
    parser.add_argument("--scope", type=Path, default=DEFAULT_SCOPE_PATH)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--embeddings-npz", type=Path, default=DEFAULT_EMBEDDINGS_NPZ)
    parser.add_argument("--embeddings-dir", type=Path, default=None)
    parser.add_argument("--shard-index", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--n-clusters", type=int, default=24)
    parser.add_argument("--min-fit-posts", type=int, default=120)
    parser.add_argument("--refit-every-weeks", type=int, default=1)
    parser.add_argument("--max-fit-window-days", type=int, default=None)
    parser.add_argument("--noise-threshold", type=float, default=0.65)
    parser.add_argument("--alignment-threshold", type=float, default=0.85)
    parser.add_argument("--model-version", type=str, default="nd4_causal_k24_v1")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logger.info("Loading scope metadata from %s", args.scope)
    scope_df = load_scope_metadata(args.scope, args.corpus)

    logger.info("Loading embeddings for %d scoped posts", len(scope_df))
    vectors, is_present = load_vectors_for_posts(
        post_ids=scope_df["post_id"].tolist(),
        embeddings_npz=args.embeddings_npz,
        embeddings_dir=args.embeddings_dir,
        shard_index=args.shard_index,
    )

    cfg = AssignmentConfig(
        n_clusters=args.n_clusters,
        min_fit_posts=args.min_fit_posts,
        refit_every_weeks=args.refit_every_weeks,
        max_fit_window_days=args.max_fit_window_days,
        noise_threshold=args.noise_threshold,
        alignment_threshold=args.alignment_threshold,
        model_version=args.model_version,
    )

    logger.info("Running causal weekly discovery & assignment (K=%d, min_fit=%d)", cfg.n_clusters, cfg.min_fit_posts)
    assignments_df, manifest = build_temporal_assignments(
        scope_df=scope_df,
        vectors=vectors,
        is_vector_present=is_present,
        config=cfg,
    )

    save_assignments_with_metadata(
        df=assignments_df,
        manifest=manifest,
        out_parquet=args.out,
        manifest_json=args.manifest_out,
    )
    s = manifest["summary"]
    logger.info(
        "Saved %d rows (%d assigned, %d noise, %d unassigned; %d candidates; zero_future_leakage=%s) to %s",
        s["output_posts"],
        s["assigned_count"],
        s["noise_count"],
        s["unassigned_count"],
        s["n_discovered_candidate_narratives"],
        s["zero_future_leakage"],
        args.out,
    )


if __name__ == "__main__":
    main()
