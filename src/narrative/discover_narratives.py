"""NarrativeCast ND3.7: reusable, non-temporal semantic clustering interface.

Inputs: a 2D embedding NumPy array and a pandas metadata table in exactly
matching row order. The module fits a local MiniBatchKMeans or uses a previously
saved, trusted model to assign local cluster IDs and cosine similarities.

IMPORTANT: local clusters are not narratives or persistent temporal IDs.
A global 2014-2020 model is NOT suitable for leakage-safe historical forecasting.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class DiscoveryConfig:
    n_clusters: int = 24
    random_state: int = 42
    batch_size: int = 2048
    n_init: int = 5
    max_iter: int = 200
    model_version: str = 'nd3_discovery_local_candidate'


def validate_inputs(embeddings: np.ndarray, metadata: pd.DataFrame):
    """Validate row-aligned embeddings/metadata without silently dropping data."""
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[0] == 0:
        raise ValueError('embeddings must be nonempty 2-D [n_posts, dim]')
    if len(metadata) != x.shape[0]:
        raise ValueError('metadata and embeddings must have the same row count')
    if 'post_id' not in metadata:
        raise ValueError('metadata must contain post_id')
    ids = metadata['post_id']
    if ids.isna().any() or ids.astype(str).str.strip().eq('').any() or not ids.astype(str).is_unique:
        raise ValueError('post_id cannot be missing, blank or duplicated')
    if not np.isfinite(x).all():
        raise ValueError('embeddings contain NaN or infinity')
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('zero-norm or invalid embeddings are not accepted')
    return normalize(x, norm='l2').astype(np.float32, copy=False)


def _assign(x: np.ndarray, metadata: pd.DataFrame, model, model_version: str):
    if model.cluster_centers_.shape[1] != x.shape[1]:
        raise ValueError('embedding dimension differs from trained model')
    labels = model.predict(x).astype(int)
    centers = normalize(np.asarray(model.cluster_centers_, dtype=np.float32))
    scores = np.einsum('ij,ij->i', x, centers[labels])
    out = metadata[['post_id']].copy()
    out['post_id'] = out['post_id'].astype(str)
    if 'publish_datetime_utc' in metadata.columns:
        dt = pd.to_datetime(metadata['publish_datetime_utc'], utc=True, errors='coerce')
        if dt.isna().any():
            raise ValueError('timestamps must be parseable, non-missing UTC datetimes')
        out['publish_datetime_utc'] = dt
    out['local_cluster_id'] = labels
    out['assignment_score'] = scores.astype(float)
    out['model_version'] = str(model_version)
    return out


def discover_narratives(embeddings: np.ndarray, metadata: pd.DataFrame,
                        config: DiscoveryConfig = DiscoveryConfig()):
    """Fit *local* clusters and return (assignments, fitted_model).

    The function name follows ND3.7's deliverable, but does not certify that
    any cluster is a narrative. Use only with an explicitly approved fit scope.
    """
    x = validate_inputs(embeddings, metadata)
    if not 2 <= config.n_clusters <= len(metadata):
        raise ValueError('n_clusters must be between 2 and number of posts')
    model = MiniBatchKMeans(n_clusters=config.n_clusters,
                            random_state=config.random_state,
                            batch_size=config.batch_size, n_init=config.n_init,
                            max_iter=config.max_iter)
    model.fit(x)
    return _assign(x, metadata, model, config.model_version), model


def assign_existing(embeddings: np.ndarray, metadata: pd.DataFrame,
                    model, model_version: str):
    """Assign using a trusted, previously fit model; DOES NOT refit it."""
    x = validate_inputs(embeddings, metadata)
    return _assign(x, metadata, model, model_version)


def make_cluster_figure(assignments: pd.DataFrame, output_png: str | Path,
                        labels: pd.DataFrame | None = None,
                        top_example_posts: pd.DataFrame | None = None,
                        title: str | None = None):
    """Plot actual row counts in supplied assignments (not inferred population).

    For official 7,244-post counts, supply full official cluster_assignments.csv.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import textwrap
    if 'local_cluster_id' in assignments:
        cluster_col = 'local_cluster_id'
    elif 'cluster_id' in assignments:
        cluster_col = 'cluster_id'
    else:
        raise ValueError('assignments require local_cluster_id or cluster_id')
    if 'post_id' not in assignments or assignments['post_id'].duplicated().any():
        raise ValueError('assignments require unique post_id')
    counts = assignments.groupby(cluster_col)['post_id'].nunique().sort_index()
    if len(counts) == 0:
        raise ValueError('empty assignments')
    fig, ax = plt.subplots(figsize=(15, 7.5))
    bars = ax.bar(np.arange(len(counts)), counts.to_numpy())
    ax.set_xticks(np.arange(len(counts)), [f'C{int(i):02d}' for i in counts.index], rotation=45)
    ax.set_ylabel('Number of POSTs in supplied assignments')
    ax.set_title(title or f'Semantic cluster sizes | {len(assignments):,} supplied POSTs')
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), str(int(n)),
                ha='center', va='bottom', fontsize=8)
    if labels is not None and {'cluster_id', 'temporary_label_en'} <= set(labels.columns):
        legend = [f'C{int(cid):02d} {str(n)[:58]}' for cid, n in
                  labels[['cluster_id','temporary_label_en']].itertuples(index=False, name=None)]
        note='Temporary topic labels (not confirmed narratives):\n'+'  |  '.join(legend)
        fig.text(0.04, 0.03, '\n'.join(textwrap.wrap(note, 170)), fontsize=7, va='bottom')
        fig.subplots_adjust(bottom=.25)
    else:
        fig.subplots_adjust(bottom=.15)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_png


def _load_arrays(npz_path: Path, csv_path: Path):
    # Source .npz expects row-order alignment with metadata CSV.
    meta = pd.read_csv(csv_path, dtype={'post_id': str})
    with np.load(npz_path, allow_pickle=False) as payload:
        vector_key = next((k for k in ('embeddings','vectors','X') if k in payload), None)
        if vector_key is None:
            raise ValueError('NPZ must contain embeddings, vectors or X')
        x = payload[vector_key]
        key = next((k for k in ('post_ids','post_id','ids') if k in payload), None)
        if key is not None and not np.array_equal(payload[key].astype(str), meta.post_id.to_numpy(dtype=str)):
            raise ValueError('NPZ post_ids do not match metadata CSV row order')
    return x, meta


def _cli():
    p = argparse.ArgumentParser(description='ND3.7 reusable local semantic clustering (not narrative tracking)')
    sub = p.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit', help='Fit on approved scope; emits candidate model and local cluster IDs')
    pred = sub.add_parser('predict', help='Use trusted fit model; never refit')
    plot = sub.add_parser('plot', help='Plot actual supplied assignment counts')
    for spec in (fit, pred):
        spec.add_argument('--embeddings', type=Path, required=True)
        spec.add_argument('--metadata', type=Path, required=True)
        spec.add_argument('--out-dir', type=Path, required=True)
        spec.add_argument('--model-version', default='nd3_discovery_local_candidate')
    fit.add_argument('--k', type=int, default=24)
    fit.add_argument('--seed', type=int, default=42)
    pred.add_argument('--model', type=Path, required=True, help='Only load trusted local joblib artifacts')
    plot.add_argument('--assignments', type=Path, required=True)
    plot.add_argument('--labels', type=Path)
    plot.add_argument('--output', type=Path, required=True)
    plot.add_argument('--title')
    args = p.parse_args()
    if args.command == 'plot':
        counts = pd.read_csv(args.assignments, dtype={'post_id': str})
        lbl = pd.read_csv(args.labels) if args.labels else None
        make_cluster_figure(counts, args.output, lbl, title=args.title)
        print(json.dumps({'status': 'PASS', 'posts_plotted': len(counts), 'output': str(args.output)}))
        return
    x, meta = _load_arrays(args.embeddings, args.metadata)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == 'fit':
        import joblib
        cfg = DiscoveryConfig(n_clusters=args.k, random_state=args.seed, model_version=args.model_version)
        result, model = discover_narratives(x, meta, cfg)
        joblib.dump(model, args.out_dir/'local_cluster_model.joblib')
        (args.out_dir/'local_model_config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')
    else:
        import joblib
        model = joblib.load(args.model) # Only trusted path, joblib is pickle-based
        result = assign_existing(x, meta, model, args.model_version)
    result.to_csv(args.out_dir/'local_cluster_assignments.csv', index=False)
    print(json.dumps({'status': 'PASS', 'rows': len(result),
                      'unique_clusters': int(result.local_cluster_id.nunique())}))


if __name__ == '__main__':
    _cli()
