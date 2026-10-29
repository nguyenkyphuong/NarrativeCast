"""
tests/test_embedding_store.py
Bộ kiểm thử tự động cho module lưu trữ và nạp lại embedding (src/embeddings/embedding_store.py).
"""

import os
import tempfile
import numpy as np
import pandas as pd

from src.embeddings.embedding_store import (
    EmbeddingMetadata,
    EmbeddingStore,
    ValidationReport,
    load_embeddings,
    save_embeddings,
    validate_embeddings,
)


def test_embedding_store_basic_and_lookups():
    vectors = np.random.randn(5, 384).astype(np.float32)
    post_ids = ["p_001", "p_002", "p_003", "p_004", "p_005"]
    meta = EmbeddingMetadata(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        num_vectors=5,
        model_version="1.0",
    )

    store = EmbeddingStore(vectors=vectors, post_ids=post_ids, metadata=meta)

    assert len(store) == 5
    assert store.embedding_dim == 384
    assert "p_001" in store
    assert "p_999" not in store

    # 1. Point lookup
    v1 = store.get_vector("p_001")
    assert v1 is not None
    assert v1.shape == (384,)
    np.testing.assert_allclose(v1, vectors[0])
    assert store.get_vector("non_existent") is None

    # 2. Batch lookup
    query_ids = ["p_002", "p_004", "missing_id"]
    batch_vecs, missing = store.get_vectors(query_ids, return_missing=True)
    assert len(batch_vecs) == 2
    assert batch_vecs.shape == (2, 384)
    assert missing == ["missing_id"]
    np.testing.assert_allclose(batch_vecs[0], vectors[1])
    np.testing.assert_allclose(batch_vecs[1], vectors[3])

    # 3. Filter by IDs
    sub_store = store.filter_by_ids(["p_003", "p_005"])
    assert len(sub_store) == 2
    assert list(sub_store.post_ids) == ["p_003", "p_005"]
    np.testing.assert_allclose(sub_store.vectors[0], vectors[2])


def test_validation_integrity_checks():
    # Trường hợp chuẩn: Đạt tất cả kiểm tra
    vectors_clean = np.random.randn(4, 384).astype(np.float32)
    ids_clean = ["p1", "p2", "p3", "p4"]
    rep = validate_embeddings(vectors_clean, ids_clean, expected_dim=384, strict=True)
    assert rep.is_valid is True
    assert len(rep.errors) == 0

    # Lỗi 1: Lệch số lượng vector và ID
    rep_len = validate_embeddings(vectors_clean, ["p1", "p2"], expected_dim=384, strict=True)
    assert rep_len.is_valid is False
    assert any("Lệch số lượng" in e for e in rep_len.errors)

    # Lỗi 2: Lệch kích thước vector
    rep_dim = validate_embeddings(vectors_clean, ids_clean, expected_dim=512, strict=True)
    assert rep_dim.is_valid is False
    assert any("không khớp" in e for e in rep_dim.errors)

    # Lỗi 3: Chứa NaN / Inf
    vectors_nan = vectors_clean.copy()
    vectors_nan[1, 10] = np.nan
    rep_nan = validate_embeddings(vectors_nan, ids_clean, expected_dim=384, strict=True)
    assert rep_nan.is_valid is False
    assert any("NaN" in e for e in rep_nan.errors)

    # Lỗi 4: ID trùng lặp
    rep_dup = validate_embeddings(vectors_clean, ["p1", "p2", "p2", "p4"], expected_dim=384, strict=True)
    assert rep_dup.is_valid is False
    assert any("trùng lặp" in e for e in rep_dup.errors)


def test_temporal_and_clustering_integration():
    vectors = np.random.randn(4, 384).astype(np.float32)
    post_ids = ["p_1", "p_2", "p_3", "p_4"]
    timestamps = [
        "2018-05-01 10:00:00+00:00",
        "2019-06-01 11:00:00+00:00",
        "2020-07-01 12:00:00+00:00",
        "2020-08-01 13:00:00+00:00",
    ]
    clusters = [0, 1, 0, 2]

    store = EmbeddingStore(
        vectors=vectors,
        post_ids=post_ids,
        timestamps=timestamps,
        cluster_ids=clusters,
    )

    # 1. Kiểm tra get_record
    rec = store.get_record("p_2")
    assert rec is not None
    assert rec["cluster_id"] == 1
    assert "2019-06-01" in str(rec["publish_datetime_utc"])

    # 2. Kiểm tra filter_by_timerange
    store_2020 = store.filter_by_timerange(start_date="2020-01-01", end_date="2020-12-31")
    assert len(store_2020) == 2
    assert list(store_2020.post_ids) == ["p_3", "p_4"]

    # 3. Kiểm tra to_clustering_dataframe
    df_clust = store.to_clustering_dataframe()
    assert len(df_clust) == 4
    assert list(df_clust.columns) == ["post_id", "is_valid", "publish_datetime_utc", "cluster_id"]
    assert list(df_clust["cluster_id"]) == [0, 1, 0, 2]


def test_save_and_load_all_formats():
    with tempfile.TemporaryDirectory() as tmpdir:
        vectors = np.random.randn(6, 384).astype(np.float32)
        post_ids = [f"post_{i}" for i in range(6)]
        timestamps = [f"2020-01-0{i+1} 00:00:00+00:00" for i in range(6)]
        clusters = [0, 1, 0, 1, 2, 2]

        meta = EmbeddingMetadata(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            num_vectors=6,
            model_version="v2.1",
            source_dataset="test_posts.csv",
        )
        store = EmbeddingStore(
            vectors=vectors,
            post_ids=post_ids,
            timestamps=timestamps,
            cluster_ids=clusters,
            metadata=meta,
        )

        # 1. Test NPZ format
        npz_file = os.path.join(tmpdir, "test.npz")
        store.save_npz(npz_file)
        loaded_npz = load_embeddings(npz_file, validate=True)
        assert len(loaded_npz) == 6
        assert loaded_npz.embedding_dim == 384
        assert loaded_npz.timestamps is not None
        assert loaded_npz.cluster_ids is not None
        assert list(loaded_npz.cluster_ids) == clusters
        np.testing.assert_allclose(loaded_npz.vectors, vectors, atol=1e-6)

        # 2. Test Parquet format
        pq_file = os.path.join(tmpdir, "test.parquet")
        store.save_parquet(pq_file)
        loaded_pq = load_embeddings(pq_file, validate=True)
        assert len(loaded_pq) == 6
        assert loaded_pq.embedding_dim == 384
        assert loaded_pq.timestamps is not None
        assert loaded_pq.cluster_ids is not None
        np.testing.assert_allclose(loaded_pq.vectors, vectors, atol=1e-6)

        # 3. Test Bundle Directory (metadata.json + embeddings.npz + embeddings.parquet)
        bundle_dir = os.path.join(tmpdir, "my_bundle")
        store.save_bundle(bundle_dir)
        assert os.path.exists(os.path.join(bundle_dir, "metadata.json"))
        assert os.path.exists(os.path.join(bundle_dir, "embeddings.npz"))

        loaded_bundle = load_embeddings(bundle_dir, validate=True)
        assert len(loaded_bundle) == 6
        assert loaded_bundle.embedding_dim == 384
        assert loaded_bundle.metadata.model_version == "v2.1"
        np.testing.assert_allclose(loaded_bundle.vectors, vectors, atol=1e-6)


if __name__ == "__main__":
    test_embedding_store_basic_and_lookups()
    test_validation_integrity_checks()
    test_temporal_and_clustering_integration()
    test_save_and_load_all_formats()
    print("[SUCCESS] ALL EMBEDDING STORE UNIT TESTS PASSED SUCCESSFULLY!")
