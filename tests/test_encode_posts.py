"""
tests/test_encode_posts.py
Bộ kiểm thử tự động cho pipeline tạo embedding bài viết (src/embeddings/encode_posts.py).
"""

import os
import tempfile
import numpy as np
import pandas as pd

from src.embeddings.encode_posts import (
    EmbeddingResult,
    PostEncoder,
    encode_posts,
    load_dataset_subset,
)


def test_post_encoder_text_validation():
    encoder = PostEncoder(model_name="mock")

    # Các trường hợp hợp lệ
    valid, text = encoder.validate_text("Valid technology discussion text")
    assert valid is True
    assert text == "Valid technology discussion text"

    # Các trường hợp không hợp lệ
    assert encoder.validate_text(None)[0] is False
    assert encoder.validate_text(np.nan)[0] is False
    assert encoder.validate_text("")[0] is False
    assert encoder.validate_text("    \n\t  ")[0] is False
    assert encoder.validate_text("[deleted]")[0] is False
    assert encoder.validate_text("[removed]")[0] is False
    assert encoder.validate_text("nan")[0] is False
    assert encoder.validate_text("null")[0] is False


def test_encode_dataframe_drop_mode():
    df = pd.DataFrame({
        "post_id": ["p1", "p2", "p3_empty", "p4_deleted", "p5"],
        "content": [
            "Open source language models are gaining popularity.",
            "Semiconductor fabrication facility expansion.",
            "",             # empty
            "[deleted]",    # placeholder
            "Antitrust lawsuit filed against tech monopoly.",
        ],
    })

    result = encode_posts(
        input_data=df,
        handle_empty="drop",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=2,
    )

    # Tiêu chí: Lọc bỏ dòng lỗi, vector có kích thước nhất quán D=384 và ánh xạ chính xác
    assert len(result.post_ids) == 3
    assert result.vectors.shape == (3, 384)
    assert list(result.post_ids) == ["p1", "p2", "p5"]
    assert result.embedding_dim == 384
    assert result.get_vector("p1") is not None
    assert result.get_vector("p3_empty") is None

    # Vector phải được chuẩn hóa L2 (norm ~ 1.0)
    norms = np.linalg.norm(result.vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_encode_dataframe_zero_mode():
    df = pd.DataFrame({
        "post_id": ["p1", "p2_empty"],
        "content": ["Artificial intelligence guardrails.", None],
    })

    result = encode_posts(
        input_data=df,
        handle_empty="zero",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=2,
    )

    # Tiêu chí: Giữ nguyên 2 dòng, gán vector 0 cho dòng lỗi
    assert len(result.post_ids) == 2
    assert result.vectors.shape == (2, 384)
    assert np.all(result.vectors[1] == 0)
    assert result.is_valid[0] is True or result.is_valid[0] == 1
    assert result.is_valid[1] is False or result.is_valid[1] == 0


def test_encode_with_usable_filter_and_timestamps():
    df = pd.DataFrame({
        "post_id": ["p_001", "p_002", "p_unusable", "p_003"],
        "content": [
            "Tech antitrust scrutiny increases in Europe.",
            "Quantum computing quantum supremacy claims.",
            "Too short",
            "Renewable energy data center powering AI.",
        ],
        "publish_datetime_utc": pd.to_datetime([
            "2020-01-01 10:00:00+00:00",
            "2020-01-02 11:00:00+00:00",
            "2020-01-03 12:00:00+00:00",
            "2020-01-04 13:00:00+00:00",
        ]),
        "is_usable_text": [True, True, False, True],
    })

    result = encode_posts(
        input_data=df,
        handle_empty="drop",
        filter_usable=True,
        model_name="mock",
        batch_size=2,
    )

    # Dòng p_unusable bị loại do is_usable_text == False
    assert len(result.post_ids) == 3
    assert list(result.post_ids) == ["p_001", "p_002", "p_003"]
    assert result.timestamps is not None
    assert len(result.timestamps) == 3
    assert str(result.get_timestamp("p_001")).startswith("2020-01-01")

    # Chuyển đổi sang EmbeddingStore
    store = result.to_embedding_store()
    assert len(store) == 3
    assert store.timestamps is not None
    assert store.validate().is_valid


def test_save_and_load_embeddings():
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame({
            "post_id": ["p_test_1", "p_test_2"],
            "content": ["First test post.", "Second test post."],
            "publish_datetime_utc": pd.to_datetime(["2020-05-01 00:00:00+00:00", "2020-05-02 00:00:00+00:00"]),
            "is_usable_text": [True, True],
        })

        result = encode_posts(
            input_data=df,
            output_dir=tmpdir,
            save_format="both",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        npz_file = os.path.join(tmpdir, "post_embeddings.npz")
        pq_file = os.path.join(tmpdir, "post_embeddings.parquet")

        assert os.path.exists(npz_file)
        assert os.path.exists(pq_file)

        # Kiểm tra đọc lại NPZ
        loaded = EmbeddingResult.load(npz_file)
        assert len(loaded.post_ids) == 2
        assert loaded.vectors.shape == (2, 384)
        assert list(loaded.post_ids) == ["p_test_1", "p_test_2"]
        assert loaded.timestamps is not None
        np.testing.assert_allclose(result.vectors, loaded.vectors, atol=1e-6)

        # Kiểm tra đọc lại Parquet
        df_loaded = pd.read_parquet(pq_file)
        assert len(df_loaded) == 2
        assert "embedding" in df_loaded.columns
        assert "publish_datetime_utc" in df_loaded.columns
        assert len(df_loaded["embedding"].iloc[0]) == 384


def test_load_dataset_subset_from_clean_parquet():
    clean_parquet = "data/processed/reddit_technology_clean.parquet"
    if os.path.exists(clean_parquet):
        # Nạp mẫu 10 dòng từ file clean thật mà không tốn RAM
        df_sub = load_dataset_subset(clean_parquet, sample_size=10, filter_usable=True)
        assert len(df_sub) == 10
        assert "post_id" in df_sub.columns
        assert "content" in df_sub.columns
        assert "publish_datetime_utc" in df_sub.columns
        assert (df_sub["is_usable_text"] == True).all()


if __name__ == "__main__":
    test_post_encoder_text_validation()
    test_encode_dataframe_drop_mode()
    test_encode_dataframe_zero_mode()
    test_encode_with_usable_filter_and_timestamps()
    test_save_and_load_embeddings()
    test_load_dataset_subset_from_clean_parquet()
    print("[SUCCESS] ALL UNIT TESTS PASSED SUCCESSFULLY!")
