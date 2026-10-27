"""
tests/test_encode_posts.py
Bộ kiểm thử tự động cho pipeline tạo embedding bài viết (src/embeddings/encode_posts.py).
"""

import os
import tempfile
import numpy as np
import pandas as pd

from src.embeddings.encode_posts import EmbeddingResult, PostEncoder, encode_posts


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


def test_save_and_load_embeddings():
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame({
            "post_id": ["p_test_1", "p_test_2"],
            "content": ["First test post.", "Second test post."],
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
        np.testing.assert_allclose(result.vectors, loaded.vectors, atol=1e-6)

        # Kiểm tra đọc lại Parquet
        df_loaded = pd.read_parquet(pq_file)
        assert len(df_loaded) == 2
        assert "embedding" in df_loaded.columns
        assert len(df_loaded["embedding"].iloc[0]) == 384


def test_encode_on_posts_sample_csv():
    sample_csv = "data/samples/posts_sample.csv"
    if os.path.exists(sample_csv):
        result = encode_posts(
            input_data=sample_csv,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=16,
        )
        assert len(result.post_ids) == 5
        assert result.vectors.shape == (5, 384)
        assert "p_rd_101" in result.to_dict()


if __name__ == "__main__":
    test_post_encoder_text_validation()
    test_encode_dataframe_drop_mode()
    test_encode_dataframe_zero_mode()
    test_save_and_load_embeddings()
    test_encode_on_posts_sample_csv()
    print("[SUCCESS] ALL UNIT TESTS PASSED SUCCESSFULLY!")
