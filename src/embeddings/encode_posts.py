"""
src/embeddings/encode_posts.py
Pipeline tạo vector embedding từ dữ liệu văn bản sạch cho đề tài NarrativeCast.

Chức năng chính:
- Nhận dữ liệu bài viết (DataFrame hoặc đường dẫn file CSV/Parquet).
- Xử lý các trường hợp văn bản rỗng, null, placeholder ([deleted], [removed]) hoặc lỗi mã hóa.
- Mã hóa vector theo batch với cấu hình mô hình (mặc định sentence-transformers/all-MiniLM-L6-v2),
  độ dài chuỗi tối đa (max_seq_length), và chuẩn hóa L2 (normalize_embeddings).
- Bảo toàn liên kết 1-1 chính xác giữa vector và post_id.
- Hỗ trợ lưu trữ dạng .npz (vectors + post_ids) và .parquet.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NarrativeCast.Embedding")

# Danh sách các chuỗi đại diện cho dữ liệu rác / placeholder không có ngữ nghĩa
DEFAULT_INVALID_PLACEHOLDERS = {
    "",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "[deleted]",
    "[removed]",
}


@dataclass
class EmbeddingResult:
    """
    Lớp lưu trữ kết quả embedding, đảm bảo liên kết chặt chẽ giữa post_id và vector.
    """
    post_ids: np.ndarray  # Mảng 1D chứa post_id dạng chuỗi
    vectors: np.ndarray   # Ma trận 2D (N, D) float32
    is_valid: np.ndarray  # Mảng 1D boolean đánh dấu tính hợp lệ
    model_name: str
    embedding_dim: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    dropped_records: Optional[pd.DataFrame] = None

    def __len__(self) -> int:
        return len(self.post_ids)

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Trả về dictionary ánh xạ từ post_id -> vector numpy."""
        return {pid: self.vectors[i] for i, pid in enumerate(self.post_ids)}

    def get_vector(self, post_id: str) -> Optional[np.ndarray]:
        """Lấy vector của một post_id cụ thể."""
        idx = np.where(self.post_ids == post_id)[0]
        if len(idx) == 0:
            return None
        return self.vectors[idx[0]]

    def save(self, output_path: Union[str, Path]) -> str:
        """
        Lưu kết quả embedding ra đĩa dạng .npz (nhẹ, tối ưu cho mảng numpy và truy xuất ma trận).
        """
        output_path = Path(output_path)
        if output_path.suffix != ".npz":
            output_path = output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        meta_json = json.dumps({
            "model_name": self.model_name,
            "embedding_dim": int(self.embedding_dim),
            "num_records": int(len(self.post_ids)),
            "num_valid": int(np.sum(self.is_valid)),
            "metadata": self.metadata,
        }, ensure_ascii=False)

        np.savez_compressed(
            output_path,
            post_ids=self.post_ids,
            vectors=self.vectors,
            is_valid=self.is_valid,
            meta=np.array([meta_json]),
        )
        logger.info(f"Đã lưu {len(self.post_ids)} vectors (dim={self.embedding_dim}) tại: {output_path}")
        return str(output_path)

    def save_parquet(
        self,
        output_path: Union[str, Path],
        source_df: Optional[pd.DataFrame] = None,
        id_col: str = "post_id",
    ) -> str:
        """
        Lưu bảng dữ liệu kèm vector embedding dưới dạng file Parquet.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Chuyển ma trận numpy thành list of lists để lưu vào cột parquet
        df_out = pd.DataFrame({
            id_col: self.post_ids,
            "embedding": [v.tolist() for v in self.vectors],
            "is_valid_embedding": self.is_valid,
        })

        if source_df is not None and id_col in source_df.columns:
            # Merge thêm metadata từ DataFrame nguồn nếu có
            other_cols = [c for c in source_df.columns if c != id_col and c not in df_out.columns]
            if other_cols:
                meta_subset = source_df[[id_col] + other_cols].drop_duplicates(subset=[id_col])
                df_out = pd.merge(df_out, meta_subset, on=id_col, how="left")

        df_out.to_parquet(output_path, index=False)
        logger.info(f"Đã lưu bảng embedding Parquet tại: {output_path}")
        return str(output_path)

    @classmethod
    def load(cls, npz_path: Union[str, Path]) -> "EmbeddingResult":
        """Nạp kết quả embedding từ file .npz đã lưu."""
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        meta_dict = {}
        if "meta" in data:
            try:
                meta_dict = json.loads(str(data["meta"][0]))
            except Exception:
                pass

        return cls(
            post_ids=data["post_ids"],
            vectors=data["vectors"],
            is_valid=data["is_valid"],
            model_name=meta_dict.get("model_name", "unknown"),
            embedding_dim=int(data["vectors"].shape[1]) if len(data["vectors"].shape) > 1 else 0,
            metadata=meta_dict.get("metadata", {}),
        )


class PostEncoder:
    """
    Lớp điều phối việc tạo vector embedding theo batch từ văn bản bài viết.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        max_seq_length: int = 256,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        invalid_placeholders: Optional[set[str]] = None,
    ):
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        self.invalid_placeholders = invalid_placeholders or DEFAULT_INVALID_PLACEHOLDERS

        self.tokenizer = None
        self.model = None
        self.device = device
        self._is_mock = model_name.lower() in ("mock", "test", "dummy")

        if not self._is_mock:
            self._init_model()

    def _init_model(self) -> None:
        """Khởi tạo tokenizer và model từ Hugging Face transformers."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Đang nạp mô hình embedding '{self.model_name}' trên thiết bị '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Xác định chiều của vector embedding
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Nạp mô hình thành công. Embedding dimension = {self.embedding_dim}")

    @staticmethod
    def _mean_pooling(model_output: Any, attention_mask: Any) -> Any:
        """
        Mean Pooling: Lấy trung bình có trọng số attention_mask trên các token embeddings.
        Đây là kỹ thuật chuẩn của SBERT (Sentence-BERT).
        """
        import torch
        token_embeddings = model_output[0]  # First element of model_output contains hidden state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def validate_text(self, text: Any) -> Tuple[bool, str]:
        """
        Kiểm tra tính hợp lệ của văn bản:
        - Phải là chuỗi ký tự hoặc có thể ép kiểu an toàn.
        - Không được là null, chuỗi rỗng hoặc chỉ chứa khoảng trắng.
        - Không thuộc danh sách chuỗi placeholder ([deleted], [removed], nan...).

        Returns:
            Tuple[is_valid (bool), reason_or_clean_text (str)]
        """
        if text is None:
            return False, "null_or_none"

        # Kiểm tra kiểu số NaN từ pandas/numpy
        if isinstance(text, float) and np.isnan(text):
            return False, "nan_float"

        if not isinstance(text, str):
            text = str(text)

        cleaned = text.strip()
        if len(cleaned) == 0:
            return False, "empty_string"

        if cleaned.lower() in self.invalid_placeholders:
            return False, f"placeholder_{cleaned.lower()}"

        return True, cleaned

    def _encode_batch(self, batch_texts: List[str]) -> np.ndarray:
        """Mã hóa một batch văn bản thành ma trận numpy (batch_size, embedding_dim)."""
        if self._is_mock:
            # Mock mode phục vụ testing siêu nhanh mà không cần tải model
            dim = 384
            rng = np.random.default_rng(seed=42)
            embs = rng.normal(size=(len(batch_texts), dim)).astype(np.float32)
            if self.normalize_embeddings:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / np.maximum(norms, 1e-9)
            return embs

        import torch
        import torch.nn.functional as F

        # Tokenize văn bản với truncation theo max_seq_length
        encoded_input = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])

            if self.normalize_embeddings:
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Tạo vector embedding cho một danh sách văn bản theo batch.
        """
        if len(texts) == 0:
            dim = getattr(self, "embedding_dim", 384)
            return np.empty((0, dim), dtype=np.float32)

        results = []
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(num_batches):
            batch = texts[i * self.batch_size : (i + 1) * self.batch_size]
            batch_emb = self._encode_batch(batch)
            results.append(batch_emb)
            if show_progress and (i + 1) % max(1, num_batches // 10) == 0:
                logger.info(f"Tiến độ mã hóa: {min(len(texts), (i + 1) * self.batch_size)}/{len(texts)} văn bản")

        return np.vstack(results)

    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "content",
        id_col: str = "post_id",
        handle_empty: Literal["drop", "zero"] = "drop",
        show_progress: bool = True,
    ) -> EmbeddingResult:
        """
        Nhận DataFrame dữ liệu sạch, xử lý các trường hợp văn bản rỗng/lỗi,
        tạo vector embedding theo batch và bảo toàn liên kết chính xác với ID bài đăng.

        Args:
            df: DataFrame chứa bài đăng.
            text_col: Tên cột văn bản (mặc định 'content').
            id_col: Tên cột định danh duy nhất (mặc định 'post_id').
            handle_empty:
                - 'drop': Loại bỏ các dòng rỗng/lỗi, chỉ trả về các bài đăng hợp lệ.
                - 'zero': Giữ nguyên toàn bộ số dòng, gán vector 0 cho các dòng rỗng/lỗi.
            show_progress: Hiển thị tiến độ xử lý.

        Returns:
            EmbeddingResult: Đối tượng chứa post_ids, vectors, is_valid và metadata.
        """
        if id_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột định danh '{id_col}' trong DataFrame!")
        if text_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột văn bản '{text_col}' trong DataFrame!")

        logger.info(f"Bắt đầu pipeline embedding cho {len(df):,} bản ghi (chế độ handle_empty='{handle_empty}')...")

        raw_ids = df[id_col].astype(str).values
        raw_texts = df[text_col].values

        valid_mask = np.zeros(len(df), dtype=bool)
        clean_texts = []
        dropped_records = []

        # 1. Kiểm tra từng văn bản và phân loại
        for idx, (pid, raw_t) in enumerate(zip(raw_ids, raw_texts)):
            is_valid, result_val = self.validate_text(raw_t)
            if is_valid:
                valid_mask[idx] = True
                clean_texts.append(result_val)
            else:
                dropped_records.append({
                    "row_index": idx,
                    id_col: pid,
                    "reason": result_val,
                    "raw_value": str(raw_t)[:100],
                })

        num_valid = len(clean_texts)
        num_invalid = len(dropped_records)
        logger.info(f"Kết quả kiểm tra văn bản: {num_valid:,} hợp lệ | {num_invalid:,} không hợp lệ / rỗng")

        if num_invalid > 0:
            reasons = pd.DataFrame(dropped_records)["reason"].value_counts().to_dict()
            logger.info(f"Phân bố lý do không hợp lệ: {reasons}")

        dropped_df = pd.DataFrame(dropped_records) if dropped_records else None

        # 2. Xử lý tạo embedding theo batch cho các văn bản hợp lệ
        if num_valid > 0:
            valid_vectors = self.encode_texts(clean_texts, show_progress=show_progress)
            dim = valid_vectors.shape[1]
        else:
            dim = getattr(self, "embedding_dim", 384)
            valid_vectors = np.empty((0, dim), dtype=np.float32)

        # 3. Gom kết quả theo chính sách handle_empty
        if handle_empty == "drop":
            final_ids = raw_ids[valid_mask]
            final_vectors = valid_vectors
            final_valid_mask = np.ones(num_valid, dtype=bool)
        elif handle_empty == "zero":
            final_ids = raw_ids
            final_valid_mask = valid_mask
            final_vectors = np.zeros((len(df), dim), dtype=np.float32)
            if num_valid > 0:
                final_vectors[valid_mask] = valid_vectors
        else:
            raise ValueError(f"handle_empty không hợp lệ: '{handle_empty}'. Phải là 'drop' hoặc 'zero'.")

        result = EmbeddingResult(
            post_ids=final_ids,
            vectors=final_vectors,
            is_valid=final_valid_mask,
            model_name=self.model_name,
            embedding_dim=dim,
            metadata={
                "total_input_rows": len(df),
                "num_valid": num_valid,
                "num_invalid": num_invalid,
                "handle_empty": handle_empty,
                "batch_size": self.batch_size,
                "max_seq_length": self.max_seq_length,
                "normalize_embeddings": self.normalize_embeddings,
            },
            dropped_records=dropped_df,
        )

        logger.info(f"Hoàn thành embedding: {len(result.post_ids):,} vectors có kích thước nhất quán D={result.embedding_dim}")
        return result


def encode_posts(
    input_data: Union[str, Path, pd.DataFrame],
    text_col: str = "content",
    id_col: str = "post_id",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    max_seq_length: int = 256,
    handle_empty: Literal["drop", "zero"] = "drop",
    output_dir: Optional[Union[str, Path]] = None,
    save_format: Literal["npz", "parquet", "both"] = "npz",
    device: Optional[str] = None,
) -> EmbeddingResult:
    """
    Hàm tiện ích cấp cao để tạo embedding từ dữ liệu bài viết và tùy chọn lưu file.

    Args:
        input_data: DataFrame hoặc đường dẫn file .csv / .parquet.
        text_col: Tên cột chứa văn bản nội dung.
        id_col: Tên cột mã định danh bài viết.
        model_name: Tên mô hình Hugging Face.
        batch_size: Kích thước mỗi batch khi suy luận.
        max_seq_length: Độ dài chuỗi token tối đa.
        handle_empty: 'drop' (bỏ qua bản ghi rỗng) hoặc 'zero' (giữ lại và gán vector 0).
        output_dir: Thư mục lưu kết quả (tùy chọn).
        save_format: 'npz', 'parquet' hoặc 'both'.
        device: 'cuda', 'cpu' hoặc None (tự động chọn).

    Returns:
        EmbeddingResult: Kết quả embedding bảo toàn liên kết với post_id.
    """
    # 1. Đọc dữ liệu nếu input là đường dẫn
    if isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
        if not input_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file đầu vào: {input_path}")
        logger.info(f"Đọc dữ liệu đầu vào từ: {input_path}")
        if input_path.suffix in (".parquet", ".pq"):
            df = pd.read_parquet(input_path)
        elif input_path.suffix in (".csv", ".txt"):
            df = pd.read_csv(input_path, keep_default_na=True)
        else:
            raise ValueError(f"Định dạng file không được hỗ trợ: {input_path.suffix}")
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data phải là DataFrame hoặc đường dẫn file (str / Path)!")

    # 2. Khởi tạo encoder và thực hiện mã hóa
    encoder = PostEncoder(
        model_name=model_name,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        device=device,
    )
    result = encoder.encode_dataframe(
        df=df,
        text_col=text_col,
        id_col=id_col,
        handle_empty=handle_empty,
    )

    # 3. Lưu kết quả nếu có output_dir
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if save_format in ("npz", "both"):
            npz_path = out_dir / "post_embeddings.npz"
            result.save(npz_path)

        if save_format in ("parquet", "both"):
            pq_path = out_dir / "post_embeddings.parquet"
            result.save_parquet(pq_path, source_df=df, id_col=id_col)

        # Lưu thêm log các bài bị loại (nếu có)
        if result.dropped_records is not None and len(result.dropped_records) > 0:
            dropped_path = out_dir / "dropped_posts.csv"
            result.dropped_records.to_csv(dropped_path, index=False)
            logger.info(f"Đã lưu danh sách bài đăng bị loại tại: {dropped_path}")

    return result


def main() -> None:
    """CLI script entrypoint."""
    parser = argparse.ArgumentParser(description="NarrativeCast — Post Embedding Pipeline")
    parser.add_argument("--input-path", type=str, default="data/samples/posts_sample.csv",
                        help="Đường dẫn file đầu vào (.csv hoặc .parquet)")
    parser.add_argument("--output-dir", type=str, default="outputs/embeddings",
                        help="Thư mục xuất file embedding (.npz hoặc .parquet)")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Tên mô hình HuggingFace (mặc định sentence-transformers/all-MiniLM-L6-v2)")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Độ dài chuỗi tối đa")
    parser.add_argument("--handle-empty", type=str, choices=["drop", "zero"], default="drop",
                        help="Chính sách xử lý văn bản rỗng: 'drop' hoặc 'zero'")
    parser.add_argument("--format", type=str, choices=["npz", "parquet", "both"], default="both",
                        help="Định dạng lưu file")
    parser.add_argument("--device", type=str, default=None, help="cpu hoặc cuda")
    parser.add_argument("--test", action="store_true", help="Chạy bộ kiểm thử tự động với dữ liệu mẫu và edge-cases")

    args = parser.parse_args()

    if args.test:
        logger.info("=== CHẠY BỘ KIỂM THỬ PIPELINE EMBEDDING TỰ ĐỘNG ===")
        # 1. Tạo test dataframe với đầy đủ các trường hợp edge cases
        test_df = pd.DataFrame({
            "post_id": ["p_001", "p_002", "p_empty", "p_whitespace", "p_deleted", "p_none", "p_valid_tech"],
            "content": [
                "Artificial intelligence and open weights models are advancing rapidly.",
                "Cloud infrastructure security and vulnerability management.",
                "",               # Rỗng
                "   \t\n   ",     # Khoảng trắng
                "[deleted]",      # Placeholder
                None,             # None
                "Antitrust regulations target big tech platform dominance.",
            ],
            "publish_date": [1704067200, 1704070800, 1704074400, 1704078000, 1704081600, 1704085200, 1704088800],
        })

        # Test mode 'drop'
        logger.info("\n--- TEST 1: CHẾ ĐỘ HANDLE_EMPTY = 'DROP' ---")
        res_drop = encode_posts(
            input_data=test_df,
            handle_empty="drop",
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
        )
        assert len(res_drop.post_ids) == 3, f"Kỳ vọng 3 bản ghi hợp lệ, nhận {len(res_drop.post_ids)}"
        assert res_drop.vectors.shape == (3, res_drop.embedding_dim), f"Kỳ vọng shape (3, {res_drop.embedding_dim})"
        assert list(res_drop.post_ids) == ["p_001", "p_002", "p_valid_tech"]
        logger.info("✅ TEST 1 ĐẠT: Lọc sạch 4 bản ghi lỗi, bảo toàn chính xác 3 bài hợp lệ kèm vector chuẩn.")

        # Test mode 'zero'
        logger.info("\n--- TEST 2: CHẾ ĐỘ HANDLE_EMPTY = 'ZERO' ---")
        res_zero = encode_posts(
            input_data=test_df,
            handle_empty="zero",
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
        )
        assert len(res_zero.post_ids) == 7, f"Kỳ vọng 7 bản ghi, nhận {len(res_zero.post_ids)}"
        assert res_zero.vectors.shape == (7, res_zero.embedding_dim)
        assert np.all(res_zero.vectors[2] == 0), "Dòng rỗng phải có vector 0"
        assert res_zero.is_valid[0] and not res_zero.is_valid[2]
        logger.info("✅ TEST 2 ĐẠT: Giữ nguyên 7 dòng, gán vector 0 và cờ invalid cho các dòng lỗi.")

        # Test trên file mẫu posts_sample.csv nếu tồn tại
        if Path(args.input_path).exists():
            logger.info(f"\n--- TEST 3: CHẠY TRÊN FILE DỮ LIỆU MẪU '{args.input_path}' ---")
            res_file = encode_posts(
                input_data=args.input_path,
                output_dir=args.output_dir,
                save_format=args.format,
                model_name=args.model_name,
                batch_size=args.batch_size,
                device=args.device,
            )
            assert len(res_file) > 0, "Kết quả từ file mẫu không được rỗng"
            logger.info(f"✅ TEST 3 ĐẠT: Xử lý thành công file mẫu, lưu kết quả tại '{args.output_dir}'.")

        logger.info("\n🎉 TẤT CẢ CÁC TIÊU CHÍ KIỂM THỬ ĐỀU ĐẠT CHUẨN 100%!")
        return

    # Chạy pipeline thông thường
    encode_posts(
        input_data=args.input_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        handle_empty=args.handle_empty,
        save_format=args.format,
        device=args.device,
    )


if __name__ == "__main__":
    main()
