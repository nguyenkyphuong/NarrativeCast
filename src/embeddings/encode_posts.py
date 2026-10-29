"""
src/embeddings/encode_posts.py
Pipeline tạo vector embedding từ dữ liệu văn bản sạch cho đề tài NarrativeCast.

Chức năng chính:
- Nhận dữ liệu bài viết (DataFrame hoặc đường dẫn file CSV/Parquet như reddit_technology_clean.parquet).
- Tự động lọc văn bản hợp lệ (chỉ lấy is_usable_text = True theo quy định của đề tài).
- Xử lý các trường hợp văn bản rỗng, null, placeholder ([deleted], [removed]) hoặc lỗi mã hóa.
- Bảo toàn liên kết 1-1 giữa vector và post_id, đồng thời lưu giữ publish_datetime_utc để phục vụ
  Temporal Narrative Tracking (Nội dung 4).
- Cấu hình mô hình embedding theo tài liệu docs/embedding_model_selection.md
  (mặc định sentence-transformers/all-MiniLM-L6-v2; hỗ trợ BAAI/bge-small-en-v1.5).
- Hỗ trợ phân tích trên mẫu nhỏ (sample_size) và lọc theo khoảng thời gian (start_date, end_date),
  tránh tính toán lãng phí trên toàn bộ dataset 11,6 triệu dòng khi chưa chốt phạm vi chủ đề.
- Đọc file Parquet tối ưu theo row group, tránh tràn bộ nhớ RAM.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NarrativeCast.EncodePosts")

# Danh sách chuỗi rác / placeholder không mang ngữ nghĩa
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
    Lớp lưu trữ kết quả embedding, đảm bảo liên kết chặt chẽ giữa post_id,
    publish_datetime_utc và vector embedding.
    """
    post_ids: np.ndarray             # Mảng 1D chứa post_id dạng chuỗi (N,)
    vectors: np.ndarray              # Ma trận 2D (N, D) float32
    is_valid: np.ndarray             # Mảng 1D boolean (N,)
    model_name: str
    embedding_dim: int
    timestamps: Optional[np.ndarray] = None  # Mảng 1D chứa timestamp (N,) phục vụ temporal tracking
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

    def get_timestamp(self, post_id: str) -> Optional[Any]:
        """Lấy timestamp của một post_id cụ thể."""
        if self.timestamps is None:
            return None
        idx = np.where(self.post_ids == post_id)[0]
        if len(idx) == 0:
            return None
        return self.timestamps[idx[0]]

    def to_embedding_store(self) -> Any:
        """
        Chuyển đổi sang đối tượng EmbeddingStore của module src.embeddings.embedding_store.
        """
        from src.embeddings.embedding_store import EmbeddingMetadata, EmbeddingStore

        meta = EmbeddingMetadata(
            model_name=self.model_name,
            embedding_dim=self.embedding_dim,
            num_vectors=len(self.post_ids),
            model_version=self.metadata.get("model_version", "1.0"),
            normalize_embeddings=self.metadata.get("normalize_embeddings", True),
            max_seq_length=self.metadata.get("max_seq_length", 256),
            source_dataset=self.metadata.get("source_dataset"),
            extra=self.metadata,
        )
        return EmbeddingStore(
            vectors=self.vectors,
            post_ids=self.post_ids,
            timestamps=self.timestamps,
            metadata=meta,
            is_valid=self.is_valid,
            validate_on_init=True,
        )

    def save(self, output_path: Union[str, Path]) -> str:
        """
        Lưu kết quả embedding ra đĩa dạng .npz (nhẹ, tối ưu mảng numpy, bảo toàn post_ids và timestamps).
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
            "has_timestamps": self.timestamps is not None,
            "metadata": self.metadata,
        }, ensure_ascii=False)

        save_dict = {
            "post_ids": self.post_ids,
            "vectors": self.vectors,
            "is_valid": self.is_valid,
            "meta": np.array([meta_json]),
        }
        if self.timestamps is not None:
            save_dict["timestamps"] = self.timestamps

        np.savez_compressed(output_path, **save_dict)
        logger.info(f"Đã lưu {len(self.post_ids):,} vectors (dim={self.embedding_dim}) tại: {output_path}")
        return str(output_path)

    def save_parquet(
        self,
        output_path: Union[str, Path],
        source_df: Optional[pd.DataFrame] = None,
        id_col: str = "post_id",
        time_col: str = "publish_datetime_utc",
    ) -> str:
        """
        Lưu bảng dữ liệu kèm vector embedding dưới dạng file Parquet.
        Bảo toàn post_id, publish_datetime_utc và embedding.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data_dict: Dict[str, Any] = {
            id_col: self.post_ids,
            "embedding": [v.tolist() for v in self.vectors],
            "is_valid_embedding": self.is_valid,
        }
        if self.timestamps is not None:
            data_dict[time_col] = self.timestamps

        df_out = pd.DataFrame(data_dict)

        if source_df is not None and id_col in source_df.columns:
            # Gộp thêm các cột metadata phụ (tránh trùng cột đã có)
            other_cols = [c for c in source_df.columns if c not in df_out.columns]
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

        timestamps = data["timestamps"] if "timestamps" in data else None

        return cls(
            post_ids=data["post_ids"],
            vectors=data["vectors"],
            is_valid=data["is_valid"],
            timestamps=timestamps,
            model_name=meta_dict.get("model_name", "unknown"),
            embedding_dim=int(data["vectors"].shape[1]) if len(data["vectors"].shape) > 1 else 0,
            metadata=meta_dict.get("metadata", {}),
        )


class PostEncoder:
    """
    Lớp điều phối việc tạo vector embedding theo batch từ văn bản bài viết.
    Hỗ trợ cấu hình mô hình theo docs/embedding_model_selection.md.
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

        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Nạp mô hình thành công. Embedding dimension = {self.embedding_dim}")

    @staticmethod
    def _mean_pooling(model_output: Any, attention_mask: Any) -> Any:
        """
        Mean Pooling: Tính trung bình có trọng số attention_mask trên các token embeddings.
        Chuẩn kỹ thuật theo Sentence-BERT (SBERT).
        """
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def validate_text(self, text: Any) -> Tuple[bool, str]:
        """
        Kiểm tra tính hợp lệ của văn bản:
        - Phải là chuỗi ký tự hoặc có thể ép kiểu chuỗi.
        - Không được là null, chuỗi rỗng hoặc chỉ chứa khoảng trắng.
        - Không thuộc danh sách chuỗi placeholder rác ([deleted], [removed], nan...).
        """
        if text is None:
            return False, "null_or_none"

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
            dim = 384
            vecs = np.random.randn(len(batch_texts), dim).astype(np.float32)
            if self.normalize_embeddings:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vecs = vecs / norms
            return vecs

        import torch
        encoded_input = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])

            if self.normalize_embeddings:
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            return sentence_embeddings.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        """
        Mã hóa danh sách văn bản theo batch.
        """
        num_texts = len(texts)
        if num_texts == 0:
            dim = getattr(self, "embedding_dim", 384)
            return np.empty((0, dim), dtype=np.float32)

        vectors_list: List[np.ndarray] = []
        for i in range(0, num_texts, self.batch_size):
            batch_slice = texts[i : i + self.batch_size]
            batch_vecs = self._encode_batch(list(batch_slice))
            vectors_list.append(batch_vecs)

            if show_progress and (i + len(batch_slice)) % max(self.batch_size * 5, 200) < self.batch_size:
                logger.info(f"Tiến độ mã hóa: {min(i + len(batch_slice), num_texts):,}/{num_texts:,} văn bản")

        return np.vstack(vectors_list)

    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "content",
        id_col: str = "post_id",
        time_col: Optional[str] = "publish_datetime_utc",
        handle_empty: Literal["drop", "zero"] = "drop",
        filter_usable: bool = True,
        show_progress: bool = True,
    ) -> EmbeddingResult:
        """
        Nhận DataFrame dữ liệu sạch, tự động lọc is_usable_text == True (nếu có),
        xử lý văn bản rỗng/lỗi, tạo vector embedding theo batch và bảo toàn post_id kèm timestamp.

        Args:
            df: DataFrame bài đăng.
            text_col: Cột văn bản (mặc định 'content').
            id_col: Cột ID duy nhất (mặc định 'post_id').
            time_col: Cột thời gian chuẩn (mặc định 'publish_datetime_utc', hoặc None).
            handle_empty: 'drop' (loại bỏ dòng rỗng/lỗi) hoặc 'zero' (gán vector 0).
            filter_usable: Nếu DataFrame có cột 'is_usable_text', chỉ giữ dòng True.
            show_progress: Hiển thị tiến độ xử lý.
        """
        if id_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột định danh '{id_col}' trong DataFrame!")
        if text_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột văn bản '{text_col}' trong DataFrame!")

        working_df = df.copy()

        # 1. Áp dụng quy tắc cốt lõi: Chỉ lấy is_usable_text == True
        if filter_usable and "is_usable_text" in working_df.columns:
            initial_count = len(working_df)
            working_df = working_df[working_df["is_usable_text"] == True]
            dropped_unusable = initial_count - len(working_df)
            if dropped_unusable > 0:
                logger.info(
                    f"Đã lọc theo quy định (is_usable_text == True): loại bỏ {dropped_unusable:,} dòng không đạt chuẩn."
                )

        logger.info(f"Bắt đầu pipeline embedding cho {len(working_df):,} bản ghi (chế độ handle_empty='{handle_empty}')...")

        raw_ids = working_df[id_col].astype(str).values
        raw_texts = working_df[text_col].values

        # Trích xuất timestamps nếu cột thời gian tồn tại
        actual_time_col = None
        raw_timestamps = None
        if time_col and time_col in working_df.columns:
            actual_time_col = time_col
            raw_timestamps = working_df[time_col].values
        elif "publish_date" in working_df.columns:
            actual_time_col = "publish_date"
            raw_timestamps = working_df["publish_date"].values

        valid_mask = np.zeros(len(working_df), dtype=bool)
        clean_texts = []
        dropped_records = []

        # 2. Kiểm tra tính hợp lệ của từng văn bản
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

        dropped_df = pd.DataFrame(dropped_records) if dropped_records else None

        # 3. Mã hóa theo batch
        if num_valid > 0:
            valid_vectors = self.encode_texts(clean_texts, show_progress=show_progress)
            dim = valid_vectors.shape[1]
        else:
            dim = getattr(self, "embedding_dim", 384)
            valid_vectors = np.empty((0, dim), dtype=np.float32)

        # 4. Gom kết quả và đồng bộ timestamps
        if handle_empty == "drop":
            final_ids = raw_ids[valid_mask]
            final_vectors = valid_vectors
            final_valid_mask = np.ones(num_valid, dtype=bool)
            final_timestamps = raw_timestamps[valid_mask] if raw_timestamps is not None else None
        elif handle_empty == "zero":
            final_ids = raw_ids
            final_valid_mask = valid_mask
            final_vectors = np.zeros((len(working_df), dim), dtype=np.float32)
            if num_valid > 0:
                final_vectors[valid_mask] = valid_vectors
            final_timestamps = raw_timestamps
        else:
            raise ValueError(f"handle_empty không hợp lệ: '{handle_empty}'. Phải là 'drop' hoặc 'zero'.")

        result = EmbeddingResult(
            post_ids=final_ids,
            vectors=final_vectors,
            is_valid=final_valid_mask,
            timestamps=final_timestamps,
            model_name=self.model_name,
            embedding_dim=dim,
            metadata={
                "total_input_rows": len(df),
                "working_rows": len(working_df),
                "num_valid": num_valid,
                "num_invalid": num_invalid,
                "handle_empty": handle_empty,
                "time_col": actual_time_col,
                "batch_size": self.batch_size,
                "max_seq_length": self.max_seq_length,
                "normalize_embeddings": self.normalize_embeddings,
            },
            dropped_records=dropped_df,
        )

        logger.info(f"Hoàn thành embedding: {len(result.post_ids):,} vectors có kích thước nhất quán D={result.embedding_dim}")
        return result


def load_dataset_subset(
    file_path: Union[str, Path],
    sample_size: Optional[int] = None,
    filter_usable: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    id_col: str = "post_id",
    text_col: str = "content",
    time_col: str = "publish_datetime_utc",
) -> pd.DataFrame:
    """
    Nạp dữ liệu an toàn từ Parquet hoặc CSV:
    - Nếu file Parquet lớn (như reddit_technology_clean.parquet 2.37GB), đọc theo row groups
      để tránh nạp cả 11,6 triệu dòng vào RAM khi thử nghiệm trên mẫu nhỏ.
    - Lọc theo is_usable_text == True và khoảng thời gian (start_date, end_date) nếu yêu cầu.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    logger.info(f"Nạp dữ liệu từ: {path} (sample_size={sample_size}, filter_usable={filter_usable})")

    if path.suffix in (".parquet", ".pq"):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        needed_cols = [id_col, text_col]
        for opt_col in [time_col, "publish_date", "is_usable_text"]:
            if opt_col in pf.schema.names and opt_col not in needed_cols:
                needed_cols.append(opt_col)

        dfs: List[pd.DataFrame] = []
        collected_rows = 0

        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=needed_cols)
            rg_df = table.to_pandas()

            if filter_usable and "is_usable_text" in rg_df.columns:
                rg_df = rg_df[rg_df["is_usable_text"] == True]

            if time_col in rg_df.columns:
                if start_date:
                    rg_df = rg_df[rg_df[time_col] >= pd.to_datetime(start_date, utc=True)]
                if end_date:
                    rg_df = rg_df[rg_df[time_col] <= pd.to_datetime(end_date, utc=True)]

            dfs.append(rg_df)
            collected_rows += len(rg_df)

            if sample_size and collected_rows >= sample_size:
                break

        if not dfs:
            return pd.DataFrame(columns=needed_cols)

        df = pd.concat(dfs, ignore_index=True)
        if sample_size and len(df) > sample_size:
            df = df.iloc[:sample_size].copy()
        return df

    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path, nrows=sample_size if sample_size else None)
        if filter_usable and "is_usable_text" in df.columns:
            df = df[df["is_usable_text"] == True]
        return df
    else:
        raise ValueError(f"Định dạng file không được hỗ trợ: {path.suffix}")


def encode_posts(
    input_data: Union[str, Path, pd.DataFrame],
    text_col: str = "content",
    id_col: str = "post_id",
    time_col: Optional[str] = "publish_datetime_utc",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    max_seq_length: int = 256,
    handle_empty: Literal["drop", "zero"] = "drop",
    filter_usable: bool = True,
    sample_size: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_format: Literal["npz", "parquet", "both"] = "npz",
    device: Optional[str] = None,
) -> EmbeddingResult:
    """
    Hàm tiện ích cấp cao để tạo embedding từ dữ liệu bài viết và tùy chọn lưu file.
    Hỗ trợ mẫu nhỏ, lọc thời gian và định dạng lưu trữ chuẩn.
    """
    # 1. Đọc dữ liệu nếu input là đường dẫn
    if isinstance(input_data, (str, Path)):
        df = load_dataset_subset(
            file_path=input_data,
            sample_size=sample_size,
            filter_usable=filter_usable,
            start_date=start_date,
            end_date=end_date,
            id_col=id_col,
            text_col=text_col,
            time_col=time_col or "publish_datetime_utc",
        )
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
        if sample_size and len(df) > sample_size:
            df = df.iloc[:sample_size].copy()
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
        time_col=time_col,
        handle_empty=handle_empty,
        filter_usable=filter_usable,
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
            result.save_parquet(
                pq_path,
                source_df=df,
                id_col=id_col,
                time_col=time_col or "publish_datetime_utc",
            )

        if result.dropped_records is not None and len(result.dropped_records) > 0:
            dropped_path = out_dir / "dropped_posts.csv"
            result.dropped_records.to_csv(dropped_path, index=False)
            logger.info(f"Đã lưu danh sách bài đăng bị loại tại: {dropped_path}")

    return result


def main() -> None:
    """CLI script entrypoint."""
    parser = argparse.ArgumentParser(description="NarrativeCast — Post Embedding Pipeline")
    parser.add_argument("--input-path", type=str, default="data/processed/reddit_technology_clean.parquet",
                        help="Đường dẫn file đầu vào (.parquet hoặc .csv)")
    parser.add_argument("--output-dir", type=str, default="outputs/embeddings",
                        help="Thư mục xuất file embedding (.npz hoặc .parquet)")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Tên mô hình (mặc định sentence-transformers/all-MiniLM-L6-v2, hoặc BAAI/bge-small-en-v1.5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Độ dài chuỗi tối đa")
    parser.add_argument("--handle-empty", type=str, choices=["drop", "zero"], default="drop",
                        help="Chính sách xử lý văn bản rỗng: 'drop' hoặc 'zero'")
    parser.add_argument("--sample-size", type=int, default=1000,
                        help="Số lượng mẫu chạy thử nghiệm (mặc định 1000, tránh chạy hết 11,6M dòng)")
    parser.add_argument("--start-date", type=str, default=None, help="Lọc thời gian bắt đầu (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Lọc thời gian kết thúc (YYYY-MM-DD)")
    parser.add_argument("--filter-usable", action="store_true", default=True,
                        help="Chỉ lấy bài đăng có is_usable_text == True")
    parser.add_argument("--format", type=str, choices=["npz", "parquet", "both"], default="npz",
                        help="Định dạng lưu file")
    parser.add_argument("--device", type=str, default=None, help="cpu hoặc cuda")
    parser.add_argument("--test", action="store_true", help="Chạy bộ kiểm thử tự động")

    args = parser.parse_args()

    if args.test:
        logger.info("=== CHẠY KIỂM THỬ PIPELINE EMBEDDING TỰ ĐỘNG ===")
        test_df = pd.DataFrame({
            "post_id": ["p_001", "p_002", "p_empty", "p_deleted", "p_unusable", "p_valid_tech"],
            "content": [
                "Artificial intelligence and open weights models are advancing rapidly.",
                "Cloud infrastructure security and vulnerability management.",
                "",
                "[deleted]",
                "Short",
                "Antitrust regulations target big tech platform dominance.",
            ],
            "publish_datetime_utc": pd.to_datetime([
                "2020-01-01 10:00:00+00:00",
                "2020-01-02 11:00:00+00:00",
                "2020-01-03 12:00:00+00:00",
                "2020-01-04 13:00:00+00:00",
                "2020-01-05 14:00:00+00:00",
                "2020-01-06 15:00:00+00:00",
            ]),
            "is_usable_text": [True, True, True, True, False, True],
        })

        res = encode_posts(
            input_data=test_df,
            handle_empty="drop",
            filter_usable=True,
            model_name="mock",
            batch_size=2,
        )

        assert len(res.post_ids) == 3, f"Kỳ vọng 3 bản ghi hợp lệ, nhận {len(res.post_ids)}"
        assert res.vectors.shape == (3, 384)
        assert list(res.post_ids) == ["p_001", "p_002", "p_valid_tech"]
        assert res.timestamps is not None
        assert len(res.timestamps) == 3
        logger.info("[PASSED] Unit tests pipeline hoàn thành thành công 100%!")
        return

    encode_posts(
        input_data=args.input_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        handle_empty=args.handle_empty,
        filter_usable=args.filter_usable,
        sample_size=args.sample_size,
        start_date=args.start_date,
        end_date=args.end_date,
        save_format=args.format,
        device=args.device,
    )


if __name__ == "__main__":
    main()
