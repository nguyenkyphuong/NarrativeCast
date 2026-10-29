"""
src/embeddings/embedding_store.py
Module quản lý lưu trữ, nạp lại và kiểm tra tính toàn vẹn của vector embedding và metadata.

Chức năng chính:
- Lưu trữ vector embedding kèm post_id, timestamps (publish_datetime_utc) và metadata mô hình,
  tránh phải tính toán lại trong các lần thử nghiệm tiếp theo (Zero Redundant Computation).
- Hỗ trợ lưu trữ kết quả phân cụm (cluster_id) liên kết với từng post_id cho các module tiếp theo.
- Hỗ trợ các định dạng lưu trữ tối ưu:
  + NumPy compressed (.npz): Rất nhẹ, nạp nhanh, tối ưu cho huấn luyện và phân cụm (BERTopic/HDBSCAN).
  + Apache Parquet (.parquet): Phù hợp truy vấn dữ liệu quan hệ, SQL joins và trích xuất đặc trưng.
  + Bundle directory: Gói thư mục chuẩn hóa gồm metadata.json, embeddings.npz và embeddings.parquet.
- Kiểm tra tính toàn vẹn dữ liệu nghiêm ngặt (ValidationReport):
  + Số lượng vector khớp chính xác số lượng post_id (và timestamps, cluster_id nếu có).
  + Số chiều vector nhất quán và khớp với metadata.
  + Không chứa giá trị NaN/Inf, không chứa post_id rỗng hoặc trùng lặp.
- Cung cấp truy xuất O(1) theo post_id, batch lookup, trích xuất theo ID (filter_by_ids)
  và trích xuất theo khoảng thời gian (filter_by_timerange) phục vụ Temporal Tracking (Module 4).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("NarrativeCast.EmbeddingStore")


@dataclass
class ValidationReport:
    """Kết quả kiểm tra tính toàn vẹn của kho embedding."""
    is_valid: bool
    num_vectors: int
    num_ids: int
    embedding_dim: int
    checks_passed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"<ValidationReport status={status} vectors={self.num_vectors:,} "
            f"dim={self.embedding_dim} errors={len(self.errors)} warnings={len(self.warnings)}>"
        )

    def summary(self) -> str:
        """Xuất chuỗi tóm tắt kết quả kiểm tra (an toàn UTF-8 trên Windows Console)."""
        lines = [
            f"=== BÁO CÁO KIỂM TRA TÍNH TOÀN VẸN EMBEDDING ===",
            f"Trạng thái: {'[PASSED] ĐẠT CHUẨN' if self.is_valid else '[FAILED] KHÔNG HỢP LỆ'}",
            f"Số lượng vector: {self.num_vectors:,}",
            f"Số lượng post_id: {self.num_ids:,}",
            f"Kích thước vector (dim): {self.embedding_dim}",
        ]
        if self.checks_passed:
            lines.append("Các tiêu chí đạt:")
            for c in self.checks_passed:
                lines.append(f"  [+] {c}")
        if self.warnings:
            lines.append("Cảnh báo (Warnings):")
            for w in self.warnings:
                lines.append(f"  [!] {w}")
        if self.errors:
            lines.append("Lỗi nghiêm trọng (Errors):")
            for e in self.errors:
                lines.append(f"  [-] {e}")
        return "\n".join(lines)


@dataclass
class EmbeddingMetadata:
    """Metadata quản lý phiên bản, nguồn gốc và cấu hình mô hình tạo embedding."""
    model_name: str
    embedding_dim: int
    num_vectors: int
    model_version: str = "1.0"
    normalize_embeddings: bool = True
    max_seq_length: int = 256
    pooling_strategy: str = "mean"
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    source_dataset: Optional[str] = None
    time_range: Optional[Tuple[str, str]] = None
    topic_scope: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi metadata thành dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Chuyển đổi metadata thành chuỗi JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingMetadata":
        """Khởi tạo từ dictionary."""
        valid_fields = {
            "model_name",
            "embedding_dim",
            "num_vectors",
            "model_version",
            "normalize_embeddings",
            "max_seq_length",
            "pooling_strategy",
            "created_at",
            "source_dataset",
            "time_range",
            "topic_scope",
            "extra",
        }
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "EmbeddingMetadata":
        """Khởi tạo từ chuỗi JSON."""
        return cls.from_dict(json.loads(json_str))


class EmbeddingStore:
    """
    Lớp quản lý lưu trữ, truy xuất và xác thực vector embedding kèm metadata.
    Đảm bảo vector luôn gắn chặt 1-1 với ID bài đăng gốc (post_id) và thời gian (publish_datetime_utc).
    """

    def __init__(
        self,
        vectors: np.ndarray,
        post_ids: Union[np.ndarray, Sequence[str]],
        timestamps: Optional[Union[np.ndarray, Sequence[Any]]] = None,
        cluster_ids: Optional[Union[np.ndarray, Sequence[Any]]] = None,
        metadata: Optional[Union[EmbeddingMetadata, Dict[str, Any]]] = None,
        is_valid: Optional[np.ndarray] = None,
        validate_on_init: bool = True,
    ):
        # 1. Chuẩn hóa vectors thành mảng float32 (N, D)
        if not isinstance(vectors, np.ndarray):
            vectors = np.asarray(vectors, dtype=np.float32)
        elif vectors.dtype not in (np.float32, np.float64, np.float16):
            vectors = vectors.astype(np.float32)

        # 2. Chuẩn hóa post_ids thành mảng chuỗi 1D (N,)
        if not isinstance(post_ids, np.ndarray):
            post_ids = np.asarray(post_ids, dtype=str)
        else:
            post_ids = post_ids.astype(str)

        self.vectors = vectors
        self.post_ids = post_ids

        # 3. Chuẩn hóa timestamps (N,) nếu có
        if timestamps is not None:
            if not isinstance(timestamps, np.ndarray):
                self.timestamps = np.asarray(timestamps)
            else:
                self.timestamps = timestamps
        else:
            self.timestamps = None

        # 4. Chuẩn hóa cluster_ids (N,) nếu có
        if cluster_ids is not None:
            if not isinstance(cluster_ids, np.ndarray):
                self.cluster_ids = np.asarray(cluster_ids)
            else:
                self.cluster_ids = cluster_ids
        else:
            self.cluster_ids = None

        # 5. Trạng thái hợp lệ (is_valid mask)
        if is_valid is None:
            self.is_valid = np.ones(len(self.post_ids), dtype=bool)
        else:
            self.is_valid = np.asarray(is_valid, dtype=bool)

        # 6. Thiết lập metadata
        dim = self.vectors.shape[1] if self.vectors.ndim > 1 else 0
        if metadata is None:
            self.metadata = EmbeddingMetadata(
                model_name="unknown",
                embedding_dim=dim,
                num_vectors=len(self.post_ids),
            )
        elif isinstance(metadata, dict):
            metadata_clean = metadata.copy()
            metadata_clean.setdefault("embedding_dim", dim)
            metadata_clean.setdefault("num_vectors", len(self.post_ids))
            metadata_clean.setdefault("model_name", "unknown")
            self.metadata = EmbeddingMetadata.from_dict(metadata_clean)
        elif isinstance(metadata, EmbeddingMetadata):
            self.metadata = metadata
        else:
            raise TypeError("metadata phải là EmbeddingMetadata hoặc dict!")

        # 7. Bộ đệm chỉ mục O(1) post_id -> index
        self._id_to_idx: Optional[Dict[str, int]] = None

        # 8. Kiểm tra tính toàn vẹn khi khởi tạo
        if validate_on_init:
            report = self.validate(strict=False)
            if not report.is_valid:
                logger.warning(f"Cảnh báo khi khởi tạo EmbeddingStore: {report.errors}")

    @property
    def embedding_dim(self) -> int:
        """Kích thước vector (chiều đặc trưng)."""
        return int(self.vectors.shape[1]) if self.vectors.ndim > 1 else 0

    @property
    def id_map(self) -> Dict[str, int]:
        """Tạo hoặc trả về bảng ánh xạ từ post_id sang chỉ số hàng (O(1))."""
        if self._id_to_idx is None:
            self._id_to_idx = {pid: i for i, pid in enumerate(self.post_ids)}
        return self._id_to_idx

    def __len__(self) -> int:
        return len(self.post_ids)

    def __contains__(self, post_id: str) -> bool:
        return post_id in self.id_map

    def attach_clusters(self, cluster_ids: Sequence[Any]) -> None:
        """
        Gắn nhãn phân cụm (cluster_id) từ module Clustering (Module 2/3) vào kho lưu trữ.
        Đáp ứng yêu cầu: 'Kết quả clustering phải lưu cluster_id tương ứng với từng post_id'.
        """
        arr = np.asarray(cluster_ids)
        if len(arr) != len(self.post_ids):
            raise ValueError(
                f"Độ dài mảng cluster_ids ({len(arr):,}) không khớp số lượng post_ids ({len(self.post_ids):,})!"
            )
        self.cluster_ids = arr
        logger.info(f"Đã gắn nhãn {len(arr):,} cluster_id vào EmbeddingStore.")

    def get_vector(self, post_id: str) -> Optional[np.ndarray]:
        """
        Truy xuất vector của một bài viết cụ thể theo post_id.
        Trả về None nếu không tìm thấy ID.
        """
        idx = self.id_map.get(str(post_id))
        if idx is None:
            return None
        return self.vectors[idx]

    def get_record(self, post_id: str) -> Optional[Dict[str, Any]]:
        """
        Truy xuất toàn bộ thông tin (vector, timestamp, cluster_id, is_valid) theo post_id.
        """
        idx = self.id_map.get(str(post_id))
        if idx is None:
            return None
        rec = {
            "post_id": str(post_id),
            "vector": self.vectors[idx],
            "is_valid": bool(self.is_valid[idx]),
        }
        if self.timestamps is not None:
            rec["publish_datetime_utc"] = self.timestamps[idx]
        if self.cluster_ids is not None:
            rec["cluster_id"] = self.cluster_ids[idx]
        return rec

    def get_vectors(
        self,
        post_ids: Sequence[str],
        return_missing: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        Truy xuất ma trận vector cho một danh sách các post_id (Batch Lookup).
        """
        indices = []
        missing = []
        for pid in post_ids:
            idx = self.id_map.get(str(pid))
            if idx is not None:
                indices.append(idx)
            else:
                missing.append(str(pid))

        if len(indices) == 0:
            found_vectors = np.empty((0, self.embedding_dim), dtype=self.vectors.dtype)
        else:
            found_vectors = self.vectors[indices]

        if return_missing:
            return found_vectors, missing
        return found_vectors

    def filter_by_ids(self, post_ids: Sequence[str]) -> "EmbeddingStore":
        """
        Trích xuất một EmbeddingStore con chỉ chứa các ID được chỉ định (bảo toàn thứ tự).
        """
        indices = []
        valid_ids = []
        for pid in post_ids:
            idx = self.id_map.get(str(pid))
            if idx is not None:
                indices.append(idx)
                valid_ids.append(str(pid))

        if len(indices) == 0:
            sub_vectors = np.empty((0, self.embedding_dim), dtype=self.vectors.dtype)
            sub_mask = np.empty((0,), dtype=bool)
            sub_timestamps = np.empty((0,)) if self.timestamps is not None else None
            sub_clusters = np.empty((0,)) if self.cluster_ids is not None else None
        else:
            sub_vectors = self.vectors[indices]
            sub_mask = self.is_valid[indices]
            sub_timestamps = self.timestamps[indices] if self.timestamps is not None else None
            sub_clusters = self.cluster_ids[indices] if self.cluster_ids is not None else None

        sub_meta = EmbeddingMetadata.from_dict(self.metadata.to_dict())
        sub_meta.num_vectors = len(valid_ids)

        return EmbeddingStore(
            vectors=sub_vectors,
            post_ids=valid_ids,
            timestamps=sub_timestamps,
            cluster_ids=sub_clusters,
            metadata=sub_meta,
            is_valid=sub_mask,
            validate_on_init=False,
        )

    def filter_by_timerange(
        self,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> "EmbeddingStore":
        """
        Trích xuất kho embedding con theo khoảng thời gian (Temporal Slicing).
        Phục vụ phân tích cửa sổ thời gian (Temporal Narrative Tracking ở Module 4)
        và tránh sử dụng dữ liệu tương lai (lookahead bias).
        """
        if self.timestamps is None:
            raise ValueError("EmbeddingStore này không chứa thông tin timestamps để lọc theo thời gian!")

        ts_series = pd.to_datetime(self.timestamps, utc=True)
        mask = np.ones(len(self), dtype=bool)

        if start_date is not None:
            s_ts = pd.to_datetime(start_date, utc=True)
            mask &= np.asarray(ts_series >= s_ts)

        if end_date is not None:
            e_ts = pd.to_datetime(end_date, utc=True)
            mask &= np.asarray(ts_series <= e_ts)

        indices = np.where(mask)[0]
        sub_vectors = self.vectors[indices]
        sub_ids = self.post_ids[indices]
        sub_timestamps = self.timestamps[indices]
        sub_clusters = self.cluster_ids[indices] if self.cluster_ids is not None else None
        sub_valid = self.is_valid[indices]

        sub_meta = EmbeddingMetadata.from_dict(self.metadata.to_dict())
        sub_meta.num_vectors = len(sub_ids)
        if start_date or end_date:
            sub_meta.time_range = (str(start_date), str(end_date))

        return EmbeddingStore(
            vectors=sub_vectors,
            post_ids=sub_ids,
            timestamps=sub_timestamps,
            cluster_ids=sub_clusters,
            metadata=sub_meta,
            is_valid=sub_valid,
            validate_on_init=False,
        )

    def validate(self, strict: bool = True) -> ValidationReport:
        """
        Kiểm tra toàn diện tính toàn vẹn của kho embedding:
        1. Số lượng vector khớp số lượng ID (len(vectors) == len(post_ids)).
        2. Kích thước vector nhất quán và khớp metadata (vectors.shape[1] == embedding_dim).
        3. Không chứa giá trị NaN hoặc Inf trong vector.
        4. Không chứa post_id rỗng, null hoặc trùng lặp.
        5. Độ dài timestamps và cluster_ids (nếu có) đồng bộ với post_ids.
        """
        report = ValidationReport(
            is_valid=True,
            num_vectors=len(self.vectors),
            num_ids=len(self.post_ids),
            embedding_dim=self.embedding_dim,
        )

        # 1. Kiểm tra độ dài vectors vs post_ids
        if len(self.vectors) != len(self.post_ids):
            report.is_valid = False
            report.errors.append(
                f"Lệch số lượng: vectors có {len(self.vectors):,} dòng, "
                f"nhưng post_ids có {len(self.post_ids):,} phần tử."
            )
        else:
            report.checks_passed.append("Số lượng vector khớp chính xác số lượng post_id.")

        # 2. Kiểm tra chiều không gian vector (dim)
        if self.vectors.ndim != 2:
            report.is_valid = False
            report.errors.append(f"Mảng vectors phải là ma trận 2D, hiện tại ndim={self.vectors.ndim}.")
        else:
            report.checks_passed.append("Mảng vectors có cấu trúc ma trận 2D chuẩn.")

        if self.metadata.embedding_dim > 0 and self.embedding_dim != self.metadata.embedding_dim:
            msg = (
                f"Kích thước vector thực tế (D={self.embedding_dim}) không khớp "
                f"với metadata (D={self.metadata.embedding_dim})."
            )
            if strict:
                report.is_valid = False
                report.errors.append(msg)
            else:
                report.warnings.append(msg)
        else:
            report.checks_passed.append(f"Kích thước vector nhất quán D={self.embedding_dim}.")

        # 3. Kiểm tra giá trị hữu hạn (NaN / Inf)
        if len(self.vectors) > 0:
            nan_count = int(np.sum(np.isnan(self.vectors)))
            inf_count = int(np.sum(np.isinf(self.vectors)))
            if nan_count > 0 or inf_count > 0:
                report.is_valid = False
                report.errors.append(
                    f"Phát hiện giá trị bất thường trong vector: {nan_count:,} NaN, {inf_count:,} Inf."
                )
            else:
                report.checks_passed.append("Không chứa giá trị NaN hoặc vô hạn (Inf) trong vector.")

        # 4. Kiểm tra ID rỗng và trùng lặp
        if len(self.post_ids) > 0:
            empty_ids = sum(1 for pid in self.post_ids if not pid or str(pid).strip() == "")
            if empty_ids > 0:
                report.is_valid = False
                report.errors.append(f"Có {empty_ids:,} post_id bị rỗng hoặc trắng.")
            else:
                report.checks_passed.append("Tất cả post_id đều có giá trị hợp lệ.")

            unique_ids_count = len(set(self.post_ids))
            if unique_ids_count != len(self.post_ids):
                dup_count = len(self.post_ids) - unique_ids_count
                msg = f"Phát hiện {dup_count:,} post_id bị trùng lặp trong kho lưu trữ."
                report.warnings.append(msg)
                if strict:
                    report.is_valid = False
                    report.errors.append(msg)
            else:
                report.checks_passed.append("Tất cả post_id đều duy nhất (unique), không trùng lặp.")

        # 5. Kiểm tra tính đồng bộ của timestamps và cluster_ids
        if self.timestamps is not None:
            if len(self.timestamps) != len(self.post_ids):
                report.is_valid = False
                report.errors.append(
                    f"Mảng timestamps ({len(self.timestamps):,}) không khớp độ dài post_ids ({len(self.post_ids):,})."
                )
            else:
                report.checks_passed.append("Mảng timestamps đồng bộ chính xác.")

        if self.cluster_ids is not None:
            if len(self.cluster_ids) != len(self.post_ids):
                report.is_valid = False
                report.errors.append(
                    f"Mảng cluster_ids ({len(self.cluster_ids):,}) không khớp độ dài post_ids ({len(self.post_ids):,})."
                )
            else:
                report.checks_passed.append("Mảng cluster_ids đồng bộ chính xác.")

        # 6. Kiểm tra mảng is_valid
        if len(self.is_valid) != len(self.post_ids):
            report.is_valid = False
            report.errors.append(
                f"Mảng is_valid ({len(self.is_valid):,}) không khớp độ dài post_ids ({len(self.post_ids):,})."
            )
        else:
            report.checks_passed.append("Mảng đánh dấu is_valid đồng bộ.")

        return report

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Chuyển thành dictionary ánh xạ post_id -> vector."""
        return {pid: self.vectors[i] for i, pid in enumerate(self.post_ids)}

    def to_dataframe(
        self,
        id_col: str = "post_id",
        time_col: str = "publish_datetime_utc",
        cluster_col: str = "cluster_id",
        include_vectors: bool = True,
    ) -> pd.DataFrame:
        """
        Chuyển thành DataFrame phục vụ liên kết giữa các module:
        - post_id
        - publish_datetime_utc (nếu có)
        - cluster_id (nếu có)
        - embedding (tùy chọn)
        """
        data: Dict[str, Any] = {
            id_col: self.post_ids,
            "is_valid": self.is_valid,
        }
        if self.timestamps is not None:
            data[time_col] = self.timestamps
        if self.cluster_ids is not None:
            data[cluster_col] = self.cluster_ids
        if include_vectors:
            data["embedding"] = [v.tolist() for v in self.vectors]

        return pd.DataFrame(data)

    def to_clustering_dataframe(self) -> pd.DataFrame:
        """
        Xuất bảng kết quả phân cụm thu gọn (không chứa vector cồng kềnh) để bàn giao cho
        Module 4 (Temporal Narrative Tracking): gồm post_id, publish_datetime_utc, cluster_id.
        """
        return self.to_dataframe(include_vectors=False)

    # -------------------------------------------------------------------------
    # Các phương thức lưu trữ (Save Methods)
    # -------------------------------------------------------------------------

    def save_npz(self, output_path: Union[str, Path], compress: bool = True) -> str:
        """
        Lưu kho embedding ra file .npz (NumPy compressed).
        Đây là định dạng tối ưu nhất về tốc độ nạp ma trận và dung lượng.
        Bảo toàn vectors, post_ids, timestamps, cluster_ids và metadata.
        """
        output_path = Path(output_path)
        if output_path.suffix != ".npz":
            output_path = output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        meta_json = self.metadata.to_json()
        save_fn = np.savez_compressed if compress else np.savez

        save_dict = {
            "vectors": self.vectors,
            "post_ids": self.post_ids,
            "is_valid": self.is_valid,
            "metadata": np.array([meta_json]),
        }
        if self.timestamps is not None:
            save_dict["timestamps"] = self.timestamps
        if self.cluster_ids is not None:
            save_dict["cluster_ids"] = self.cluster_ids

        save_fn(output_path, **save_dict)
        logger.info(f"Đã lưu NPZ embedding store ({len(self):,} vectors) tại: {output_path}")
        return str(output_path)

    def save_parquet(
        self,
        output_path: Union[str, Path],
        source_df: Optional[pd.DataFrame] = None,
        id_col: str = "post_id",
        time_col: str = "publish_datetime_utc",
        cluster_col: str = "cluster_id",
    ) -> str:
        """
        Lưu kho embedding ra file Apache Parquet, tích hợp với bảng dữ liệu gốc.
        """
        output_path = Path(output_path)
        if output_path.suffix not in (".parquet", ".pq"):
            output_path = output_path.with_suffix(".parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_out = self.to_dataframe(
            id_col=id_col,
            time_col=time_col,
            cluster_col=cluster_col,
            include_vectors=True,
        )

        if source_df is not None and id_col in source_df.columns:
            other_cols = [c for c in source_df.columns if c != id_col and c not in df_out.columns]
            if other_cols:
                meta_subset = source_df[[id_col] + other_cols].drop_duplicates(subset=[id_col])
                df_out = pd.merge(df_out, meta_subset, on=id_col, how="left")

        df_out.to_parquet(output_path, index=False)
        logger.info(f"Đã lưu Parquet embedding store ({len(self):,} rows) tại: {output_path}")
        return str(output_path)

    def save_bundle(self, dir_path: Union[str, Path], save_parquet: bool = True) -> str:
        """
        Lưu trọn gói (Bundle Directory) chuẩn hóa cho một phiên bản embedding:
        dir_path/
          ├── metadata.json
          ├── embeddings.npz
          └── embeddings.parquet (tùy chọn)
        """
        bundle_dir = Path(dir_path)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # 1. Lưu metadata.json độc lập để có thể xem nhanh bằng text editor
        meta_file = bundle_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(self.metadata.to_json(indent=2))

        # 2. Lưu embeddings.npz
        npz_file = bundle_dir / "embeddings.npz"
        self.save_npz(npz_file, compress=True)

        # 3. Lưu embeddings.parquet
        if save_parquet:
            pq_file = bundle_dir / "embeddings.parquet"
            self.save_parquet(pq_file)

        logger.info(f"Đã lưu hoàn chỉnh Embedding Bundle tại thư mục: {bundle_dir}")
        return str(bundle_dir)

    def save(
        self,
        output_path_or_dir: Union[str, Path],
        format: Literal["npz", "parquet", "bundle"] = "npz",
    ) -> str:
        """Phương thức lưu đa định dạng linh hoạt."""
        if format == "npz":
            return self.save_npz(output_path_or_dir)
        elif format == "parquet":
            return self.save_parquet(output_path_or_dir)
        elif format == "bundle":
            return self.save_bundle(output_path_or_dir)
        else:
            raise ValueError(f"Định dạng không được hỗ trợ: '{format}'. Chọn 'npz', 'parquet' hoặc 'bundle'.")

    # -------------------------------------------------------------------------
    # Các phương thức nạp lại (Load Methods)
    # -------------------------------------------------------------------------

    @classmethod
    def load_npz(cls, npz_path: Union[str, Path], validate: bool = True) -> "EmbeddingStore":
        """Nạp kho embedding từ file .npz đã lưu."""
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file .npz: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        metadata = None
        if "metadata" in data:
            metadata = EmbeddingMetadata.from_json(str(data["metadata"][0]))
        elif "meta" in data:
            # Hỗ trợ nạp ngược tương thích với file sinh từ encode_posts.py
            try:
                raw_meta = json.loads(str(data["meta"][0]))
                metadata = EmbeddingMetadata(
                    model_name=raw_meta.get("model_name", "unknown"),
                    embedding_dim=int(raw_meta.get("embedding_dim", data["vectors"].shape[1])),
                    num_vectors=int(len(data["post_ids"])),
                    extra=raw_meta.get("metadata", {}),
                )
            except Exception:
                pass

        is_valid = data["is_valid"] if "is_valid" in data else None
        timestamps = data["timestamps"] if "timestamps" in data else None
        cluster_ids = data["cluster_ids"] if "cluster_ids" in data else None

        store = cls(
            vectors=data["vectors"],
            post_ids=data["post_ids"],
            timestamps=timestamps,
            cluster_ids=cluster_ids,
            metadata=metadata,
            is_valid=is_valid,
            validate_on_init=validate,
        )
        logger.info(f"Nạp thành công {len(store):,} vectors (dim={store.embedding_dim}) từ: {npz_path}")
        return store

    @classmethod
    def load_parquet(
        cls,
        parquet_path: Union[str, Path],
        id_col: str = "post_id",
        time_col: str = "publish_datetime_utc",
        cluster_col: str = "cluster_id",
        vec_col: str = "embedding",
        validate: bool = True,
    ) -> "EmbeddingStore":
        """Nạp kho embedding từ file Apache Parquet."""
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file parquet: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        if id_col not in df.columns:
            raise KeyError(f"Không tìm thấy cột ID '{id_col}' trong file parquet: {parquet_path}")

        post_ids = df[id_col].astype(str).values

        if vec_col in df.columns:
            vectors = np.array(df[vec_col].tolist(), dtype=np.float32)
        else:
            dim_cols = [c for c in df.columns if c.startswith("dim_")]
            if dim_cols:
                dim_cols_sorted = sorted(dim_cols, key=lambda c: int(c.split("_")[1]))
                vectors = df[dim_cols_sorted].values.astype(np.float32)
            else:
                raise KeyError(f"Không tìm thấy cột vector ('{vec_col}' hoặc 'dim_0, dim_1...') trong file parquet!")

        is_valid = df["is_valid"].values if "is_valid" in df.columns else None
        timestamps = df[time_col].values if time_col in df.columns else None
        cluster_ids = df[cluster_col].values if cluster_col in df.columns else None

        store = cls(
            vectors=vectors,
            post_ids=post_ids,
            timestamps=timestamps,
            cluster_ids=cluster_ids,
            metadata=EmbeddingMetadata(
                model_name="parquet_source",
                embedding_dim=vectors.shape[1],
                num_vectors=len(vectors),
                source_dataset=str(parquet_path),
            ),
            is_valid=is_valid,
            validate_on_init=validate,
        )
        logger.info(f"Nạp thành công {len(store):,} vectors (dim={store.embedding_dim}) từ parquet: {parquet_path}")
        return store

    @classmethod
    def load_bundle(cls, dir_path: Union[str, Path], validate: bool = True) -> "EmbeddingStore":
        """Nạp kho embedding từ thư mục Bundle."""
        bundle_dir = Path(dir_path)
        if not bundle_dir.is_dir():
            raise NotADirectoryError(f"Đường dẫn không phải là thư mục: {bundle_dir}")

        npz_file = bundle_dir / "embeddings.npz"
        meta_file = bundle_dir / "metadata.json"

        if not npz_file.exists():
            raise FileNotFoundError(f"Không tìm thấy file embeddings.npz trong bundle: {bundle_dir}")

        store = cls.load_npz(npz_file, validate=False)

        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                store.metadata = EmbeddingMetadata.from_json(f.read())

        if validate:
            report = store.validate(strict=True)
            if not report.is_valid:
                raise ValueError(f"Dữ liệu bundle không hợp lệ: {report.errors}")

        logger.info(f"Nạp thành công Bundle từ: {bundle_dir}")
        return store

    @classmethod
    def load(cls, path_or_dir: Union[str, Path], validate: bool = True) -> "EmbeddingStore":
        """
        Hàm nạp thông minh tự động nhận diện định dạng:
        - Nếu là thư mục: nạp theo Bundle.
        - Nếu là file .npz: nạp theo NPZ.
        - Nếu là file .parquet: nạp theo Parquet.
        """
        path = Path(path_or_dir)
        if not path.exists():
            raise FileNotFoundError(f"Đường dẫn không tồn tại: {path}")

        if path.is_dir():
            return cls.load_bundle(path, validate=validate)
        elif path.suffix == ".npz":
            return cls.load_npz(path, validate=validate)
        elif path.suffix in (".parquet", ".pq"):
            return cls.load_parquet(path, validate=validate)
        else:
            raise ValueError(f"Định dạng không được hỗ trợ để nạp: {path}")


# -----------------------------------------------------------------------------
# Standalone Functions
# -----------------------------------------------------------------------------

def save_embeddings(
    vectors: np.ndarray,
    post_ids: Union[np.ndarray, Sequence[str]],
    output_path: Union[str, Path],
    timestamps: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    cluster_ids: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    metadata: Optional[Union[EmbeddingMetadata, Dict[str, Any]]] = None,
    format: Literal["npz", "parquet", "bundle"] = "npz",
    is_valid: Optional[np.ndarray] = None,
    validate: bool = True,
) -> str:
    """Hàm tiện ích lưu vector và metadata ra đĩa."""
    store = EmbeddingStore(
        vectors=vectors,
        post_ids=post_ids,
        timestamps=timestamps,
        cluster_ids=cluster_ids,
        metadata=metadata,
        is_valid=is_valid,
        validate_on_init=validate,
    )
    return store.save(output_path, format=format)


def load_embeddings(
    path_or_dir: Union[str, Path],
    validate: bool = True,
) -> EmbeddingStore:
    """Hàm tiện ích nạp kho embedding từ đĩa và kiểm tra tính toàn vẹn."""
    return EmbeddingStore.load(path_or_dir, validate=validate)


def validate_embeddings(
    vectors: np.ndarray,
    post_ids: Union[np.ndarray, Sequence[str]],
    expected_dim: Optional[int] = None,
    strict: bool = True,
) -> ValidationReport:
    """Hàm tiện ích độc lập để kiểm tra tính toàn vẹn của vector và danh sách post_id."""
    meta = EmbeddingMetadata(
        model_name="validation_check",
        embedding_dim=expected_dim if expected_dim else (vectors.shape[1] if vectors.ndim > 1 else 0),
        num_vectors=len(vectors),
    )
    store = EmbeddingStore(
        vectors=vectors,
        post_ids=post_ids,
        metadata=meta,
        validate_on_init=False,
    )
    return store.validate(strict=strict)
