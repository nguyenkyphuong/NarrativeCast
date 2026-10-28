"""
src/embeddings/embedding_store.py
Module quản lý lưu trữ, nạp lại và kiểm tra tính toàn vẹn của vector embedding và metadata.

Chức năng chính:
- Lưu trữ vector embedding kèm metadata phiên bản mô hình, tránh phải tính toán lại trong các thử nghiệm.
- Hỗ trợ các định dạng lưu trữ tối ưu: NumPy compressed (.npz), Apache Parquet (.parquet), và Bundle directory kèm metadata.json.
- Kiểm tra tính toàn vẹn dữ liệu nghiêm ngặt: số lượng vector, độ dài post_id, kích thước vector nhất quán,
  không chứa NaN/Inf, không trùng lặp ID.
- Cung cấp truy xuất O(1) theo post_id, batch lookup và lọc trích xuất tập con (subset filtering).
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
            f"<ValidationReport status={status} vectors={self.num_vectors} "
            f"dim={self.embedding_dim} errors={len(self.errors)} warnings={len(self.warnings)}>"
        )

    def summary(self) -> str:
        """Xuất chuỗi tóm tắt kết quả kiểm tra."""
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
    """Metadata quản lý phiên bản và cấu hình mô hình tạo embedding."""
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
        # Lọc các trường hợp hợp lệ để tránh lỗi khi có trường lạ
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
    Đảm bảo vector luôn gắn chặt 1-1 với ID bài đăng gốc.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        post_ids: Union[np.ndarray, Sequence[str]],
        metadata: Optional[Union[EmbeddingMetadata, Dict[str, Any]]] = None,
        is_valid: Optional[np.ndarray] = None,
        validate_on_init: bool = True,
    ):
        # 1. Chuẩn hóa vectors thành mảng float32
        if not isinstance(vectors, np.ndarray):
            vectors = np.asarray(vectors, dtype=np.float32)
        elif vectors.dtype not in (np.float32, np.float64, np.float16):
            vectors = vectors.astype(np.float32)

        # 2. Chuẩn hóa post_ids thành mảng chuỗi 1D
        if not isinstance(post_ids, np.ndarray):
            post_ids = np.asarray(post_ids, dtype=str)
        else:
            post_ids = post_ids.astype(str)

        self.vectors = vectors
        self.post_ids = post_ids

        # 3. Trạng thái hợp lệ (is_valid mask)
        if is_valid is None:
            self.is_valid = np.ones(len(self.post_ids), dtype=bool)
        else:
            self.is_valid = np.asarray(is_valid, dtype=bool)

        # 4. Thiết lập metadata
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

        # 5. Bộ đệm chỉ mục O(1) post_id -> index
        self._id_to_idx: Optional[Dict[str, int]] = None

        # 6. Kiểm tra tính toàn vẹn khi khởi tạo nếu được yêu cầu
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

    def get_vector(self, post_id: str) -> Optional[np.ndarray]:
        """
        Truy xuất vector của một bài viết cụ thể theo post_id.
        Trả về None nếu không tìm thấy ID.
        """
        idx = self.id_map.get(str(post_id))
        if idx is None:
            return None
        return self.vectors[idx]

    def get_vectors(
        self,
        post_ids: Sequence[str],
        return_missing: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        Truy xuất ma trận vector cho một danh sách các post_id (Batch Lookup).

        Args:
            post_ids: Danh sách các ID cần lấy.
            return_missing: Nếu True, trả về kèm danh sách các ID không tìm thấy.

        Returns:
            np.ndarray: Ma trận vector cho các ID tìm thấy.
            (tùy chọn) List[str]: Danh sách các ID bị thiếu.
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
        else:
            sub_vectors = self.vectors[indices]
            sub_mask = self.is_valid[indices]

        sub_meta = EmbeddingMetadata.from_dict(self.metadata.to_dict())
        sub_meta.num_vectors = len(valid_ids)

        return EmbeddingStore(
            vectors=sub_vectors,
            post_ids=valid_ids,
            metadata=sub_meta,
            is_valid=sub_mask,
            validate_on_init=False,
        )

    def validate(self, strict: bool = True) -> ValidationReport:
        """
        Kiểm tra toàn diện tính toàn vẹn của kho embedding:
        1. Số lượng vector khớp số lượng ID (len(vectors) == len(post_ids)).
        2. Kích thước vector nhất quán và khớp metadata (vectors.shape[1] == embedding_dim).
        3. Không chứa giá trị NaN hoặc Inf trong vector.
        4. Không chứa post_id rỗng, null hoặc trùng lặp.
        5. Độ dài mảng is_valid khớp số lượng bản ghi.
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

        # 5. Kiểm tra mảng is_valid
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

    def to_dataframe(self, id_col: str = "post_id") -> pd.DataFrame:
        """Chuyển thành DataFrame có cột post_id, embedding và is_valid."""
        return pd.DataFrame({
            id_col: self.post_ids,
            "embedding": [v.tolist() for v in self.vectors],
            "is_valid": self.is_valid,
        })

    # -------------------------------------------------------------------------
    # Các phương thức lưu trữ (Save Methods)
    # -------------------------------------------------------------------------

    def save_npz(self, output_path: Union[str, Path], compress: bool = True) -> str:
        """
        Lưu kho embedding ra file .npz (NumPy compressed).
        Đây là định dạng tối ưu nhất về tốc độ nạp ma trận và dung lượng.
        """
        output_path = Path(output_path)
        if output_path.suffix != ".npz":
            output_path = output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        meta_json = self.metadata.to_json()
        save_fn = np.savez_compressed if compress else np.savez

        save_fn(
            output_path,
            vectors=self.vectors,
            post_ids=self.post_ids,
            is_valid=self.is_valid,
            metadata=np.array([meta_json]),
        )
        logger.info(f"Đã lưu NPZ embedding store ({len(self):,} vectors) tại: {output_path}")
        return str(output_path)

    def save_parquet(
        self,
        output_path: Union[str, Path],
        source_df: Optional[pd.DataFrame] = None,
        id_col: str = "post_id",
    ) -> str:
        """
        Lưu kho embedding ra file Apache Parquet, tích hợp với bảng dữ liệu gốc.
        """
        output_path = Path(output_path)
        if output_path.suffix not in (".parquet", ".pq"):
            output_path = output_path.with_suffix(".parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_out = self.to_dataframe(id_col=id_col)

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

        # 2. Lưu vectors.npz
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
        source_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Hàm tiện ích lưu kho embedding theo định dạng mong muốn.
        """
        path = Path(output_path_or_dir)
        if format == "bundle" or (path.is_dir() and not path.suffix):
            return self.save_bundle(path)
        elif format == "parquet" or path.suffix in (".parquet", ".pq"):
            return self.save_parquet(path, source_df=source_df)
        elif format == "npz" or path.suffix == ".npz":
            return self.save_npz(path)
        else:
            raise ValueError(f"Định dạng không được hỗ trợ: {format}")

    # -------------------------------------------------------------------------
    # Các phương thức nạp lại (Load Methods)
    # -------------------------------------------------------------------------

    @classmethod
    def load_npz(cls, npz_path: Union[str, Path], validate: bool = True) -> "EmbeddingStore":
        """Nạp kho embedding từ file .npz."""
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        meta_dict = {}
        if "metadata" in data:
            try:
                meta_dict = json.loads(str(data["metadata"][0]))
            except Exception as e:
                logger.warning(f"Không đọc được metadata trong npz: {e}")
        elif "meta" in data:  # Hỗ trợ backward compatibility với encode_posts
            try:
                meta_dict = json.loads(str(data["meta"][0]))
            except Exception:
                pass

        vectors = data["vectors"]
        post_ids = data["post_ids"]
        is_valid = data.get("is_valid", None)

        store = cls(
            vectors=vectors,
            post_ids=post_ids,
            metadata=meta_dict,
            is_valid=is_valid,
            validate_on_init=validate,
        )
        return store

    @classmethod
    def load_parquet(
        cls,
        parquet_path: Union[str, Path],
        id_col: str = "post_id",
        embedding_col: str = "embedding",
        valid_col: str = "is_valid",
        validate: bool = True,
    ) -> "EmbeddingStore":
        """Nạp kho embedding từ file Apache Parquet."""
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file Parquet: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        if id_col not in df.columns:
            raise ValueError(f"Cột ID '{id_col}' không tồn tại trong Parquet!")
        if embedding_col not in df.columns:
            raise ValueError(f"Cột embedding '{embedding_col}' không tồn tại trong Parquet!")

        post_ids = df[id_col].astype(str).values
        raw_embs = df[embedding_col].values

        # Chuyển list of arrays/lists thành ma trận 2D
        vectors = np.array(raw_embs.tolist(), dtype=np.float32)

        is_valid = None
        if valid_col in df.columns:
            is_valid = df[valid_col].values
        elif "is_valid_embedding" in df.columns:
            is_valid = df["is_valid_embedding"].values

        dim = vectors.shape[1] if vectors.ndim > 1 else 0
        metadata = EmbeddingMetadata(
            model_name="loaded_from_parquet",
            embedding_dim=dim,
            num_vectors=len(post_ids),
            source_dataset=str(parquet_path.name),
        )

        return cls(
            vectors=vectors,
            post_ids=post_ids,
            metadata=metadata,
            is_valid=is_valid,
            validate_on_init=validate,
        )

    @classmethod
    def load_bundle(cls, dir_path: Union[str, Path], validate: bool = True) -> "EmbeddingStore":
        """Nạp kho embedding từ thư mục Bundle chứa metadata.json và embeddings.npz."""
        bundle_dir = Path(dir_path)
        if not bundle_dir.is_dir():
            raise NotADirectoryError(f"Không phải thư mục hợp lệ: {bundle_dir}")

        meta_file = bundle_dir / "metadata.json"
        npz_file = bundle_dir / "embeddings.npz"
        if not npz_file.exists():
            # Thử tìm file .npz bất kỳ trong thư mục
            npz_candidates = list(bundle_dir.glob("*.npz"))
            if npz_candidates:
                npz_file = npz_candidates[0]
            else:
                # Thử tìm file parquet
                pq_candidates = list(bundle_dir.glob("*.parquet"))
                if pq_candidates:
                    return cls.load_parquet(pq_candidates[0], validate=validate)
                raise FileNotFoundError(f"Không tìm thấy file embeddings (.npz / .parquet) trong: {bundle_dir}")

        store = cls.load_npz(npz_file, validate=False)

        # Nếu có file metadata.json riêng, ưu tiên nạp metadata từ đó
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    store.metadata = EmbeddingMetadata.from_json(f.read())
            except Exception as e:
                logger.warning(f"Không thể đọc metadata.json: {e}")

        if validate:
            report = store.validate(strict=False)
            if not report.is_valid:
                logger.warning(f"Cảnh báo khi nạp Bundle: {report.errors}")

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
    metadata: Optional[Union[EmbeddingMetadata, Dict[str, Any]]] = None,
    format: Literal["npz", "parquet", "bundle"] = "npz",
    is_valid: Optional[np.ndarray] = None,
    validate: bool = True,
) -> str:
    """
    Hàm tiện ích lưu vector và metadata ra đĩa.
    """
    store = EmbeddingStore(
        vectors=vectors,
        post_ids=post_ids,
        metadata=metadata,
        is_valid=is_valid,
        validate_on_init=validate,
    )
    return store.save(output_path, format=format)


def load_embeddings(
    path_or_dir: Union[str, Path],
    validate: bool = True,
) -> EmbeddingStore:
    """
    Hàm tiện ích nạp kho embedding từ đĩa và kiểm tra tính toàn vẹn.
    """
    return EmbeddingStore.load(path_or_dir, validate=validate)


def validate_embeddings(
    vectors: np.ndarray,
    post_ids: Union[np.ndarray, Sequence[str]],
    expected_dim: Optional[int] = None,
    strict: bool = True,
) -> ValidationReport:
    """
    Hàm tiện ích độc lập để kiểm tra tính toàn vẹn của vector và danh sách post_id.
    """
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
