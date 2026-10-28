# Hướng Dẫn Cấu Trúc Lưu Trữ & Tái Sử Dụng Vector Embedding (Embedding Storage Guide)

- **Đề tài:** NarrativeCast — Predicting Emerging Narratives in Social Media Using Temporal Semantic Representations
- **Module mã nguồn:** [`src/embeddings/embedding_store.py`](../src/embeddings/embedding_store.py)
- **Tài liệu liên quan:** [`docs/embedding_input_requirements.md`](./embedding_input_requirements.md) · [`docs/data_cleaning_rules.md`](./data_cleaning_rules.md) · [`docs/experimental_plan.md`](./experimental_plan.md)

---

## 1. Mục Tiêu & Tầm Quan Trọng Của Việc Lưu Trữ Embedding

Trong quy trình nghiên cứu của **NarrativeCast**, việc chuyển đổi văn bản mạng xã hội thành vector ngữ nghĩa thông qua các mô hình ngôn ngữ (như Sentence-BERT `all-MiniLM-L6-v2`) là bước tiêu tốn nhiều thời gian và tài nguyên phần cứng (GPU/CPU) nhất:
- Với tập mẫu 5,000 bài: mất khoảng vài giây trên GPU hoặc vài phút trên CPU.
- Với tập dữ liệu thật của Reddit Technology (hàng trăm nghìn đến hàng triệu bài viết): quá trình mã hóa có thể mất từ vài giờ đến nhiều ngày.

Nếu mỗi lần thử nghiệm thay đổi thuật toán phân cụm (BERTopic / HDBSCAN) hoặc thay đổi cửa sổ dự báo ($h \in \{3, 7, 14\}$ ngày) mà phải tạo lại embedding từ đầu, nghiên cứu sẽ bị tắc nghẽn nghiêm trọng.

Do đó, **thiết kế lưu trữ embedding** giải quyết các bài toán cốt lõi:
1. **Tránh tính toán lại (Zero Redundant Computation):** Chỉ tạo vector một lần duy nhất cho mỗi bài đăng/văn bản sạch, sau đó nạp lại tức thì trong các lần thử nghiệm tiếp theo.
2. **Bảo toàn liên kết 1-1 giữa Vector và `post_id`:** Xác định chính xác vector nào thuộc về bài đăng nào, hỗ trợ truy xuất $O(1)$ và lọc theo danh sách ID.
3. **Quản lý phiên bản & Nguồn gốc mô hình (Model Lineage & Versioning):** Ghi nhận rõ mô hình nào, phiên bản nào, tham số chuẩn hóa ra sao đã sinh ra vector.
4. **Kiểm tra tính toàn vẹn (Data Integrity):** Ngăn chặn việc nạp phải vector bị lỗi, lệch số lượng hoặc chứa giá trị bất thường (`NaN`, `Inf`).

---

## 2. So Sánh & Lựa Chọn Định Dạng Lưu Trữ

Dưới đây là bảng so sánh các phương án lưu trữ vector phổ biến trong khoa học dữ liệu và học máy:

| Định dạng | Tốc độ nạp (Load Speed) | Dung lượng nén | Khả năng truy vấn quan hệ | Lưu trữ Metadata | Đánh giá phù hợp với NarrativeCast |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NumPy Compressed (`.npz`)** | **Cực nhanh** (hàng mili-giây) | **Rất tốt** (nén zlib) | Hạn chế (truy xuất ma trận theo chỉ số index) | Tốt (lưu JSON kèm mảng) | **Khuyên dùng cho khâu huấn luyện mô hình** (BERTopic, Clustering, Feature Matrix cho Time-series). |
| **Apache Parquet (`.parquet`)** | Nhanh | **Xuất sắc** (Snappy / ZSTD) | **Rất mạnh** (tích hợp cột metadata: timestamp, author, interaction) | Rất tốt (schema có kiểu rõ ràng) | **Khuyên dùng cho kho lưu trữ dữ liệu tập trung (Data Lake / Warehouse)** và truy vấn bằng DuckDB/Polars/PySpark. |
| **JSON / CSV** | Rất chậm | Kém (dung lượng tăng gấp 3-5 lần) | Tốt | Tốt | **Không phù hợp** cho vector số thực chiều lớn (làm phình to dung lượng và mất độ chính xác số học). |
| **Vector DB (FAISS / Chroma / Milvus)** | Nhanh (truy vấn láng giềng gần nhất ANN) | Trung bình | Tùy hệ thống | Khá | **Chưa cần thiết** ở giai đoạn phân tích offline theo cửa sổ thời gian (tăng độ phức tạp cài đặt không cần thiết). |

### 🎯 Chiến Lược Lưu Trữ Lai (Hybrid Storage Strategy):
Dự án NarrativeCast áp dụng mô hình lưu trữ lai kết hợp ưu điểm của cả hai định dạng:
1. **Định dạng `.npz`:** Dành cho các tác vụ tính toán ma trận học máy (nạp ma trận $\mathbf{X} \in \mathbb{R}^{N \times D}$ vào RAM chỉ trong 0.05 giây).
2. **Định dạng `.parquet`:** Dành cho việc lưu trữ liên kết với các trường thông tin bài viết (`publish_date`, `community`, `user_id`, v.v.).
3. **Định dạng `metadata.json`:** Cho phép nhà nghiên cứu đọc nhanh thông số cấu hình mô hình mà không cần nạp dữ liệu nhị phân.

---

## 3. Cấu Trúc Thư Mục Tiêu Chuẩn Trong Dự Án

Các kho lưu trữ embedding được tổ chức theo cấu trúc phân cấp phiên bản tại thư mục `data/embeddings/`:

```text
NarrativeCast/
├── data/
│   ├── samples/
│   │   └── posts_sample.csv              # Dữ liệu nguồn mẫu
│   └── embeddings/
│       └── all-minilm-l6-v2__reddit_tech_v1/  # Bundle thư mục theo tên mô hình & phiên bản
│           ├── metadata.json             # File thông tin mô hình, phiên bản, thời gian tạo
│           ├── embeddings.npz            # Ma trận vectors (N, 384), post_ids (N,), is_valid (N,)
│           └── embeddings.parquet        # Bảng dữ liệu quan hệ kèm cột vector embedding
```

---

## 4. Đặc Tả Metadata & Quản Lý Phiên Bản (Model Lineage)

File `metadata.json` lưu giữ đầy đủ thông số kỹ thuật của lần mã hóa:

```json
{
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "model_version": "1.0",
  "embedding_dim": 384,
  "num_vectors": 5000,
  "normalize_embeddings": true,
  "max_seq_length": 256,
  "pooling_strategy": "mean",
  "created_at": "2026-10-28T17:15:00+00:00",
  "source_dataset": "madoc_reddit_technology_cleaned.parquet",
  "extra": {
    "device": "cuda",
    "batch_size": 64,
    "handle_empty": "drop",
    "clean_pipeline_version": "v1.0"
  }
}
```

---

## 5. Hướng Dẫn Sử Dụng & Code Mẫu Thực Chiến

Module [`src/embeddings/embedding_store.py`](../src/embeddings/embedding_store.py) cung cấp lớp `EmbeddingStore` và các hàm tiện ích nạp/lưu.

### 5.1. Lưu Vector và Metadata Sau Khi Mã Hóa
```python
import numpy as np
from src.embeddings.embedding_store import EmbeddingStore, EmbeddingMetadata

# Giả sử bạn có vectors từ PostEncoder và danh sách post_ids
vectors = np.random.randn(100, 384).astype(np.float32)
post_ids = [f"p_{i:04d}" for i in range(100)]

# Tạo metadata phiên bản
metadata = EmbeddingMetadata(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_version="1.0",
    embedding_dim=384,
    num_vectors=len(post_ids),
    normalize_embeddings=True,
    max_seq_length=256,
    source_dataset="data/samples/posts_sample.csv"
)

# Khởi tạo kho lưu trữ
store = EmbeddingStore(
    vectors=vectors,
    post_ids=post_ids,
    metadata=metadata
)

# Cách 1: Lưu thành Bundle trọn gói (npz + parquet + metadata.json)
store.save_bundle("data/embeddings/minilm_v1")

# Cách 2: Lưu riêng lẻ file .npz
store.save_npz("outputs/embeddings/my_embeddings.npz")
```

---

### 5.2. Nạp Lại Vector & Tự Động Kiểm Tra Tính Toàn Vẹn
```python
from src.embeddings.embedding_store import load_embeddings

# Tự động nhận diện file .npz, .parquet hoặc thư mục bundle
store = load_embeddings("data/embeddings/minilm_v1", validate=True)

print(f"Số lượng vector: {len(store):,}")
print(f"Kích thước vector: {store.embedding_dim}")
print(f"Mô hình sử dụng: {store.metadata.model_name}")

# Chạy kiểm tra tính toàn vẹn độc lập
report = store.validate(strict=True)
print(report.summary())
```

---

### 5.3. Truy Xuất Vector Theo `post_id` ($O(1)$ Lookup)
```python
# 1. Truy xuất vector của 1 bài viết cụ thể
vec = store.get_vector("p_rd_101")
if vec is not None:
    print(f"Vector shape: {vec.shape}")

# 2. Truy xuất theo batch danh sách ID bài viết
target_ids = ["p_0001", "p_0005", "p_9999"]
batch_vectors, missing_ids = store.get_vectors(target_ids, return_missing=True)
print(f"Số vector tìm thấy: {len(batch_vectors)}")
print(f"Các ID không có trong kho: {missing_ids}")

# 3. Trích xuất một kho con (Sub-store) chỉ chứa các bài trong cửa sổ thời gian
window_store = store.filter_by_ids(["p_0001", "p_0002"])
```

---

### 5.4. Tái Sử Dụng Cho Phân Cụm BERTopic & Dự Báo Mà Không Cần Encode Lại
```python
from bertopic import BERTopic
from src.embeddings.embedding_store import load_embeddings
import pandas as pd

# 1. Nạp bảng dữ liệu bài viết và kho vector tương ứng
df_posts = pd.read_parquet("data/processed/clean_posts.parquet")
store = load_embeddings("data/embeddings/minilm_v1")

# 2. Lấy danh sách vector khớp chính xác theo thứ tự các dòng trong df_posts
post_ids = df_posts["post_id"].tolist()
aligned_embeddings = store.get_vectors(post_ids)

# 3. Đưa thẳng vào BERTopic mà KHÔNG CẦN CHẠY LẠI MÔ HÌNH EMBEDDING
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(
    documents=df_posts["content"].tolist(),
    embeddings=aligned_embeddings  # Dùng trực tiếp vector đã lưu
)
```

---

## 6. Bộ Quy Tắc Kiểm Tra Tính Toàn Vẹn (Integrity Rules)

Khi gọi phương thức `store.validate(strict=True)`, hệ thống kiểm tra bắt buộc 5 tiêu chí:
1. **Đồng bộ độ dài:** `len(vectors) == len(post_ids)`. Mỗi ID phải có duy nhất một vector tương ứng.
2. **Kích thước nhất quán:** `vectors.ndim == 2` và `vectors.shape[1] == metadata.embedding_dim`. Mọi vector phải có số chiều bằng nhau ($D=384$), không có vector bị cắt cụt.
3. **Giá trị số học hợp lệ:** Không chứa bất kỳ giá trị `NaN`, `None` hoặc vô hạn `Inf` nào trong ma trận vector.
4. **Định danh duy nhất:** Tất cả các `post_id` phải là chuỗi không rỗng và không bị trùng lặp trong cùng một kho lưu trữ.
5. **Đồng bộ cờ hợp lệ:** Mảng boolean `is_valid` có cùng kích thước với số lượng bài viết để phân biệt vector thực và vector đệm (zero vector).

---

## 7. Kết Luận

Thiết kế lưu trữ với `EmbeddingStore` giúp dự án **NarrativeCast**:
- Tiết kiệm 100% thời gian tạo embedding cho các thử nghiệm phân cụm và dự báo kế tiếp.
- Đảm bảo tính tái lập (reproducibility) khoa học nhờ lưu trữ metadata phiên bản mô hình rõ ràng.
- Tương thích liền mạch với cả hai hệ sinh thái: **NumPy ma trận hiệu năng cao** và **Apache Parquet quản lý dữ liệu lớn**.
