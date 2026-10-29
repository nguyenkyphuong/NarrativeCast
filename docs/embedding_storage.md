# Hướng Dẫn Cấu Trúc Lưu Trữ & Tái Sử Dụng Vector Embedding (Embedding Storage Guide)

- **Đề tài:** NarrativeCast — Predicting Emerging Narratives in Social Media Using Temporal Semantic Representations
- **Module mã nguồn:** [`src/embeddings/embedding_store.py`](../src/embeddings/embedding_store.py) · [`src/embeddings/encode_posts.py`](../src/embeddings/encode_posts.py)
- **Tài liệu phối hợp:** [`docs/processed_data_schema.md`](./processed_data_schema.md) · [`docs/embedding_model_selection.md`](./embedding_model_selection.md) · [`docs/embedding_input_requirements.md`](./embedding_input_requirements.md)

---

## 1. Mục Tiêu & Nguyên Tắc Không Tính Toán Lại (Zero Redundant Computation)

Trong quy trình nghiên cứu của **NarrativeCast**, việc chuyển đổi văn bản mạng xã hội thành vector ngữ nghĩa thông qua các mô hình ngôn ngữ sâu (Sentence Transformers) là bước tiêu tốn nhiều thời gian và tài nguyên phần cứng (GPU/CPU) nhất:
- Với tập dữ liệu thật của Reddit r/technology (`data/processed/reddit_technology_clean.parquet` gồm 11.597.712 dòng, 2,37 GB), quá trình mã hóa toàn bộ đòi hỏi hàng chục giờ tính toán GPU hoặc nhiều ngày CPU.
- Nếu mỗi lần thử nghiệm thay đổi thuật toán phân cụm (BERTopic, HDBSCAN, k-Means), tinh chỉnh số lượng cụm, hoặc thay đổi cửa sổ dự báo ($h \in \{3, 7, 14\}$ ngày) mà phải tạo lại vector từ đầu, tiến độ nghiên cứu sẽ bị gián đoạn nghiêm trọng.

Do đó, **thiết kế lưu trữ embedding** giải quyết các bài toán kỹ thuật cốt lõi:
1. **Tránh tính toán lại (Zero Redundant Computation):** Chỉ tạo vector một lần duy nhất cho mỗi bài đăng/văn bản sạch, sau đó nạp lại tức thì (trong vài mili-giây) ở các lần thử nghiệm tiếp theo.
2. **Bảo toàn liên kết 1-1 giữa Vector, `post_id` và Thời gian:** Xác định chính xác vector nào thuộc về bài đăng nào, hỗ trợ truy xuất $O(1)$, batch lookup và trích xuất theo ID.
3. **Phục vụ Temporal Narrative Tracking:** Lưu trữ đồng bộ `publish_datetime_utc` kèm vector để chia cửa sổ thời gian (time-slice) mà không cần truy vấn ngược lại file dữ liệu gốc.
4. **Quản lý phiên bản & Nguồn gốc mô hình (Model Lineage & Versioning):** Ghi nhận rõ mô hình nào, phiên bản nào, tham số chuẩn hóa ra sao đã sinh ra vector.
5. **Kiểm tra tính toàn vẹn (Data Integrity):** Tự động phát hiện lỗi lệch số lượng, lệch số chiều vector, giá trị bất thường (`NaN`, `Inf`) hoặc ID bị trùng lặp.

---

## 2. Các Lưu Ý Nghiên Cứu Quan Trọng (Research Considerations)

### 2.1. Phân biệt cốt lõi: Topic ≠ Narrative
Một trong những cạm bẫy phổ biến trong phân tích truyền thông xã hội là đồng nhất chủ đề thảo luận với luận điểm:
- **Topic (Chủ đề):** Đại diện cho lĩnh vực, thực thể hoặc sự kiện được bàn luận (ví dụ: `AI Regulation`, `Semiconductor Manufacturing`, `Data Privacy`).
- **Narrative (Luận điểm / Dòng diễn ngôn):** Đại diện cho lập trường, thông điệp hoặc góc nhìn diễn giải cụ thể bên trong chủ đề đó. Ví dụ trong cùng một chủ đề `AI Regulation`, tồn tại ít nhất 2 narrative đối lập:
  + Narrative A: *"AI Regulation Stifles Innovation and Weakens Global Competitiveness"* (Lập trường phản đối kiểm soát gắt gao).
  + Narrative B: *"Urgent Global Guardrails Are Essential to Prevent Catastrophic AI Risks"* (Lập trường yêu cầu quản lý an toàn khẩn cấp).

> [!IMPORTANT]
> **Không mặc định mỗi cluster từ embedding là một narrative!**  
> Các thuật toán clustering thuần vector ngữ nghĩa (như BERTopic/HDBSCAN) thường gom các bài viết có cùng trường từ vựng vào một cụm chủ đề (Topic Cluster). Để xác định một Narrative, các module tiếp theo cần kết hợp phân tích phân phối thực thể, cực tính cảm xúc (`sentiment_vader`), lập trường (stance) và diễn tiến thời gian.

### 2.2. Bảo toàn trục thời gian & Chống rò rỉ dữ liệu tương lai (Lookahead Bias)
- **Chuẩn hóa thời gian:** Luôn dùng `publish_datetime_utc` làm trục thời gian chuẩn hóa.
- **Tránh Lookahead Bias:** Khi huấn luyện mô hình dự báo narrative mới nổi ở thời điểm $T$, tuyệt đối không sử dụng vector, tần suất cụm hay thống kê xuất hiện sau thời điểm $T$. Lớp `EmbeddingStore` cung cấp hàm `filter_by_timerange(start_date, end_date)` giúp trích xuất chính xác lát cắt dữ liệu trong quá khứ một cách an toàn.

---

## 3. Thống Nhất Quy Chuẩn Giữa Các Module

| Quy chuẩn | Yêu cầu kỹ thuật | Module thực thi |
| :--- | :--- | :--- |
| **Dữ liệu đầu vào** | Đọc từ `data/processed/reddit_technology_clean.parquet`. **Chỉ lấy dòng có `is_usable_text == True`** (10.844.716 dòng). Xem chi tiết tại [`docs/processed_data_schema.md`](./processed_data_schema.md). | Module 1 (Preprocessing / Embedding) |
| **Mô hình embedding** | Tuân thủ lựa chọn tại [`docs/embedding_model_selection.md`](./embedding_model_selection.md): <br>• Mặc định: `sentence-transformers/all-MiniLM-L6-v2` (384 chiều, 256 tokens max, tối ưu Reddit). <br>• Dự phòng ngữ cảnh dài: `BAAI/bge-small-en-v1.5` (384 chiều, 512 tokens max). | Module 1 (Embedding Pipeline) |
| **Bảo toàn ID** | Mọi vector phải liên kết chặt chẽ với `post_id`. Hỗ trợ truy xuất $O(1)$ qua hash index. | Module 1 & Module 2 |
| **Bàn giao cho Clustering** | Module Phân cụm (Clustering / Topic Modeling) nạp vector trực tiếp từ kho mà không tạo lại. Kết quả phân cụm bắt buộc phải gán `cluster_id` tương ứng với từng `post_id`. | Module 2 (Clustering) |
| **Bàn giao cho Temporal Tracking** | Sử dụng hàm `store.attach_clusters(cluster_ids)` và `store.to_clustering_dataframe()` để xuất bảng `[post_id, publish_datetime_utc, cluster_id]`. | Module 3 / Module 4 (Temporal Tracking) |

---

## 4. Chiến Lược Lưu Trữ Lai (Hybrid Storage Strategy)

| Định dạng | Tốc độ nạp (Load Speed) | Dung lượng nén | Khả năng truy vấn quan hệ | Đánh giá & Mục đích sử dụng trong NarrativeCast |
| :--- | :---: | :---: | :---: | :--- |
| **NumPy Compressed (`.npz`)** | **Cực nhanh** (<0.1s cho 100k vector) | **Rất tốt** (nén zlib) | Hạn chế (truy xuất ma trận theo index) | **Khuyên dùng cho khâu huấn luyện mô hình**: nạp ma trận $\mathbf{X} \in \mathbb{R}^{N \times D}$ vào GPU/CPU cho BERTopic, UMAP, HDBSCAN. |
| **Apache Parquet (`.parquet`)** | Nhanh | **Xuất sắc** (Snappy) | **Rất mạnh** (tích hợp `post_id`, `publish_datetime_utc`, `cluster_id`) | **Khuyên dùng cho Data Warehouse / Analysis**: phân tích quan hệ bảng bằng DuckDB, Polars hoặc Pandas. |
| **Bundle Directory (`bundle/`)** | Nhanh | Toàn diện | Rất mạnh | Thư mục đóng gói chuẩn hóa: `metadata.json` + `embeddings.npz` + `embeddings.parquet`. |

---

## 5. Cấu Trúc Thư Mục Tiêu Chuẩn

Các file embedding được lưu trữ tại thư mục `outputs/embeddings/` hoặc `data/embeddings/`:

```text
NarrativeCast/
├── data/
│   ├── processed/
│   │   └── reddit_technology_clean.parquet       # File dữ liệu sạch dùng chung (2,37 GB)
│   └── embeddings/
│       └── all-minilm-l6-v2__sample_10k/         # Bundle embedding của một lần chạy thử nghiệm
│           ├── metadata.json                     # Thông tin mô hình, phiên bản, thời gian tạo
│           ├── embeddings.npz                    # Ma trận vector, post_id, timestamps, cluster_id
│           └── embeddings.parquet                # Bảng dữ liệu quan hệ kèm vector
```

> [!WARNING]
> **Quy định Git:** Tuyệt đối không push file `.parquet` lớn hoặc file `.npz` vector lên GitHub remote repo. Tất cả các định dạng dữ liệu lớn đã được khai báo loại trừ trong `.gitignore`. Các thành viên chỉ chia sẻ code, notebook và tài liệu báo cáo.

---

## 6. Hướng Dẫn Thực Hành & Code Mẫu

### 6.1. Tạo Embedding từ Dữ Liệu Sạch (Chạy Thử Nghiệm Mẫu Nhỏ)
```python
from src.embeddings.encode_posts import encode_posts

# Chạy thử nghiệm trên mẫu 1,000 bài (chỉ lấy is_usable_text == True)
result = encode_posts(
    input_data="data/processed/reddit_technology_clean.parquet",
    sample_size=1000,
    filter_usable=True,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=64,
    output_dir="outputs/embeddings",
    save_format="both",  # Sinh cả .npz và .parquet
)

print(f"Hoàn thành mã hóa: {len(result)} bài viết, D={result.embedding_dim}")
```

### 6.2. Nạp Lại Vector & Tái Sử Dụng Cho Phân Cụm (BERTopic)
```python
from src.embeddings.embedding_store import load_embeddings

# 1. Nạp kho embedding đã lưu mà không tốn công tính toán lại
store = load_embeddings("outputs/embeddings/post_embeddings.npz", validate=True)
print(f"Đã nạp {len(store):,} vectors, chiều D={store.embedding_dim}")

# 2. Truy xuất ma trận vector cho thuật toán gom cụm
vectors = store.vectors          # np.ndarray shape (N, 384)
post_ids = store.post_ids        # np.ndarray shape (N,)
timestamps = store.timestamps    # np.ndarray shape (N,) chứa publish_datetime_utc

# 3. Chạy phân cụm (ví dụ BERTopic hoặc HDBSCAN)
from sklearn.cluster import HDBSCAN
clusterer = HDBSCAN(min_cluster_size=15, metric="cosine")
cluster_labels = clusterer.fit_predict(vectors)

# 4. Gắn nhãn cluster_id vào store và xuất bảng cho Module 4
store.attach_clusters(cluster_labels)
df_tracking = store.to_clustering_dataframe()
print(df_tracking.head())
# Output:
#    post_id                   publish_datetime_utc  cluster_id
# 0  a2e5ce4e-... 2020-01-01 10:00:00+00:00           3
# 1  05da00c8-... 2020-01-01 10:05:00+00:00           0
```

### 6.3. Trích Xuất Lát Cắt Cửa Sổ Thời Gian (Temporal Slicing)
```python
# Trích xuất dữ liệu của riêng năm 2019 để phân tích diễn biến luận điểm
store_2019 = store.filter_by_timerange(
    start_date="2019-01-01",
    end_date="2019-12-31 23:59:59"
)
print(f"Số lượng bài đăng năm 2019: {len(store_2019):,}")
```

### 6.4. Kiểm Tra Tính Toàn Vẹn (Integrity Verification)
```python
from src.embeddings.embedding_store import validate_embeddings

report = store.validate(strict=True)
print(report.summary())
assert report.is_valid, f"Kho embedding không hợp lệ: {report.errors}"
```
