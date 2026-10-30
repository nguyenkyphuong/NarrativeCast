# Quy cách cấu trúc dữ liệu sạch (Processed Data Schema) — NarrativeCast

> **File chuẩn:** `data/processed/reddit_technology_clean.parquet`  
> **Quy mô:** 11.597.712 dòng | 12 cột | Kích thước đĩa: ≈ 2,37 GB  
> **Nguồn gốc:** Làm sạch từ `data/raw/reddit_technology_madoc-001.parquet` theo pipeline `src/preprocessing/clean_data.py` (Tham chiếu: `outputs/reports/cleaning_audit_report.md`).

---

## 1. Bảng quy cách cột (Schema Definition)

| Tên cột | Kiểu dữ liệu (PyArrow / Pandas) | Null count | Mục đích & Ràng buộc sử dụng |
| :--- | :--- | :--- | :--- |
| `post_id` | `string` / `object` | 0 | **Khóa chính duy nhất (UUID string)** định danh bài đăng (POST) hoặc bình luận (COMMENT). Bắt buộc liên kết 1-1 với vector embedding. |
| `publish_date` | `int64` | 0 | UNIX timestamp nguyên bản tính theo giây. |
| `publish_datetime_utc` | `timestamp[ns, tz=UTC]` | 0 | **Thời gian chuẩn hóa UTC** (2014-01-01 -> 2020-12-31). Phục vụ chia cửa sổ thời gian (time-slice) và theo dõi tiến trình luận điểm (Temporal Narrative Tracking). |
| `user_id` | `string` / `object` | 0 | Mã ẩn danh của tác giả bài viết. |
| `interaction_type` | `category` (`POST`, `COMMENT`) | 0 | Loại tương tác: `POST` (768.483 dòng ~ 6,63%) hoặc `COMMENT` (10.829.229 dòng ~ 93,37%). |
| `parent_id` | `string` / `object` | 9.561.196 | ID bài viết/bình luận cha mà bình luận này phản hồi. Bằng `null` nếu là bài đăng gốc. |
| `parent_user_id` | `string` / `object` | 9.554.752 | Mã tác giả của bài viết/bình luận cha. |
| `content` | `string` | 0 | **Văn bản nội dung đã dọn dẹp**: đã trích xuất URL, chuẩn hóa khoảng trắng, gỡ bỏ emoji hỏng / HTML entity, dọn dẹp các mẫu rác. |
| `content_hash` | `uint64` | 0 | Mã băm nội dung phục vụ khử trùng văn bản nhanh chóng. |
| `word_count` | `int32` | 0 | Số lượng từ trong văn bản sau làm sạch (Mean: 37,17, Median: 20, Max: 5.145). |
| `is_usable_text` | `bool` | 0 | **Cờ chất lượng nội dung**: <br>• `True`: 10.844.716 dòng (93,51%) — Văn bản đạt chuẩn ngữ nghĩa.<br>• `False`: 752.996 dòng (6,49%) — Văn bản quá ngắn (<3 từ), chứa placeholder, bot boilerplate hoặc không mang giá trị phân tích. |
| `sentiment_vader` | `float32` | 0 | Điểm chỉ số cảm xúc VADER compound trong đoạn [-1.0, 1.0]. |

---

## 2. Nguyên tắc bắt buộc đối với module Embedding & Clustering

1. **Lọc dữ liệu hợp lệ (`is_usable_text = True`)**:
   - **Bắt buộc**: Pipeline embedding chỉ trích xuất và tính toán vector cho những bản ghi có `is_usable_text == True`.
   - Không tạo vector cho 752.996 bản ghi `is_usable_text == False` để tránh gây nhiễu không gian vector và lãng phí tài nguyên tính toán.

2. **Bảo toàn ID và Thời gian (`post_id` & `publish_datetime_utc`)**:
   - Vector embedding sinh ra phải đi kèm `post_id` để đồng đội thực hiện gom cụm (Clustering / Topic Modeling).
   - Phải giữ lại `publish_datetime_utc` để phục vụ phân tích diễn tiến thời gian (Temporal Narrative Tracking ở Module 4). Không đánh mất cột thời gian khi lưu trữ vector.

3. **Tính bất biến của Schema**:
   - Tuyệt đối không tự ý đổi tên cột, chỉnh sửa kiểu dữ liệu hoặc ghi đè file `reddit_technology_clean.parquet`.

4. **Tránh quá tải bộ nhớ khi xử lý**:
   - File có kích thước 2,37 GB (11,6 triệu dòng). Không nạp toàn bộ file vào RAM cùng lúc nếu tài nguyên máy hạn chế.
   - Sử dụng streaming hoặc đọc theo row group / batch với `pyarrow.parquet.ParquetFile`.
