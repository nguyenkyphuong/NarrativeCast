# Dataset Card: MADOC (Multi-Platform Aggregated Dataset of Online Communities)

## 1. Nguồn & Giấy phép (Source & License)
- **Tên bộ dữ liệu:** MADOC (Multi-Platform Aggregated Dataset of Online Communities)
- **Nguồn:** Zenodo / Google Drive (`ie403_dataset_zenodo`)
- **Giấy phép (License):** Creative Commons Attribution International 4.0 License (CC-BY 4.0)
- **Liên hệ tác giả:** madoc@ctrust.ac.rs
- **Mục đích thu thập:** Phục vụ nghiên cứu khoa học xã hội tính toán (computational social science), phân tích động lực mạng xã hội đa nền tảng, theo dõi sự lan truyền hành vi và kiểm duyệt nội dung.

## 2. Quy mô & Định dạng (Scale & Format)
- **Định dạng dữ liệu:** Apache Parquet (`.parquet`) nén theo từng cộng đồng.
- **Tổng quy mô toàn bộ bộ dữ liệu:**
  - **18.9 triệu** bài viết (posts)
  - **236 triệu** bình luận (comments)
  - **23.1 triệu** người dùng duy nhất (unique users)
- **Chi tiết từng nền tảng & Khung thời gian:**
  - **Reddit (2014 – 2020):** 247.6M tương tác (14.5M posts, 233.1M comments), 22.6M users.
  - **Voat (2013 – 2020):** 1.2M tương tác (0.4M posts, 0.8M comments), 0.1M users.
  - **Bluesky (2023 – 2024):** 2.8M tương tác (0.9M posts, 0.9M comments), 0.2M users.
  - **Koo (2020 – 2023):** 4.3M tương tác (3.1M posts, 1.2M comments), 0.2M users.

## 3. Cấu trúc trường dữ liệu (Data Schema)
Mỗi bản ghi đại diện cho một tương tác (post, comment, repost) chứa 13 trường thông tin chuẩn hóa:

| Tên trường (Field) | Kiểu / Định dạng | Mô tả chi tiết |
| :--- | :--- | :--- |
| `post_id` | String (Anonymized) | Mã định danh duy nhất của bài viết/bình luận (đã mã hóa ẩn danh) |
| `publish_date` | Timestamp (UNIX) | Mốc thời gian tạo bài đăng (tính bằng giây từ epoch) |
| `user_id` | String (Anonymized) | Mã định danh người tạo nội dung (đã mã hóa UUID) |
| `parent_id` | String / NA | ID bài viết/bình luận cha (NA nếu là bài đăng chính) |
| `parent_user_id` | String / NA | ID của người tạo bài viết/bình luận cha (NA nếu là bài gốc) |
| `content` | String | Nội dung văn bản (đã trích xuất các liên kết URL) |
| `url` | String | Các đường dẫn URL bên ngoài được trích dẫn trong bài |
| `language` | String | Mã ngôn ngữ phát hiện (hiện tại 100% là `'en'` - Tiếng Anh) |
| `interaction_type` | Enum | Loại tương tác: `'POST'`, `'COMMENT'`, hoặc `'REPOST'` |
| `platform` | Enum | Nền tảng nguồn: `'reddit'`, `'voat'`, `'bluesky'`, hoặc `'koo'` |
| `community` | String / NA | Tên cộng đồng/subreddit/subverse (NA với Bluesky/Koo) |
| `sentiment_vader` | Float (-1.0 đến 1.0) | Điểm phân tích cảm xúc VADER |
| `strict_filter` | Binary (0 / 1) | Cờ đánh dấu nội dung khớp từ khóa lọc nghiêm ngặt |

## 4. Hạn chế & Quyền riêng tư (Limitations & Ethics)
- **Độ bao phủ cộng đồng:** Bao gồm cả cộng đồng phổ thông (*funny, gaming, technology, videos...*) và các cộng đồng đặc thù/tranh cãi (*fatpeoplehate, GreatAwakening, MensRights...*).
- **Quyền riêng tư:** Toàn bộ ID người dùng đã được ẩn danh hóa bằng thuật toán UUID hashing. Không chứa thông tin định danh cá nhân (PII) trực tiếp.
- **Giới hạn ngôn ngữ:** Chỉ hỗ trợ nội dung bằng Tiếng Anh (`'en'`).
