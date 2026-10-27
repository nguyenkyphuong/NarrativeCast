# Cleaning Audit Report -- NarrativeCast

> **Input:** `data/raw/reddit_technology_madoc-001.parquet`  
> **Output:** `data/processed/reddit_technology_clean.parquet`

---

## 1. Overview

| Metric        |     Before |      After |              Delta |
| :------------ | ---------: | ---------: | -----------------: |
| Total rows    | 11,766,134 | 11,597,712 | -168,422 (-1.431%) |
| Total columns |         13 |         12 |                 -1 |
| Memory (MB)   |    9,312.7 |    6,477.7 |           -2,835.0 |

**Interaction type (after cleaning):**

- `COMMENT`: 10,829,229 (93.37%)
- `POST`: 768,483 (6.63%)

---

## 2. Rows Dropped by Reason

| Reason                   |       Count | % of Original |
| :----------------------- | ----------: | ------------: |
| post_duplicate           |      58,223 |       0.4948% |
| garbage_empty_whitespace |      40,059 |       0.3405% |
| bot_automod_boilerplate  |      39,746 |       0.3378% |
| garbage_lt_3_alpha       |      15,994 |       0.1359% |
| bot_spam_test_message    |      13,596 |       0.1156% |
| bot_deleted_by_user      |         798 |       0.0068% |
| garbage_pseudo_null      |           6 |       0.0001% |
| **TOTAL DROPPED**        | **168,422** |    **1.431%** |

---

## 3. Column Changes

| Column                 | Action  | Detail                                                              |
| :--------------------- | :------ | :------------------------------------------------------------------ |
| `community`            | DROPPED | Zero-variance / unused (Constant 'technology')                      |
| `language`             | DROPPED | Zero-variance / unused (Constant 'en'/'English')                    |
| `platform`             | DROPPED | Zero-variance / unused (Constant 'reddit')                          |
| `strict_filter`        | DROPPED | Zero-variance / unused (100% null)                                  |
| `url`                  | DROPPED | Zero-variance / unused (87.86% null / unused in embedding pipeline) |
| `content_hash`         | ADDED   | dtype: `uint64`                                                     |
| `is_usable_text`       | ADDED   | dtype: `bool`                                                       |
| `publish_datetime_utc` | ADDED   | dtype: `datetime64[ns, UTC]`                                        |
| `word_count`           | ADDED   | dtype: `int32`                                                      |

---

## 4. Text Quality After Cleaning

| Metric                 |               Value |
| :--------------------- | ------------------: |
| word_count mean        |               37.17 |
| word_count median      |                  20 |
| word_count P05         |                   4 |
| word_count P95         |                 123 |
| word_count P99         |                 252 |
| word_count max         |                5145 |
| is_usable_text = True  | 10,844,716 (93.51%) |
| is_usable_text = False |     752,996 (6.49%) |
| Unique content_hash    | 11,375,527 (98.08%) |

---

## 5. Temporal Integrity Check

| Check      | Result                                                 |
| :--------- | :----------------------------------------------------- |
| Date range | 2014-01-01 00:00:16+00:00 -> 2020-12-31 23:59:25+00:00 |

---

## 6. Memory Profile

| Stage          |     Memory (MB) |
| :------------- | --------------: |
| Raw load       |         9,312.7 |
| After pipeline |         6,477.7 |
| Saved          | 2,835.0 (30.4%) |

---

## 7. Output Schema

|  #  | Column                 | Dtype                 |
| :-: | :--------------------- | :-------------------- |
|  1  | `post_id`              | `object`              |
|  2  | `publish_date`         | `int64`               |
|  3  | `publish_datetime_utc` | `datetime64[ns, UTC]` |
|  4  | `user_id`              | `object`              |
|  5  | `interaction_type`     | `category`            |
|  6  | `parent_id`            | `object`              |
|  7  | `parent_user_id`       | `object`              |
|  8  | `content`              | `string`              |
|  9  | `content_hash`         | `uint64`              |
| 10  | `word_count`           | `int32`               |
| 11  | `is_usable_text`       | `bool`                |
| 12  | `sentiment_vader`      | `float32`             |

---

## 8. Summary of Key Cleaning Actions & Evidence-Based Rationale

1. **Thu gọn cột & Loại bỏ trường không sử dụng:**
   - Xóa bỏ hoàn toàn 5 cột hằng số, rỗng hoặc không phục vụ mục tiêu mô hình hóa: `strict_filter` (100% null), `platform` (hằng số 'reddit'), `community` (hằng số 'technology'), `language` (hằng số 'en'/'English'), và `url` (87.86% null, không mang giá trị ngữ nghĩa cho SBERT). Giúp thu gọn bảng dữ liệu từ 13 cột gốc xuống 12 cột chuẩn đầu ra (đồng thời bổ sung 4 cột mới: `publish_datetime_utc`, `word_count`, `is_usable_text`, `content_hash`) và tiết kiệm hàng GB RAM.

2. **Chuẩn hóa Thời gian & Bảo toàn Thứ tự:**
   - Chuyển `publish_date` (UNIX epoch giây) sang `publish_datetime_utc` (`datetime64[ns, UTC]`), xác nhận dữ liệu nằm trọn trong khoảng 2014–2020 (2,557 ngày liên tục, 0 gap) và kiểm định thứ tự thời gian tăng dần tuyệt đối (`is_monotonic_increasing`) để chia cửa sổ trượt $W_{hist}=7$ ngày hoàn toàn không bị rò rỉ dữ liệu (data leakage).

3. **Tối ưu Kiểu dữ liệu (Downcasting):**
   - Ép kiểu sớm `interaction_type` từ `object` sang `category` (chỉ có 2 giá trị `POST` và `COMMENT`) và `sentiment_vader` từ `float64` sang `float32`, tiết kiệm ngay ~630 MB RAM trước khi bước vào các phép biến đổi văn bản nặng.

4. **Làm sạch Văn bản Không phá hủy (Steps 6–12):**
   - Giải mã HTML entities bằng cách chỉ áp dụng `html.unescape` trên các dòng có chứa '&' (856,739 dòng chứa `&gt;`, `&amp;`... chiếm 7.28%), tối ưu tốc độ gấp 10 lần và tránh rác từ vựng `amp`.
   - Chuẩn hóa Unicode NFC (`.str.normalize('NFC')`) và loại bỏ ký tự điều khiển/lỗi mã hóa.
   - Bóc tách link Markdown `[text](url)` -> `text` (bảo toàn trọn vẹn ngữ nghĩa anchor text), xóa URL thô (`https?://\S+`).
   - Xóa ký tự đánh dấu định dạng Markdown (`**`, `~~`, `` ` ``, và dấu `>` đầu dòng `r'(?m)^>\s*'`) trong khi giữ nguyên toàn bộ chữ bên trong.
   - Cắt hậu tố `[removed]` và `[deleted]` bị hệ thống Reddit nối dính ở cuối tiêu đề bài POST (83,214 lần) bằng `r'\s*\[(removed|deleted)\]\s*$'`.
   - Chuẩn hóa khoảng whitespace (`\s+` -> ' ') và `strip()`.

5. **Lọc bỏ Rác, Bot & Khử trùng lặp POST (Loại bỏ tổng cộng 168,422 dòng - 1.431%):**
   - Loại bỏ 40,059 dòng rỗng/chỉ khoảng trắng sau làm sạch, 15,994 dòng `< 3 ký tự chữ cái` (không mang ngữ nghĩa), và 6 dòng giả-null (`"nan"`, `"none"`).
   - Loại bỏ 39,746 dòng thông báo xóa bài tự động của `AutoModerator`, 13,596 dòng bot spam tiếng Pháp (`"Message de test vraie hein"`), và 798 dòng `[deleted by user]`.
   - Khử trùng lặp chính xác trên POST (`post_duplicate`): loại bỏ 58,223 bài `POST` trùng nội dung (giữ bài đầu tiên theo thời gian) để ngăn ngừa BERTopic hình thành các cụm chủ đề nhân tạo (artificial clusters), trong khi vẫn giữ nguyên các `COMMENT` trùng lặp tự nhiên của cộng đồng.

6. **Tạo Cột Metadata Hỗ trợ Pipeline sau:**
   - Tạo `word_count` (`int32`, dùng `.str.count(r'\S+')` không tốn auxiliary RAM).
   - Tạo cờ `is_usable_text` (`bool`): phân hóa ngưỡng theo bản chất tương tác (`POST >= 3 từ`, `COMMENT >= 5 từ`). Đạt 10,844,716 dòng khả dụng (93.51%) cho SBERT/BERTopic mà vẫn giữ đủ toàn bộ 11,597,712 dòng để đếm chính xác tổng lượng tương tác $N_t$ trong công thức Prominence $P(n, W_t)$.
   - Tạo `content_hash` (`uint64` murmur hash qua `pd.util.hash_pandas_object`): xác định 11,375,527 chuỗi văn bản duy nhất (98.08%), cho phép bước SBERT Embedding chỉ cần nhúng 1 lần cho mỗi hash rồi map lại, tiết kiệm tối đa thời gian tính toán.
