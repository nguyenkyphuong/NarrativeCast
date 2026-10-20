# Embedding Input Requirements — NarrativeCast (MADOC · Reddit r/technology)

> **Phạm vi và nguồn kiểm tra**
> - Schema: `dataset_card.md` (MADOC, 13 trường chuẩn) và `generate_sample_data.py`.
> - Số liệu toàn bộ dataset: `01_dataset_summary_dashboard.pdf` (chỉ có số tổng hợp, **không có văn bản thật**).
> - Thống kê văn bản: tính trực tiếp trên `madoc_sample.parquet` (5.000 dòng, 15 cột).
>
> **Lưu ý quan trọng:** `madoc_sample.parquet` là dữ liệu **tổng hợp (synthetic)** do `generate_sample_data.py` sinh ra từ 36 câu mẫu, không phải trích từ Reddit thật. Vì vậy các con số bên dưới chỉ kiểm chứng được *schema và pipeline*, **không phản ánh chất lượng văn bản của 11,77 triệu dòng thật**. Các mục cần kiểm tra trên dữ liệu thật được đánh dấu **[CẦN KIỂM TRA TRÊN DỮ LIỆU THẬT]**.

---

## 1. Kết luận nhanh

| Câu hỏi | Kết luận |
| :--- | :--- |
| Văn bản sample có dùng được để tạo embedding không? | **Có về mặt kỹ thuật**: 0 null, 0 rỗng, 0 chỉ-khoảng-trắng, không URL/HTML/ký tự lỗi, độ dài 15–24 từ. |
| Có dùng sample để đánh giá chất lượng embedding/mô hình không? | **Không nên**: 92,8% dòng trùng nội dung, chỉ 360 văn bản duy nhất (36 câu lõi), và văn bản chứa sẵn tín hiệu nhãn (xem mục 4). |
| Dữ liệu thật đã sẵn sàng chưa? | **Chưa kết luận được**: chưa có văn bản thật để thống kê rỗng/ngắn/bất thường (xem mục 6). |

---

## 2. Trường văn bản cần dùng

| Mục | Quy định |
| :--- | :--- |
| **Trường đầu vào embedding** | `content` |
| **Kiểu dữ liệu** | `string` (parquet `BYTE_ARRAY`/UTF-8), `optional` theo schema file nhưng sample có 0 null |
| **Mã hóa** | UTF-8; sample không có ký tự thay thế `U+FFFD` |
| **Trường KHÔNG đưa vào văn bản embedding** | `url` (đã tách khỏi `content`), `topic`, `ground_truth_narrative` (nhãn, chỉ có trong sample, **không thuộc 13 trường chuẩn**), `sentiment_vader`, `strict_filter`, mọi ID |
| **Trường giữ lại làm metadata đi kèm vector** | `post_id` (khóa), `publish_date` (cửa sổ thời gian), `interaction_type`, `parent_id`, `user_id` |
| **Trường chỉ dùng để lọc/kiểm tra** | `language` (100% `en` theo dataset card), `platform`, `community` |

---

## 3. Thống kê văn bản trên `madoc_sample.parquet`

### 3.1 Cấu trúc

| Chỉ số | Giá trị |
| :--- | ---: |
| Số dòng | 5.000 |
| Số cột | 15 (13 chuẩn + `topic` + `ground_truth_narrative`) |
| `post_id` trùng | 0 |
| `interaction_type` | POST 2.364 (47,3%) · COMMENT 2.636 (52,7%) · REPOST 0 |
| `language` / `platform` / `community` | 100% `en` / `reddit` / `technology` |
| Khoảng thời gian | 2024-01-01 → 2024-02-29 (60 ngày) |

### 3.2 Văn bản rỗng, ngắn, bất thường

| Kiểm tra | Số dòng | Ghi chú |
| :--- | ---: | :--- |
| `content` null | 0 | |
| `content` chuỗi rỗng `""` | 0 | |
| `content` chỉ chứa khoảng trắng | 0 | |
| Chuỗi giả-null (`nan`, `none`, `null`, `N/A`) | 0 | |
| `[deleted]` / `[removed]` | 0 | |
| < 3 từ / < 5 từ / < 10 từ | 0 / 0 / 0 | Từ ngắn nhất: 15 từ |
| < 20 ký tự / < 50 ký tự | 0 / 0 | Ngắn nhất: 124 ký tự |
| Chứa URL trong `content` | 0 | Khớp mô tả "đã trích xuất URL" |
| Chứa thẻ HTML / entity (`&amp;`…) | 0 / 0 | |
| Ký tự điều khiển / xuống dòng / khoảng trắng thừa | 0 / 0 / 0 | |
| Ký tự non-ASCII (emoji, dấu…) | 0 | Không đại diện cho Reddit thật |
| `content` trùng với `url` | 0 | |
| `@mention` | 0 | |
| Chứa `r/technology` | 2.636 | Toàn bộ COMMENT (chuỗi mẫu cố định, xem mục 4) |

### 3.3 Độ dài

| Chỉ số | Từ | Ký tự |
| :--- | ---: | ---: |
| Min | 15 | 124 |
| Trung bình | 19,0 | 154,4 |
| Trung vị | 19 | 155 |
| P95 / P99 | 22 / 23 | — |
| Max | 24 | 179 |
| Ước lượng token (ký tự ÷ 4) | max ≈ 45, trung bình ≈ 39 | |

Mọi dòng đều nằm xa dưới giới hạn 128/256/512 token của các mô hình embedding phổ biến, nên sample **không phát sinh truncation**.

> **Sample ngắn hơn dữ liệu thật:** dashboard báo trung bình **37,1 từ/dòng** cho toàn bộ dataset, so với 19,0 từ trong sample. Không dùng độ dài sample để chọn `max_seq_length`.

### 3.4 Trùng lặp

| Chỉ số | Giá trị |
| :--- | ---: |
| Văn bản duy nhất | **360 / 5.000** |
| Dòng trùng nội dung | **4.640 (92,8%)** |
| Câu lõi duy nhất (sau khi bỏ tiền tố/hậu tố mẫu) | **36** |
| Văn bản duy nhất mỗi narrative | 60 (đồng đều cho cả 6 narrative) |
| Văn bản lặp nhiều nhất | 31 lần |

---

## 4. Rủi ro rò rỉ nhãn và nhiễu cấu trúc (riêng với sample)

Các điểm này làm embedding của sample **dễ phân tách giả tạo** theo narrative và theo loại tương tác:

1. **Tiền tố chủ đề trong POST:** 2.364/2.364 POST bắt đầu bằng `[<topic>]`, trùng 100% với cột `topic`. Đây là nhãn chủ đề nằm sẵn trong văn bản.
2. **Từ khóa → narrative 1–1:** 30 từ khóa chèn vào văn bản (`Regarding <keyword>:` / `Discussion around <keyword>.`) mỗi từ khóa chỉ thuộc đúng 1 narrative. Mỗi câu lõi cũng chỉ thuộc đúng 1 narrative.
3. **Khuôn mẫu theo loại tương tác:** COMMENT luôn có `Regarding …:` ở đầu và `This is crucial for r/technology.` ở cuối. POST luôn có `[topic]` ở đầu và `Discussion around …` ở cuối. Mô hình embedding sẽ nhóm theo khuôn mẫu này thay vì theo ý nghĩa.
4. **Tỷ lệ khuôn mẫu:** khoảng 37% số ký tự trung bình của mỗi văn bản là khuôn mẫu lặp lại.

**Khuyến nghị cho sample:** khi dùng sample để thử pipeline embedding, tạo thêm cột `content_core` (bỏ `[topic] `, `Regarding …: `, ` This is crucial for r/technology.`, ` Discussion around …`) và embed cột đó. Không dùng embedding của sample để kết luận về khả năng dự đoán narrative.

---

## 5. Đối chiếu schema thực tế với dataset card

| Mục | Dataset card | Sample parquet | Xử lý đề xuất |
| :--- | :--- | :--- | :--- |
| Số trường | 13 | 15 (+`topic`, `ground_truth_narrative`) | Loại 2 trường thừa khỏi mọi bước tạo embedding |
| `publish_date` | UNIX giây | `int64` | Chuyển sang datetime UTC khi tạo cửa sổ |
| `parent_id`, `parent_user_id` | `String / NA` | `null` thật (2.364 dòng POST) | Khi đọc dữ liệu thật, kiểm tra `NA` là null hay chuỗi `"NA"` |
| `url` | String | chuỗi rỗng `""` ở 3.020 dòng (60,4%), không có null | Chuẩn hóa `""` → null khi cần, nhưng `url` không vào embedding |
| `community` | `String / NA` | luôn `technology` | Không có NA ở sample |
| `interaction_type` | `POST`/`COMMENT`/`REPOST` | chỉ POST, COMMENT | Kiểm tra REPOST trên dữ liệu thật |
| Thứ tự thời gian | Không nói | **1.479 dòng sai thứ tự** `publish_date` (do thêm offset ngẫu nhiên 0–3.600 giây sau khi sắp xếp) | **Luôn sắp xếp theo `publish_date`** trước khi chia cửa sổ, không dựa vào thứ tự dòng |
| Tỷ lệ POST/COMMENT | — | 47,3% / 52,7% | Dữ liệu thật: 847.657 POST (7,2%) / 10.918.477 COMMENT (92,8%), lệch hẳn so với sample |

Các số trong dashboard khớp nội bộ: 847.657 + 10.918.477 = 11.766.134 dòng; 2.557 ngày ≈ 7 năm, phù hợp khung 2014–2020 của Reddit trong dataset card. Dataset card không nêu số dòng riêng cho r/technology nên chưa đối chiếu được con số này với card.

---

## 6. Quy tắc xử lý trước khi tạo embedding

Thứ tự áp dụng đề xuất. Ngưỡng là **đề xuất khởi điểm**, cần chỉnh sau khi xem phân phối trên dữ liệu thật.

| # | Bước | Quy tắc | Trạng thái trên sample |
| :-: | :--- | :--- | :--- |
| 1 | Kiểm tra kiểu | `content` phải là `str`; ép các giá trị khác kiểu thành lỗi, không ngầm chuyển | Đạt |
| 2 | Chuẩn hóa Unicode | NFKC, bỏ ký tự điều khiển và `U+FFFD`, giải mã HTML entity (`&amp;`, `&gt;`…) | Không cần |
| 3 | Chuẩn hóa khoảng trắng | Gộp khoảng trắng/xuống dòng liên tiếp thành một dấu cách, `strip()` | Không cần |
| 4 | Loại placeholder | Bỏ dòng có `content` ∈ {`[deleted]`, `[removed]`, `nan`, `none`, `null`, ``} (không phân biệt hoa thường) | 0 dòng |
| 5 | Dọn dấu vết tách URL | Dataset card nói URL đã được trích khỏi `content`. Kiểm tra phần còn sót: `[text]()`, `()`, khoảng trắng đôi, câu cụt | **[CẦN KIỂM TRA TRÊN DỮ LIỆU THẬT]** |
| 6 | Loại văn bản quá ngắn | Đề xuất: loại nếu < 3 từ; gắn cờ `short` nếu < 5 từ để xem xét riêng | 0 dòng |
| 7 | Loại văn bản không có nội dung chữ | Loại nếu sau khi bỏ ký tự không phải chữ cái còn < 3 ký tự (chỉ emoji/dấu câu/số) | 0 dòng |
| 8 | Xử lý văn bản quá dài | Đặt `max_seq_length` theo mô hình; ghi nhận số dòng bị cắt. Mặc định cắt phần đuôi, không loại | Không phát sinh (max ≈ 45 token) |
| 9 | Xử lý trùng lặp | Giữ **mọi** dòng cho thống kê khối lượng thảo luận theo thời gian. Chỉ tính embedding một lần cho mỗi văn bản duy nhất rồi ánh xạ lại bằng hash của `content` | Tiết kiệm 92,8% tính toán ở sample |
| 10 | Kiểm tra ngôn ngữ | Dataset card nói 100% `en`; vẫn nên lấy mẫu kiểm tra bằng bộ phát hiện ngôn ngữ vì nhãn có thể sai | **[CẦN KIỂM TRA TRÊN DỮ LIỆU THẬT]** |
| 11 | Loại trường rò rỉ | Không nối `topic`/`ground_truth_narrative`/từ khóa vào văn bản. Với sample, dùng `content_core` (mục 4) | Cần áp dụng |
| 12 | Sắp xếp và lưu | Sắp xếp theo `publish_date` tăng dần, lưu `post_id` cùng vector, ghi tên mô hình, phiên bản, `max_seq_length` và tham số chuẩn hóa | Cần áp dụng |

---

## 7. Checklist kiểm tra trước khi chạy embedding trên dữ liệu thật

**[CẦN KIỂM TRA TRÊN DỮ LIỆU THẬT]** — hiện chưa có số liệu, các mục sau cần được thống kê và điền vào:

- [ ] Số dòng `content` null / rỗng / chỉ khoảng trắng (đối chiếu với tổng 11.766.134)
- [ ] Số dòng `[deleted]` / `[removed]` và các placeholder khác
- [ ] Phân phối số từ: min, P1, P5, trung vị, P95, P99, max (đối chiếu với trung bình 37,1 từ ở dashboard)
- [ ] Số dòng dưới 3 từ, dưới 5 từ, dưới 10 từ — tách riêng theo POST và COMMENT
- [ ] Tỷ lệ dòng vượt `max_seq_length` của mô hình đã chọn
- [ ] Tỷ lệ văn bản trùng lặp chính xác và trùng gần đúng (copypasta, bot, AutoModerator)
- [ ] Dấu vết markdown/HTML: `&amp;`, `&gt;`, `>` đầu dòng, `**`, `[ ]( )`, xuống dòng
- [ ] Phần sót sau khi tách URL (dấu ngoặc rỗng, khoảng trắng đôi)
- [ ] Ký tự lỗi mã hóa, emoji, văn bản không phải tiếng Anh dù `language = 'en'`
- [ ] Giá trị `NA` của `parent_id`, `parent_user_id`, `community` là null thật hay chuỗi `"NA"`
- [ ] Có dòng `REPOST` không, và `content` của chúng có rỗng hay lặp văn bản gốc không
- [ ] Sự phân bố `publish_date` (khoảng trống thời gian, dòng ngoài khoảng 2014–2020, thứ tự)


