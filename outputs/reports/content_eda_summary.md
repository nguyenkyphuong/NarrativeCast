# NarrativeCast — Content & Sub-domain EDA Summary Report

> **Source:** `notebooks/03_content_subdomain_eda.ipynb`  
> **Dataset:** `reddit_technology_madoc-001.parquet` — 11,766,134 rows × 13 cols (Chỉ nạp 7 cols -> Tối ưu RAM từ 6,664 MB xuống 5,973 MB)  
> **Khung thời gian:** 2014-01-01 -> 2020-12-31 (84 tháng liên tục)  
> **Trạng thái:** Hoàn thành Bước 2 (Đã chạy thực nghiệm trên 11.76M dòng, kiểm định đầy đủ 12 biểu đồ PDF và đồng bộ 100% số liệu thực tế)

---

## Part A: Kiểm định Chất lượng Văn bản Thực tế (Text Quality Audit for `clean_data.py`)

### 1. Placeholder & Garbage Content Detection (Section 1)

- **Biểu đồ PDF:** `outputs/content_eda/01_placeholder_garbage.pdf`
- **Số liệu thực nghiệm (`POST = 847,657` | `COMMENT = 10,918,477`):**

| Mẫu Rác / Placeholder (`Pattern`)       | Số lượng `POST` | Tỷ lệ `POST` (%) | Số lượng `COMMENT` | Tỷ lệ `COMMENT` (%) |
| :-------------------------------------- | --------------: | ---------------: | -----------------: | ------------------: |
| `< 3 alpha chars` (Dưới 3 chữ cái)      |           3,858 |          0.4551% |              8,969 |             0.0821% |
| `Empty / NaN` (Rỗng / Khuyết)           |               0 |          0.0000% |                 38 |             0.0003% |
| `Whitespace-only` (Chỉ khoảng trắng)    |               0 |          0.0000% |                 14 |             0.0001% |
| `Empty brackets []()` (Ngoặc rỗng)      |               0 |          0.0000% |                 11 |             0.0001% |
| `[deleted]` (Khớp toàn chuỗi)           |               0 |          0.0000% |                 10 |             0.0001% |
| `Pseudo-null (nan/none/null)`           |               0 |          0.0000% |                  4 |             0.0000% |
| `[removed]` (Khớp toàn chuỗi)           |               0 |          0.0000% |                  1 |             0.0000% |
| **Union (6 nhóm rác rỗng/placeholder)** |           **0** |      **0.0000%** |             **78** |         **0.0007%** |

- **Nhận xét & Quy tắc làm sạch:**
  - Các dòng placeholder độc lập chiếm tỷ lệ cực nhỏ (78 dòng), nhưng nội dung siêu ngắn `< 3 ký tự chữ cái` xuất hiện ở **3,858 bài POST (0.46%)** và **8,969 COMMENT (0.08%)**.
  - Đặc biệt, đối chiếu với Section 4 và Section 6.1 cho thấy trong dữ liệu MADOC, chuỗi `[removed]` thường bị nối dính vào đuôi tiêu đề bài `POST` (xuất hiện **83,214 lần** trong nhóm Unclassified) thay vì đứng một mình. Do đó, `clean_data.py` cần vừa lọc bỏ các dòng `< 3 alpha chars`, vừa dùng Regex cắt bỏ hậu tố `\[removed\]$` và `\[deleted\]$` ở cuối chuỗi.

### 2. HTML Entities, Markdown Residue & URL Artifacts (Section 2)

- **Biểu đồ PDF:** `outputs/content_eda/02_content_artifacts.pdf`
- **Số liệu thực nghiệm trên 11,766,134 dòng:**

| Loại Nhiễu Kỹ thuật (`Artifact`) | Số dòng bị ảnh hưởng (`Affected Rows`) | Tỷ lệ (`Pct %`) | Ghi chú chi tiết                                       |
| :------------------------------- | -------------------------------------: | --------------: | :----------------------------------------------------- |
| `HTML entity (&amp; &gt; etc.)`  |                                856,739 |          7.281% | `&gt;` (1,076,295), `&amp;` (300,242), `&lt;` (22,171) |
| `URL in content body`            |                                592,480 |          5.035% | Đường dẫn `https?://\S+`                               |
| `Markdown bold/italic (**)`      |                                531,611 |          4.518% | Cú pháp nhấn mạnh `**...**`                            |
| `Markdown link [text](url)`      |                                323,889 |          2.753% | Cần giữ lại `text`, xóa `(url)`                        |
| `Markdown strikethrough (~~)`    |                                 21,603 |          0.184% | Cú pháp gạch ngang `~~...~~`                           |
| `Inline code (backticks)`        |                                  4,037 |          0.034% | Cặp dấu `` `...` ``                                    |
| `Empty brackets artifact`        |                                     15 |          0.000% | Cặp ngoặc rỗng `[]()` sót lại                          |
| `Markdown blockquote (>)`        |                                      0 |          0.000% | Đã bị mã hóa hoàn toàn thành `&gt;`                    |

- **Nhận xét & Quy tắc làm sạch:**
  - Việc chạy `html.unescape()` trước tiên là bắt buộc để giải mã hơn **1.07 triệu** ký tự `&gt;` và **300 ngàn** ký tự `&amp;` (ngăn chặn từ rác `amp` làm nhiễu không gian vector). Tiếp theo, cần chuyển đổi `[text](url)` thành `text` và xóa toàn bộ URL độc lập cũng như các ký tự định dạng Markdown.

### 3. Phân phối Độ dài Văn bản: POST vs COMMENT (Section 3)

- **Biểu đồ PDF:** `outputs/content_eda/03_wordcount_post_vs_comment.pdf`
- **Bảng thống kê phân vị `word_count` và kiểm định ngưỡng cắt cụt SBERT:**

| Chỉ số thống kê (`Metric`)                    |  Tập `POST` ($N = 847,657$) | Tập `COMMENT` ($N = 10,918,477$) |
| :-------------------------------------------- | --------------------------: | -------------------------------: |
| **Mean $\pm$ Std**                            |        $10.99 \pm 25.05$ từ |             $39.02 \pm 55.62$ từ |
| **Min / P05 / P10**                           |                1 / 3 / 4 từ |                     0 / 4 / 6 từ |
| **P25 / Median (P50) / P75**                  |           7 / **9** / 12 từ |              11 / **22** / 46 từ |
| **P90 / P95 / P99 / Max**                     | 16 / 21 / **43** / 5,149 từ |    87 / 127 / **258** / 4,733 từ |
| **Quá ngắn: `< 3 words`**                     |              23,335 (2.75%) |                  263,941 (2.42%) |
| **Quá ngắn: `< 5 words`**                     |             91,225 (10.76%) |                  777,533 (7.12%) |
| **Vượt ngưỡng SBERT 256-tok (`> 200 words`)** |           **1,187 (0.14%)** |              **204,241 (1.87%)** |
| **Vượt ngưỡng SBERT 512-tok (`> 400 words`)** |             **523 (0.06%)** |               **33,957 (0.31%)** |

- **Nhận xét & Quy tắc làm sạch:**
  - Tiêu đề `POST` rất ngắn gọn (Median = 9 từ, P99 = 43 từ), do đó chỉ nên lọc ở ngưỡng `< 3 từ` (2.75%) để không làm mất hơn 10.76% tiêu đề tin tức ngắn có từ 3–4 từ. Với `COMMENT` (Median = 22 từ), có thể áp dụng ngưỡng lọc `< 5 từ` (7.12%) để loại bỏ các bình luận cảm thán cụt lủn.
  - Mô hình SBERT `all-MiniLM-L6-v2` (256 tokens $\approx$ 200 từ) bảo toàn trọn vẹn ngữ nghĩa của **99.86% bài POST** và **98.13% COMMENT**, hoàn toàn không cần đánh đổi sang các mô hình 512 tokens nặng nề hơn.

### 4. Phân tích Nội dung Trùng lặp & Định vị Bot / Power-Users (Section 4)

- **Biểu đồ PDF:** `outputs/content_eda/04_duplicate_content.pdf`
- **Số liệu thực nghiệm:**
  - Tổng số dòng liên quan đến trùng lặp chính xác: **385,663 dòng (3.28%)** trên toàn tập (chứa 11,474,216 chuỗi duy nhất).
  - Phân tách theo loại tương tác: **POST có 95,206 dòng trùng lặp (11.23%)** | **COMMENT có 289,251 dòng trùng lặp (2.65%)**.
- **Bảng đối chiếu chéo Top chuỗi trùng lặp với Top-20 Power-Users (từ Bài 02):**

| Chuỗi nội dung trùng lặp (`Content truncated`)            | Tổng số bản sao | Số bản sao từ Top-20 Users | Tỷ lệ từ Power-Users (%) | Bản chất nội dung          |
| :-------------------------------------------------------- | --------------: | -------------------------: | -----------------------: | :------------------------- |
| `Message de test vraie hein[removed]`                     |          13,596 |                     13,596 |               **100.0%** | Bot spam/test trên POST    |
| `Thank you for your submission! Unfortunately...` (Mẫu 1) |           8,757 |                      8,486 |                **96.9%** | AutoModerator xóa bài      |
| `Thank you for your submission! Unfortunately...` (Mẫu 2) |           4,812 |                      2,711 |                **56.3%** | AutoModerator xóa bài      |
| `Thank you for your submission! Unfortunately...` (Mẫu 3) |           3,116 |                      2,491 |                **79.9%** | AutoModerator xóa bài      |
| `Thank you for your submission! Unfortunately...` (Mẫu 4) |           2,369 |                      2,017 |                **85.1%** | AutoModerator xóa bài      |
| `Source?`                                                 |           1,818 |                         12 |                 **0.7%** | Câu hỏi ngắn của user thật |
| `Thank you for your submission! Unfortunately...` (Mẫu 5) |           1,792 |                        691 |                **38.6%** | AutoModerator xóa bài      |
| `Thank you for your submission! Unfortunately...` (Mẫu 6) |           1,695 |                      1,667 |                **98.3%** | AutoModerator xóa bài      |
| `Thank you for your submission! Unfortunately...` (Mẫu 7) |           1,515 |                        637 |                **42.0%** | AutoModerator xóa bài      |
| `Why?`                                                    |           1,493 |                         15 |                 **1.0%** | Câu hỏi ngắn của user thật |

- **Nhận xét & Quy tắc làm sạch:**
  - Kết quả đối chiếu xác nhận chính xác nghi vấn từ `02_temporal_eda.ipynb`: các tài khoản đăng hàng chục ngàn bài trong Top-20 Power-Users thực chất là **AutoModerator** và **Spam Bot**. Việc lọc bỏ các mẫu câu boilerplate của AutoModerator và khử trùng lặp `POST` trong `clean_data.py` (thông qua cột băm `content_hash`) sẽ loại bỏ hoàn toàn các cụm chủ đề giả tạo (artificial clusters) trước khi chạy BERTopic.

---

## Part B: Khám phá Ngách Chủ đề & Thu hẹp Phạm vi (Sub-domain Discovery & Scoping)

### 5. Bộ từ điển Từ khóa (Keyword Taxonomy Definition - Section 5)

Bao gồm 6 ngách chủ đề chính được chuẩn hóa với ranh giới từ `\b` phù hợp giai đoạn 2014–2020:

1. **AI & Automation**
2. **Cybersecurity & Privacy**
3. **Big Tech Antitrust & Regulation**
4. **Semiconductors & Hardware**
5. **Clean Energy, EVs & Climate Tech**
6. **Telecom, Net Neutrality, Web & Crypto**

### 6. Quy mô Ngách trên Tập POST & Ma trận Giao thoa (Section 6)

- **Biểu đồ PDF:** `outputs/content_eda/05_subdomain_volume_posts.pdf` & `outputs/content_eda/06_subdomain_overlap_heatmap.pdf`
- **Thống kê quy mô bài đăng `POST` ($N = 847,657$) và mức độ giao thoa:**
  1. **Big Tech Antitrust & Regulation:** **113,204 POSTs (13.3%)** — Quy mô áp đảo nhất toàn tập dữ liệu.
  2. **Telecom, Net Neutrality, Web & Crypto:** **46,100 POSTs (5.4%)** — Đứng thứ 2 về quy mô bài đăng mở đầu.
  3. **Semiconductors & Hardware:** **33,236 POSTs (3.9%)** — Đứng thứ 3.
  4. **Cybersecurity & Privacy:** **28,159 POSTs (3.3%)** — Đứng thứ 4.
  5. **Clean Energy, EVs & Climate Tech:** **17,296 POSTs (2.0%)** — Đứng thứ 5.
  6. **AI & Automation:** **14,387 POSTs (1.7%)** — Đứng thứ 6.
- **Đặc điểm chồng lấn (Overlap Heatmap):** Các ngách có độ phân tách tốt (mọi cặp giao thoa đều `< 18%`). Cặp có giao thoa tự nhiên cao nhất là _Cybersecurity & Privacy_ nằm trong _Big Tech_ (**17.1%**), _AI & Automation_ nằm trong _Big Tech_ (**13.2%**), và _Semiconductors_ giao thoa với _Telecom_ (**13.0%**).

### 6.1. Khai phá Tự động theo Hướng Bottom-Up trên Nhóm Unclassified (Section 6.1)

- **Biểu đồ PDF:** `outputs/content_eda/06b_unclassified_top_ngrams.pdf`
- **Kết quả kiểm định phần dư:**
  - Các n-gram có tần suất cao nhất trong tập `POST` chưa phân loại bao gồm: hậu tố `removed` (**83,214**), các từ khóa câu hỏi hỗ trợ kỹ thuật / tiêu dùng cá nhân như `support` (**32,265**), `new` (**27,778**), `number` (**26,013**), `free` (**21,006**), `phone` (**20,385**), `best` (**20,077**), `app` (**19,750**), `tech` (**19,539**), `technology` (**18,446**), mã lỗi HTML `amp` (**16,965**), và cụm spam tiếng Pháp `message test` / `vraie hein` (**13,596**).
  - **Kết luận:** Không có chủ đề chính sách hay sự kiện công nghệ lớn nào bị bỏ sót. Bộ từ điển 6 ngách đã bao phủ trọn vẹn các luồng thảo luận nghị sự cốt lõi của `r/technology`.

### 7. Tính Liên tục và Độ phủ Thời gian 7 Năm (2014–2020 - Section 7)

- **Biểu đồ PDF:** `outputs/content_eda/07_subdomain_temporal_coverage.pdf`
- **Kết quả kiểm định:**
  - Cả 6 ngách chủ đề đều đạt **độ phủ 84/84 tháng (100%)**, không có tháng nào bị rơi xuống dưới ngưỡng 10 bài `POST`.
  - **Big Tech Antitrust & Regulation** luôn duy trì ở dải cao nhất (**900 -> ~2,400 POSTs/tháng**).
  - **Telecom, Net Neutrality, Web & Crypto** ghi nhận cú vọt lịch sử lên tới **~1,600 POSTs/tháng vào cuối năm 2017**, khớp hoàn toàn với đợt sóng tranh luận bãi bỏ Net Neutrality của FCC đã phát hiện ở `02_temporal_eda.ipynb`.

### 8. Độ Phân cực Cảm xúc & Tiềm năng Narrative (Section 8)

- **Biểu đồ PDF:** `outputs/content_eda/08_sentiment_by_subdomain.pdf` & `outputs/content_eda/09_sentiment_violin_subdomain.pdf`
- **Bảng số liệu thực nghiệm đầy đủ (POST toàn tập & COMMENT mẫu 15% = 1,637,771 dòng):**

| Ngách Chủ đề (`Sub-domain`)           | Loại (`Type`)     | Quy mô ($N$) |        Mean | Median |        Std |   Neg (%) | Neu (%) | Pos (%) |
| :------------------------------------ | :---------------- | -----------: | ----------: | -----: | ---------: | --------: | ------: | ------: |
| **Cybersecurity & Privacy**           | **POST**          |       28,159 | **-0.0224** | 0.0000 | **0.3810** | **35.1%** |   36.0% |   28.8% |
| AI & Automation                       | POST              |       14,387 |      0.1223 | 0.0000 |     0.3510 |     15.2% |   45.0% |   39.8% |
| Telecom, Net Neutrality, Web & Crypto | POST              |       46,100 |      0.0354 | 0.0000 |     0.3508 |     22.6% |   47.3% |   30.1% |
| Big Tech Antitrust & Regulation       | POST              |      113,204 |      0.0568 | 0.0000 |     0.3483 |     22.3% |   42.4% |   35.3% |
| Clean Energy, EVs & Climate Tech      | POST              |       17,296 |      0.0956 | 0.0000 |     0.3282 |     15.4% |   48.5% |   36.1% |
| Semiconductors & Hardware             | POST              |       33,236 |      0.0742 | 0.0000 |     0.3236 |     15.3% |   54.1% |   30.6% |
| **AI & Automation**                   | **COMMENT (15%)** |       12,213 |      0.1191 | 0.1193 | **0.5900** |     34.6% |   13.2% |   52.2% |
| **Cybersecurity & Privacy**           | **COMMENT (15%)** |       52,202 |      0.0465 | 0.0000 | **0.5825** | **39.8%** |   13.0% |   47.2% |
| **Big Tech Antitrust & Regulation**   | **COMMENT (15%)** |      164,464 |      0.0871 | 0.0302 | **0.5710** |     36.5% |   13.8% |   49.7% |
| Semiconductors & Hardware             | COMMENT (15%)     |       30,109 |      0.1497 | 0.1548 |     0.5696 |     32.1% |   14.4% |   53.4% |
| Telecom, Net Neutrality, Web & Crypto | COMMENT (15%)     |      104,347 |      0.0870 | 0.0000 |     0.5646 |     36.4% |   14.6% |   48.9% |
| Clean Energy, EVs & Climate Tech      | COMMENT (15%)     |       36,252 |      0.1607 | 0.1989 |     0.5632 |     31.5% |   13.8% |   54.7% |

- **Kết quả kiểm định:**
  - Trên tập `POST` (Biểu đồ `08`), **Cybersecurity & Privacy** là ngách duy nhất có trung bình cảm xúc âm (`Mean = -0.0224` hay `-0.022`) và đạt **độ phân cực cảm xúc cao nhất (`Std = 0.3810` hay `0.381`)**. Theo sát về độ phân cực là **AI & Automation** (`Std = 0.3510`), **Telecom** (`Std = 0.3508`) và **Big Tech** (`Std = 0.3483`).
  - Trên biểu đồ Violin so sánh `POST` vs `COMMENT` (Biểu đồ `09`), trong khi `POST` tập trung mạnh ở vùng trung tính (`0.00`), thì `COMMENT` ở cả 6 ngách đều bị kéo dãn mạnh về hai cực âm/dương (`Q1` lùi về `-0.30 -> -0.45`, `Q3` vọt lên `+0.55 -> +0.65`).
  - **Độ phân cực vọt xà ở COMMENT:** Trên mẫu 15% đại diện, độ lệch chuẩn `Std` của `COMMENT` ở cả 6 ngách tăng vọt lên dải **`0.563 -> 0.590`**, chứng minh rằng sự phân cực quan điểm (narrative divergence) diễn ra mãnh liệt nhất ở tầng bình luận của người dùng.

### 9. Bảng Điểm Đa Tiêu chí & Đề xuất Ngách Tối ưu (Section 9)

- **Biểu đồ PDF:** `outputs/content_eda/10_subdomain_radar.pdf` & `outputs/content_eda/11_subdomain_ranking.pdf`
- **Bảng số liệu thô và Điểm chuẩn hóa Radar (Normalized scores):**

| Ngách Chủ đề (`Sub-domain`)               | POST Vol (raw / norm) | Coverage (raw / norm) | Polarization (raw / norm) | Amplification (raw / norm) | Điểm Tổng hợp (`Weighted Score`) |
| :---------------------------------------- | :-------------------: | :-------------------: | :-----------------------: | :------------------------: | :------------------------------: |
| **Big Tech Antitrust & Regulation**       |  113,204 / **1.000**  |  100.0% / **0.500**   |    0.3483 / **0.430**     |      9.69 / **0.427**      |            **0.618**             |
| **Cybersecurity & Privacy**               |  28,159 / **0.139**   |  100.0% / **0.500**   |    0.3810 / **1.000**     |     12.36 / **0.710**      |            **0.559**             |
| **Telecom, Net Neutrality, Web & Crypto** |  46,100 / **0.321**   |  100.0% / **0.500**   |    0.3508 / **0.474**     |     15.09 / **1.000**      |            **0.540**             |
| **Clean Energy, EVs & Climate Tech**      |  17,296 / **0.029**   |  100.0% / **0.500**   |    0.3282 / **0.080**     |     13.97 / **0.881**      |            **0.330**             |
| **AI & Automation**                       |  14,387 / **0.000**   |  100.0% / **0.500**   |    0.3510 / **0.477**     |      5.66 / **0.000**      |            **0.244**             |
| **Semiconductors & Hardware**             |  33,236 / **0.191**   |  100.0% / **0.500**   |    0.3236 / **0.000**     |      6.04 / **0.040**      |            **0.190**             |

- **Giải thích Radar `10_subdomain_radar.pdf`:**
  - `Coverage (84mo)_norm = 0.50` cho cả 6 ngách vì 100% độ phủ đều đạt ở cả 6 ngách.
  - _Big Tech_ độc tôn về Volume (`1.000`).
  - _Cybersecurity_ dẫn đầu tuyệt đối về Polarization (`1.000`).
  - _Telecom_ dẫn đầu tuyệt đối về Amplification (`1.000`).
- **Quyết định cuối cùng cho Pipeline (`clean_data.py` / `filter_subdomain.py`):** Thu hẹp phạm vi nghiên cứu từ toàn bộ `r/technology` vào **Big Tech Antitrust & Regulation** kết hợp với **Cybersecurity & Privacy** (và có thể mở rộng sang **Telecom / Net Neutrality** cho bài toán bắt đỉnh sự kiện đột biến). Tổ hợp này vừa đảm bảo mật độ chuỗi thời gian dày đặc xuyên suốt 84 tháng, vừa chứa đựng các cuộc xung đột quan điểm sâu sắc nhất để mô hình hóa vòng đời Narrative (RQ1 -> RQ3).

---

## Part C: Actionable Recommendations for `clean_data.py` (Section 10)

|  STT  | Hạng mục xử lý                                                              | Bằng chứng số liệu từ EDA (11,766,134 dòng)                                                                                                                                                                                                                       | Quy tắc triển khai cụ thể trong `scripts/preprocessing/clean_data.py`                                                                                                                                                                                                                                                                       |
| :---: | :-------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1** | **Giải mã HTML Entities**                                                   | **856,739 dòng (7.281%)** chứa mã HTML (`&gt;` 1,076,295 lần; `&amp;` 300,242 lần; `&lt;` 22,171 lần). Từ rác `amp` xuất hiện 16,965 lần trong Top N-grams.                                                                                                       | Chạy `html.unescape()` đầu tiên trên cột `content` trước mọi bước xử lý chuỗi khác.                                                                                                                                                                                                                                                         |
| **2** | **Làm sạch URL & Markdown**                                                 | **592,480 dòng (5.035%)** chứa URL thô; **531,611 dòng (4.518%)** chứa `**`; **323,889 dòng (2.753%)** chứa link `[text](url)`; **21,603 dòng (0.184%)** chứa `~~`.                                                                                               | Dùng Regex đổi `[text](url)` -> `text`, xóa toàn bộ `https?://\S+`, xóa ký tự đánh dấu `**`, `~~`, `` ` `` và dấu trích dẫn `>` đầu dòng.                                                                                                                                                                                                   |
| **3** | **Cắt hậu tố `[removed]` dính liền, Lọc Placeholder & Sinh `content_hash`** | Chỉ có **78 dòng** khớp chính xác hoàn toàn với `[deleted]`/`[removed]`/rỗng, nhưng từ khóa `removed` xuất hiện tới **83,214 lần** ở đuôi các bài `POST` chưa phân loại.                                                                                          | Dùng Regex xóa hậu tố `\[removed\]$` và `\[deleted\]$` bị nối vào cuối tiêu đề `POST`; loại bỏ các dòng rỗng, khoảng trắng hoặc giả-null (`"nan"`, `"none"`, `"null"`). Đồng thời tạo cột `content_hash` (MD5 hex của chuỗi đã chuẩn hóa) theo chuẩn `docs/embedding_input_requirements.md` để hỗ trợ deduplicate và tái sử dụng embedding. |
| **4** | **Loại bỏ Bot, Spam & AutoModerator**                                       | **385,663 dòng (3.28%)** trùng lặp chính xác (riêng `POST` là **11.23%**). Chuỗi spam `"Message de test vraie hein[removed]"` lặp lại **13,596 lần** (100% từ Top-20 user); 13/20 mẫu đầu là thông báo xóa bài của AutoModerator (38.6% -> 98.3% từ Top-20 user). | Lọc bỏ toàn bộ dòng khớp mẫu AutoModerator (`^Thank you for your (submission\|comment)! Unfortunately, it has been removed`), chuỗi `"Message de test vraie hein"`, `"[deleted by user]"` và khử trùng lặp chính xác trên `POST` qua `content_hash`.                                                                                        |
| **5** | **Ngưỡng độ dài tối thiểu (`word_count`)**                                  | `< 3 alpha chars`: 3,858 POST (0.46%) & 8,969 COMMENT (0.08%). `< 3 từ`: 23,335 POST (2.75%) & 263,941 COMMENT (2.42%). `< 5 từ`: 91,225 POST (10.76%) & 777,533 COMMENT (7.12%).                                                                                 | Loại bỏ mọi dòng có `< 3 ký tự chữ cái`. Áp dụng ngưỡng lọc độ dài linh hoạt: giữ `POST >= 3 từ` (bảo toàn tiêu đề tin tức ngắn) và `COMMENT >= 5 từ` (loại bỏ câu cảm thán cụt lủn như `"Source?"`, `"Why?"`, `"What?"`).                                                                                                                  |
| **6** | **Cấu hình SBERT `max_seq_length`**                                         | Chỉ có **0.14% POST (1,187 bài)** và **1.87% COMMENT (204,241 dòng)** vượt quá 200 từ (~256 tokens). P99 của POST = 43 từ, P99 của COMMENT = 258 từ.                                                                                                              | Giữ nguyên mô hình `all-MiniLM-L6-v2` với `max_seq_length = 256 tokens` — bảo toàn trọn vẹn ngữ nghĩa của **99.86% POST** và **98.13% COMMENT** với tốc độ nhúng nhanh nhất.                                                                                                                                                                |

---

## Danh mục 12 Biểu đồ PDF Xuất ra (`outputs/content_eda/`)

1. `01_placeholder_garbage.pdf`: Tỷ lệ nội dung rác và placeholder theo loại bài viết (`< 3 alpha chars` chiếm 0.46% POST và 0.08% COMMENT).
2. `02_content_artifacts.pdf`: Tần suất các artifact kỹ thuật (HTML entities 7.28%, URL 5.04%, Markdown bold 4.52%, Markdown link 2.75%).
3. `03_wordcount_post_vs_comment.pdf`: Phân phối `word_count` (POST Median=9, P99=43; COMMENT Median=22, P99=258) và ngưỡng cắt cụt SBERT 256-tok (~200w).
4. `04_duplicate_content.pdf`: Top 20 chuỗi văn bản trùng lặp nhiều nhất (dẫn đầu bởi spam `"Message de test vraie hein[removed]"` 13,596 lần và các mẫu AutoModerator).
5. `05_subdomain_volume_posts.pdf`: Quy mô bài viết POST theo 6 ngách chủ đề (Big Tech dẫn đầu với 113,204 bài - 13.3%).
6. `06_subdomain_overlap_heatmap.pdf`: Ma trận tỷ lệ chồng lấn giữa các ngách chủ đề (cao nhất là Cybersecurity trùng với Big Tech ở mức 17.1%).
7. `06b_unclassified_top_ngrams.pdf`: Top 30 n-grams trích xuất tự động từ nhóm POST Unclassified (phát hiện `removed` 83,214 lần và các từ khóa helpdesk chung chung).
8. `07_subdomain_temporal_coverage.pdf`: Biến động lưu lượng 6 ngách theo 84 tháng (2014–2020), nổi bật cú spike của Telecom/Net Neutrality cuối năm 2017.
9. `08_sentiment_by_subdomain.pdf`: So sánh giá trị cảm xúc trung bình và độ phân cực (`Std`), với Cybersecurity & Privacy âm nhất (`-0.0224`) và phân cực mạnh nhất (`0.3810`).
10. `09_sentiment_violin_subdomain.pdf`: Phân phối cảm xúc dạng Violin đối xứng giữa POST (tập trung ở 0.0) và COMMENT (kéo giãn mạnh về 2 cực với Std 0.563 -> 0.590).
11. `10_subdomain_radar.pdf`: Biểu đồ Radar so sánh 6 ngách trên 4 tiêu chí chuẩn hóa (Volume, Coverage, Polarization, Amplification).
12. `11_subdomain_ranking.pdf`: Bảng xếp hạng điểm số tổng hợp (Big Tech: 0.618, Cybersecurity: 0.559, Telecom: 0.540).
