# NarrativeCast — Temporal EDA Summary Report
> **Source:** `notebooks/02_temporal_eda.ipynb` 
> **Dataset:** `reddit_technology_madoc-001.parquet` — 11,766,134 rows × 13 cols (2.4 GB)
> **Khung thời gian:** 2014-01-01 → 2020-12-31 (2,557 ngày liên tục)


---

## 1. Temporal Continuity & Order (Section 1)

| Metric | Giá trị thực tế | Ý nghĩa kỹ thuật |
|---|---|---|
| Timeline | 2014-01-01 → 2020-12-31 | 7 năm hoàn chỉnh |
| Total days | **2,557 ngày** | Không có khoảng trống (0 gaps) |
| Missing days | **0 ngày** | Chuỗi thời gian liên tục 100% |
| Out-of-order rows | **0 dòng** | Dữ liệu MADOC đã được sắp xếp tăng dần theo `publish_date` |
| Max gap | **1.0 ngày** | Không có ngày nào bị đứt đoạn |
| Daily volume min / mean / max | **1,372 / 4,602 / 17,054** bài/ngày | Lưu lượng nền luôn dồi dào |
| Available sliding windows ($W_{hist}=7$) | **2,551 cửa sổ** | $2,557 - 7 + 1$ cửa sổ trượt khả dụng |

**Đề xuất:** Mỗi cửa sổ lịch sử $W_{hist}=7$ ngày luôn có đầy đủ 7 ngày dữ liệu, **không cần kỹ thuật padding** hay drop window thiếu dữ liệu.

---

## 2. Day-of-Week & Hour-of-Day Seasonality (Section 2)

| Metric | Giá trị thực tế | Ý nghĩa kỹ thuật |
|---|---|---|
| Weekend/Weekday ratio | **0.762** | Cuối tuần giảm **~23.8%** so với ngày trong tuần |
| Weekday avg volume | **~1.80M** tương tác | Mon–Fri lưu lượng cao (giờ làm việc) |
| Weekend avg volume | **~1.37M** tương tác | Sat–Sun lưu lượng giảm rõ rệt |
| Peak hours (UTC) | **15:00–19:00 UTC** | Tương ứng giờ làm việc buổi sáng – đầu giờ chiều Bắc Mỹ |
| Trough hours (UTC) | **08:00–10:00 UTC** | Đáy tương tác trong ngày |
| Dịch chuyển hành vi qua 7 năm | 2014–2016: đỉnh Thứ 4 – Thứ 5 (~16.0%–16.7%)<br>2018–2019: đỉnh Thứ 3 (~16.8%)<br>2020: đường phân phối phẳng hơn, Thứ 7 tăng lên ~13.8% (so với ~10.7% năm 2014) do đại dịch COVID-19 / WFH |

**Đề xuất:** 
- $W_{hist}=7$ ngày bao trọn vẹn 1 chu kỳ tuần (5 ngày làm việc + 2 ngày nghỉ), giúp triệt tiêu hoàn toàn day-of-week bias.
- Với $h=7$ và $h=14$ ngày (bội số của 7), mục tiêu dự báo $y_{n,t,h}$ miễn nhiễm với hiệu ứng ngày cuối tuần. Với $h=3$ ngày, việc chuẩn hóa bằng **Prominence $P(n, W_t) = N_{n,t}/N_t$** giúp triệt tiêu sự dao động khối lượng tuyệt đối do lệch ngày bắt đầu trong tuần.

---

## 3. POST vs COMMENT & Amplification Ratio (Section 3)

| Metric | Giá trị thực tế | Ý nghĩa kỹ thuật |
|---|---|---|
| Total POST | **847,657** (7.2%) | Bài viết mở đầu |
| Total COMMENT | **10,918,477** (92.8%) | Bình luận thảo luận |
| Overall Amplification | **12.88** COMMENT/POST | Mức độ tương tác chuyên sâu cao |
| Weekly Amplification median / mean | **12.73** / **13.97** | Phân phối lệch phải |
| Weekly Amplification max | **146.21** | Đỉnh nhọn bất thường quý 1/2014 |
| Days with `post_count = 0` | **11 ngày** | Khuyết cục bộ POST trong Pushshift đầu năm 2014 |
| Dịch chuyển cấu trúc 2018–2020 | POST giảm từ ~2,500–3,200/tuần (2016–2017) xuống ~1,200–2,000/tuần (2019–đầu 2020). Ngược lại, trung vị `COMMENT/POST` tăng mạnh từ đáy năm 2017 (~10.0) lên 2018 (~12.0), 2019 (~15.5) và đạt đỉnh năm 2020 (~17.0, phân vị 75% vượt trên 22 COMMENT/POST) |

**Đề xuất:** Giai đoạn 2018–2020 cộng đồng chuyển dịch sang thảo luận sâu gấp rưỡi trên mỗi chủ đề. Khi thiết kế đặc trưng, bắt buộc tách riêng `post_volume`, `comment_volume` và `amplification_ratio`, đồng thời xử lý 11 ngày khuyết `POST` ở `clean_data.py`.

---

## 4. `parent_id` Missingness (Section 4)

| Metric | Giá trị thực tế | Ý nghĩa kỹ thuật |
|---|---|---|
| COMMENT with `parent_id` | **2,072,940** (19.0%) | Tập con dựng được luồng đối thoại |
| COMMENT missing `parent_id` | **8,845,537** (81.0%) | Vấn đề hệ thống của Pushshift MADOC |
| POST missing `parent_id` | **847,657** (100.0%) | Đúng kỳ vọng (bài gốc không có cha) |
| Monthly miss rate: mean / std | **80.9%** / **1.9%** | Rất ổn định qua 84 tháng |
| Monthly miss rate: min / max | **77.6%** / **89.0%** | Không có đợt đứt gãy đột biến |

**Đề xuất:** Tỷ lệ thiếu mang tính hệ thống. **KHÔNG** loại bỏ COMMENT thiếu `parent_id` vì vẫn mang đầy đủ giá trị nội dung, cảm xúc và thời gian. Sử dụng 2.07M bản ghi có cha hợp lệ làm mẫu đại diện cho phân tích Reply Lag.

---

## 5. User Dynamics & Participation (Section 5)

| Metric | Giá trị thực tế | Ý nghĩa kỹ thuật |
|---|---|---|
| Unique users toàn tập | **1,263,268** tài khoản | Quy mô cộng đồng lớn |
| Pearson $r$ (Users ↔ Volume) | **0.9760** ($p < 10^{-300}$) | Tương quan tuyến tính gần hoàn hảo → Bùng phát tự nhiên (organic participation) |
| Posts/User: mean / median | **1.68** / **1.67** bài/user/ngày | Ổn định quanh 1.67; đỉnh nhọn giữa 2017 vọt > 2.05 (ngày 2017-08-08 đạt 2.70) và cuối 2018 vọt > 1.90 |
| Posts/User distribution: P50 / P90 / P99 / Max | **2 / 15 / 129 / 20,471** bài/user | Long-tail cực mạnh |
| Single-post users | **44.9%** (566,876 users) | Gần một nửa tài khoản chỉ bình luận đúng 1 lần |
| Bot / Power users phát hiện | Top 1: **20,471 bài**, Top 2: **17,530 bài** | AutoModerator / News Bot → Cần lọc bỏ ở Content EDA |
| New User Ratio theo tuần | Đầu 2014 vọt >80% do Left-censoring (bắt đầu thu thập); giai đoạn 2015–2020 ổn định và tăng nhẹ về cuối kỳ |

**Đề xuất:** Đã sửa công thức tính mẫu số `weekly_total = df.set_index('publish_date')['user_id'].resample('W').nunique()` để tránh đếm trùng người dùng hoạt động nhiều ngày. Đưa `user_diversity` (tỷ lệ user mới trong $W_{hist}$) vào tập đặc trưng Khối B (Weng et al.).

---

## 6. Reply Lag & Discussion Lifespan (Section 6)

| Metric | Giá trị thực tế | Ý nghĩa kỹ thuật |
|---|---|---|
| Valid reply pairs | **2,072,940 cặp** | Match rate 100%, Valid rate 100% |
| Median reply lag | **6.29 giờ** (~0.26 ngày) | Trung vị phản hồi |
| Mean reply lag | **11.14 giờ** (~0.46 ngày) | Bị kéo bởi đuôi dài |
| Phân vị độ trễ | P25 = **3.13h**, P75 = **10.79h**, P90 = **15.78h**, P99 = **36.99h** (~1.5 ngày) | 75% bình luận xuất hiện trong 11h |
| `< 1 giờ` | **10.83%** | Phản xạ tức thì |
| `< 6 giờ` | **47.89%** | Gần một nửa phản hồi trong 6h đầu |
| `< 24 giờ (1 ngày)` | **97.71%** | Gần như toàn bộ thảo luận hoàn tất trong ngày |
| `< 3 ngày` ($h=3$) | **99.47%** | Bắt trọn 99.47% động lực thảo luận của thread |
| `< 7 ngày` ($h=7$) | **99.71%** | Chỉ tăng thêm **+0.24%** so với mốc 3 ngày |
| `< 14 ngày` ($h=14$) | **99.79%** | Chỉ tăng thêm **+0.08%** so với mốc 7 ngày (còn 0.21% sau 14 ngày) |
| Tính ổn định qua 7 năm | Trung vị ổn định **~6.0–6.5 giờ** xuyên suốt 2014–2020 (giai đoạn 2019–2020 Q3 nhích nhẹ lên ~11.5h, râu trên tới ~24h) |

**Ý nghĩa khoa học giải thích RQ2:**
- **Chu kỳ bán rã cực nhanh:** 97.71% bình luận hoàn tất trong 24 giờ đầu, 99.47% kết thúc trong 3 ngày.
- Ở $h=3$ ngày, quán tính luồng thảo luận hiện tại còn tác động trực tiếp. Ở $h=7$ và $h=14$ ngày, các bài đăng cũ đã hoàn toàn tắt; một **Narrative** chỉ có thể tiếp tục nổi bật nếu có **các bài đăng mới (new posts/threads) tiếp tục khởi phát xoay quanh cùng mạch ngữ nghĩa đó**. Điều này khẳng định sự cần thiết của Temporal Semantic Representations (BERTopic/Embedding) để liên kết các bài đăng rời rạc theo thời gian.

---

## 7. Sentiment Trajectory & Polarization (Section 7)

| Metric | Giá trị thực tế | Ý nghĩa kỹ thuật |
|---|---|---|
| Overall mean sentiment | **+0.05** | Trung lập – hơi tích cực |
| Dịch chuyển cảm xúc theo thời gian | 2014–2015: tích cực nhất (+0.053 đến +0.092, phủ xanh đậm)<br>2016–2018: giảm dần (+0.034 đến +0.072)<br>2019–2020: giảm sâu (+0.009 đến +0.066, chuyển tông cam/đỏ, đáy tháng 06/2020 là **+0.009**) | Xu hướng "lạnh đi" (critical shift), chuyển từ hào hứng công nghệ sang hoài nghi, phê phán độc quyền Big Tech, quyền riêng tư |
| Sự gia tăng mức độ phân cực (`Std Dev`) | Tăng bậc thang từ **~0.466–0.475** (2014) lên **0.485–0.503** (2017–2020), với các đỉnh vượt **0.500** vào giữa 2017, giữa 2019 và giữa 2020 | Thảo luận giai đoạn sau chứa đựng các luồng quan điểm đối lập gay gắt hơn hẳn |
| Phân hóa POST vs COMMENT | POST luôn tích cực hơn (đỉnh +0.14~+0.15 vào cuối 2016 – giữa 2017, hạ về +0.05~+0.08 vào 2019–2020). COMMENT luôn thấp hơn và chạm mốc **0.00** vào giữa năm 2020 | Độ lệch POST vs COMMENT là đặc trưng đo lường mức độ phản biện/tranh cãi |

**Đề xuất:** Đưa `sentiment_mean` và `sentiment_std` (polarization) trong $W_{hist}$ vào tập đặc trưng thời gian cốt lõi.

---

## 8. Spike / Anomaly Detection (Section 8)

| Metric | Giá trị thực tế | Ý nghĩa kỹ thuật |
|---|---|---|
| Spike days ($Z > 3$) | **24 ngày** (0.94% tổng số ngày) | Sự kiện hiếm (rare events) phân bố đều 7 năm |
| Spike days ($Z > 4$) | **3 ngày** | Các sự kiện chấn động công nghệ |
| Top 1: **2017-10-26** | $Z = 4.70$, Vol = 14,978 bài, Users = 6,444 | Sentiment chuyển âm (**-0.09**), Std cao (**0.53**), Amplification vọt lên **34.49** |
| Top 2: **2016-09-08** | $Z = 4.40$, Vol = **17,054 bài** (kỷ lục), Users = **8,794** | Kỷ lục số người dùng trong 1 ngày, Amplification = **32.05** |
| Top 3: **2020-01-20** | $Z = 4.11$, Vol = 10,879 bài, Users = 7,431 | Amplification cao nhất nhóm spike (**60.12**), Sentiment âm (**-0.02**) |
| Hành vi tại điểm bùng phát | 22/24 ngày spike có `Posts/User` nằm trong dải bình thường (1.46 – 2.11 bài/user); chỉ có ngày 2017-08-08 (2.70) và 2017-10-26 (2.32) cao hơn | Khẳng định đợt bùng phát là do hiệu ứng đám đông tự nhiên (viral), không phải bot spam |
| Mức độ phân cực tại spike | Phần lớn các ngày spike đi kèm `Sentiment Std` cao (**0.485 – 0.540**) và hệ số khuếch đại lớn (**20 – 60 COMMENT/POST**) | Tranh cãi dữ dội xoay quanh sự kiện |

---

## 9. Đề xuất Thiết kế Pipeline & Gợi ý Thu hẹp Phạm vi (Scoping)

1. **Khung thời gian trọng tâm (Temporal Scoping):**
   - Đề xuất ưu tiên giai đoạn **2018–2020** (hoặc mở rộng **2017–2020**): Đây là giai đoạn có độ sâu thảo luận cao nhất (`Amplification` tăng từ 12 lên 17), tính phân cực đối kháng rõ nét nhất (`Sentiment Std` ~0.49–0.50), và hoàn toàn không bị lỗi kỹ thuật khuyết `POST` như đầu năm 2014.
2. **Bộ đặc trưng thời gian cốt lõi (Core Temporal Features):**
   - `Z-score Volume / Growth Momentum`: Đo lường tốc độ gia tăng thảo luận.
   - `Amplification Ratio (COMMENT/POST)`: Đo chiều sâu tương tác.
   - `Sentiment Mean & Polarization (Std)`: Đo lập trường và mức độ tranh luận trong $W_{hist}=7$ ngày.
   - Sử dụng thống kê bền vững (**Median**) khi tổng hợp trong cửa sổ để kháng nhiễu trước các ngày bùng phát (spikes).
3. **Danh mục 16 Biểu đồ PDF đã xuất (`outputs/temporal_eda/`):**
   - `01_temporal_continuity_heatmap.pdf`
   - `02_weekly_hourly_seasonality.pdf`
   - `03_dow_hour_heatmap.pdf`
   - `04_dow_stability_by_year.pdf` *(Phiên bản mới: Heatmap + Line chart tab10)*
   - `05_amplification_ratio_trend.pdf`
   - `06_amplification_by_year_boxplot.pdf`
   - `07_parent_id_missing_over_time.pdf`
   - `08_user_activity_dynamics.pdf`
   - `09_new_user_ratio_trend.pdf` *(Phiên bản mới: Đã sửa mẫu số nunique thực tế)*
   - `10_reply_lag_distribution.pdf`
   - `11_reply_lag_by_year.pdf`
   - `12_sentiment_trajectory.pdf`
   - `13_sentiment_heatmap_year_month.pdf`
   - `14_sentiment_by_type_trajectory.pdf`
   - `15_spike_detection_zscore.pdf`
   - `16_spike_characteristics.pdf`

## 9. Đề xuất Thiết kế Pipeline & Gợi ý Thu hẹp Phạm vi (Scoping)

1. **Khung thời gian trọng tâm (Temporal Scoping):**
   - Đề xuất ưu tiên giai đoạn **2018–2020** (hoặc mở rộng **2017–2020**): Đây là giai đoạn có độ sâu thảo luận cao nhất (`Amplification` tăng từ 12 lên 17), tính phân cực đối kháng rõ nét nhất (`Sentiment Std` ~0.49–0.50), và hoàn toàn không bị lỗi kỹ thuật khuyết `POST` như đầu năm 2014.
2. **Bộ đặc trưng thời gian cốt lõi (Core Temporal Features):**
   - `Z-score Volume / Growth Momentum`: Đo lường tốc độ gia tăng thảo luận.
   - `Amplification Ratio (COMMENT/POST)`: Đo chiều sâu tương tác.
   - `Sentiment Mean & Polarization (Std)`: Đo lập trường và mức độ tranh luận trong $W_{hist}=7$ ngày.
   - Sử dụng thống kê bền vững (**Median**) khi tổng hợp trong cửa sổ để kháng nhiễu trước các ngày bùng phát (spikes).
3. **Danh mục 16 Biểu đồ PDF đã xuất (`outputs/temporal_eda/`):**
   - `01_temporal_continuity_heatmap.pdf`
   - `02_weekly_hourly_seasonality.pdf`
   - `03_dow_hour_heatmap.pdf`
   - `04_dow_stability_by_year.pdf` 
   - `05_amplification_ratio_trend.pdf`
   - `06_amplification_by_year_boxplot.pdf`
   - `07_parent_id_missing_over_time.pdf`
   - `08_user_activity_dynamics.pdf`
   - `09_new_user_ratio_trend.pdf` 
   - `10_reply_lag_distribution.pdf`
   - `11_reply_lag_by_year.pdf`
   - `12_sentiment_trajectory.pdf`
   - `13_sentiment_heatmap_year_month.pdf`
   - `14_sentiment_by_type_trajectory.pdf`
   - `15_spike_detection_zscore.pdf`
   - `16_spike_characteristics.pdf`

