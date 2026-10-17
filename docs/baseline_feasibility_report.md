# Báo Cáo Nghiên Cứu Khả Thi: Xây Dựng Baseline (Baseline Feasibility Report)

- **Đề tài:** NarrativeCast — Predicting Emerging Narratives in Social Media Using Temporal Semantic Representations
- **Nghiên cứu:** Kiểm tra khả năng xây dựng baseline (Baseline Feasibility Study)
- **Tài liệu quy chiếu:** [`docs/problem_specification.md`](./problem_specification.md) & [`docs/related_work.md`](./related_work.md)
- **Sản phẩm bàn giao (Deliverable):** [`notebooks/01_baseline_feasibility.ipynb`](../notebooks/01_baseline_feasibility.ipynb)
- **Tập dữ liệu mẫu:** [`data/sample/madoc_sample.parquet`](../data/sample/madoc_sample.parquet)

---

## 1. Mục Tiêu Nghiên Cứu & Nhiệm Vụ Đặt Ra

Theo đặc tả bài toán tại [docs/problem_specification.md](./problem_specification.md) và tổng quan nghiên cứu liên quan tại [docs/related_work.md](./related_work.md), nghiên cứu khả thi xây dựng baseline nhằm giải quyết các mục tiêu trọng tâm:
1. **Xác định các trường dữ liệu cần thiết:** Dựa trên cấu trúc 13 trường của bộ dữ liệu đa nền tảng MADOC (*Reddit, Bluesky, Voat, Koo*) và phạm vi trọng tâm là **bài đăng Reddit thuộc cộng đồng Technology (`r/technology`)**, phân tích các trường bắt buộc và mở rộng.
2. **Thiết kế cơ chế cửa sổ thời gian (Temporal Windowing) & Đo lường mức độ nổi bật (Prominence):**
   - Phân biệt rõ **Narrative** và **Topic** (ví dụ Topic *AI Governance* bao gồm nhiều narrative có lập trường đối lập).
   - Đo lường mức độ nổi bật chuẩn hóa $P(n, W_t) = \frac{N_{n,t}}{N_t}$ và mục tiêu tương lai $y_{n,t,h} = P(n, (t, t+h])$.
3. **Cài đặt & Thử nghiệm các phương án Baseline trên đa đường chân trời (Multi-horizon $h \in \{3, 7, 14\}$ ngày):**
   - *Popularity Baseline:* Xếp hạng theo khối lượng bài đăng $N_{n,t}$ trong cửa sổ hiện tại $W_t$.
   - *Persistence Baseline:* Giả định mức độ nổi bật hiện tại tiếp tục duy trì trong tương lai.
   - *Growth Momentum Baseline:* Xếp hạng theo gia tốc tăng trưởng tức thời.
   - Trực tiếp khảo sát câu hỏi nghiên cứu **RQ2**: Hiệu quả dự đoán thay đổi như thế nào giữa các khoảng thời gian 3, 7 và 14 ngày.
4. **Cung cấp bằng chứng sơ bộ về tính khả thi (Feasibility Evidence):** Chứng minh nhóm có thể tạo dữ liệu đầu vào ($X$) và đầu ra ($Y$) cho ít nhất một baseline dự đoán.
5. **Ghi rõ các phần chưa làm được (Limitations & Unresolved Challenges):** Đối chiếu với khung 3 khối chức năng trong `related_work.md` (BERTopic [1], Weng et al. [2], TFT [3]).
6. **Đóng góp ý tưởng vào bài báo nghiên cứu:** Soạn thảo đoạn văn (~0.1 cột) cho **Phần 5 — Experimental Setup** mô tả phương pháp baseline, đồng thời tuân thủ nguyên tắc không đưa kết quả thử sơ bộ vào bảng kết quả chính thức khi chưa kiểm chứng toàn diện trên tập dữ liệu hoàn chỉnh.

---

## 2. Xác Định Các Trường Dữ Liệu Cần Thiết (Data Schema Contract)

Dựa trên [docs/dataset_card.md](./dataset_card.md) và [docs/problem_specification.md](./problem_specification.md), bảng dưới đây phân loại 13 trường dữ liệu chuẩn hóa của MADOC theo mức độ cần thiết:

| Tên trường (Field) | Kiểu dữ liệu | Nhóm vai trò | Cần cho Baseline? | Đánh giá & Ràng buộc kỹ thuật |
| :--- | :--- | :--- | :---: | :--- |
| `publish_date` | Timestamp (UNIX) | **Trục thời gian** | **Bắt buộc** | Trục thời gian chuẩn hóa để cắt lát cửa sổ $W_{\text{hist}} = [t-k, t]$ và $W_{\text{future}} = (t, t+h]$, ngăn chặn hoàn toàn rò rỉ dữ liệu (data leakage). |
| `content` | String (Text) | **Nội dung văn bản** | **Bắt buộc** | Văn bản để trích xuất narrative, phân tích từ khóa và biểu diễn ngữ nghĩa (SBERT theo BERTopic [1]). |
| `post_id` | String | **Định danh duy nhất** | **Bắt buộc** | Khóa định danh dùng để đếm số lượng bài $N_{n,t}$ và tổng số bài $N_t$ trong cửa sổ. |
| `community` | String / NA | **Lọc phạm vi dữ liệu** | **Bắt buộc** | Lọc dữ liệu thuộc cộng đồng `technology` theo Mục 2 của `problem_specification.md`. |
| `platform` | Enum | **Lọc nền tảng** | **Bắt buộc** | Lọc dữ liệu `reddit` theo phạm vi nghiên cứu của đề tài. |
| `interaction_type` | Enum (`POST`, `COMMENT`) | **Trọng số tương tác** | Khuyến nghị | Phân biệt bài mở đầu luồng (`POST`) và thảo luận mở rộng (`COMMENT`). |
| `user_id` | String (UUID Hash) | **Mạng xã hội người dùng** | Khuyến nghị (Weng et al.) | Tính số người dùng duy nhất, tính các chỉ số *adoption dominance* và *adoption entropy* [2]. |
| `parent_id` | String / NA | **Cấu trúc cascade** | Khuyến nghị (Weng et al.) | Xây dựng đồ thị tương tác phản hồi (reply network), đo lường độ phân nhánh thảo luận. |
| `parent_user_id` | String / NA | **Tương tác trực tiếp** | Tùy chọn | Xác định mối quan hệ phản hồi giữa hai tác giả trong đồ thị xã hội. |
| `sentiment_vader` | Float (-1.0 đến 1.0) | **Đặc trưng cảm xúc** | Mở rộng | Bổ trợ nhận diện khung lập trường (stance/framing) của narrative. |
| `strict_filter` | Binary (0 / 1) | **Bộ lọc chất lượng** | Khuyến nghị | Cờ loại bỏ nội dung rác hoặc bài viết nằm ngoài tiêu chí nghiên cứu. |
| `url` | String | **Nguồn tham chiếu** | Tùy chọn | Phân tích nguồn báo chí công nghệ bên ngoài (Ars Technica, The Verge,...). |
| `language` | String (`en`) | **Kiểm soát ngôn ngữ** | Tiêu chuẩn | Đảm bảo tính nhất quán của dữ liệu đầu vào tiếng Anh. |

> **Kết luận:** Hợp đồng dữ liệu tối thiểu cho baseline bao gồm: `publish_date`, `content`, `post_id`, `community == 'technology'`, `platform == 'reddit'`.

---

## 3. Kiến Trúc Cửa Sổ Thời Gian & Định Nghĩa Toán Học

### 3.1. Phân biệt Narrative và Topic
Theo Mục 2 của `problem_specification.md`, **Narrative khác Topic**:
- Một **Topic** (ví dụ: *AI Governance*) là chủ đề thảo luận chung.
- Một **Narrative** là mạch diễn giải hoặc quan điểm cụ thể. Cùng một Topic có thể bao gồm các Narrative đối lập:
  - Narrative A: `AI_Regulation_Stifles_Innovation` (Quản lý bóp nghẹt đổi mới sáng tạo).
  - Narrative B: `AI_Safety_Demands_Urgent_Guardrails` (Cần quy chuẩn an toàn khẩn cấp).

### 3.2. Đo lường Mức độ nổi bật (Prominence Score) & Mục tiêu Dự đoán
- **Mức độ nổi bật tại cửa sổ $W_t$:**
  $$P(n, W_t) = \frac{N_{n,t}}{N_t}$$
- **Mục tiêu dự đoán tại thời điểm $t$ cho khoảng $h \in \{3, 7, 14\}$ ngày:**
  $$y_{n,t,h} = P(n, (t, t+h])$$
  *(Tại thời điểm $t$, chỉ được sử dụng dữ liệu $\le t$. Dữ liệu tương lai chỉ dùng để chấm điểm ground truth).*

### 3.3. Các Mô hình Baseline Đã Xây Dựng
1. **Popularity Baseline:** $\hat{R}_{n,t,h} = \text{Rank}(N_{n,t})$
2. **Persistence Baseline:** $\hat{R}_{n,t,h} = \text{Rank}(P(n, W_t))$
3. **Growth Momentum Baseline:** $\hat{R}_{n,t,h} = \text{Rank}\left(\frac{N_{n,t} - N_{n,t-1}}{N_{n,t-1} + 1}\right)$

---

## 4. Kết Quả Thực Nghiệm Sơ Bộ Đa Đường Chân Trời (Trả lời RQ2)

Thực nghiệm được thực hiện trên mẫu dữ liệu Reddit `r/technology` gồm **5,000 bài viết/bình luận** trải dài **60 ngày**, áp dụng cơ chế cửa sổ trượt (Rolling Forecast).

Bảng kết quả trung bình qua các mốc $h \in \{3, 7, 14\}$ ngày:

| Đường chân trời ($h$) | Mô hình Baseline | Spearman Correlation ($\rho$) | Top-1 Accuracy | Top-3 Hit Rate | Rank MAE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$h = 3$ ngày** | **Persistence Baseline** | **0.7922** | **0.7273 (72.7%)** | **0.8485 (84.9%)** | **0.6061** |
| | **Popularity Baseline** | **0.7922** | **0.7273 (72.7%)** | **0.8485 (84.9%)** | **0.6061** |
| | Growth Momentum Baseline | 0.4805 | 0.7273 (72.7%) | 0.6667 (66.7%) | 1.2727 |
| **$h = 7$ ngày** | **Persistence Baseline** | 0.6635 | 0.7273 (72.7%) | 0.7576 (75.8%) | 0.9545 |
| | **Popularity Baseline** | 0.6635 | 0.7273 (72.7%) | 0.7576 (75.8%) | 0.9545 |
| | Growth Momentum Baseline | 0.4364 | 0.7273 (72.7%) | 0.6970 (69.7%) | 1.3485 |
| **$h = 14$ ngày** | **Persistence Baseline** | 0.6156 | 0.6364 (63.6%) | 0.6970 (69.7%) | 1.1212 |
| | **Popularity Baseline** | 0.6156 | 0.6364 (63.6%) | 0.6970 (69.7%) | 1.1212 |
| | Growth Momentum Baseline | 0.4857 | 0.6364 (63.6%) | 0.7273 (72.7%) | 1.2727 |

### 💡 Phát hiện thực nghiệm quan trọng (Trả lời RQ2):
1. **Sự suy giảm quán tính theo thời gian:** Hệ số tương quan $\rho$ của *Persistence/Popularity* suy giảm rõ rệt từ **0.7922** ở $h=3$ ngày xuống **0.6635** ở $h=7$ ngày và chỉ còn **0.6156** ở $h=14$ ngày; sai số xếp hạng trung bình (Rank MAE) tăng gấp đôi từ 0.60 lên 1.12.
2. **Ý nghĩa khoa học:** Giả định mức độ phổ biến có quán tính chỉ đúng trong ngắn hạn ($h=3$). Ở các khoảng thời gian trung và dài hạn ($h=7, 14$), các phương pháp dựa trên tần suất đơn giản không còn duy trì được độ chính xác cao. Kết quả này củng cố trực tiếp luận điểm cốt lõi của đề tài: **Cần kết hợp biểu diễn ngữ nghĩa thời gian (BERTopic) và cấu trúc lan truyền (Weng et al.) để dự báo tốt hơn các narrative tương lai**.

---

## 5. Nghiệm Thu Tiêu Chí (Acceptance Criteria Check)

### ✅ Tiêu chí 1: Có bằng chứng sơ bộ rằng nhóm có thể tạo đầu vào và đầu ra cho ít nhất một baseline
- **Đầu vào ($X$):** Pipeline đã trích xuất thành công lịch sử bài đăng Reddit `r/technology`, cắt lát $W_{\text{hist}} = [t-k, t]$ và tính ma trận khối lượng $N_{n,t}$.
- **Đầu ra ($Y$):** Tính toán chính xác giá trị ground-truth $y_{n,t,h} = P(n, (t, t+h])$, xếp hạng thực tế $R_{n,t,h}$ và vector thứ hạng dự đoán $\hat{R}_{n,t,h}$.
- **Đánh giá:** Quy trình chạy tự động 100%, không bị data leakage, tạo đầy đủ đồ thị và bảng số liệu định lượng trong [`notebooks/01_baseline_feasibility.ipynb`](../notebooks/01_baseline_feasibility.ipynb).

### ⚠️ Tiêu chí 2: Ghi rõ phần chưa làm được (Limitations & Remaining Challenges)
Đối chiếu với 3 khối chức năng trong [docs/related_work.md](./related_work.md):
1. **Khối A (Phát hiện & Biểu diễn Narrative - BERTopic [1]):**
   - Chưa tích hợp đầy đủ pipeline BERTopic (SBERT `all-MiniLM-L6-v2` $\rightarrow$ UMAP $\rightarrow$ HDBSCAN $\rightarrow$ c-TF-IDF).
   - Chưa có module phân loại lập trường/khung diễn giải (stance/framing classification) để tách narrative khỏi topic thô.
   - Chưa xử lý hiện tượng **Semantic Drift** (sự dịch chuyển ngữ nghĩa của narrative theo thời gian).
2. **Khối B (Đặc trưng Cấu trúc Cộng đồng - Weng et al. [2]):**
   - Chưa xây dựng đồ thị tương tác phản hồi (reply network) từ `user_id` và `parent_id`.
   - Chưa tính các đặc trưng cấu trúc lan truyền cốt lõi: *usage dominance*, *adoption entropy*, *số cộng đồng bị lây nhiễm*.
3. **Khối C (Dự báo Đa đường chân trời - TFT [3]):**
   - Chưa huấn luyện mô hình Temporal Fusion Transformer (TFT) với đầu ra quantile (P10, P50, P90).
   - Chưa có cơ chế diễn giải tự động qua trọng số Variable Selection và Self-Attention.
4. **Quy mô xử lý dữ liệu lớn:**
   - Mới kiểm thử trên mẫu 5,000 bài viết Reddit `r/technology`. Chưa xây dựng pipeline xử lý phân tán (PySpark / Polars Streaming) cho toàn bộ 18.9 triệu bài viết và 236 triệu bình luận của bộ dữ liệu MADOC gốc.

---

## 6. Đóng Góp Bài Báo (Section 5 — Experimental Setup)

Đoạn văn dưới đây được chuẩn bị với dung lượng khoảng **0.1 cột chuẩn bài báo khoa học IEEE/ACM (~160 từ)** để đưa vào **Phần 5 — Experimental Setup**:

### Bản tiếng Anh (Chuẩn Academic Paper Draft — Section 5.X):
> **5.X Baseline Methods**  
> To establish an empirical lower bound for evaluating temporal semantic representations, we formulate three statistical and heuristic baselines operating over sliding temporal observation windows $W_{\text{hist}} = [t-k, t]$ across forecast horizons $h \in \{3, 7, 14\}$ days:  
> *(1) Popularity Baseline:* Ranks candidate narratives in descending order of observed post volume $N_{n,t}$ within the immediate observation window $W_t$.  
> *(2) Persistence Baseline:* Assumes discussion prominence exhibits short-term temporal inertia, predicting that future prominence $y_{n,t,h} = P(n, (t, t+h])$ maintains the current observed relative share $P(n, W_t) = N_{n,t} / N_t$.  
> *(3) Growth Momentum Baseline:* Ranks narratives by their immediate acceleration velocity $G_{n,t} = (N_{n,t} - N_{n,t-1}) / (N_{n,t-1} + 1)$, prioritizing fast-surging narratives.  
> Predictions are evaluated against the ground-truth prominence ranking using Spearman's rank correlation ($\rho$), Top-$K$ prediction hit rate, and rank Mean Absolute Error (MAE). These baselines provide the foundational benchmark to quantify the predictive gain contributed by temporal semantic embeddings and community structure.

*Ghi chú học thuật:* Các kết quả thử nghiệm sơ bộ trong notebook này đóng vai trò xác nhận tính khả thi phương pháp (feasibility check) và sẽ **không** được đưa vào bảng kết quả chung cuộc của bài báo cho đến khi hoàn thành huấn luyện và kiểm thử trên toàn bộ tập dữ liệu MADOC.
