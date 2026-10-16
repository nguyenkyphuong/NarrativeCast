# Related Work cho NarrativeCast

> **NarrativeCast**: dự đoán những *narrative* trên mạng xã hội có khả năng trở nên nổi bật trong tương lai, dựa trên **nội dung** và **diễn biến hoạt động trong quá khứ**.

Tài liệu này tổng hợp ba bài báo nền tảng, mỗi bài ứng với một khối chức năng của NarrativeCast:

| Khối chức năng | Bài báo | Vai trò |
|---|---|---|
| (1) Phát hiện narrative từ nội dung | **BERTopic** [1] | Gom nhóm văn bản thành các chủ đề và theo dõi chúng theo thời gian |
| (2) Đặc trưng cấu trúc cộng đồng, dự đoán độ viral sớm | **Weng et al.** [2] | Cho thấy cấu trúc cộng đồng dự báo được độ lan rộng |
| (3) Dự báo chuỗi thời gian đa đường chân trời, có diễn giải | **Temporal Fusion Transformer** [3] | Mô hình dự báo kết hợp biến tĩnh, biến biết trước và biến quan sát được |

---

## 1. BERTopic: Neural topic modeling with a class-based TF-IDF procedure [1]

### Nhiệm vụ
Khám phá các chủ đề tiềm ẩn trong tập văn bản. Bài báo tiếp cận topic modeling như một bài toán **clustering trên embedding** thay vì mô hình túi từ như LDA/NMF, vốn bỏ qua quan hệ ngữ nghĩa giữa các từ.

### Cách làm
Pipeline gồm ba bước độc lập (mục 3 của bài báo):
1. **Document embedding**: dùng Sentence-BERT (thí nghiệm chính dùng `all-mpnet-base-v2`). Bài báo cho biết có thể thay bằng embedding khác miễn là mô hình đã được fine-tune cho độ tương đồng ngữ nghĩa.
2. **Giảm chiều và phân cụm**: UMAP giảm chiều (để giảm "curse of dimensionality"), sau đó HDBSCAN phân cụm. HDBSCAN cho phép gán văn bản vào nhóm **nhiễu/outlier** để không ép văn bản không liên quan vào chủ đề nào.
3. **Biểu diễn chủ đề bằng c-TF-IDF**: gộp mọi văn bản trong một cụm thành một "tài liệu", rồi tính độ quan trọng của từ theo cụm: `W(t,c) = tf(t,c) · log(1 + A / tf(t))`, với `A` là số từ trung bình mỗi lớp. Có thể gộp dần chủ đề nhỏ nhất vào chủ đề giống nhất để giảm về số chủ đề mong muốn.

**Dynamic topic modeling** (mục 4): fit BERTopic một lần trên toàn bộ corpus để có chủ đề toàn cục, sau đó tính biểu diễn cục bộ tại mỗi mốc thời gian bằng cách nhân tần suất từ của mốc đó với IDF toàn cục (Eq. 3). Cách này không cần embed và cluster lại. Có thêm tùy chọn **smoothing**: chuẩn hóa L1 rồi lấy trung bình c-TF-IDF của bước `t` và `t-1`.

### Kết quả chính
- Đánh giá bằng NPMI coherence (TC) và topic diversity (TD) trên 20 NewsGroups, BBC News và tweet của Trump (44.253 tweet, 2009–2021), chạy từ 10 đến 50 chủ đề.
- Trên dữ liệu tweet (tiền xử lý nhẹ), BERTopic-MPNET đạt TC = 0,066, cao nhất trong bảng, so với LDA −0,011 và CTM 0,009. Trên hai tập tin tức được tiền xử lý kỹ, BERTopic cạnh tranh nhưng không luôn dẫn đầu. TD luôn thấp hơn CTM.
- Hiệu năng ổn định giữa các mô hình embedding SBERT. Bài báo gợi ý `all-MiniLM-L6-v2` là điểm cân bằng tốt giữa tốc độ và chất lượng.
- Dynamic topic modeling: trên tập Trump, TC = 0,079 so với LDA Sequence 0,009. Giả định "topic tiến hóa tuyến tính" (smoothing) không ảnh hưởng đáng kể tới điểm số.

### Ưu điểm
- Tách rời embedding, clustering và biểu diễn chủ đề, nên có thể thay từng khối mà không cần chạy lại toàn bộ (ví dụ đổi n-gram mà không cần cluster lại).
- Hoạt động tốt trên văn bản ngắn như tweet.
- Có sẵn cơ chế dynamic topic, và c-TF-IDF cục bộ tính nhanh vì không phải embed lại.
- Chất lượng tăng theo các mô hình ngôn ngữ mới.

### Hạn chế (do tác giả nêu)
- Mỗi văn bản chỉ thuộc **một** chủ đề (có thể dùng ma trận xác suất của HDBSCAN để xấp xỉ phần nào).
- Biểu diễn chủ đề vẫn là túi từ, nên các từ trong một chủ đề có thể dư thừa. Tác giả đề xuất Maximal Marginal Relevance nhưng chưa thử.
- NPMI có thể không tương quan với đánh giá của con người đối với mô hình neural [Hoyle et al., 2021, được trích trong [1]].
- Thời gian chạy đo trên máy có GPU. Không có GPU thì sẽ chậm hơn đáng kể.

### Khả năng áp dụng vào NarrativeCast
**Cao, là khối lõi của phần phát hiện narrative.** Tuy nhiên cần lưu ý một khoảng cách quan trọng: BERTopic tìm **chủ đề** (cùng nói về cái gì), còn *narrative* thường bao hàm cả **khung diễn giải, lập trường và tác nhân**. Hai bài đăng cùng chủ đề có thể thuộc hai narrative đối lập. Phần 5 bên dưới đề xuất cách xử lý.

---

## 2. Virality Prediction and Community Structure in Social Networks [2]

### Nhiệm vụ
Dự đoán một **meme** (ở đây là hashtag Twitter) có trở nên viral hay không, chỉ dựa trên **mẫu lan truyền giai đoạn đầu** nhìn qua cấu trúc cộng đồng của mạng xã hội. Độ viral được đo bằng độ phổ biến (số tweet hoặc số người dùng).

### Cách làm
- **Dữ liệu**: mẫu 10% tweet công khai (24/03–25/04/2012), 121.807.378 tweet tiếng Anh, 14,6 triệu người dùng, hơn 10,39 triệu hashtag. Mạng vô hướng, không trọng số, xây từ quan hệ follow hai chiều của 595.460 người dùng. Phát hiện cộng đồng bằng **Infomap** (kiểm tra lại bằng link clustering).
- **Meme mới**: hashtag có dưới 20 tweet trong tháng trước.
- **Đo độ tập trung vào cộng đồng**: *usage dominance* và *adoption dominance* (tỷ lệ tweet/người dùng thuộc cộng đồng trội nhất), *usage entropy* và *adoption entropy* (phân bố qua các cộng đồng), và *số lần tiếp xúc trung bình* trước khi người dùng chấp nhận meme (đo sức mạnh của social reinforcement).
- **Bốn mô hình nền (M1–M4)**: lấy mẫu ngẫu nhiên, cascade đơn giản (simple contagion), reinforcement xã hội, homophily. So sánh dữ liệu thật với các mô hình này để biết meme lan như "simple" hay "complex contagion".
- **Mô hình dự đoán**: Random Forest (500 cây, mỗi cây dùng 4 đặc trưng ngẫu nhiên, 10-fold cross validation), chỉ dùng **50 tweet đầu tiên** của mỗi hashtag. Nhãn viral: meme thuộc top theo ngưỡng phân vị θ ∈ {70, 80, 90}. Các đặc trưng gồm:
  1. số người dùng sớm và số hàng xóm chưa bị nhiễm của họ,
  2. số cộng đồng đã bị "nhiễm",
  3. usage entropy và adoption entropy,
  4. tỷ lệ tương tác giữa người dùng trong cùng cộng đồng.

### Kết quả chính
- Phần lớn meme **không viral** lan như complex contagion, tức là bị "mắc kẹt" trong cộng đồng. Số ít meme **viral** lan như simple contagion, xuyên qua nhiều cộng đồng.
- Khi phát hiện meme viral nhất theo người dùng (θ_U = 90), mô hình dựa trên cộng đồng đạt precision 0,62 và recall 0,42, so với 0,17 và 0,13 của mô hình không dùng cộng đồng, và 0,09 và 0,09 của đoán ngẫu nhiên (Fig. 5). Tác giả tóm tắt: chính xác gấp khoảng 7 lần ngẫu nhiên và hơn 3 lần mô hình "community-blind".
- Kết quả bền vững khi đổi phương pháp phát hiện cộng đồng hoặc loại mạng (retweet, mention), theo phần Supplementary Information của bài báo.

### Ưu điểm
- Không cần nội dung văn bản, chỉ cần một mẫu nhỏ dữ liệu sớm.
- Đặc trưng có ý nghĩa giải thích rõ (entropy, dominance, số cộng đồng bị nhiễm).
- Có thiết kế baseline rõ ràng (random, community-blind) để chứng minh giá trị của đặc trưng cộng đồng.

### Hạn chế
- Đơn vị phân tích là **hashtag**, không phải narrative. Một narrative có thể trải qua nhiều hashtag và nhiều cách diễn đạt.
- Bài toán là **phân loại nhị phân** dựa trên 50 tweet đầu, không phải dự báo diễn biến theo thời gian.
- Mạng xã hội là **tĩnh** (follow hai chiều), dữ liệu Twitter năm 2012, chỉ tiếng Anh.
- Ngưỡng viral là phân vị tương đối trong tập dữ liệu, không phải ngưỡng tuyệt đối.
- Không dùng nội dung, nên bỏ lỡ tín hiệu ngữ nghĩa.
- Cần có đồ thị xã hội và thông tin cộng đồng. Với nền tảng không công khai đồ thị follow thì phải dùng đồ thị tương tác (retweet, reply) thay thế, và bài báo chỉ kiểm chứng điều này ở mức bổ sung.

### Khả năng áp dụng vào NarrativeCast
**Cao, là nguồn đặc trưng "hoạt động" cho NarrativeCast.** Các đặc trưng entropy, dominance, số cộng đồng bị nhiễm và tỷ lệ tương tác nội cộng đồng có thể tính cho mỗi narrative trong mỗi cửa sổ thời gian, rồi đưa vào mô hình dự báo. Giả thuyết cốt lõi (narrative lan rộng qua nhiều cộng đồng thì dễ nổi bật) phù hợp trực tiếp với mục tiêu của NarrativeCast.

---

## 3. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting [3]

### Nhiệm vụ
**Dự báo đa đường chân trời** (multi-horizon): dự đoán giá trị tại nhiều bước tương lai cùng lúc, trong điều kiện có ba loại đầu vào không đồng nhất: biến **tĩnh** (metadata), biến **biết trước trong tương lai** (ví dụ ngày lễ) và biến **chỉ quan sát được trong quá khứ**. Đồng thời mô hình phải **diễn giải được**.

### Cách làm
Kiến trúc TFT (Fig. 2) gồm các thành phần:
- **Gated Residual Network (GRN)** và **Gated Linear Unit (GLU)**: cho phép mô hình bỏ qua các tầng không cần thiết (độ sâu thích ứng).
- **Variable selection network** theo từng mẫu: chọn biến quan trọng tại mỗi bước thời gian, riêng cho biến tĩnh, biến quá khứ và biến tương lai.
- **Static covariate encoders**: tạo bốn vector ngữ cảnh (`c_s, c_e, c_c, c_h`) điều kiện hóa chọn biến, xử lý cục bộ và làm giàu đặc trưng thời gian.
- **LSTM encoder-decoder** xử lý cục bộ (thay thế positional encoding), sau đó **interpretable multi-head attention** học phụ thuộc dài hạn. Các head chia sẻ ma trận value, nên trọng số attention có thể được đọc như một ensemble.
- **Đầu ra quantile** (10%, 50%, 90%), huấn luyện bằng quantile loss. Đánh giá bằng **q-Risk** (P50, P90).

### Kết quả chính
- Thực nghiệm trên Electricity, Traffic, Retail (Favorita) và Volatility. TFT thấp hơn trung bình **7% ở P50** và **9% ở P90** so với mô hình tốt nhất tiếp theo. Trên Retail, P50 là 0,354 so với 0,379 của MQRNN. Trên Volatility, P50 là 0,039 so với 0,042.
- Ablation: các thành phần xử lý thời gian (local processing và self-attention) có ảnh hưởng lớn nhất. Gating đặc biệt có lợi với tập nhỏ và nhiễu (Volatility).
- **Ba ca sử dụng diễn giải**: (i) độ quan trọng biến toàn cục từ trọng số variable selection (ví dụ trên Retail, `log sales` quá khứ, khuyến mãi và ngày lễ quốc gia nổi bật); (ii) mẫu thời gian bền vững từ attention (chu kỳ ngày, tuần); (iii) nhận diện **regime** và sự kiện lớn qua khoảng cách Bhattacharyya `dist(t)` giữa mẫu attention tại `t` với mẫu trung bình. Trên S&P 500, `dist(t)` tăng quanh khủng hoảng 2008.

### Ưu điểm
- Xử lý tự nhiên cả biến tĩnh, biến biết trước và biến quá khứ, rất giống cấu trúc dữ liệu của bài toán narrative.
- Cho **khoảng dự báo** (quantile), hữu ích cho quyết định rủi ro.
- Diễn giải ở mức toàn bộ dữ liệu, không chỉ từng mẫu.
- Có mã nguồn mở và hướng dẫn tái lập.

### Hạn chế
- Chi phí huấn luyện không nhỏ: mô hình tối ưu cho Electricity mất hơn 6 giờ trên một GPU V100.
- Cần một chuỗi thời gian số được căn chỉnh theo bước đều. Các dataset trong bài đều là chuỗi số đã sạch (điện, giao thông, bán lẻ, biến động giá).
- Diễn giải dựa trên attention và trọng số chọn biến, mang tính **mô tả hành vi mô hình**, chưa phải quan hệ nhân quả. (Nhận xét của người tổng hợp, không phải kết luận của bài báo.)
- Không có thành phần xử lý văn bản. Nội dung phải được mã hóa thành đặc trưng số trước khi vào mô hình.
- Cần chọn lại kiến trúc cho từng tập dữ liệu. Bài báo cho thấy local processing có hại trên Electricity.

### Khả năng áp dụng vào NarrativeCast
**Cao, là ứng viên cho mô hình dự báo chính.** Mỗi narrative là một "entity", chuỗi hoạt động của nó (volume, số người dùng, đặc trưng cộng đồng) là biến quan sát, các thuộc tính narrative là biến tĩnh, và lịch (ngày trong tuần, sự kiện đã biết) là biến biết trước.

---

## 4. So sánh và bổ trợ

| | BERTopic [1] | Weng et al. [2] | TFT [3] |
|---|---|---|---|
| Đầu vào chính | Văn bản | Đồ thị xã hội + dòng tweet sớm | Chuỗi thời gian nhiều loại biến |
| Đầu ra | Chủ đề, từ khóa, chủ đề theo thời gian | Nhãn viral/không viral | Dự báo quantile nhiều bước |
| Dùng nội dung | Có | Không | Không (cần đặc trưng số) |
| Dùng cấu trúc cộng đồng | Không | Có (cốt lõi) | Không trực tiếp |
| Dùng diễn biến theo thời gian | Một phần (dynamic topic) | Chỉ 50 tweet đầu | Có (cốt lõi) |
| Diễn giải | Từ khóa c-TF-IDF | Đặc trưng RF, entropy | Variable selection, attention |
| Đơn vị phân tích | Cụm văn bản | Hashtag | Entity (chuỗi) |

Ba bài **bổ trợ nhau**: [1] xác định *cái gì* đang được nói, [2] cho biết *ai và cộng đồng nào* đang lan truyền, [3] dự báo *sẽ phát triển thế nào theo thời gian*. Không bài nào đủ cho NarrativeCast. Khoảng trống chung là **không bài nào định nghĩa hay theo dõi "narrative" như một thực thể ổn định theo thời gian**.

---

## 5. Đề xuất hướng kỹ thuật ban đầu cho NarrativeCast

> Phần này là **đề xuất của người tổng hợp**, dựa trên ba bài báo ở trên. Cần kiểm chứng bằng thực nghiệm. Các lựa chọn cụ thể (ví dụ mô hình embedding đa ngôn ngữ) chưa được ba bài báo kiểm chứng.

### 5.1. Tổng quan pipeline

```
Bài đăng MXH (nội dung + metadata + tương tác)
        │
        ▼
[A] Phát hiện & theo dõi narrative      (BERTopic [1])
        │   gán mỗi bài đăng → narrative_id
        ▼
[B] Xây chuỗi thời gian theo narrative  (cửa sổ giờ/ngày)
        │   volume, người dùng, tăng trưởng + đặc trưng cộng đồng [2]
        │   + đặc trưng nội dung (embedding, c-TF-IDF)
        ▼
[C] Dự báo độ nổi bật                   (baseline [2] → TFT [3])
        │   quantile của hoạt động ở H bước tới
        ▼
[D] Xác suất "nổi bật" + diễn giải
```

### 5.2. Khối A: Phát hiện và theo dõi narrative
- Dùng BERTopic: SBERT → UMAP → HDBSCAN → c-TF-IDF. Với văn bản tiếng Việt hoặc đa ngôn ngữ, cần chọn mô hình embedding đa ngôn ngữ (đề xuất thử nghiệm, chưa được bài báo kiểm chứng).
- **Tránh rò rỉ thông tin tương lai**: bài báo fit BERTopic trên toàn bộ corpus rồi mới phân tích theo thời gian. NarrativeCast cần **fit chỉ trên giai đoạn huấn luyện**, sau đó gán narrative cho dữ liệu mới bằng bước transform, và cập nhật định kỳ. Nếu không, mô hình sẽ vô tình "biết" narrative nào sẽ nổi bật.
- **Narrative khác topic**: hai hướng mở rộng để thử, ban đầu có thể chỉ thử một:
  - gắn nhãn khung/lập trường bằng một bước phân loại hoặc LLM trên từng cụm, hoặc tách cụm con theo lập trường;
  - dùng biểu diễn c-TF-IDF có n-gram dài hơn và MMR để giảm dư thừa (hai điểm bài báo [1] nêu là hạn chế).
- **Xử lý nhóm outlier (topic −1)** của HDBSCAN: đo tỷ lệ bài đăng rơi vào nhóm này, vì trên mạng xã hội nó có thể rất lớn. Có thể gán lại bằng xác suất mềm.
- **Nối narrative qua các cửa sổ thời gian**: dùng độ tương đồng giữa centroid embedding hoặc vector c-TF-IDF để khớp narrative giữa các lần cập nhật, và gộp/tách theo ngưỡng.

### 5.3. Khối B: Xây đặc trưng theo narrative và theo thời gian
- **Đặc trưng hoạt động** (quan sát, quá khứ): số bài đăng, số người dùng duy nhất, tốc độ tăng trưởng, tỷ lệ retweet/reply, tỷ lệ tài khoản mới.
- **Đặc trưng cộng đồng** (theo [2], tính cho mỗi narrative và mỗi cửa sổ): usage/adoption dominance, usage/adoption entropy, số cộng đồng bị nhiễm, tỷ lệ tương tác nội cộng đồng. Đồ thị cộng đồng xây từ đồ thị tương tác (retweet, reply, mention), phát hiện cộng đồng bằng Infomap hoặc Louvain. Nên **cố định phân vùng cộng đồng** từ giai đoạn huấn luyện và chỉ cập nhật định kỳ.
- **Đặc trưng nội dung**: embedding trung bình của narrative (có thể giảm chiều), độ phân tán nội bộ cụm, độ mới (khoảng cách tới narrative cũ gần nhất), cảm xúc.
- **Biến tĩnh**: nền tảng, ngôn ngữ, chủ đề lớn, thời điểm xuất hiện, hồ sơ cộng đồng xuất hiện đầu tiên.
- **Biến biết trước**: ngày trong tuần, giờ, ngày lễ và sự kiện đã lên lịch.

### 5.4. Khối C: Định nghĩa mục tiêu và mô hình
**Định nghĩa "nổi bật"** (cần chốt sớm vì quyết định toàn bộ bài toán): lấy từ [2], một narrative nổi bật nếu tổng hoạt động trong H ngày tới thuộc top-θ% (θ ∈ {70, 80, 90}) các narrative cùng giai đoạn. Cách này tương đối so với tập dữ liệu, nên cần ghi rõ khi báo cáo.

**Lộ trình mô hình theo từng bước:**
1. **Baseline thống kê**: persistence/naive, ARIMA/ETS trên volume.
2. **Baseline dạng bảng theo [2]**: Random Forest (hoặc Gradient Boosting) dùng đặc trưng từ N bài đăng đầu hoặc từ cửa sổ gần nhất, so sánh *có/không có* đặc trưng cộng đồng, và có/không có đặc trưng nội dung. Đây là thí nghiệm ablation tái hiện tinh thần [2].
3. **TFT** [3]: dự báo quantile (P10/P50/P90) volume ở H bước tương lai, với biến tĩnh, biến quá khứ và biến biết trước như mục 5.3. Từ phân phối quantile có thể suy ra xác suất vượt ngưỡng "nổi bật", hoặc huấn luyện thêm đầu phân loại.
4. So sánh với các baseline deep learning (Seq2Seq, DeepAR) nếu đủ thời gian, vì [3] dùng chúng làm chuẩn so sánh.

**Vấn đề khởi đầu lạnh (cold start)**: narrative mới chưa có đủ lịch sử, trong khi TFT cần cửa sổ nhìn lại dài. Giải pháp: dùng mô hình dạng bảng theo [2] cho giai đoạn rất sớm (N bài đầu), chuyển sang TFT khi đủ lịch sử, hoặc huấn luyện TFT với cửa sổ nhìn lại ngắn và đệm giá trị bằng 0 kèm cờ "có dữ liệu".

### 5.5. Khối D: Đánh giá và diễn giải
- **Chia dữ liệu theo thời gian** (huấn luyện < kiểm định < kiểm tra), không xáo trộn ngẫu nhiên.
- **Chỉ số**: precision, recall, PR-AUC cho nhãn nổi bật; q-Risk P50/P90 cho dự báo; thêm *lead time* (dự đoán sớm được bao lâu trước khi narrative nổi bật). Với dữ liệu mất cân bằng, nên báo cáo thêm PR-AUC, không chỉ accuracy.
- **Baseline bắt buộc**: đoán ngẫu nhiên, mô hình chỉ dùng độ phổ biến sớm, mô hình "community-blind" (theo [2]).
- **Diễn giải**: trọng số variable selection của TFT cho biết nhóm đặc trưng nào (nội dung hay cộng đồng hay hoạt động) quan trọng. Attention cho thấy chu kỳ và độ trễ. `dist(t)` kiểu [3] có thể thử để phát hiện thời điểm hành vi lan truyền chuyển chế độ.
- **Đánh giá chất lượng narrative**: NPMI và topic diversity như [1], kèm đánh giá thủ công một mẫu nhỏ vì bài [1] cũng lưu ý các chỉ số tự động chỉ là ước lượng.

### 5.6. Rủi ro và câu hỏi mở
| Rủi ro | Hướng xử lý ban đầu |
|---|---|
| Rò rỉ tương lai qua clustering/community | Fit trên train, transform trên test |
| Narrative trôi nghĩa, tách/gộp theo thời gian | Khớp narrative bằng centroid, đánh giá độ ổn định |
| Nhãn "nổi bật" phụ thuộc ngưỡng | Thử nhiều ngưỡng, báo cáo độ nhạy |
| Thiếu đồ thị xã hội | Dùng đồ thị tương tác thay cho follow |
| Chi phí tính toán (embedding + TFT) | Dùng MiniLM, giới hạn số narrative theo hoạt động tối thiểu |
| Dữ liệu một nền tảng, một thời kỳ | Ghi rõ phạm vi, kiểm tra trên giai đoạn khác |

**Câu hỏi cần chốt cùng nhóm**: (1) định nghĩa chính xác "narrative" và "nổi bật"; (2) nền tảng và ngôn ngữ dữ liệu; (3) khoảng dự báo H và độ phân giải thời gian; (4) có đồ thị follow hay chỉ có đồ thị tương tác.

---

## Tài liệu tham khảo

**Ba bài báo chính**

[1] M. Grootendorst. *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794 [cs.CL], 2022. Mã nguồn: https://github.com/MaartenGr/BERTopic

[2] L. Weng, F. Menczer, Y.-Y. Ahn. *Virality Prediction and Community Structure in Social Networks.* Scientific Reports 3, 2522 (2013). DOI: 10.1038/srep02522.

[3] B. Lim, S. Ö. Arık, N. Loeff, T. Pfister. *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* arXiv:1912.09363v3 [stat.ML], 2020 (International Journal of Forecasting, 2021). Mã nguồn: https://github.com/google-research/google-research/tree/master/tft

**Tài liệu được trích dẫn trong ba bài trên và liên quan trực tiếp đến đề xuất** (chưa đọc trực tiếp, chỉ nêu theo cách các bài báo trích dẫn):

- N. Reimers, I. Gurevych. *Sentence-BERT: Sentence embeddings using siamese BERT-networks.* EMNLP 2019. (embedding trong [1])
- L. McInnes, J. Healy, J. Melville. *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.* 2018; L. McInnes, J. Healy, S. Astels. *hdbscan: Hierarchical density based clustering.* JOSS 2(11):205, 2017. (giảm chiều và phân cụm trong [1])
- A. Hoyle et al. *Is automated topic model evaluation broken? The incoherence of coherence.* NeurIPS 2021. (hạn chế của NPMI, được nêu trong [1])
- S. Terragni et al. *OCTIS: Comparing and optimizing topic models is simple!* EACL 2021. (khung thực nghiệm của [1])
- M. Rosvall, C. T. Bergstrom. *Maps of random walks on complex networks reveal community structure.* PNAS 105, 1118–1123 (2008). (Infomap, dùng trong [2])
- D. Centola. *The spread of behavior in an online social network experiment.* Science 329, 1194–1197 (2010). (complex contagion, nền tảng lý thuyết của [2])
- L. Breiman. *Random forests.* Machine Learning 45, 5–32 (2001). (mô hình dự đoán trong [2])
- D. Salinas et al. *DeepAR: Probabilistic forecasting with autoregressive recurrent networks.* Int. J. Forecasting (2019); R. Wen et al. *A multi-horizon quantile recurrent forecaster.* NIPS 2017 Time Series Workshop; S. Li et al. *Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting.* NeurIPS 2019. (các baseline dự báo trong [3])
