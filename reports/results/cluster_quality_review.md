# Báo cáo Đánh giá Chất lượng Phân cụm Narrative (Version 1.0)

## 1. Tổng quan Tập Mẫu Phân tầng (Sampling Overview)

Báo cáo này đánh giá chất lượng của kết quả phân cụm Narrative (SBERT + UMAP 5D + HDBSCAN + Soft Assignment) trên tập dữ liệu ngách `bigtech_antitrust` năm 2017. Tập mẫu đánh giá gồm 53 bài viết/bình luận được trích xuất phân tầng từ 7 cụm đại diện:

- **Cụm Lớn (Large):** Cụm 635 (2.137 bài), Cụm 443 (2.050 bài)
- **Cụm Vừa (Medium):** Cụm 572 (76 bài), Cụm 53 (75 bài)
- **Cụm Nhỏ (Small):** Cụm 640 (30 bài), Cụm 735 (30 bài)
- **Cụm Nhiễu (Noise):** Cụm -1 (31.648 bài)

Trong mỗi cụm hợp lệ, các mẫu được trích xuất đại diện cho 3 vai trò: _Core_Prototype_ (bài viết gần tâm cụm nhất), _Cluster_Boundary_ (bài viết xa tâm cụm nhất) và _Contrastive_Stance_ (bài viết có độ lệch cảm xúc cao hoặc chứa liên từ tương phản).

---

## 2. Đánh giá Độ mạch lạc: Topic vs. Narrative (Topical Coherence vs. Stance Homogeneity)

Kết quả phân tích 53 mẫu thực tế củng cố nhận định cốt lõi của dự án: **không phải cụm nào cũng hình thành một Narrative hoàn chỉnh**.

### 2.1 Các cụm đạt mức Coherent Narrative

Các cụm nhỏ thường thể hiện độ mạch lạc nội cụm (Intra-Cosine) rất cao và có cùng một góc nhìn hoặc lập trường rõ rệt.

- **Cụm 640 ("Community surprise upon learning Google is a subsidiary of Alphabet"):** Độ tương đồng tâm rất cao (trung bình 0.824). Các mẫu đều là những câu ngắn thể hiện cùng một sự kiện (_"TIL Google has a parent company..."_). Đây là một Narrative sự kiện rất hẹp và đồng nhất.
- **Cụm 735 ("Advocating for Startpage as a privacy-focused alternative to Google Search"):** Không chỉ chia sẻ chủ đề về công cụ tìm kiếm, các bài viết ở cụm này đều mang chung một lập trường khuyên dùng Startpage để bảo vệ quyền riêng tư (_"And if you want Google search results without the above mentioned Google features, there's always Startpage.com"_).

### 2.2 Các cụm dừng ở mức Broad Topic

Ngược lại, một số cụm chỉ mang tính chất phân loại chủ đề (Topic) mà chưa định hình được một câu chuyện (Narrative) cụ thể.

- **Cụm 572 ("SSL/TLS certificate validation mechanisms, trust in CAs, and web security warnings"):** Mặc dù các bài viết Prototype chia sẻ chủ đề về SSL và Certificate Authority (CA) với độ tương đồng rất cao (0.92 - 0.99), nhưng nội dung của chúng mang tính chất giải thích kỹ thuật (Boilerplate/Explanatory) hơn là một mạch tranh luận có lập trường đối đầu.

---

## 3. Phân tích Cụm Trộn lẫn (Mixed-Narrative Clusters)

Đây là thách thức lớn nhất của thuật toán hiện tại, đặc biệt xảy ra ở các cụm lớn (Tier Large). Các cụm này gộp chung nhiều bài viết cùng thảo luận về một sự kiện nhưng lại mang các lập trường (Stance) hoàn toàn trái ngược nhau:

- **Cụm 443 ("iPhone battery degradation, performance throttling, and right to repair"):**
  - Mặc dù tập trung vào sự kiện Batterygate của Apple, cụm này chứa các bài viết bảo vệ hãng (_"Exactly. They are improving user experience and conducting standard battery management practices..."_ - Sentiment: -0.56) nằm chung với các bài viết thuyết âm mưu chỉ trích Apple (_"I believe it's a little more sinister than that. They're changing the problem..."_ - Sentiment: -0.59).
  - Sự xuất hiện của các bài viết có gắn cờ `has_contrastive = True` cho thấy cần một bước **Sub-clustering** (Phân cụm con) dựa trên điểm phân cực cảm xúc (Sentiment Std) để tách các bài viết ủng hộ và phản đối thành các Narrative riêng biệt.
- **Cụm 635 ("Debate over 'Fake News' definition, CNN's credibility, and media polarization"):** Tương tự, cụm này bao trùm chủ đề "Fake News" nhưng các góc nhìn bên trong rất đa dạng và xung đột, phản ánh một "Topic" lớn hơn là một "Narrative" thống nhất.

---

## 4. Phân tích Cụm Nhiễu (Noise) & Cụm Nhỏ lẻ

- **Cụm -1 (Noise_or_Boilerplate):** Các mẫu được lấy từ cụm nhiễu (_"Maybe I'm mistaken, but I believe Apple actually filed for a patent..."_ hoặc _"I have several family members who fall into this..."_) cho thấy chúng không hoàn toàn là văn bản rác hay vô nghĩa. Đây là các bình luận hợp lệ, có quan điểm, nhưng phong cách diễn đạt tự do hoặc không đủ số lượng bài viết tương đồng lân cận (`min_cluster_size = 15`) để hình thành một mật độ dày trên không gian UMAP 5D.
- Cơ chế Gán mềm (Soft Assignment với ngưỡng $\\tau = 0.65$) trong Notebook 04 đã cứu lại hơn 39.000 bài viết khỏi nhóm nhiễu này một cách hiệu quả, nhưng vẫn giữ lại được ranh giới an toàn cho 22.7% bài viết quá rời rạc.

---

## 5. Giới hạn & Định hướng Version 2.0

Quá trình đánh giá 53 mẫu đã chỉ ra những điểm yếu của thuật toán hiện tại, trong đó bao gồm những vấn đề có thể khắc phục ở Version 2.0 và những rào cản bất khả kháng của hệ thống:

### 5.1 Các nâng cấp khả thi cho Version 2.0

- **Định danh Narrative:** Các `human_interpreted_label` hiện tại được gán dựa trên đánh giá định tính. Sau khi hoàn tất `domain_scoping.md`, danh pháp này sẽ được đồng bộ với chuẩn chung của toàn dự án.
- **Sub-clustering:** Thuật toán HDBSCAN trên không gian UMAP hiện tại thuần túy gom cụm theo Semantic Similarity. Điều này dẫn đến sự hình thành các Mixed-Narratives. Version 2.0 cần bổ sung cơ chế phân tách thứ cấp dựa trên `sentiment_std` và `has_contrastive`.

### 5.2 Hạn chế (Hard Limitations)

- **Lexicon-based NLP Limitations:** Các công cụ nhẹ như VADER không thể xử lý triệt để **Implicit Sarcasm**, **Implicit Multi-aspect** (khen chê nhiều khía cạnh nhưng không dùng từ nối), hoặc các Meme đặc thù của Reddit. Việc sử dụng LLMs để khắc phục hoàn toàn rào cản này là không khả thi do **Compute Bottleneck** (Hạ tầng Kaggle GPU không thể scale cho 11.5 triệu dòng văn bản).
- **Context Fragmentation:** Cột `parent_id` bị khuyết 81% ở các bản ghi bình luận. Đây là **Irreversible Data Loss** từ dataset gốc. Việc mất cấu trúc cây hội thoại khiến hệ thống không thể xác định chính xác đối tượng tham chiếu của các bình luận ngắn (ví dụ: _"I totally agree"_), làm giảm độ chính xác khi phân loại lập trường.
