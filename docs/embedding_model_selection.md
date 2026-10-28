# So sánh mô hình sentence embedding — NarrativeCast (MADOC · Reddit r/technology)

## 1. Yêu cầu rút ra từ dữ liệu

| Yêu cầu | Căn cứ | Hệ quả khi chọn mô hình |
| :--- | :--- | :--- |
| Chỉ tiếng Anh | Dataset card: `language` = `'en'` 100% | Không cần mô hình đa ngữ. Đa ngữ chỉ làm tăng kích thước. |
| Văn bản ngắn, vài chục từ | Dashboard: trung bình **37,1 từ/dòng** (≈ 48 token, ước lượng 1,3 token/từ). Sample: tối đa ≈ 45 token | `max_seq_length` 256 là đủ cho phần lớn dòng. Phần đuôi dài của dữ liệu thật **chưa biết** (xem `embedding_input_requirements.md`), nên ưu tiên mô hình có giới hạn ≥ 256, tốt nhất 512. |
| Quy mô lớn | **11.766.134 dòng** (2,4 GB) | Thông lượng và dung lượng vector là ràng buộc chính. Mô hình càng lớn càng tốn thời gian chạy toàn bộ. |
| Văn bản Reddit, giọng thân mật | r/technology, 847.657 bài + 10.918.477 bình luận | Ưu tiên mô hình có dữ liệu huấn luyện gồm bình luận Reddit (họ `all-*` của sentence-transformers có dùng Reddit comments theo mô tả mô hình). |
| Phải chạy lại được | Yêu cầu của nhóm | Mô hình tải được từ Hugging Face, chạy offline trên CPU, giấy phép dùng tự do, không phụ thuộc API trả phí. |
| Tài nguyên máy của nhóm | **Chưa biết** | Giả định bảo thủ: máy CPU, không GPU. Nhóm cần điền cấu hình thật vào mục 5. |

---

## 2. Các mô hình đã cân nhắc

Số liệu lấy từ nguồn thứ cấp (mục 9), chưa kiểm tra lại bằng model card gốc.

| Mô hình | Tham số | Số chiều | Giới hạn token | Ngôn ngữ | Giấy phép | Ghi chú dùng |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `all-MiniLM-L6-v2` | 22,7M | 384 | 256 | Anh | Apache 2.0 | Huấn luyện trên 1,17 tỷ cặp câu. Văn bản quá 256 token bị cắt. Không cần prefix. |
| `bge-small-en-v1.5` | 33,4M | 384 | 512 | Anh | MIT | Không cần prefix cho văn bản đối xứng. MTEB trung bình 62,17. |
| `all-mpnet-base-v2` | ≈109M | 768 | 384 | Anh | Apache 2.0 | Huấn luyện trên hơn 1 tỷ cặp câu, có bình luận Reddit. Văn bản quá 384 token bị cắt. |
| `bge-base-en-v1.5` | ≈109M | 768 | 512 | Anh | MIT (một nguồn ghi Apache 2.0, cần kiểm tra) | MTEB trung bình 63,55. |
| `nomic-embed-text-v1.5` | 137M | 768 (cắt được 64–512) | 8.192 | Anh | Apache 2.0 | **Bắt buộc prefix** (`clustering: `…). Cần `trust_remote_code=True`, tức chạy mã từ kho mô hình. |
| `embeddinggemma-300m` | ≈308M | 768 (cắt được 128–512) | 2.048 | Đa ngữ (100+) | Gemma Terms of Use (không phải MIT/Apache) | Có thể phải chấp nhận điều khoản trên Hugging Face. Nặng nhất trong nhóm Anh-ngữ cần xét. |
| `bge-m3` | chưa rõ (nguồn ghi file ≈2,27 GB) | 1.024 | 8.192 | Đa ngữ (100+) | MIT | Chỉ để tham chiếu cận trên về kích thước. |
| TF-IDF (1–2 gram) | — | độ dài từ vựng | không giới hạn | tùy văn bản | — | **Baseline từ vựng**, không phải mô hình ngữ nghĩa. Cho biết bài toán khó đến đâu nếu chỉ dùng từ trùng nhau. |

**Nêu tên nhưng không đưa vào thử nghiệm:**
- `e5-small-v2`, `gte-small`: cùng tầng (≈33M, 384 chiều) với `bge-small`. Chỉ có nguồn thứ cấp, nên thêm vào chỉ làm tăng số biến mà chưa có lý do dữ liệu.
- API (`text-embedding-3-*`): không chạy offline, không chạy lại được khi đổi phiên bản, và chi phí tăng theo 11,77 triệu dòng.

**Lưu ý khi đọc điểm MTEB:** các điểm trên đến từ nhiều trang, có thể khác phiên bản/tập con MTEB, nên **không so trực tiếp giữa các mô hình**. MTEB cũng không đo việc phân biệt narrative trong một cộng đồng, nên không dùng điểm này làm căn cứ chọn.

---

## 3. Lý do chọn và loại

| Mô hình | Quyết định | Lý do (theo yêu cầu ở mục 1) |
| :--- | :--- | :--- |
| `all-MiniLM-L6-v2` | **Đưa vào thử** | Nhỏ nhất (22,7M, 384 chiều), phù hợp nếu thông lượng là nút thắt. Có dữ liệu Reddit. Rủi ro: giới hạn 256 token. |
| `bge-small-en-v1.5` | **Đưa vào thử** | Cùng 384 chiều nhưng giới hạn 512 token, dư địa cho phần đuôi dài chưa biết. MIT. Rủi ro: lớn hơn MiniLM ≈ 47% số tham số nên dự kiến chậm hơn (**chưa đo**). |
| `all-mpnet-base-v2` | **Đưa vào thử (mốc chất lượng)** | Cho biết tăng chất lượng bao nhiêu khi gấp ≈5 lần tham số. Có dữ liệu Reddit. Dung lượng vector gấp đôi (768 chiều). |
| `nomic-embed-text-v1.5` | **Thử nếu còn thời gian** | Cho phép cắt chiều để giảm dung lượng, nhưng cần prefix và `trust_remote_code`, làm giảm tính tái lập và an toàn. Context 8.192 token thừa cho dữ liệu này. |
| `bge-base-en-v1.5` | Dự phòng | Cùng tầng với `all-mpnet-base-v2`. Chỉ thêm nếu cần so hai họ mô hình ở cùng kích thước. |
| `embeddinggemma-300m` | Không đưa vào vòng đầu | Đa ngữ (dữ liệu chỉ có tiếng Anh), ≈308M tham số, giấy phép riêng. Xét lại khi nhóm có GPU. |
| `bge-m3` | Loại | Đa ngữ, context 8K, file ≈2,27 GB: toàn bộ năng lực dư so với dữ liệu 37 từ tiếng Anh. |
| API đóng | Loại | Không chạy lại offline, chi phí theo 11,77 triệu dòng. |

**Đề xuất tạm thời (giả thuyết cần kiểm chứng, không phải kết luận):** thử trước `bge-small-en-v1.5` làm mặc định vì có dư địa 512 token ở kích thước 384 chiều; dùng `all-MiniLM-L6-v2` làm phương án nhanh nếu thông lượng không đủ; dùng `all-mpnet-base-v2` làm mốc chất lượng. Chọn cuối cùng theo quy tắc ở mục 7.

---

## 4. Giới hạn ngôn ngữ

- Các mô hình `*-en-*` và họ `all-*` huấn luyện chủ yếu bằng tiếng Anh. Dataset card cam đoan 100% `en`, nhưng nhãn này do bộ phát hiện ngôn ngữ gán, nên **cần lấy mẫu kiểm tra trên dữ liệu thật** (đã có trong checklist của `embedding_input_requirements.md`).
- Lẫn mã (code-switching), tiếng lóng, viết tắt Reddit, emoji có thể bị tách thành nhiều token lạ. Mức ảnh hưởng với từng mô hình **chưa đo**, vì sample tổng hợp không có các hiện tượng này.
- `all-MiniLM-L6-v2` có nền là mô hình không phân biệt hoa thường (uncased) theo nguồn đã đọc, nên mất thông tin chữ hoa (ví dụ từ viết hoa nhấn mạnh).
- Mô hình đa ngữ (`embeddinggemma-300m`, `bge-m3`) chỉ có lợi nếu dữ liệu thật hóa ra có tỷ lệ đáng kể văn bản không phải tiếng Anh.

---

## 5. Yêu cầu tài nguyên

### 5.1 Dung lượng (tính được, không cần đo)

Dung lượng vector cho **11.766.134 dòng** (nếu chỉ lưu một vector mỗi dòng, chưa tính chỉ mục):

| Số chiều | fp32 | fp16 |
| ---: | ---: | ---: |
| 384 | 18,07 GB | 9,04 GB |
| 768 | 36,15 GB | 18,07 GB |
| 1.024 | 48,19 GB | 24,10 GB |

Dung lượng file trọng số (≈ tham số × 4 byte, fp32, ước tính): MiniLM-L6 ≈ 91 MB · bge-small ≈ 134 MB · all-mpnet / bge-base ≈ 436 MB · nomic ≈ 548 MB · EmbeddingGemma ≈ 1,2 GB.

**Giảm khối lượng tính toán:** văn bản trùng lặp chỉ cần embed một lần rồi ánh xạ lại qua hash của `content`. Ở sample, mức tiết kiệm là 92,8% (360 văn bản duy nhất trên 5.000 dòng). Mức tiết kiệm ở dữ liệu thật **chưa biết**.

### 5.2 Thời gian và RAM (cần đo)

| Mô hình | texts/giây (CPU) | RAM đỉnh | Giờ ước tính cho 11,77 triệu dòng |
| :--- | :--- | :--- | :--- |
| `all-MiniLM-L6-v2` | **[CHƯA ĐO]** | **[CHƯA ĐO]** | **[CHƯA ĐO]** |
| `bge-small-en-v1.5` | **[CHƯA ĐO]** | **[CHƯA ĐO]** | **[CHƯA ĐO]** |
| `all-mpnet-base-v2` | **[CHƯA ĐO]** | **[CHƯA ĐO]** | **[CHƯA ĐO]** |
| `nomic-embed-text-v1.5` | **[CHƯA ĐO]** | **[CHƯA ĐO]** | **[CHƯA ĐO]** |

Script tự điền các cột này (dùng văn bản ghép dài ≈ 37 từ để mô phỏng độ dài thật). **Máy của nhóm:** CPU: ___ nhân · RAM: ___ GB · GPU: ___ (nếu có).

---

## 6. Kết quả thử trên mẫu nhỏ

### 6.1 Thiết kế thử nghiệm

- **Dữ liệu:** `madoc_sample.parquet`, 5.000 dòng → chỉ **36 câu lõi duy nhất**, 6 narrative, mỗi narrative 6 câu.
- **Phương pháp:** leave-one-sentence-out. Giữ lại toàn bộ biến thể của một câu lõi, phân loại câu đó vào narrative bằng các câu còn lại (nearest-centroid và 1-NN theo cosine). Phép thử này đo khả năng khái quát sang **câu chưa thấy**, không đo khả năng nhớ khuôn mẫu.
- **Hai chế độ văn bản:**
  - `core`: chỉ câu lõi (đã bỏ `[topic]`, `Regarding …:`, `… r/technology.`, `Discussion around …`).
  - `raw`: nguyên cột `content` (còn tiền tố, từ khóa và khuôn mẫu — nguồn rò rỉ nhãn đã nêu trong `embedding_input_requirements.md`).
- **Mức ngẫu nhiên:** 1/6 = 16,7%.
- **Độ phân giải:** chế độ `core` chỉ có 36 điểm, mỗi câu sai/đúng đổi accuracy 2,8 điểm phần trăm. Chênh lệch dưới 3 câu coi là nhiễu.

### 6.2 Kết quả

| Mô hình | Chế độ | Acc nearest-centroid | Acc 1-NN | Độ tách (intra − inter cos) |
| :--- | :--- | ---: | ---: | ---: |
| TF-IDF (baseline) | `core` (n=36) | 47,2% | 52,8% | 0,006 |
| TF-IDF (baseline) | `raw` (n=360) | 97,2% | 100,0% | 0,158 |
| `all-MiniLM-L6-v2` | `core` / `raw` | **[CHƯA ĐO]** | **[CHƯA ĐO]** | **[CHƯA ĐO]** |
| `bge-small-en-v1.5` | `core` / `raw` | **[CHƯA ĐO]** | **[CHƯA ĐO]** | **[CHƯA ĐO]** |
| `all-mpnet-base-v2` | `core` / `raw` | **[CHƯA ĐO]** | **[CHƯA ĐO]** | **[CHƯA ĐO]** |
| `nomic-embed-text-v1.5` | `core` / `raw` | **[CHƯA ĐO]** | **[CHƯA ĐO]** | **[CHƯA ĐO]** |

### 6.3 Cách đọc kết quả đã có

- Với TF-IDF, accuracy tăng từ ≈ 47–53% (`core`) lên 97–100% (`raw`) chỉ vì thêm khuôn mẫu và từ khóa. Đây là bằng chứng đo được rằng **văn bản `raw` của sample làm bài toán dễ giả tạo**. Mọi so sánh mô hình phải dùng chế độ `core`, nếu không các mô hình sẽ đều gần 100% và không phân biệt được nhau.
- TF-IDF ở chế độ `core` vẫn cao hơn mức ngẫu nhiên (16,7%) nhưng độ tách cosine gần bằng 0 (0,006), nghĩa là nó đúng nhờ vài từ trùng chứ không có cấu trúc ngữ nghĩa. Mô hình neural có ý nghĩa nếu vượt rõ baseline này ở `core`.
- Các nhầm lẫn của TF-IDF ở `core` rải đều giữa nhiều cặp narrative (xem file JSON). Hai narrative cùng chủ đề `AI Governance` (`AI_Regulation_Stifles_Innovation` và `AI_Safety_Demands_Urgent_Guardrails`) là cặp khó nhất về mặt ý nghĩa; nên xem riêng cặp này khi có kết quả neural.
- TF-IDF được fit trên chính tập đánh giá (transductive), nên hơi lạc quan cho baseline.

---

## 7. Quy tắc chọn mô hình (định trước khi có kết quả)

Chọn **mô hình nhỏ nhất** thỏa đồng thời:

1. **Chất lượng:** accuracy nearest-centroid ở chế độ `core` thấp hơn mô hình tốt nhất không quá 3 câu (≈ 8,3 điểm phần trăm). Chênh lệch nhỏ hơn mức này coi là hòa vì n = 36.
2. **Thông lượng:** thời gian ước tính cho 11,77 triệu dòng không vượt ngân sách của nhóm (nhóm điền: ___ giờ).
3. **Bộ nhớ:** RAM đỉnh không quá 50% RAM máy chạy.
4. **Giấy phép và tái lập:** giấy phép cho phép dùng trong dự án; không yêu cầu API ngoài; ưu tiên không cần `trust_remote_code`.
5. **Hòa:** ưu tiên giới hạn token lớn hơn (ít bị cắt), sau đó số chiều nhỏ hơn (ít dung lượng).

Nếu **không mô hình nào** vượt rõ baseline TF-IDF ở `core`, kết luận là mẫu tổng hợp không đủ để phân biệt mô hình, và cần chạy lại phép thử trên mẫu văn bản Reddit thật (vài nghìn dòng có nhãn hoặc nhãn tạm).

---

## 8. Cách chạy lại

```bash
pip install sentence-transformers scikit-learn pandas pyarrow psutil

python scripts/benchmark_embeddings.py \
  --input data/sample/madoc_sample.parquet \
  --models tfidf all-MiniLM-L6-v2 bge-small-en-v1.5 all-mpnet-base-v2 nomic-embed-text-v1.5 \
  --out outputs/embedding_benchmark.json
```

- Script chạy trên CPU (`device="cpu"`), mỗi mô hình lỗi được ghi riêng và không làm hỏng cả lần chạy.
- Để tái lập: ghi lại `pip freeze`, phiên bản `sentence-transformers`, và mã commit (revision) Hugging Face của từng mô hình đã tải.
- **Phần neural của script chưa được chạy thử** vì thiếu thư viện trong môi trường soạn thảo. Phần TF-IDF, tải dữ liệu, tách câu lõi, đánh giá và bắt lỗi đã chạy thật. Lần chạy đầu của nhóm nên kiểm tra kỹ phần prefix (đặc biệt `nomic-embed-text-v1.5` và `embeddinggemma-300m`) và số chiều/`max_seq_length` in ra có khớp bảng mục 2 không.

---

## 9. Nguồn thông số (thứ cấp, cần đối chiếu model card gốc)

- `all-MiniLM-L6-v2`: superlinked.com/glossary/what-is-all-minilm-l6-v2 · memgraph.com/docs/advanced-algorithms/available-algorithms/embeddings
- `bge-small-en-v1.5` / `bge-base-en-v1.5`: zilliz.com/ai-models/bge-base-en-v1.5 · llmreference.com/model/bge-small-en-v1.5 · inference.readthedocs.io (xinference, bge-small-en-v1.5)
- `bge-m3`: datalearner.com/en/ai-models/pretrained-models/BGE-M3-Embedding
- `all-mpnet-base-v2`: zilliz.com/ai-models/all-mpnet-base-v2 · promptlayer.com/models/all-mpnet-base-v2
- `nomic-embed-text-v1.5`: promptlayer.com/models/nomic-embed-text-v15 · localaimaster.com/models/nomic-embed-text
- `embeddinggemma-300m`: mixpeek.com/model/google/embeddinggemma-300m · datalearner.com/en/ai-models/pretrained-models/embeddinggemma-300m

**Điểm chưa thống nhất giữa các nguồn:** giấy phép của `bge-base-en-v1.5` (MIT hay Apache 2.0); số tham số của `bge-m3` (chỉ biết kích thước file); số tham số `embeddinggemma-300m` (một nguồn ghi 308M, một trang khác ghi sai là 3,0B).

## 10. Hạn chế của tài liệu này

- Chưa có số đo nào cho mô hình neural (thông lượng, RAM, chất lượng).
- Sample là dữ liệu tổng hợp với 36 câu lõi: đủ để kiểm tra script và phát hiện rò rỉ nhãn, không đủ để xếp hạng mô hình một cách chắc chắn.
- Thông số mô hình từ nguồn thứ cấp.
- Chưa biết cấu hình máy của nhóm, nên chưa chốt được ngân sách thời gian và RAM.
