# NarrativeCast — Experimental Plan

> **Trạng thái:** Kế hoạch thí nghiệm phiên bản 1.0 — cần cập nhật sau khi xác minh dữ liệu MADOC thực tế.  
> **Mục đích:** Chốt phạm vi, câu hỏi nghiên cứu, phương pháp, baseline, cách chia dữ liệu theo thời gian và tiêu chí đánh giá trước khi triển khai mô hình chính.

## 1. Phạm vi nghiên cứu đã chốt

### 1.1. Bài toán nghiên cứu

Nghiên cứu xem liệu việc kết hợp **đặc trưng ngữ nghĩa** của nội dung với **đặc trưng diễn biến theo thời gian** có dự đoán thứ hạng mức độ nổi bật của các narrative trong tương lai tốt hơn các baseline dựa trên độ phổ biến hiện tại hay không.

Đây là bài toán **dự báo mức độ nổi bật tương đối**, không phải khẳng định chắc chắn một narrative sẽ xuất hiện, cũng không phải dự đoán cảm xúc hay quan hệ nhân quả.

### 1.2. Phạm vi dữ liệu

- **Nguồn dữ liệu dự kiến:** MADOC, bộ dữ liệu được mô tả trong `docs/dataset_card.md`.
- **Nền tảng/phạm vi phân tích chính:** Reddit, cộng đồng `technology`, ngôn ngữ tiếng Anh.
- **Đơn vị dữ liệu:** một bài đăng hoặc bình luận có nội dung và thời gian đăng hợp lệ.
- **Trường tối thiểu:** `post_id`, `publish_date`, `content`, `platform`, `community`, `language`.
- **Trường bổ trợ nếu chất lượng cho phép:** `user_id`, `parent_id`, `interaction_type`, `sentiment_vader`, `url`.
- **Cửa sổ lịch sử ban đầu:** 7 ngày gần nhất trước thời điểm dự đoán.
- **Khoảng dự đoán:** 3, 7 và 14 ngày.
- **Phạm vi tính toán:** bắt đầu từ mẫu nhỏ đã xác minh, sau đó mở rộng theo khả năng tài nguyên. Không mặc định phải xử lý toàn bộ MADOC ngay từ thí nghiệm đầu tiên.

**Điều kiện trước khi chạy thí nghiệm chính:** kiểm tra lại nguồn gốc và giấy phép từ trang Zenodo gốc; xác nhận các tệp đang dùng là dữ liệu MADOC thực hay dữ liệu mẫu sinh để phát triển. Không suy rộng kết quả từ dữ liệu mô phỏng thành kết luận về dữ liệu mạng xã hội thực.

### 1.3. Định nghĩa vận hành

- **Topic:** chủ đề rộng mà các bài đăng cùng đề cập.
- **Narrative (trong thí nghiệm):** nhóm bài đăng có nội dung ngữ nghĩa tương đồng và thể hiện một mạch diễn giải tương đối nhất quán. Hai narrative có thể thuộc cùng một topic nhưng khác lập trường/khung diễn giải.
- **Prominence:** tỷ lệ bài đăng được gán cho narrative \(n\) trong cửa sổ \(W\):

  \[
  P(n,W)=\frac{N_{n,W}}{N_W}
  \]

  \(N_{n,W}\) là số bài đăng được gán cho narrative \(n\); \(N_W\) là tổng số bài đăng hợp lệ trong cửa sổ, kể cả bài không được gán vào narrative đang xét.
- **Mục tiêu dự báo:** tại thời điểm \(t\), dự đoán thứ hạng hoặc prominence của narrative trong cửa sổ tương lai \((t,t+h]\), với \(h\in\{3,7,14\}\) ngày.
- **Narrative nổi bật:** trong phiên bản đầu, định nghĩa chính là narrative nằm trong **Top- K theo prominence tương lai**. Chọn \(K=3\) cho chỉ số Top-3 Hit Rate; đồng thời báo cáo thứ hạng và prominence liên tục để tránh phụ thuộc hoàn toàn vào một ngưỡng nhị phân.

Không dùng từ “probability” cho điểm xếp hạng nếu mô hình chưa được hiệu chỉnh xác suất. Ban đầu, output là **score/rank**; chỉ báo cáo xác suất khi có bước calibration và kiểm tra calibration riêng.

## 2. Câu hỏi nghiên cứu

- **RQ1 — Giá trị của thông tin ngữ nghĩa:** Mô hình dùng đặc trưng ngữ nghĩa và thời gian có dự đoán tốt hơn các baseline chỉ dựa vào độ phổ biến/lịch sử hay không?
- **RQ2 — Ảnh hưởng của khoảng dự đoán:** Hiệu quả thay đổi thế nào giữa \(h=3\), \(h=7\) và \(h=14\) ngày?
- **RQ3 — Đóng góp của đặc trưng:** Đặc trưng nào (prominence hiện tại, tăng trưởng, độ biến động ngữ nghĩa, hoặc đặc trưng hoạt động) liên quan nhiều nhất đến hiệu quả dự đoán?

## 3. Giả thuyết có thể kiểm chứng

- **H1:** Mô hình kết hợp ngữ nghĩa và thời gian đạt kết quả tốt hơn ít nhất một baseline đơn giản trên tập kiểm tra theo thời gian.
- **H2:** Dự đoán ở horizon dài hơn có xu hướng khó hơn, nhưng mức suy giảm phải được xác định từ kết quả thực nghiệm, không mặc định là đúng.
- **H3:** Đặc trưng tăng trưởng và thay đổi ngữ nghĩa có thể bổ sung thông tin cho prominence hiện tại. Giả thuyết này được kiểm tra bằng ablation, không xem là kết luận trước thực nghiệm.

## 4. Phương pháp dự kiến

### Bước A — Xác minh và tiền xử lý dữ liệu

1. Đọc metadata và giấy phép của nguồn dữ liệu gốc.
2. Lọc `platform == reddit`, `community == technology`, `language == en` khi các trường có giá trị đáng tin cậy.
3. Chuyển `publish_date` từ Unix timestamp sang thời gian UTC; sắp xếp tăng dần.
4. Loại bản ghi thiếu `content` hoặc timestamp không hợp lệ; kiểm tra ID trùng. Không xóa bài chỉ vì nội dung trùng nếu chưa xác định đó là bản ghi trùng thật hay các bài đăng khác nhau.
5. Báo cáo số bản ghi trước/sau lọc, khoảng thời gian, missing values, duplicates, số bài theo ngày/tuần và các giới hạn lấy mẫu.

### Bước B — Khám phá và theo dõi narrative

- **Phương án ưu tiên để thử:** Sentence-BERT embeddings + BERTopic (embedding → UMAP/HDBSCAN → biểu diễn từ khóa c-TF-IDF).
- Xem cụm BERTopic là **cụm chủ đề/narrative ứng viên**, không tự động coi mỗi topic là narrative hoàn chỉnh.
- Kiểm tra thủ công một mẫu bài và từ khóa đại diện để đánh giá tính mạch lạc; ghi lại các cụm không rõ nghĩa và outlier.
- Giữ cách gán narrative nhất quán giữa các cửa sổ. Với phiên bản đầu, fit cụm trên tập phát triển trước rồi gán dữ liệu về sau vào cụm đã xác định nếu kỹ thuật hỗ trợ; không fit clustering trên toàn bộ dữ liệu rồi dùng nhãn tương lai như thể nhãn đó đã biết tại thời điểm dự đoán.
- Nếu chưa đủ nguồn lực để đánh giá narrative bằng người gán nhãn, gọi đơn vị dự báo là **narrative candidate / semantic cluster** và nêu rõ giới hạn này.

### Bước C — Tạo chuỗi đặc trưng

Với mỗi narrative và mỗi thời điểm dự đoán, tính từ dữ liệu lịch sử:
- `volume`: số bài đăng trong cửa sổ 7 ngày;
- `prominence`: tỷ lệ bài đăng của narrative trong cửa sổ;
- `growth`: thay đổi volume so với cửa sổ 7 ngày liền trước, có làm trơn mẫu số;
- `momentum`: tốc độ tăng tương đối;
- `semantic_change` (nếu khả thi): thay đổi giữa embedding/centroid của narrative ở các cửa sổ liên tiếp;
- `activity_features` (tùy chọn): số người dùng duy nhất hoặc số reply, chỉ dùng nếu các trường tương ứng đủ tin cậy.

Chỉ dùng dữ liệu có timestamp không muộn hơn thời điểm dự đoán \(t\). Các giá trị trong \((t,t+h]\) chỉ được dùng để tạo ground truth.

### Bước D — Mô hình

Triển khai theo thứ tự từ đơn giản đến phức tạp:

1. **Popularity baseline:** xếp hạng theo volume trong cửa sổ hiện tại.
2. **Persistence baseline:** xếp hạng theo prominence hiện tại.
3. **Growth Momentum baseline:** xếp hạng theo tốc độ tăng so với cửa sổ liền trước.
4. **Mô hình đề xuất ban đầu:** một mô hình học máy gọn (ưu tiên Random Forest hoặc Gradient Boosting) sử dụng các đặc trưng thời gian và ngữ nghĩa đã trích xuất.

Chỉ thêm mô hình phức tạp như Temporal Fusion Transformer nếu số lượng chuỗi và số cửa sổ thời gian thực tế đủ lớn. Không xem TFT là yêu cầu bắt buộc của phiên bản đầu vì mô hình này có chi phí và yêu cầu dữ liệu cao hơn.

## 5. Thiết kế đánh giá

### 5.1. Chia dữ liệu

Dùng **chronological rolling-origin evaluation** (đánh giá cuốn chiếu theo thời gian):
1. Tại mỗi mốc \(t\), chỉ dùng lịch sử đến \(t\) để tạo đặc trưng và huấn luyện/cập nhật mô hình.
2. Dự đoán các khoảng tương lai 3, 7 hoặc 14 ngày.
3. So sánh với dữ liệu thực tế sau khi khoảng dự đoán kết thúc.
4. Dịch mốc \(t\) về phía trước và lặp lại.
5. Dùng các mốc dự đoán đủ xa nhau hoặc điều chỉnh cách tổng hợp sai số để tránh coi các cửa sổ tương lai chồng lấn là những mẫu độc lập.

Không chia ngẫu nhiên các bài đăng thành train/test vì cách đó có thể để dữ liệu tương lai lọt vào quá trình huấn luyện.

### 5.2. Ground truth

Với mỗi narrative đã có thể nhận diện tại thời điểm \(t\), tính prominence thực tế trong \((t,t+h]\). Các narrative không có bài trong cửa sổ tương lai nhận volume bằng 0; cần quy định rõ cách xử lý narrative mới chỉ xuất hiện trong tương lai, vì mô hình không thể xếp hạng một narrative chưa có trong tập ứng viên nếu không có cơ chế phát hiện narrative mới.

Trong phiên bản đầu, đánh giá chính trên tập **narrative ứng viên đã được xác định từ lịch sử**. Khả năng phát hiện narrative hoàn toàn mới được ghi nhận là giới hạn/mở rộng tiếp theo, không trộn lẫn với bài toán xếp hạng narrative đã biết.

### 5.3. Chỉ số đánh giá

| Chỉ số | Ý nghĩa | Cách diễn giải |
|---|---|---|
| Spearman rank correlation (\(\rho\)) | Mức tương quan giữa thứ hạng dự đoán và thứ hạng thực tế | Cao hơn tốt hơn |
| Top-1 accuracy | Tỷ lệ mốc mà narrative dự đoán đứng đầu trùng narrative đứng đầu thực tế | Cao hơn tốt hơn |
| Top-3 Hit Rate | Tỷ lệ mốc mà narrative đứng đầu thực tế nằm trong Top-3 dự đoán | Cao hơn tốt hơn |
| Mean Absolute Rank Error (Rank MAE) | Sai lệch thứ hạng trung bình | Thấp hơn tốt hơn |
| MAE của prominence (bổ sung) | Sai số giữa prominence dự đoán và thực tế nếu mô hình dự báo giá trị số | Thấp hơn tốt hơn |

Báo cáo kết quả riêng cho từng horizon, số mốc đánh giá, và trung bình/tổng hợp trên các mốc. Khi có thể, bổ sung khoảng tin cậy hoặc độ biến thiên giữa các mốc. Không chỉ báo cáo một chỉ số thuận lợi.

### 5.4. So sánh công bằng

- Mọi mô hình dùng cùng các mốc dự đoán, cùng tập narrative ứng viên và cùng ground truth.
- Baseline không được dùng đặc trưng tương lai.
- Các bước fit embedding/clustering/chuẩn hóa hoặc chọn đặc trưng phải tuân thủ ranh giới train/test theo thời gian.
- Nếu một horizon không có đủ dữ liệu tương lai, loại mốc đó theo quy tắc đã công bố và báo cáo số mốc còn lại.
- Không so sánh số liệu từ dữ liệu mô phỏng với số liệu từ dữ liệu thật như thể cùng một benchmark.

## 6. Thí nghiệm ablation

Sau khi pipeline chính chạy được, thực hiện lần lượt:
1. **Temporal-only:** volume, prominence, growth/momentum.
2. **Semantic + temporal:** thêm đặc trưng embedding/semantic change.
3. **Semantic + temporal + activity:** thêm user/reply features nếu khả dụng.

Mục tiêu là xác định liệu phần ngữ nghĩa có cải thiện kết quả so với thông tin thời gian đơn thuần hay không. Chỉ thêm từng nhóm đặc trưng khi dữ liệu thực sự hỗ trợ.

## 7. Các giới hạn và rủi ro đã xác định

1. **Narrative không có nhãn sẵn:** MADOC không được mô tả là có nhãn narrative chuẩn. BERTopic tạo topic/cluster, không đảm bảo tự động nhận diện stance hoặc narrative.
2. **Mẫu dữ liệu:** mẫu hiện tại trong repo cần được xác minh nguồn gốc trước khi dùng cho kết luận thực nghiệm.
3. **Tính nhất quán theo thời gian:** cluster có thể đổi nghĩa, tách/gộp hoặc biến mất; cần cơ chế gán và theo dõi cụm.
4. **Dữ liệu lệch:** lượng bài theo thời gian/cộng đồng có thể không đồng đều; prominence phụ thuộc tổng lượng bài trong cửa sổ.
5. **Cửa sổ chồng lấn:** các lần dự đoán gần nhau có thể dùng chung bài tương lai; cần báo cáo rõ và tránh đánh giá quá lạc quan.
6. **Xếp hạng narrative mới:** narrative chỉ xuất hiện sau thời điểm dự đoán không nằm trong candidate set lịch sử.
7. **Độ bao phủ:** phiên bản đầu chỉ xét Reddit `technology`, tiếng Anh; không tuyên bố tổng quát cho mọi nền tảng.

## 8. Trạng thái baseline hiện tại và lưu ý quan trọng

Repo có `notebooks/01_baseline_feasibility.ipynb` và `docs/baseline_feasibility_report.md`, trong đó mẫu đang được dùng có 5.000 bản ghi, Reddit `technology`, khoảng 60 ngày, cùng trường `ground_truth_narrative`.

**Trước khi trích dẫn các kết quả baseline trong báo cáo khoa học, nhóm phải xác minh xem `madoc_sample.parquet` có phải mẫu lấy trực tiếp từ MADOC hay được tạo bằng `scripts/generate_sample_data.py`.** Notebook cho thấy có nhánh tự sinh dữ liệu nếu file không tồn tại, và các nhãn `ground_truth_narrative` không nằm trong schema MADOC được mô tả ở `dataset_card.md`. Vì vậy:
- Nếu dữ liệu được sinh mô phỏng, các chỉ số hiện tại chỉ chứng minh **pipeline có thể chạy**, không chứng minh năng lực dự báo trên dữ liệu mạng xã hội thật.
- Nếu dữ liệu được trích từ MADOC nhưng nhãn narrative do nhóm gán/thêm, phải ghi rõ quy trình tạo nhãn và kiểm tra chất lượng.
- Không trình bày các con số baseline hiện có như kết quả cuối cùng cho MADOC trước khi xác minh.

Ngoài ra, Popularity và Persistence hiện có thể tạo cùng thứ hạng nếu volume và prominence được tính trên cùng cửa sổ, vì \(P(n,W)=N_{n,W}/N_W\) với mẫu số chung. Đây là hành vi kỳ vọng về mặt toán học, không phải hai baseline độc lập trong điều kiện đó.

## 9. Tiêu chí hoàn thành nội dung 1

Nội dung 1 được xem là hoàn thành khi:
- [ ] Đã xác minh nguồn, giấy phép, schema và nguồn gốc chính xác của mẫu đang dùng.
- [ ] Đã chốt phạm vi Reddit `technology`, đơn vị phân tích và định nghĩa prominence.
- [ ] Đã nêu rõ cách tạo/gán narrative và giới hạn của nhãn.
- [ ] Đã chốt horizons 3/7/14 ngày, cửa sổ lịch sử 7 ngày và cách đánh giá cuốn chiếu.
- [ ] Đã mô tả 3 baseline và mô hình đề xuất ban đầu.
- [ ] Đã định nghĩa metrics và quy tắc tránh data leakage.
- [ ] Đã phân biệt kết quả kiểm tra tính khả thi trên mẫu với kết quả khoa học trên dữ liệu thật.
- [ ] Đã ghi rõ rủi ro, giới hạn và các thí nghiệm ablation.

## 10. Kế hoạch triển khai tiếp theo

1. **Chốt dữ liệu:** kiểm tra script sinh mẫu, nguồn tải, metadata và giấy phép.
2. **Tái lập profiling:** chạy lại `notebooks/01_data_profiling.ipynb` trên mẫu đã xác minh; lưu thống kê vào dataset card.
3. **Sửa baseline evaluation:** xác nhận cách tạo nhãn narrative, xử lý candidate mới và cách chọn các mốc dự đoán; kiểm tra rolling-origin không dùng thông tin tương lai.
4. **Chạy baseline trên dữ liệu thật:** báo cáo metrics theo horizon và số mốc đánh giá.
5. **Thử BERTopic/embedding:** đánh giá chất lượng cụm bằng kiểm tra mẫu trước khi đưa vào dự báo.
6. **So sánh mô hình và ablation:** chỉ sau khi baseline và gán narrative đã ổn định.
