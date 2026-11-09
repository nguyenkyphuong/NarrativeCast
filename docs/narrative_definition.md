# NarrativeCast — Định nghĩa và quy tắc nhận diện narrative

**Nội dung 3 · Task 6 — Xem xét tiêu chí narrative**  
**Sản phẩm:** `docs/narrative_definition.md`  
**Phiên bản:** v1.0 · 2026-11-09  
**Ngách đã chọn:** **Net Neutrality – FCC Regulation** (Chính sách trung lập mạng Internet)  
**Trạng thái:** Bộ quy tắc đề xuất dùng chung; các narrative minh họa **chưa phải nhãn đã nghiệm thu trên dữ liệu**.

## 1. Mục đích và phạm vi

Tài liệu quy định cách **nhận diện, diễn giải và đặt tên narrative** nhất quán cho các cụm/ngữ liệu Reddit thuộc ngách Net Neutrality, phục vụ báo cáo, demo và theo dõi theo thời gian.

- **Domain:** Telecom, Net Neutrality, Web & Crypto.
- **Subdomain/selected scope:** Net Neutrality – FCC Regulation.
- **Nguồn:** Reddit r/technology. Khám phá toàn bộ POST giai đoạn 2014–2020; cấu hình `selected_scope.json` phiên bản `scope_v1` trước đó khảo sát POST + COMMENT giai đoạn 2014–2018. Hai phạm vi **không được trộn số liệu** nếu chưa thống nhất phiên bản tập dữ liệu.
- **Ranh giới:** Task này định nghĩa quy tắc; **không thực hiện thay ND3.4b clustering** và không mặc nhiên xác nhận bất kỳ nhãn nào.

## 2. Định nghĩa thao tác

**Topic (chủ đề):** Nội dung/vấn đề được đề cập — *bài viết đang nói về điều gì?*  
Ví dụ: `Net Neutrality`, `FCC`, `ISP`, `paid prioritization`.

**Stance (lập trường):** Hướng ủng hộ, phản đối hoặc trung lập đối với một đối tượng/chính sách cụ thể — *người viết đồng tình với điều gì?*

**Frame (khung diễn giải):** Khía cạnh được nhấn mạnh để hiểu vấn đề (quyền lợi người dùng, tự do ngôn luận, cạnh tranh, quản lý nhà nước...). Một frame có thể được nhiều narrative dùng với những kết luận khác nhau.

**Narrative (mạch diễn giải/luận điểm):** Một **thông điệp hoặc cách diễn giải có thể nhận diện và lặp lại giữa nhiều văn bản**, liên kết vấn đề/đối tượng với **một khẳng định cụ thể** về hậu quả, nguyên nhân, trách nhiệm, giá trị hoặc giải pháp; thường hàm chứa lập trường hoặc hàm ý hành động.

> **Quy tắc cốt lõi:** Một narrative không chỉ là từ khóa, chủ đề, cảm xúc, tên thực thể hay một cụm embedding. Phải diễn đạt được nó thành **một mệnh đề có nội dung** và tìm được bằng chứng trong các bài viết.

**Narrative prominence (mức độ nổi bật):** Độ phổ biến của narrative trong **một cửa sổ thời gian xác định**, tính từ các văn bản đã được gán/kiểm chứng, cần báo cáo cả số lượng và tỷ lệ trên mẫu số thích hợp. **Emerging** là sự tăng nổi bật theo thời gian theo tiêu chí định lượng được chốt trước ở module dự báo, **không đồng nghĩa** với narrative có tổng số bài lớn nhất.

## 3. Quy tắc nhận diện: các bước bắt buộc

Đơn vị xét là **văn bản riêng lẻ** (`post_id` + `content` + `publish_datetime_utc`), không gán cùng narrative cho toàn bộ một cluster khi chưa đọc ví dụ và kiểm tra độ thuần.

1. **Đúng phạm vi?** Văn bản bàn về Net Neutrality hoặc cơ chế quản lý lưu lượng Internet liên quan. Nếu chỉ nói FCC, Internet, giá mạng hay một công ty viễn thông mà không có luận điểm liên quan, đánh dấu `OUT_OF_SCOPE` hoặc `UNCERTAIN`.
2. **Có khẳng định diễn giải?** Xác định **ai/cái gì**, **khẳng định gì**, và **vì sao/hệ quả/giải pháp nào**. Nếu chỉ tường thuật lịch họp, đường link hoặc tên chính sách thì gán `TOPIC_ONLY`.
3. **Có thể tóm tắt thành một câu không?** Viết câu tường minh, không thêm ý mà văn bản không hỗ trợ. Không suy luận lập trường chỉ từ cảm xúc, tiêu đề gây chú ý hoặc tên tác giả.
4. **Có lặp lại đủ để xem là narrative không?** Đối chiếu **ít nhất 3 văn bản khác nhau** chứa cùng khẳng định cốt lõi để đưa vào danh sách **ứng viên** (ngưỡng thực hành sơ bộ kế thừa bước khám phá; không phải ngưỡng xác nhận chất lượng thống kê). Cần xét độ độc lập nguồn và loại bản sao, bot/spam.
5. **Các bài có cùng claim không?** Một cluster có thể chứa nhiều narrative đối lập; một narrative có thể trải trên nhiều cluster. Tách khi khác khẳng định cốt lõi, không tách chỉ vì từ vựng khác nhau.
6. **Ghi độ chắc chắn và bằng chứng.** Lưu ID bài, câu trích ngắn/hay diễn giải bằng chứng, trường hợp phản ví dụ; gán `PROVISIONAL` cho đến khi được review. Nếu mơ hồ, giữ `UNCERTAIN` thay vì ép nhãn.

### Điều kiện tối thiểu để công bố một narrative trong báo cáo/demo

- Có **mệnh đề rõ** (đối tượng + khẳng định/quan hệ + hệ quả/giải pháp), khác với nhãn chủ đề.
- Có **ít nhất 3 văn bản khác nhau** minh họa (điều kiện sơ bộ, **không** thay thế đánh giá chất lượng trên mẫu rộng hơn).
- Có ví dụ ủng hộ và kiểm tra ví dụ lệch/chồng lấn; có quyết định `ACCEPT / REVISE / REJECT` của người rà soát.
- Có **ID ổn định**, tên tiếng Anh nhất quán và mô tả tiếng Việt; ghi rõ `first_observed` theo dữ liệu có sẵn, không tuyên bố ngày thực tế narrative ra đời.

Nếu thiếu điều kiện, chỉ gọi là **candidate narrative / frame hypothesis**, không gọi là narrative đã xác nhận.

## 4. Ví dụ đúng và không đúng

*Các câu sau là ví dụ minh họa do người soạn tạo, **không phải trích dẫn Reddit** và không chứng minh narrative đã tồn tại trong bộ dữ liệu.*

| Văn bản / cách gọi minh họa | Phân loại | Vì sao |
|---|---|---|
| “Net neutrality and the FCC” | `TOPIC_ONLY` | Chỉ nêu đề tài/thực thể, chưa có khẳng định |
| “The FCC voted on the rules today.” | `TOPIC_ONLY` | Tin sự kiện, chưa có thông điệp diễn giải |
| “This decision is terrible.” | `UNCERTAIN` | Cảm xúc tiêu cực nhưng không rõ quyết định, lý do và claim |
| “Without net neutrality, ISPs can favor paid fast lanes and disadvantage smaller websites.” | `NARRATIVE_CANDIDATE` | Claim về cơ chế và bất lợi cạnh tranh |
| “Repealing these rules restores market freedom and removes unnecessary government control.” | `NARRATIVE_CANDIDATE` | Claim phản đối quy định với logic thị trường/nhà nước |
| “Keeping an open Internet protects people's ability to access and share information freely.” | `NARRATIVE_CANDIDATE` | Claim bảo vệ tự do truy cập và biểu đạt |
| “Contact your representatives to oppose the repeal and defend an open Internet.” | `NARRATIVE_CANDIDATE` | Claim/hàm ý hành động huy động cộng đồng |
| Một bài chỉ có từ “Comcast”, “FCC” hoặc “internet bill” | `OUT_OF_SCOPE`/`UNCERTAIN` | Từ khóa đơn lẻ không đủ chứng minh đúng ngách hoặc narrative |

Một văn bản có thể đề cập nhiều luận điểm. Khi triển khai: lưu tối đa một **primary narrative** cho bảng phân tích đơn giản, đồng thời giữ `secondary_narrative_ids` nếu có bằng chứng; không nhân số văn bản khi tính tổng lưu lượng.

## 5. Bộ narrative giả thuyết ban đầu trong Net Neutrality

Đây là **nhóm ứng viên để kiểm tra**, không phải danh sách nhãn gold hay kết quả clustering của Phước:

| ID tạm | Tên tiếng Anh cho báo cáo/demo | Mệnh đề cốt lõi cần tìm bằng chứng |
|---|---|---|
| `NN_OPEN_ACCESS` | Open Internet as a Public Right | Bảo vệ Net Neutrality giúp người dùng tiếp cận và chia sẻ thông tin công bằng/tự do |
| `NN_ISP_POWER` | ISP Fast Lanes Harm Fair Competition | Cho phép ưu tiên trả phí có thể trao quyền bất cân xứng cho ISP và gây bất lợi cho dịch vụ nhỏ |
| `NN_DEREGULATION` | Net Neutrality Rules as Government Overreach | Hủy/bớt quy định được cho là giảm can thiệp nhà nước, khuyến khích thị trường/đầu tư |
| `NN_MOBILIZATION` | Public Mobilization to Defend Net Neutrality | Sự tham gia của cộng đồng và gây sức ép với đại diện dân cử có thể bảo vệ Internet mở |
| `NN_POLITICAL` | Net Neutrality as a Political Accountability Issue | Tranh luận được diễn giải qua trách nhiệm và quyết định của các phe/nhà hoạch định chính sách |

`NN_POLITICAL` dễ biến thành chỉ một **frame**. Chỉ giữ làm narrative riêng nếu dữ liệu chứng minh được **một claim nhất quán** về trách nhiệm/hệ quả chính trị; nếu không, chuyển thành thuộc tính `frame=political_accountability`. Tương tự, thay nhãn hoặc gộp/tách các ứng viên theo review thực tế.

## 6. Quy tắc đặt tên thống nhất

**Quy cách bắt buộc:**

- Tên ngắn bằng **tiếng Anh** cho chart/demo: `[đối tượng hoặc tác nhân] + [khẳng định/hệ quả/giải pháp]`, ưu tiên 4–9 từ, tránh chỉ viết `FCC`, `Privacy`, `Positive`, `Cluster 7`.
- Mỗi narrative có **stable ID** (`NN_<SHORT_CODE>`) và một câu định nghĩa tiếng Việt. Đổi tên hiển thị **không đổi ID**; nếu nghĩa thay đổi, ghi lịch sử version và ánh xạ.
- Tên **trung tính**, diễn đạt claim của người đăng, không biến thành khẳng định thực tế của nhóm; tránh tên giật gân hay đánh giá đúng/sai.
- Không đặt tên dựa trên **sentiment** đơn thuần (`Positive/Negative`) hay **stance** đơn thuần (`Pro/Anti`) mà thiếu đối tượng và luận điểm.
- Không lấy tên từ một bài lẻ hoặc một cụm k-means; dựa trên nhiều ví dụ và ghi `evidence_post_ids`.
- Khi một cụm trộn nhiều claim: gắn nhãn `MIXED` hoặc tách narrative theo bằng chứng; cụm không rõ: `UNRESOLVED`; không ép 1 cluster = 1 narrative.

**Ví dụ:** `cluster_44` ❌ → `ISP Fast Lanes Harm Fair Competition` ✅ *nếu* ví dụ trong cụm thực sự cùng bảo vệ claim này.  
`Pro Net Neutrality` ⚠️ → `Open Internet as a Public Right` ✅ nếu văn bản thực sự nói đến quyền truy cập/tự do, không chỉ bày tỏ ủng hộ.

## 7. Mẫu bảng đầu ra/đối soát

Mỗi narrative được xác nhận nên có tối thiểu các trường:

| Trường | Ý nghĩa |
|---|---|
| `narrative_id` | ID ổn định, ví dụ `NN_ISP_POWER` |
| `display_name_en` | Tên theo mẫu ở mục 6 |
| `definition_vi` | Một câu claim bằng tiếng Việt |
| `frame` / `stance_target` | Khung diễn giải / đối tượng của lập trường (nếu xác định được) |
| `evidence_post_ids` | ID các bài làm bằng chứng, không chỉ ví dụ gần centroid |
| `counterexamples_post_ids` | Bài cùng chủ đề nhưng không thuộc narrative |
| `status` | `PROVISIONAL`, `ACCEPT`, `REVISE`, `REJECT` |
| `review_note` | Lý do giữ, sửa, gộp hoặc loại |
| `version` | Phiên bản định nghĩa/nội dung |

Ở mức bài đăng: giữ `post_id`, `publish_datetime_utc`, `cluster_id` (nếu có), `narrative_id` hoặc `UNRESOLVED`, `evidence_span`/`claim_summary`, và `review_status`.

## 8. Quy tắc theo dõi thời gian và giới hạn

- Sử dụng UTC và **cửa sổ thời gian thống nhất** (ví dụ tuần); báo cáo **counts và tỷ lệ** vì tổng lượng bài từng kỳ thay đổi.
- Bảo toàn cả narrative ít bài và nhãn `UNRESOLVED`/`TOPIC_ONLY`; **không ép toàn bộ bài ngách vào các narrative**.
- Với bài gồm nhiều narrative, định nghĩa rõ cách đếm: đếm `primary` một lần trong bảng chính; multi-label báo cáo riêng.
- **Tránh lookahead bias:** với dự báo tại thời điểm T, không dùng văn bản, thống kê, taxonomy tinh chỉnh hoặc bằng chứng từ tương lai để đưa ra quyết định tại T. Việc rà soát toàn kỳ hiện tại chỉ phục vụ **khám phá hồi cứu**, không tự chứng minh kết quả dự báo ngoài mẫu.
- POST và COMMENT nếu đưa vào chung phải có chính sách đếm/phân tầng rõ. Bộ embedding hiện có chỉ bao phủ POST; không ngầm suy rằng COMMENT đã được embedding.

## 9. Hướng dẫn sử dụng cho nhóm

1. **Phước / ND3.4b–5:** xuất cluster IDs và các ví dụ có tính đa dạng; đối chiếu bảng tiêu chí mục 3–4; báo cáo cụm thuần, trộn hoặc nhiễu. Không buộc mỗi cluster có đúng một narrative.
2. **Phương / ND3.6:** duyệt `narrative_id`, `display_name_en`, `definition_vi`, trường hợp chồng lấn; cập nhật bảng tên và version khi nhận mẫu thực tế.
3. **Nhóm báo cáo/demo:** chỉ gọi narrative **đã xác nhận** khi có evidence và quyết định review; các tên tại mục 5 hiện là giả thuyết.

## 10. Điều kiện nghiệm thu ND3.6

- [x] Có định nghĩa thao tác phân biệt **topic – stance – frame – narrative**.
- [x] Có tiêu chí nhận diện có thể thực hành và xử lý trường hợp mơ hồ/nhiễu.
- [x] Có ví dụ đúng/sai, phân biệt rõ ví dụ minh họa với dữ liệu Reddit thật.
- [x] Có quy tắc đặt tên, stable ID và hợp đồng bằng chứng cho báo cáo/demo.
- [x] Có quy định về diễn biến thời gian và tránh dùng tương lai trong dự báo.
- [ ] **Chưa thực hiện trong Task 6:** thẩm định nhãn từ cluster và POST thực tế (thuộc phối hợp bước clustering/review sau đó).

**Tài liệu nội bộ tham chiếu:** `docs/domain_scoping.md`, `configs/selected_scope.json`, `docs/embedding_storage.md`, bảng phân công NarrativeCast Nội dung 3. Các ví dụ/nhãn giả thuyết trong tài liệu do người soạn đề xuất, không phải số liệu kết quả thực nghiệm.
