# ND3.4a — Khám phá chủ đề sơ bộ và thu hẹp phạm vi

Trạng thái: **PASS**. Run: `scope_v1`. Chọn duy nhất: **Chính sách trung lập mạng Internet**.

## Căn cứ và tái sử dụng
Notebook EDA cũ: `notebooks/03_content_subdomain_eda.ipynb`; báo cáo `outputs/reports/content_eda_summary.md`.
Dùng dữ liệu sạch sẵn có: `/content/drive/MyDrive/NarrativeCast/data/processed/reddit_technology_clean.parquet`; không chạy lại cleaning/embedding.
Khảo sát trên [2014-01-01, 2019-01-01) UTC.
Các ngưỡng là đề xuất nghiệm thu được lưu trước khi scan, không phải yêu cầu đã có sẵn trong Excel.

## Quyết định
Chọn chính sách trung lập mạng Internet. Khảo sát có 98.261 bản ghi, 260 tuần đầy đủ, dense_week_fraction=0.996154, weekly_p10=40.9, weekly_median=228.5 và không có tuần trống. Review AI hỗ trợ: 90/100 mẫu positive đúng phạm vi; 3/30 mẫu near_miss còn liên quan. Có 5 framing sơ bộ, mỗi framing có ít nhất 3 văn bản khác nhau: cạnh tranh công bằng, tự do biểu đạt, đảng phái và bầu cử, huy động cộng đồng, và phản đối quy định bị coi là không cần thiết. Ví dụ có tranh luận về fast lanes, chặn website và gây sức ép lên đại diện dân cử. Bốn dòng NetNeutralityBot không được dùng làm bằng chứng framing. Big Tech không đạt precision; digital privacy không đạt near_miss audit. Các framing chưa phải nhãn narrative chuẩn. Đây là review AI hỗ trợ, không phải chú thích độc lập của hai người.

## Số liệu đo trong lần chạy này
- 98,261 bản ghi; POST=6,538, COMMENT=91,723.
- 260 tuần đầy đủ; tỷ lệ tuần đạt mật độ: 99.6%.
- Mẫu khớp: 100 dòng; tỷ lệ đúng phạm vi: 90.0%.
- Mẫu gần nhưng không khớp: 30 dòng; tỷ lệ còn liên quan: 10.0%. Không phải recall toàn tập.
- 5 giả thuyết mạch diễn giải đủ bằng chứng sơ bộ; chưa phải nhãn narrative được nghiệm thu.

## Phạm vi và hợp đồng bàn giao
Quy tắc lọc chính xác ở `configs/selected_scope.json`; chỉ một ngách được chọn.
POST/COMMENT được xét theo nội dung riêng; không tự suy comment thuộc ngách từ parent_id thiếu.
Không thay đổi tiền xử lý đã hoàn thành; không tạo lại vector ở task này.
Số lượng là các bản ghi khớp từ khóa, không phải ground truth số bài liên quan.

## Giới hạn và trạng thái bước sau
Review từ khóa có thể bỏ sót diễn đạt khác; framing hints do người đọc gán cần được rà soát về ý nghĩa.
EDA toàn kỳ đã được xem trước đó; không tuyên bố tập test tương lai hoàn toàn chưa từng được khảo sát.
Kiểm tra ID trong task này không thay thế kiểm tra unique ID toàn tập của ND2.
Chưa phân cụm, chưa đánh giá cụm, chưa dự báo. Dừng tại nghiệm thu ND3.4a.

## Bằng chứng nội bộ
`data/interim/task4/scope_v1/acceptance.json`, `manifest.json`, `domain_review.csv`, `framing_evidence.csv`.
Không commit văn bản Reddit hoặc toàn bộ review mẫu khi chưa kiểm tra quyền chia sẻ dữ liệu.
