# Báo cáo Đánh giá Chất lượng Dữ liệu Sau Làm sạch (Data Quality Notes)

Tài liệu này tổng hợp và giải thích các biến động dữ liệu trước và sau quá trình làm sạch đối với bộ dữ liệu **MADOC (Multi-Platform Aggregated Dataset of Online Communities)** phục vụ dự án NarrativeCast.

---

## 1. So sánh Tóm tắt Trạng thái Dữ liệu

- **Tổng bài đăng (Posts):** Giảm từ **19.85M** xuống **18.90M** (~4.8% bị loại bỏ do nội dung rỗng, spam, hoặc bài bị xóa).
- **Tổng bình luận (Comments):** Giảm từ **246.21M** xuống **236.00M** (~4.15% bị loại bỏ, chủ yếu là các chuỗi nhãn `[deleted]` và `[removed]` trên Reddit/Voat).
- **Người dùng duy nhất (Unique Users):** Giảm từ **24.50M** xuống **23.10M** sau khi chuẩn hóa định dạng ID và hợp nhất các tài khoản ảo/bot rác.

---

## 2. Giải thích Biến động & Quy tắc Tiền xử lý

### a. Nguyên nhân biến động số lượng bản ghi
1. **Lọc văn bản không hợp lệ:** Loại bỏ các bài viết có độ dài dưới 3 ký tự, chứa toàn ký tự đặc biệt hoặc chỉ chứa đường dẫn URL không kèm văn bản mô tả.
2. **Loại bỏ nội dung rác/hệ thống:** Loại bỏ các bài đăng tự động từ bot hệ thống và các phản hồi bị người dùng hoặc quản trị viên xóa (`[deleted]`, `[removed]`).
3. **Lọc trùng lặp (Deduplication):** Loại bỏ các bản ghi trùng lặp tuyệt đối về `post_id` do lỗi thu thập dữ liệu nhiều lần (multi-scraping).

### b. Xử lý giá trị khuyết thiếu (Missing Values)
- **`community` (Cộng đồng):** Tỷ lệ thiếu 100% đối với nền tảng Bluesky và Koo là **hợp lệ kỹ thuật**, do kiến trúc mạng xã hội này không chia phân cấp sub-community như Reddit (subreddit) hay Voat (subverse).
- **`parent_id` & `parent_user_id`:** Tỷ lệ thiếu 100% đối với bài đăng gốc (`POST`) là **hợp lệ**, chỉ các bình luận (`COMMENT`) mới chứa thông tin liên kết này.

---

## 3. Hạn chế Còn tồn tại & Khuyến nghị

1. **Lệch phân bố nền tảng (Platform Imbalance):** Dữ liệu Reddit chiếm tỷ trọng áp đảo (>85% tổng số bản ghi), cần áp dụng kỹ thuật lấy mẫu (sampling) hoặc trọng số khi phân tích so sánh liên nền tảng.
2. **Định dạng thời gian:** Tồn tại một số nhỏ bản ghi lệch thời gian (time drift) do lệch múi giờ giữa các máy chủ thu thập. Khuyến nghị luôn dùng chuẩn UTC / Unix Timestamp khi truy vấn.
3. **Độ tin cậy điểm Sentiment:** Điểm VADER Sentiment được gán mặc định tốt nhất cho tiếng Anh; các bài đăng bằng ngôn ngữ khác cần tiền xử lý bổ sung bằng mô hình đa ngôn ngữ (như mBERT/XLM-RoBERTa).
