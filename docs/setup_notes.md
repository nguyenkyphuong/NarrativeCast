# Hướng dẫn Chạy thử Dữ liệu Mẫu & Thiết lập Môi trường (Setup Notes)
Tài liệu này hướng dẫn cách đọc dữ liệu mẫu `data/samples/posts_sample.csv`, môi trường Python cần thiết và các lưu ý tái lập cho nhóm nghiên cứu dự án NarrativeCast.
---
## 1. Môi trường & Thư viện yêu cầu
- **Python Version:** `>= 3.8`
- **Cài đặt thư viện:**
```bash
pip install pandas pyarrow vaderSentiment
```
---
## 2. Cách đọc dữ liệu mẫu (Code Python)
Các thành viên có thể chạy đoạn code mẫu sau để đọc và kiểm tra dữ liệu:
```python
import pandas as pd

# Đọc file dữ liệu mẫu CSV
df_sample = pd.read_csv('data/samples/posts_sample.csv', keep_default_na=True)

# Hiển thị thông tin tổng quan
print("--- THÔNG TIN CẤU TRÚC DỮ LIỆU ---")
print(df_sample.info())

print("\n--- 5 BẢN GHI MẪU ---")
print(df_sample.head())

# Kiểm tra phân bố nền tảng
print("\n--- THỐNG KÊ THEO NỀN TẢNG ---")
print(df_sample['platform'].value_counts())
```
---
## 3. Các trường hợp bất thường & Lưu ý kỹ thuật
1. **Xử lý giá trị rỗng (`NA`):**
- Các trường `parent_id`, `parent_user_id`, `url`, và `community` chứa giá trị `NA` đối với các bài đăng gốc (`POST`) hoặc các nền tảng không chia cộng đồng (như Bluesky, Koo).
- Khi dùng `pandas.read_csv()`, đảm bảo xử lý `NA` thành `NaN` phù hợp để tránh lỗi kiểu dữ liệu.
2. **Định dạng Thời gian (`publish_date`):**
- Lưu trữ dưới dạng Unix Timestamp (giây từ Epoch).
- Để chuyển sang dạng `datetime` trong Python:
```python
df_sample['datetime'] = pd.to_datetime(df_sample['publish_date'], unit='s')
```
3. **Lưu ý về Bảo mật Dữ liệu (Quy định bắt buộc):**
- **Tuyệt đối KHÔNG** đẩy các file dữ liệu gốc dạng nén (.parquet) dung lượng lớn lên GitHub.
- Chỉ sử dụng thư mục `data/samples/` để lưu trữ các mẫu dữ liệu nhỏ (`.csv` hoặc `.parquet` dưới 5MB) nhằm phục vụ kiểm thử code.
