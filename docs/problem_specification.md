# NarrativeCast — Problem Specification

## 1. Problem Statement

NarrativeCast hướng đến bài toán dự đoán những *narratives* trên mạng xã hội có khả năng trở nên nổi bật trong tương lai, dựa trên nội dung và diễn biến hoạt động trong quá khứ.

Vấn đề cốt lõi là mức độ phổ biến hiện tại chưa chắc phản ánh xu hướng tương lai. Vì vậy, nghiên cứu xem xét liệu việc kết hợp thông tin ngữ nghĩa và thông tin thời gian có giúp dự đoán tốt hơn so với các phương pháp dựa trên mức độ phổ biến hoặc tính liên tục của xu hướng hiện tại hay không.

## 2. Object of Study

- **Đối tượng theo dõi:** Các narratives được hình thành từ những bài đăng có nội dung liên quan trên mạng xã hội.
- **Phạm vi dữ liệu chính:** Bài đăng Reddit thuộc cộng đồng Technology.
- **Đơn vị phân tích:** Narrative trong từng cửa sổ thời gian.
- **Khái niệm narrative:** Một mạch diễn giải, quan điểm hoặc thông điệp chung được thể hiện qua nhiều bài đăng liên quan. Narrative không đồng nghĩa với topic; một topic có thể bao gồm nhiều narratives khác nhau.

*Ví dụ minh họa:* Topic “AI trong giáo dục” có thể chứa narrative “AI hỗ trợ cá nhân hóa việc học” và narrative “AI khiến người học phụ thuộc vào công nghệ”.

## 3. Input and Output

### Input

- Nội dung văn bản của bài đăng.
- Thời điểm đăng bài.
- Các bài đăng được sắp xếp theo những cửa sổ thời gian liên tiếp.
- Các đặc trưng ngữ nghĩa và hoạt động lịch sử được trích xuất từ dữ liệu quá khứ.

### Output

- Danh sách narratives được dự đoán có khả năng nổi bật trong tương lai.
- Điểm dự đoán hoặc thứ hạng của từng narrative.
- Khoảng dự đoán tương lai: 3, 7 hoặc 14 ngày.

Cách biểu diễn narrative, cách tính điểm dự đoán và ngưỡng xác định một narrative là “nổi bật” sẽ được cụ thể hóa trong phần phương pháp và thiết kế thực nghiệm.

## 4. Detection vs. Prediction

| Nội dung | Phát hiện narrative hiện tại | Dự đoán narrative tương lai |
|---|---|---|
| Mục tiêu | Xác định các narratives đang xuất hiện trong dữ liệu đã quan sát | Ước lượng narratives nào sẽ nổi bật trong khoảng thời gian tiếp theo |
| Dữ liệu sử dụng | Bài đăng trong cửa sổ hiện tại và quá khứ | Chỉ thông tin có sẵn trước thời điểm dự đoán |
| Kết quả | Các nhóm nội dung/narratives hiện tại | Danh sách hoặc thứ hạng narratives dự kiến nổi bật |
| Đánh giá | Chất lượng nhận diện và biểu diễn narrative | Mức độ chính xác của dự đoán trên dữ liệu tương lai |

Dự đoán phải sử dụng thông tin có trước thời điểm dự đoán. Dữ liệu trong cửa sổ tương lai chỉ được dùng để xác định mục tiêu đánh giá, không được đưa vào đặc trưng đầu vào.

## 5. Operational Definition of Narrative Prominence

### 5.1. Narrative Assignment

Một *narrative* \(n\) là một nhóm bài đăng thể hiện cùng một mạch diễn giải hoặc thông điệp về một chủ đề. Mỗi bài đăng được gán vào một narrative theo phương pháp nhận diện đã xác định trong phần Methodology. Những bài đăng không thuộc narrative nào được xác định sẽ không được tính vào số bài đăng của narrative đó.

Để phép đo có ý nghĩa theo thời gian, cách nhận diện và gán nhãn narrative cần được duy trì nhất quán giữa cửa sổ lịch sử và cửa sổ đánh giá tương lai.

### 5.2. Prominence Score

Gọi \(W_t\) là cửa sổ thời gian dùng để dự đoán, \(N_{n,t}\) là số bài đăng trong \(W_t\) được gán cho narrative \(n\), và \(N_t\) là tổng số bài đăng hợp lệ được quan sát trong cùng cửa sổ.

Mức độ nổi bật của narrative \(n\) trong cửa sổ \(W_t\) được định nghĩa là:

\[
P(n,W_t)=\frac{N_{n,t}}{N_t}
\]

Trong đó:

- \(N_{n,t}\): số bài đăng hợp lệ được gán cho narrative \(n\) trong cửa sổ \(W_t\).
- \(N_t\): tổng số bài đăng hợp lệ trong cửa sổ \(W_t\), bao gồm cả bài đăng không được gán vào narrative đang xét.
- \(P(n,W_t)\): tỷ lệ bài đăng của narrative \(n\) trong toàn bộ bài đăng hợp lệ của cửa sổ, có giá trị từ 0 đến 1.

Ví dụ, nếu một cửa sổ có 10.000 bài đăng hợp lệ và 800 bài được gán cho narrative \(n\), thì \(P(n,W_t)=0{,}08\), tức narrative này chiếm 8% số bài đăng trong cửa sổ đó.

### 5.3. Prediction Target

Tại thời điểm dự đoán \(t\), hệ thống chỉ được sử dụng dữ liệu có thời điểm không muộn hơn \(t\). Với khoảng dự đoán \(h\in\{3,7,14\}\) ngày, hệ thống dự đoán mức độ nổi bật của narrative trong cửa sổ tương lai \((t,t+h]\).

Giá trị mục tiêu dùng để đánh giá được tính từ dữ liệu thực tế trong cửa sổ tương lai:

\[
y_{n,t,h}=P(n,(t,t+h])
\]

Giá trị này chỉ được dùng làm *ground truth* khi đánh giá, không được sử dụng làm đầu vào của dự đoán tại thời điểm \(t\).

### 5.4. Meaning and Limitation

Trong nghiên cứu này, *prominence* biểu thị **tỷ trọng thảo luận dành cho một narrative trong một khoảng thời gian xác định**, không đồng nghĩa với số lượng người ủng hộ, mức độ lan truyền giữa người dùng, độ tích cực của quan điểm hay ảnh hưởng xã hội.

Định nghĩa trên đo lường mức độ hiện diện tương đối của narrative. Việc xác định một narrative có được xem là “nổi bật” theo nghĩa phân loại nhị phân hay không, chẳng hạn thuộc nhóm top-\(K\), cần được quy định riêng trong thiết kế thực nghiệm.

## 6. Research Objective and Questions

**Mục tiêu:** Đánh giá liệu việc kết hợp biểu diễn ngữ nghĩa và thông tin thời gian có cải thiện khả năng dự đoán narratives nổi bật trong tương lai so với các baseline đơn giản hay không.

- **RQ1:** Phương pháp kết hợp thông tin ngữ nghĩa và thời gian có dự đoán tốt hơn các baseline hay không?
- **RQ2:** Hiệu quả dự đoán thay đổi như thế nào giữa các khoảng thời gian 3, 7 và 14 ngày?
- **RQ3:** Đặc điểm nội dung và hoạt động lịch sử nào có liên quan đến khả năng dự đoán thành công hoặc thất bại?

## 7. Baselines

- **Popularity baseline:** Xếp hạng narratives dựa trên số lượng bài đăng trong cửa sổ hiện tại.
- **Persistence baseline:** Giả định thứ hạng mức độ nổi bật hiện tại tiếp tục duy trì trong cửa sổ tương lai.

Các baseline được sử dụng làm mốc so sánh để xác định giá trị bổ sung của phương pháp đề xuất.

## 8. Intended Use

NarrativeCast hướng đến hỗ trợ nhà nghiên cứu hoặc người phân tích theo dõi diễn biến thảo luận trên mạng xã hội, nhận diện narratives có khả năng nổi bật trong tương lai và ưu tiên các nội dung cần quan sát thêm.

Hệ thống không nhằm khẳng định chắc chắn một narrative sẽ xuất hiện, cũng không tự suy ra quan hệ nhân quả hoặc mức độ ảnh hưởng xã hội chỉ từ tần suất bài đăng.

## 9. Scope and Contributions

Nghiên cứu chính tập trung vào Reddit Technology để giới hạn phạm vi và đánh giá bài toán trong một cộng đồng cụ thể.

Các đóng góp dự kiến:

1. Đặc tả bài toán dự đoán mức độ nổi bật của narratives theo thời gian.
2. Xây dựng quy trình kết hợp thông tin ngữ nghĩa và diễn biến lịch sử để dự đoán narratives tương lai.
3. Đánh giá phương pháp bằng cách so sánh với các baseline tại nhiều khoảng thời gian dự đoán.

Đây là các đóng góp dự kiến; mức độ khả thi và kết quả cụ thể sẽ được xác nhận qua thực nghiệm.