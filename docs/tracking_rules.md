# NarrativeCast - Đặc tả theo dõi Narrative theo thứ tự thời gian

## Trạng thái và mục đích

Phạm vi áp dụng: Bộ máy theo dõi narrative theo thứ tự thời gian, không sử dụng dữ liệu tương lai (cấu hình theo `configs/forecast.yaml` và đầu vào `data/interim/narrative_assignments.parquet`).

Tài liệu này mô tả bộ máy theo dõi narrative theo thời gian không dùng dữ liệu tương lai trong `src/tracking/track_narratives.py`, gồm các tham số cấu hình, công thức tính toán, máy trạng thái vòng đời, quy ước giá trị `0` và `NaN` cho pipeline downstream, ranh giới chống data leakage, schema đầu ra và kết quả kiểm định qua `scripts/audit_tracking_registry.py` cùng `data/interim/narrative_registry_manifest.json`.

## Phạm vi dữ liệu, phân chia thời gian và lưới lịch UTC

### Định danh dữ liệu và phạm vi

- Phiên bản phạm vi dữ liệu: `expanded_v1` (`scope_status: PROVISIONAL_AI_ASSISTED`, `human_verified: false` trong `configs/forecast.yaml`).
- Miền dữ liệu và giai đoạn: Thảo luận về Net Neutrality (`data/D4/net_neutrality_scope_expanded_v1.csv`), giai đoạn `2014-2020`.
- Tổng số bài đăng (`POST`) trong phạm vi: `7,331` (`7,244` bài gốc cộng thêm `44` bài bổ sung theo từ khóa và `43` bài bổ sung theo ngữ nghĩa).
- Số bài đăng `ASSIGNED` được theo dõi: `6,411` bài trên `63` narrative và `20,841` bản ghi dòng thời gian.
- Các bài đăng hợp lệ không thuộc cụm nào (`NOISE` và `UNASSIGNED` có `is_usable_text == True`) vẫn được giữ lại trong mẫu số đếm theo tuần (`window_total_valid_posts`) để tỷ trọng nổi bật của các ứng viên không bị phóng đại sai lệch.

### Lưới lịch UTC

UTC là múi giờ duy nhất được sử dụng (`configs/forecast.yaml`). Mỗi cửa sổ thời gian là một khoảng nửa mở `[window_start, window_end)` dài 7 ngày từ Thứ Hai tuần này đến Thứ Hai tuần kế tiếp, neo tại mốc `00:00:00+00:00`.

Theo cấu hình trong manifest và kết quả kiểm định:

- `calendar_weeks_total`: `364` tuần lịch UTC, bao phủ từ `2014-01-06T00:00:00+00:00` đến `2020-12-28T00:00:00+00:00`.
- `warmup_weeks_without_candidates`: `2` tuần warmup đầu tiên (`2014-01-06` đến `2014-01-20`) trước cửa sổ `birth` đầu tiên của narrative (`2014-01-20` đến `2014-01-27`). Cả `2` tuần warmup này đều chứa bài đăng hợp lệ trong `data/interim/narrative_assignments.parquet` (`2` tuần warmup không rỗng) nhưng có `0` bài `ASSIGNED` và `0` ứng viên vì bước phát hiện ứng viên ở upstream chưa có đủ quan sát lịch sử để fit mô hình.
- `active_registry_windows`: `362` tuần registry (`2014-01-20T00:00:00+00:00` đến `2020-12-28T00:00:00+00:00`), chia thành:
  - `339` tuần hoạt động không rỗng (`window_total_valid_posts > 0`).
  - `23` tuần lịch rỗng (`window_total_valid_posts == 0`).
- Đối soát tổng số tuần: `339` tuần hoạt động không rỗng + `2` tuần warmup không rỗng = `341` tuần không rỗng (khớp đúng với số cặp `(window_start, window_end)` phân biệt trong `narrative_assignments.parquet`), và `341` tuần không rỗng + `23` tuần rỗng = `364` tuần lịch tổng cộng.

### Phân chia tập dữ liệu theo thời gian

Các phân vùng đánh giá theo trình tự thời gian được khóa trong `configs/forecast.yaml`:

- Tập huấn luyện (`Train`): `2014-2017` (`train_years: [2014, 2015, 2016, 2017]`).
- Tập kiểm định (`Validation`): `2018` (`validation_years: [2018]`).
- Tập kiểm thử (`Test`): `2019-2020` (`test_years: [2019, 2020]`).
- Chính sách phân chia: `shuffle: false`, `fit_future_data: false`, `rolling_origin_past_only: true`.

## Kiến trúc tracking hai chế độ

Lớp `NarrativeTracker` trong `src/tracking/track_narratives.py` hỗ trợ hai chế độ chạy riêng biệt thông qua `TrackingConfig.tracking_mode`. Các module downstream cần phân biệt rõ cơ chế gán ID giữa hai chế độ này.

### Chế độ 1 (mặc định): `registry_from_assignments`

Đây là chế độ vận hành chính thức được ghi nhận trong `data/interim/narrative_registry_manifest.json`.

- **Lọc và gom nhóm đầu vào**: Chọn các dòng thỏa điều kiện `(assignment_status == "ASSIGNED") & is_usable_text` và loại bỏ các giá trị `narrative_id` thuộc `("NOISE", "UNASSIGNED")`. Các batch được gom nhóm theo trình tự thời gian dựa trên `("window_start", "window_end", "narrative_id")` với `sort=True`.
- **Kế thừa định danh**: Không gọi `match_clusters_to_registry()`. Kế thừa trực tiếp các mã định danh ổn định từ upstream (`CAND_XXXX`) trong cột `narrative_id` đầu vào.
- **Vai trò của registry**: Ghi nhận narrative theo đúng thứ tự thời gian (mỗi narrative chỉ bắt đầu xuất hiện từ cửa sổ `birth` của nó trở đi), cập nhật vòng đời 5 trạng thái, tính batch centroid L2-normalized, cập nhật EMA centroid, theo dõi `centroid_drift`, `similarity_to_previous_centroid` và `intra_window_dispersion`, kiểm tra dấu hiệu tách/gộp trong cùng tuần (`_diagnose_split_merge`), và lưu các snapshot centroid theo từng cutoff.
- **Chẩn đoán theo cặp trong cùng cửa sổ**: Với mọi cặp narrative đang hoạt động `(id_a, id_b)` trong cùng một tuần (tam giác trên `i < j` của danh sách `active_in_window` đã sắp xếp), nếu cosine similarity `>= 0.75` (`split_merge_threshold`) thì gán cờ `SplitMergeFlag.MERGE_CANDIDATE` cho cả hai narrative và ghi sự kiện `potential_merge_or_split`; nếu cosine similarity thuộc `[0.60, 0.75)` thì ghi sự kiện `unlinked_near_miss`.

### Chế độ 2 (liên kết lại độc lập): `relink_local_clusters`

Chế độ này hỗ trợ liên kết lại độc lập các cụm cục bộ hàng tuần qua các cửa sổ liên tiếp.

- **Lọc và gom nhóm đầu vào**: Yêu cầu cột `local_cluster_id` có mặt trong `assignments_df` (ném `KeyError` nếu thiếu), lọc `(assignment_status == "ASSIGNED") & is_usable_text` với `local_cluster_id.astype(int) != -1`, và gom nhóm theo `("window_start", "window_end", "local_cluster_id")` với `sort=True`.
- **Khớp greedy 1-to-1 theo cosine similarity (`match_clusters_to_registry`)**:
  - Tính ma trận cosine similarity giữa các batch centroid L2-normalized của cụm cục bộ ở cửa sổ hiện tại với các vector `current_centroid` L2-normalized của toàn bộ thực thể đang có trong `self.registry`.
  - Các cạnh ứng viên phải thỏa điều kiện `similarity >= alignment_threshold` (`0.85`).
  - Các ứng viên được sắp xếp theo quy tắc ưu tiên 4 mức ổn định (`highest_sim_then_oldest_then_id`):
    1. `similarity` giảm dần (`-sim`)
    2. `first_seen_utc` tăng dần (ưu tiên narrative xuất hiện sớm nhất)
    3. `lifetime_post_count` giảm dần (ưu tiên tổng số bài tích lũy cao nhất)
    4. `narrative_id` tăng dần theo thứ tự từ điển
       (dùng chỉ số cụm hiện tại làm tiêu chí phụ cuối cùng để kết quả luôn ổn định).
  - Các cạnh được chấp nhận theo thuật toán greedy 1-to-1 (mỗi cụm hiện tại và mỗi thực thể trong registry chỉ được ghép tối đa một lần).
  - Các cụm cục bộ chưa ghép được sẽ được cấp mã định danh tuần tự mới theo định dạng `f"{id_prefix}_{next_id_num:04d}"` (ví dụ `CAND_0001`).
- **Gắn cờ ứng viên tách/gộp tại ngưỡng `split_merge_threshold = 0.75`**:
  - **Chẩn đoán gộp**: Nếu một cụm hiện tại có `sim >= 0.75` với nhiều hơn một thực thể đã có trong registry (`len(high) > 1`), cụm đó được gắn cờ `MERGE_CANDIDATE` và ghi sự kiện kiểm định `potential_merge`.
  - **Chẩn đoán tách**: Nếu một thực thể trong registry có `sim >= 0.75` với nhiều hơn một cụm hiện tại (`len(high) > 1`), toàn bộ các cụm hiện tại liên quan sẽ nhận cờ `SPLIT_CANDIDATE` thông qua `_combine_split_merge_flags()`, nâng cấp thành `SPLIT_AND_MERGE` nếu cụm đó đồng thời là `MERGE_CANDIDATE`, và ghi sự kiện kiểm định `potential_split`.
  - Trong `track_all_windows()`, cờ từ bộ khớp liên cửa sổ được hợp nhất với cờ từ `_diagnose_split_merge()` trong cùng cửa sổ thông qua `_combine_split_merge_flags()`.
- **Ghi nhận vùng xám `[0.60, 0.85)` không bỏ sót và không ghi trùng**:
  - Mọi cặp `(current_cluster, registry_entity)` tham gia vào sự kiện `potential_merge` hoặc `potential_split` đều được đưa vào `flagged_pairs`.
  - Bất kỳ cặp nào có `0.60 <= sim < alignment_threshold` (`0.85`) mà không thuộc `flagged_pairs` đều được ghi nhận là `unlinked_near_miss`.
  - Nhờ vậy, cả các cặp trong `[0.60, 0.75)` lẫn các cặp near-miss 1-1 chưa gắn cờ trong `[0.75, 0.85)` đều được ghi lại đầy đủ, đồng thời tránh ghi trùng các cặp đã nằm trong nhóm cờ tách/gộp.

### Cấu hình siêu tham số được khóa (`TrackingConfig`)

| Tham số                   | Giá trị                             | Mô tả                                                                        |
| :------------------------ | :---------------------------------- | :--------------------------------------------------------------------------- |
| `tracking_mode`           | `"registry_from_assignments"`       | Chế độ tracking mặc định trong cấu hình chạy chuẩn và manifest               |
| `alignment_threshold`     | `0.85`                              | Ngưỡng cosine similarity tối thiểu cho phép khớp greedy 1-to-1               |
| `ema_alpha`               | `0.30`                              | Trọng số trung bình trượt hàm mũ áp dụng lên batch centroid hiện tại         |
| `dormant_threshold_weeks` | `12`                                | Ngưỡng số tuần im lặng liên tiếp để chuyển từ `inactive` sang `dormant`      |
| `split_merge_threshold`   | `0.75`                              | Ngưỡng cosine similarity để gắn cờ ứng viên tách hoặc gộp                    |
| `tie_break_strategy`      | `"highest_sim_then_oldest_then_id"` | Quy tắc ưu tiên 4 mức tái lập được cho `match_clusters_to_registry`          |
| `id_prefix`               | `"CAND"`                            | Tiền tố cho các mã định danh narrative mới được cấp                          |
| `embedding_dim`           | `384`                               | Số chiều kỳ vọng của vector bài đăng và vector centroid                      |
| `random_state`            | `42`                                | Seed tái lập cho bước lấy mẫu phân tầng kiểm tra chất lượng (QA)             |
| `calendar_start_utc`      | `"2014-01-06"`                      | Mốc bắt đầu (bao gồm) của tuần lịch UTC đầu tiên                             |
| `calendar_end_utc`        | `"2020-12-28T00:00:00+00:00"`       | Cận trên loại trừ của lưới lịch trong `TrackingConfig`                       |

## Định nghĩa toán học và quy ước giá trị `0` vs `NaN`

### Chuẩn hóa L2 bắt buộc và kiểm tra lỗi sớm (fail-fast)

- Không tự động thay thế vector lỗi bằng vector 0; hàm sẽ báo lỗi ValueError ngay khi gặp đầu vào không hợp lệ.
- `_l2_normalize_rows()` yêu cầu ma trận 2D hữu hạn, không rỗng và mọi hàng có L2 norm `> 1e-12`; nếu vi phạm sẽ ném `ValueError`.
- `load_embeddings_lookup()` lọc các vector lưu trữ bằng điều kiện `is_valid & np.all(np.isfinite(vectors), axis=1) & (norms > 1e-12)` và ném `ValueError` nếu bất kỳ bài đăng `ASSIGNED` bắt buộc nào thiếu vector hợp lệ.
- `NarrativeTracker.track_all_windows()` kiểm tra lại độc lập từng embedding của mọi bài `ASSIGNED` hợp lệ (`ndim == 1`, không rỗng, hữu hạn, `norm > 1e-12`) trước khi xử lý các cửa sổ thời gian.
- `compute_normalized_centroid()` và bước cập nhật EMA cũng ném `ValueError` nếu trung bình hoặc vector sau cập nhật có norm không hữu hạn hoặc `<= 1e-12`.

### Tính tái lập chính xác (bitwise deterministic) khi hoán vị thứ tự dòng

Trong `_pregroup_assigned_batches()`, danh sách `post_id` bên trong mỗi batch `(window_start, window_end, group_col)` được sắp xếp theo thứ tự từ điển (`sorted(grp["post_id"].tolist())`). Cả `compute_normalized_centroid()` và `_compute_intra_window_dispersion()` đều tích lũy tổng và trung bình bằng `float64` theo đúng thứ tự bài đăng đã sắp xếp trước khi ép kiểu kết quả đầu ra sang `float32`. Cách này giúp kết quả không đổi từng bit ngay cả khi thứ tự dòng trong tệp Parquet đầu vào bị xáo trộn.

### Các công thức cốt lõi

Gọi $B_t$ là danh sách đã sắp xếp gồm $n_t = |B_t| \ge 1$ bài đăng `ASSIGNED` hợp lệ của narrative $c$ trong cửa sổ $t$, với các vector embedding L2-normalized $\hat{\mathbf{v}}_i \in \mathbb{R}^{384}$:

1. **Batch centroid $\mathbf{b}_t$**:

   $$\bar{\mathbf{v}}_t = \frac{1}{n_t} \sum_{i \in B_t} \hat{\mathbf{v}}_i, \qquad \mathbf{b}_t = \frac{\bar{\mathbf{v}}_t}{\|\bar{\mathbf{v}}_t\|_2}$$

   (tích lũy $\bar{\mathbf{v}}_t$ ở `float64`, lưu $\mathbf{b}_t$ dưới dạng `float32`).

2. **EMA centroid $\mathbf{c}_t$ ($\alpha = 0.30$)**:
   - Tại trạng thái `birth`: $\mathbf{c}_t = \mathbf{b}_t$.
   - Tại các cửa sổ `active` hoặc `reactivated`:

     $$\tilde{\mathbf{c}}_t = (1 - \alpha)\,\mathbf{c}_{t-1} + \alpha\,\mathbf{b}_t, \qquad \mathbf{c}_t = \frac{\tilde{\mathbf{c}}_t}{\|\tilde{\mathbf{c}}_t\|_2}$$

     (lưu $\mathbf{c}_t$ dưới dạng `float32`).

   - Tại các cửa sổ im lặng (`inactive` hoặc `dormant`): $\mathbf{c}_t = \mathbf{c}_{t-1}$ (đóng băng).

3. **Độ tương đồng với centroid trước đó (`similarity_to_previous_centroid`)**:
   - Tại các cửa sổ `active` hoặc `reactivated`: cosine similarity giữa EMA centroid trước đó $\mathbf{c}_{t-1}$ và batch centroid thô hiện tại $\mathbf{b}_t$:

     $$s_t = \operatorname{clip}(\mathbf{c}_{t-1} \cdot \mathbf{b}_t,\,-1.0,\,1.0)$$

     (Trong chế độ `relink_local_clusters`, sử dụng giá trị cosine similarity đã tính trước từ `match_clusters_to_registry()`.)

   - Tại trạng thái `birth` (`age_weeks == 0`): gán giá trị quy ước `1.0`.
   - Tại các cửa sổ im lặng (`current_window_posts == 0`): `NaN`.

4. **Độ dịch chuyển centroid (`centroid_drift`)**:
   - Tại các cửa sổ `active` hoặc `reactivated`: độ lệch cosine giữa EMA centroid trước đó $\mathbf{c}_{t-1}$ và EMA centroid sau cập nhật $\mathbf{c}_t$:

     $$d_t = 1.0 - \operatorname{clip}(\mathbf{c}_{t-1} \cdot \mathbf{c}_t,\,-1.0,\,1.0)$$

   - Tại trạng thái `birth` và các cửa sổ im lặng (`inactive` hoặc `dormant`): `0.0`.

5. **Độ phân tán nội cửa sổ (`intra_window_dispersion`)**:
   - Khi $n_t \ge 2$:

     $$\text{dispersion}_t = 1.0 - \frac{1}{n_t} \sum_{i \in B_t} \operatorname{clip}(\hat{\mathbf{v}}_i \cdot \hat{\mathbf{b}}_t,\,-1.0,\,1.0)$$

     (tính trung bình ở `float64`, ép kiểu kết quả sang `float32`).

   - Khi $n_t = 1$: `0.0`.
   - Khi $n_t = 0$: `NaN`.

### Quy ước giá trị `0` vs `NaN` cho pipeline downstream

Các pipeline tạo feature ở downstream cần phân biệt rõ giữa giá trị thiếu hoặc không xác định (`NaN`) với các giá trị bằng không hợp lệ như số bài bằng 0 hoặc dispersion bằng `0.0`. Không dùng `df.fillna(0)` đồng loạt lên toàn bộ bảng registry.

| Cột                               | `window_total_valid_posts == 0` (23 tuần rỗng) | `window_total_valid_posts > 0` và `current_window_posts == 0` (im lặng) | `birth` (`age_weeks == 0`, $n_t \ge 1$)        | `active` / `reactivated` ($n_t = 1$) | `active` / `reactivated` ($n_t \ge 2$) |
| :-------------------------------- | :--------------------------------------------- | :---------------------------------------------------------------------- | :--------------------------------------------- | :----------------------------------- | :------------------------------------- |
| `window_total_valid_posts`        | `0`                                            | `> 0` (`ASSIGNED + NOISE + UNASSIGNED` hợp lệ)                          | `>= n_t`                                       | `>= 1`                               | `>= 2`                                 |
| `current_window_posts`            | `0`                                            | `0`                                                                     | `n_t >= 1`                                     | `1`                                  | `n_t >= 2`                             |
| `current_window_share`            | `NaN` (mẫu số không xác định)                  | `0.0`                                                                   | `n_t / window_total_valid_posts`               | `1 / window_total_valid_posts`       | `n_t / window_total_valid_posts`       |
| `intra_window_dispersion`         | `NaN`                                          | `NaN`                                                                   | `0.0` nếu `n_t == 1`, ngược lại theo công thức | `0.0`                                | theo công thức trong `[0.0, 2.0]`      |
| `similarity_to_previous_centroid` | `NaN`                                          | `NaN`                                                                   | `1.0` (giá trị quy ước)                        | theo công thức trong `[-1.0, 1.0]`   | theo công thức trong `[-1.0, 1.0]`     |
| `centroid_drift`                  | `0.0` (centroid đóng băng)                     | `0.0` (centroid đóng băng)                                              | `0.0`                                          | theo công thức trong `[0.0, 2.0]`    | theo công thức trong `[0.0, 2.0]`      |

> **Lưu ý với trạng thái `BIRTH` trên cột `similarity_to_previous_centroid`**:
> Tại dòng `birth` (`lifecycle_status == "birth"`, `age_weeks == 0`), giá trị `similarity_to_previous_centroid = 1.0` và `centroid_drift = 0.0` chỉ là giá trị khởi tạo quy ước chứ không phải độ tương đồng đo giữa hai tuần. Vì vậy, bước tạo feature và huấn luyện mô hình cần tách riêng hoặc lọc kèm điều kiện `lifecycle_status != "birth"` (hoặc `age_weeks > 0`) để không nhầm lẫn narrative mới xuất hiện với narrative đã tồn tại ổn định.

- `14` cột bắt buộc của registry không được phép chứa giá trị null; chỉ `3` cột `current_window_share`, `intra_window_dispersion` và `similarity_to_previous_centroid` được phép chứa `NaN` theo đúng các điều kiện nêu trên.

## Máy trạng thái vòng đời

Mỗi `NarrativeEntity` chuyển trạng thái theo quy tắc cố định qua từng tuần UTC dựa trên 5 trạng thái trong `LifecycleStatus`:

| Trạng thái    | Điều kiện chuyển vào                                                                         | Hành vi centroid và các chỉ số                                                                                                                      | Bộ đếm tuần im lặng                                                                                | Sự kiện kiểm định được ghi                |
| :------------ | :------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- | :---------------------------------------- |
| `birth`       | Cửa sổ đầu tiên mà `nid` xuất hiện trong `self.registry` (`p_curr >= 1`)                     | `current_centroid = batch_centroid`; `centroid_drift = 0.0`; `similarity_to_previous_centroid = 1.0`; `age_weeks = 0`                               | `consecutive_silent_weeks = 0`; `max_consecutive_silent_weeks = 0`                                 | `birth`                                   |
| `active`      | `p_curr >= 1` và trạng thái trước thuộc `{birth, active, reactivated}`                       | Cập nhật EMA centroid (`alpha = 0.30`); `centroid_drift` và `similarity_to_previous_centroid` hữu hạn                                               | `consecutive_silent_weeks = 0`; `active_weeks_count += 1`                                          | `linked_active`                           |
| `inactive`    | `p_curr == 0` và `consecutive_silent_weeks` sau khi tăng `< 12`                              | `current_centroid` đóng băng; `centroid_drift = 0.0`; `similarity_to_previous_centroid = NaN`; `intra_window_dispersion = NaN`                      | `consecutive_silent_weeks += 1`; cập nhật `max_consecutive_silent_weeks`                           | Không có                                  |
| `dormant`     | `p_curr == 0` và `consecutive_silent_weeks` sau khi tăng `>= 12` (`dormant_threshold_weeks`) | `current_centroid` đóng băng; `centroid_drift = 0.0`; `similarity_to_previous_centroid = NaN`; `intra_window_dispersion = NaN`                      | `consecutive_silent_weeks += 1`; cập nhật `max_consecutive_silent_weeks`                           | Không có                                  |
| `reactivated` | `p_curr >= 1` và trạng thái trước thuộc `{inactive, dormant}`                                | Cập nhật EMA centroid (`alpha = 0.30`) đối chiếu với `current_centroid` đã đóng băng; `centroid_drift` và `similarity_to_previous_centroid` hữu hạn | `consecutive_silent_weeks = 0`; bảo toàn `max_consecutive_silent_weeks`; `active_weeks_count += 1` | `reactivation` (gồm `silent_weeks_prior`) |

Các điều kiện kiểm tra được giữ nguyên khi chuyển sang trạng thái im lặng (`_update_silent_entities`):

- Vector centroid được giữ đóng băng qua các tuần im lặng liên tiếp (`centroid_drift = 0.0`), tránh làm sai lệch phép đo cosine similarity khi narrative hoạt động trở lại (`reactivated`).
- `max_consecutive_silent_weeks` luôn không giảm trong suốt vòng đời của mỗi thực thể và khớp chính xác với `registry_df.groupby("narrative_id")["consecutive_silent_weeks"].max()` trên mọi narrative trong `narrative_registry_summary.parquet`.

## Ràng buộc thứ tự thời gian và quy ước bàn giao downstream

### Ràng buộc thứ tự thời gian và primary key

- `_finalize_timeline()` kiểm tra tính duy nhất của primary key ghép `(window_end, narrative_id)` (ném `ValueError` nếu có bản ghi trùng lặp).
- `_finalize_timeline()` kiểm tra thứ tự thời gian trên mọi dòng dữ liệu xuất ra:

  `first_seen_utc <= last_active_utc < window_end`

  và ném `ValueError` nếu bất kỳ dòng nào vi phạm một trong hai điều kiện trên.

- Mỗi narrative chỉ gia nhập registry tại cửa sổ `birth` của nó (`window_end == birth_window_end`) và không xuất hiện ở các cửa sổ sớm hơn (`20,841` dòng timeline được sinh ra trên `362` tuần hoạt động, so với `362 * 63 = 22,806` ô của lưới hồi cứu toàn giai đoạn).

> **Lưu ý chống data leakage — Không dùng `narrative_registry_summary.parquet` làm feature huấn luyện**:
> Không đưa `data/interim/narrative_registry_summary.parquet` vào làm feature huấn luyện hoặc join bảng này vào tập dữ liệu mô hình tại bất kỳ mốc thời gian nào. Tệp summary tổng hợp thống kê trên toàn bộ giai đoạn `2014-2020` (như `last_active_utc`, `total_posts`, `active_weeks`, `final_lifecycle_status`, `mean_weekly_posts`, `max_consecutive_silent_weeks`), nên chỉ dùng cho mục đích báo cáo và kiểm định. Mọi pipeline trích xuất feature và huấn luyện mô hình chỉ được đọc dữ liệu từ `narrative_registry.parquet` cắt tại `window_end <= cutoff_utc` thông qua `load_frozen_registry_at_cutoff()`.

### Khác biệt quy ước ngày kết thúc (CLI vs `TrackingConfig`)

Người triển khai phải tuân thủ đúng ngữ nghĩa của tham số ngày kết thúc trên hai giao diện trong `src/tracking/track_narratives.py`:

- **Cờ CLI `--calendar-end-utc`**: Nhận giá trị là **ngày Thứ Hai cuối cùng** (`2020-12-21`, khớp với `time.last_complete_week` trong `configs/forecast.yaml`). Bên trong `main()`, giá trị `2020-12-21` từ `parse_args()` được quy đổi bằng `pd.Timestamp(args.calendar_end_utc, tz="UTC") + pd.Timedelta(days=7)` thành `2020-12-28T00:00:00+00:00`.
- **`TrackingConfig.calendar_end_utc`**: Biểu diễn **cận trên loại trừ** (`"2020-12-28T00:00:00+00:00"` như được ghi nhận trong `narrative_registry_manifest.json`), được `_build_calendar_week_grid()` duyệt bằng vòng lặp `while cursor < end:`. Mã gọi khởi tạo trực tiếp `TrackingConfig` phải truyền `"2020-12-28T00:00:00+00:00"` (hoặc `last_complete_week + 7 days`) để tuần cuối cùng `[2020-12-21, 2020-12-28)` không bị bỏ sót.

### Lưu trữ `.npz` đóng băng và hàm đọc theo cutoff chính thức

- `save_centroids_npz()` kiểm tra hợp lệ và chuẩn hóa L2-normalized toàn bộ snapshot centroid qua `_l2_normalize_rows()`, lưu các mảng (`window_end_utc`, `narrative_id`, `centroids` kiểu `float32`) bằng `np.savez_compressed` không dùng Python object pickling (bắt buộc truyền `allow_pickle=False` khi đọc).
- `_finalize_timeline()` sắp xếp `timeline_df` theo `["window_end", "narrative_id"]` bằng `kind="mergesort"` và sắp xếp lại `centroid_records` theo đúng phép hoán vị chỉ số dòng đó, bảo đảm khớp 1:1 cả số dòng lẫn thứ tự giữa `narrative_registry.parquet` và `narrative_registry_centroids.npz`.
- Các module trích xuất feature và dự báo ở downstream bắt buộc phải đọc trạng thái ứng viên duy nhất thông qua `load_frozen_registry_at_cutoff(registry_path, centroids_path, cutoff_utc)`:
  - Chuẩn hóa `cutoff_utc` (dạng naive hoặc kèm múi giờ như `+07:00`) về UTC bằng `_to_utc()`.
  - Lọc chính xác `window_end == cutoff`, nạp `centroids_path` với `allow_pickle=False`, sắp xếp theo `narrative_id`, kiểm tra khớp bằng `np.array_equal` giữa các vector `narrative_id` của Parquet và `.npz`, và trả về `(filtered_df, centroids_float32)`.
  - Mã downstream không được đọc các dòng có `window_end > cutoff_utc`.

## Dữ liệu đầu ra, schema và kết quả kiểm định

### Danh mục tài nguyên và tệp dữ liệu

| Vai trò                        | Đường dẫn                                         | Định dạng / Chi tiết                                                                                    |
| :----------------------------- | :------------------------------------------------ | :------------------------------------------------------------------------------------------------------ |
| Dữ liệu gán cụm đầu vào        | `data/interim/narrative_assignments.parquet`      | Gán cụm mức bài đăng (`7,331` dòng tổng cộng, `6,411` dòng `ASSIGNED`)                                  |
| Embedding đầu vào              | `data/interim/scope_expanded_v1_embeddings.npz`   | Vector bài đăng `384`-D (`allow_pickle=False`)                                                          |
| Metadata phạm vi đầu vào       | `data/D4/net_neutrality_scope_expanded_v1.csv`    | Tập bài đăng và nội dung văn bản thuộc phạm vi Net Neutrality mở rộng                                   |
| Cấu hình giao thức             | `configs/forecast.yaml`                           | Phạm vi dữ liệu, biên lịch UTC, các năm chia tập, quy tắc biến mục tiêu prominence                      |
| Dòng thời gian registry đầu ra | `data/interim/narrative_registry.parquet`         | `20,841` dòng, `17` cột, nén Snappy, nhúng metadata của manifest (`b"narrativecast_registry_metadata"`) |
| Bảng tổng hợp thực thể đầu ra  | `data/interim/narrative_registry_summary.parquet` | `63` dòng, `10` cột, nén Snappy (chỉ phục vụ báo cáo/kiểm định, không dùng làm feature huấn luyện)      |
| Kho lưu trữ centroid đầu ra    | `data/interim/narrative_registry_centroids.npz`   | Ma trận `(20841, 384)` `float32` unit-norm khớp 1:1 với timeline, `allow_pickle=False`                  |
| Manifest đầu ra                | `data/interim/narrative_registry_manifest.json`   | Metadata cấu hình, số liệu thống kê tổng hợp và `audit_events_by_type`                                  |
| Mẫu cặp QA đầu ra              | `reports/results/tracking_pairs_sample.csv`       | Mẫu cặp kiểm tra chất lượng phân tầng thuộc giai đoạn phát triển (`48` dòng, `16` dòng mỗi tầng)        |

### Schema 17 cột của `data/interim/narrative_registry.parquet`

| #   | Tên cột                           | Dtype                 | Cho phép Null | Định nghĩa                                                                                                                  |
| :-- | :-------------------------------- | :-------------------- | :------------ | :-------------------------------------------------------------------------------------------------------------------------- |
| 1   | `window_start`                    | `datetime64[us, UTC]` | Không         | Mốc Thứ Hai `00:00:00+00:00` bắt đầu (bao gồm) của cửa sổ 7 ngày                                                            |
| 2   | `window_end`                      | `datetime64[us, UTC]` | Không         | Mốc Thứ Hai `00:00:00+00:00` kết thúc (loại trừ, là gốc cutoff `t`); thành phần của primary key                             |
| 3   | `narrative_id`                    | `string`              | Không         | Mã định danh ứng viên ổn định (`CAND_XXXX`); thành phần của primary key                                                     |
| 4   | `lifecycle_status`                | `category`            | Không         | Một trong các trạng thái `birth`, `active`, `inactive`, `dormant`, `reactivated`                                            |
| 5   | `current_window_posts`            | `int32`               | Không         | Số bài `ASSIGNED` hợp lệ của `narrative_id` trong `[window_start, window_end)`                                              |
| 6   | `window_total_valid_posts`        | `int32`               | Không         | Tổng số bài hợp lệ trong phạm vi (`ASSIGNED + NOISE + UNASSIGNED`) trong cửa sổ                                             |
| 7   | `current_window_share`            | `float32`             | Có            | `current_window_posts / window_total_valid_posts`; bằng `NaN` khi và chỉ khi `window_total_valid_posts == 0`                |
| 8   | `cumulative_posts`                | `int64`               | Không         | Tổng tích lũy bài `ASSIGNED` hợp lệ tính đến `window_end`                                                                   |
| 9   | `age_weeks`                       | `int32`               | Không         | `max(0, int((window_end - birth_window_end) // 7 days))`                                                                    |
| 10  | `consecutive_silent_weeks`        | `int32`               | Không         | Số tuần im lặng liên tiếp có `current_window_posts == 0` kết thúc tại `window_end`                                          |
| 11  | `first_seen_utc`                  | `datetime64[us, UTC]` | Không         | Mốc thời gian của bài `ASSIGNED` hợp lệ sớm nhất thuộc narrative này                                                        |
| 12  | `last_active_utc`                 | `datetime64[us, UTC]` | Không         | Mốc thời gian của bài `ASSIGNED` hợp lệ muộn nhất tính đến `window_end`                                                     |
| 13  | `centroid_drift`                  | `float32`             | Không         | `1.0 - clip(dot(prev_c, new_c), -1.0, 1.0)`; bằng `0.0` tại `birth` và các tuần im lặng                                     |
| 14  | `similarity_to_previous_centroid` | `float32`             | Có            | `clip(dot(prev_c, batch_centroid), -1.0, 1.0)`; bằng `1.0` tại `birth`; `NaN` khi và chỉ khi `current_window_posts == 0`    |
| 15  | `intra_window_dispersion`         | `float32`             | Có            | `1.0 - mean_float64(cosines)`; bằng `0.0` khi `current_window_posts == 1`; `NaN` khi và chỉ khi `current_window_posts == 0` |
| 16  | `is_in_candidate_set`             | `bool`                | Không         | Chỉ báo boolean (`True` cho mọi dòng ứng viên được ghi vào registry)                                                        |
| 17  | `split_merge_flag`                | `category`            | Không         | Một trong các cờ `NORMAL`, `SPLIT_CANDIDATE`, `MERGE_CANDIDATE`, `SPLIT_AND_MERGE`                                          |

### Schema 10 cột của `data/interim/narrative_registry_summary.parquet`

> Bảng tổng hợp này chỉ dùng cho báo cáo và kiểm định. Không dùng tệp này làm feature huấn luyện hay đầu vào mô hình vì số liệu đã tổng hợp trên toàn giai đoạn `2014-2020`.

| #   | Tên cột                        | Dtype                 | Định nghĩa                                                                            |
| :-- | :----------------------------- | :-------------------- | :------------------------------------------------------------------------------------ |
| 1   | `narrative_id`                 | `string`              | Mã định danh ứng viên ổn định được sắp xếp tăng dần                                   |
| 2   | `first_seen_utc`               | `datetime64[us, UTC]` | Mốc thời gian của bài `ASSIGNED` hợp lệ sớm nhất                                      |
| 3   | `last_active_utc`              | `datetime64[us, UTC]` | Mốc thời gian của bài `ASSIGNED` hợp lệ muộn nhất                                     |
| 4   | `birth_window_start`           | `datetime64[us, UTC]` | Mốc thời gian bắt đầu của cửa sổ `birth`                                              |
| 5   | `birth_window_end`             | `datetime64[us, UTC]` | Mốc thời gian kết thúc của cửa sổ `birth`                                             |
| 6   | `total_posts`                  | `int64`               | Tổng số bài `ASSIGNED` hợp lệ trong toàn vòng đời (`lifetime_post_count`)             |
| 7   | `active_weeks`                 | `int64`               | Tổng số tuần có `current_window_posts >= 1` (`active_weeks_count`)                    |
| 8   | `final_lifecycle_status`       | `category`            | Trạng thái vòng đời tại cửa sổ lịch cuối cùng (`2020-12-28T00:00:00+00:00`)           |
| 9   | `mean_weekly_posts`            | `float64`             | `total_posts / max(1, active_weeks)` (số bài trung bình trên mỗi tuần hoạt động)      |
| 10  | `max_consecutive_silent_weeks` | `int64`               | Giá trị `consecutive_silent_weeks` lớn nhất ghi nhận trên dòng thời gian của thực thể |

### Tổng hợp manifest và thống kê sự kiện kiểm định

- **Phân bố `lifecycle_status_distribution` ở cửa sổ cuối** (`63` narrative):
  - `dormant`: `52`
  - `inactive`: `11`
- **`audit_events_by_type`** (`7,110` sự kiện kiểm định tổng cộng):
  - `birth`: `63`
  - `linked_active`: `751`
  - `potential_merge_or_split`: `1,240`
  - `reactivation`: `1,372`
  - `unlinked_near_miss`: `3,684`

### Đặc tả lấy mẫu cặp QA (`reports/results/tracking_pairs_sample.csv`)

Hàm `export_tracking_review_sample()` trích xuất một tập mẫu kiểm tra chất lượng có thể tái lập, không rò rỉ dữ liệu test và không trùng cặp:

- **Giới hạn trong giai đoạn phát triển**: Lọc các sự kiện kiểm định thỏa điều kiện `window_end_utc <= 2019-01-01T00:00:00Z` (giá trị `window_end_utc` lớn nhất trong mẫu thực tế là `2018-06-25T00:00:00+00:00`, `< 2019-01-01T00:00:00Z`), loại bỏ toàn bộ các cửa sổ thuộc giai đoạn `Test` (`2019-2020`).
- **Ba tầng phân nhóm chính xác**: `linked_active`, `potential_merge_or_split` (mở rộng cả sự kiện `potential_merge` và `potential_split` khi chạy ở chế độ `relink_local_clusters`) và `unlinked_near_miss`.
- **Không ép kiểu sự kiện**: Các sự kiện `birth` (và `reactivation`) bị loại khỏi tập sự kiện được hỗ trợ và không bao giờ bị ép chuyển sang `linked_active`.
- **Khử trùng lặp vô hướng xuyên tầng**: Xử lý các tầng theo thứ tự ưu tiên `("linked_active", "potential_merge_or_split", "unlinked_near_miss")` với khóa chuẩn hóa `(utc_iso, min(id1, id2), max(id1, id2))`, ngăn chặn trùng lặp cặp `(window_end_utc, narrative_id_1, narrative_id_2)` cả trong từng tầng lẫn giữa các tầng bất kể thứ tự ID.
- **Tham số lấy mẫu và kết quả quan sát được**: `target_rows = 50`, `min_rows = 30`, `random_state = 42`, `per_stratum = max(target_rows // 3, 10) = 16`. Tệp `reports/results/tracking_pairs_sample.csv` được tạo ra gồm `48` cặp (`16` `linked_active`, `16` `potential_merge_or_split`, `16` `unlinked_near_miss`), `0` giá trị `NaN` ở cột độ tương đồng hoặc mã định danh và `0` cặp trùng lặp có thứ tự hoặc vô hướng.

### Trạng thái kiểm định và nghiệm thu

- **Kiểm định dữ liệu đầu ra (`scripts/audit_tracking_registry.py`)**: Đạt `8/8` tiêu chí `PASS`:
  1. Độ phủ cửa sổ lịch (`362` tuần hoạt động + `2` tuần warmup = `364` tuần lịch; `339` tuần hoạt động không rỗng + `2` tuần warmup không rỗng = `341` tuần không rỗng; `23` tuần rỗng; các tuần warmup có `0` bài `ASSIGNED`).
  2. Đối soát số lượng bài `ASSIGNED` (`6,411` bài `ASSIGNED` trên tổng `7,331` bài trong phạm vi; khớp chính xác tổng số bài theo từng narrative giữa bảng gán cụm, bảng timeline và bảng tổng hợp).
  3. Điều kiện kiểm tra Null / `NaN` và miền giá trị chỉ số (không có giá trị null trên `14` cột bắt buộc; điều kiện `NaN` chính xác cho `current_window_share` trong `[0, 1]`, `intra_window_dispersion` và `similarity_to_previous_centroid` trong `[-1, 1]`).
  4. Tính toàn vẹn của centroid và embedding nguồn (`allow_pickle=False`; khớp 1:1 về số dòng và thứ tự giữa `narrative_registry_centroids.npz` và `narrative_registry.parquet`; `0` giá trị `NaN`/`Inf`; L2 norm của centroid nằm trong `[0.9999, 1.0001]`).
  5. Kiểm tra thực thi hàm `load_frozen_registry_at_cutoff()` tại các cutoff của tập `Train` và `Validation`, bao gồm tính tương đương khi quy đổi múi giờ ngoài UTC (từ `+07:00` sang UTC với `atol=1e-7`).
  6. Tính duy nhất của primary key `(window_end, narrative_id)`, thứ tự thời gian (`first_seen_utc <= last_active_utc < window_end`) và tính nhất quán của `max_consecutive_silent_weeks`.
  7. Tính toàn vẹn của tệp CSV mẫu cặp QA (`>= 30` dòng, `< 2019-01-01T00:00:00Z`, đúng `3` tầng sự kiện, `0` giá trị `NaN`, `0` cặp trùng lặp có thứ tự hoặc vô hướng xuyên tầng).
  8. Kiểm tra định tính `10` cặp cụm liên kết qua hai tuần liên tiếp kèm trích đoạn nội dung văn bản từ `data/D4/net_neutrality_scope_expanded_v1.csv`.
- **Bộ kiểm thử đơn vị tự động**: Đạt `62/62` unit test trong kho mã nguồn (`PASS`), bao gồm `40` unit test kiểm tra module theo dõi narrative.
