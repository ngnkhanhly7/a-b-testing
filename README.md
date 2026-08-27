# A/B Testing Platform

## Giá trị kinh doanh

Peeking không kiểm soát làm tăng tỷ lệ dương tính giả từ 5% danh nghĩa lên
17.5% (đo được ở checkpoint 4) — nghĩa là cứ khoảng 6 experiment "có ý
nghĩa thống kê" mà một team tự peek mỗi ngày kết luận, thì có thể có tới 1
experiment thực chất không có tác dụng gì, dẫn tới ship nhầm một thay đổi
không mang lại giá trị (hoặc tệ hơn, có hại) dựa trên "bằng chứng" giả.
Chi phí của một lần ship sai như vậy — công sức kỹ thuật, thời gian
roadmap, và rủi ro với guardrail metric (VD retention, revenue) — thường
lớn hơn nhiều chi phí xây dựng nền tảng ngăn lỗi này bằng thiết kế thay vì
trông chờ kỷ luật cá nhân của người phân tích.

## Vấn đề

Nhiều team tự làm A/B test thủ công và mắc phải các lỗi thống kê phổ biến:
peeking (nhìn kết quả mỗi ngày và dừng ngay khi thấy "có ý nghĩa"), cỡ mẫu
chưa đủ trước khi kết luận, và chia nhóm A/B bị lệch do lỗi hệ thống mà
không ai kiểm tra. Nền tảng này cho phép bất kỳ team nào định nghĩa một
experiment, tự động chia người dùng ngẫu nhiên (deterministic hashing), thu
log sự kiện, và trả về kết luận thống kê đáng tin cậy — với mục tiêu chính
là **tránh những lỗi đó bằng thiết kế**, không phải bằng kỷ luật cá nhân của
người phân tích.

## Kiến trúc

```
src/
├── simulator.py       # sinh traffic giả có ground truth biết trước (checkpoint quan trọng nhất)
├── assignment.py       # chia nhóm A/B bằng consistent hashing + Sample Ratio Mismatch check
├── stats_engine.py     # two-proportion z-test / Welch's t-test, CI, effect size
├── sequential.py       # chứng minh + sửa lỗi peeking bằng O'Brien-Fleming boundary
├── power.py             # tính cỡ mẫu cần thiết, cảnh báo khi cỡ mẫu chưa đủ
├── pipeline.py           # ghép: assign (hashing) -> SRM check -> primary metric (gated bởi sequential nếu bật) -> sample size -> guardrail metrics
└── api/main.py           # FastAPI: POST /experiments, POST /experiments/{id}/events, GET /experiments/{id}/results (mỗi lần gọi = 1 "look", tự áp dụng sequential testing), POST /experiments/{id}/extend (nâng max_looks cho experiment đang chạy)
```

`pipeline.py` là nơi khép kín toàn bộ luồng: mỗi lần `GET /results` được gọi
tính là một lần "peek" (`look_number`), và nếu experiment có cấu hình
`max_looks`, kết luận về metric chính không còn dựa vào ngưỡng p<alpha tĩnh
mà được so với ngưỡng O'Brien-Fleming tương ứng với lần peek đó — nghĩa là
một team có thể xem `/results` bao nhiêu lần tuỳ ý trong lúc experiment đang
chạy mà không làm tăng tỷ lệ dương tính giả, đúng như checkpoint 4 chứng
minh. Mặc định `max_looks=20` khi tạo experiment qua API.

## Cách kiểm chứng công cụ (điểm nhấn chính)

Đây là project xây **công cụ đo lường**, không phải model dự đoán — "đúng"
nghĩa là *kết luận thống kê chính xác*. Vì vậy trước khi tin công cụ chạy
đúng trên dữ liệu thật, mọi module được kiểm chứng bằng dữ liệu mô phỏng có
đáp án biết trước (`src/simulator.py`), theo 3 kịch bản cố định:

| Kịch bản | p_control | p_treatment | Công cụ phải kết luận |
|---|---|---|---|
| 1. Khác biệt rõ ràng | 10% | 15% | Có ý nghĩa thống kê |
| 2. Không khác biệt | 10% | 10% | Không đủ bằng chứng (không dương tính giả) |
| 3. Khác biệt nhỏ | 10% | 10.3% | Chỉ phát hiện được nếu đủ cỡ mẫu (power analysis) |

`src/sequential.py` đi xa hơn: nó **đo bằng số** tỷ lệ dương tính giả khi
"peek" mỗi ngày trên kịch bản 2 (không có khác biệt thật), cho thấy tỷ lệ
này cao hơn hẳn 5% danh nghĩa, rồi áp dụng O'Brien-Fleming boundary và đo
lại — kết quả lưu ở `reports/sequential_validation.md`.

### Kết quả kiểm chứng thực tế (đã chạy)

| Checkpoint | Kết quả |
|---|---|
| 3 — Stats engine | Kịch bản 1: p≈4.4e-23 (phát hiện đúng). Kịch bản 2: p=0.86 (không báo dương tính giả). Kịch bản 3: p=0.83 ở cỡ mẫu 20k/nhóm (chưa đủ mẫu để phát hiện — đúng như lý thuyết, xem checkpoint 5). Chi tiết: `reports/stats_engine_validation.md`. |
| 4 — Peeking | Peek thủ công mỗi ngày: tỷ lệ dương tính giả **17.5%** (so với 5% danh nghĩa). Sau khi áp dụng O'Brien-Fleming: **6.5%**. Chi tiết: `reports/sequential_validation.md`. |
| 5 — Power analysis | Cỡ mẫu cần thiết để phát hiện 10%→10.3% (power 80%): **159,059 user/nhóm**. Empirical power qua 200 lần mô phỏng: chỉ **32%** ở 30% cỡ mẫu yêu cầu, **85%** ở đúng cỡ mẫu tính toán — khớp lý thuyết. |
| 6 — Cookie Cats (dữ liệu thật) | Retention ngày 7 giảm có ý nghĩa thống kê ở nhóm gate_40 (p=0.0016, -0.82 điểm %); retention ngày 1 và số vòng chơi không khác biệt có ý nghĩa. Khớp với các phân tích công khai đã biết trước. Chi tiết: `reports/cookie_cats_analysis.md`. |
| 8 — Guardrail | Demo: metric chính (CTR) thắng +50%, nhưng guardrail (thời gian session) bị phát hiện giảm 25% có ý nghĩa — hệ thống cảnh báo đúng "REGRESSION". |
| 7 — API + sequential end-to-end | Gọi `/results` 5 lần liên tiếp qua đúng API path (experiment hiệu ứng nhỏ 10% vs 11%): 3 lần đầu trả về "continue", lần thứ 4 chuyển sang "stop_significant" -- kiểm chứng pipeline hoàn chỉnh: assign (hashing) → log event → sequential-safe kết luận, không phải test rời rạc từng module. |

Trong lúc kiểm chứng, phát hiện và sửa 3 lỗi/lỗ hổng thật: (1) bản demo power
analysis đầu tiên chỉ chạy 1 lần mô phỏng nên kết luận có thể đúng/sai do
may rủi — đã sửa thành Monte Carlo 200 lần lặp đo empirical power; (2) các
kiểu dữ liệu numpy (`numpy.bool_`, `numpy.int64`...) không serialize được
qua FastAPI — đã ép kiểu về Python thuần trong `assignment.py` và
`stats_engine.py`; (3) `pipeline.py` và API ban đầu **không thực sự dùng**
sequential testing (checkpoint 4) hay consistent hashing (checkpoint 2) —
`/results` chỉ chạy z-test tĩnh, nghĩa là gọi API nhiều lần trong lúc
experiment đang chạy vẫn dính đúng lỗi peeking mà project được thiết kế để
ngăn. Đã sửa: `analyze_experiment` nhận `look_number`, gate kết luận qua
`sequential.py`, và API đếm mỗi lần gọi `/results` làm một lần "look".

### Đợt tối ưu & rà soát lần 2

- **Hiệu năng:** `simulator.py` từng sinh log bằng vòng lặp Python theo từng
  user -- chậm ~0.9s/318k user, khiến Monte Carlo trong `power.py` (200 lần
  lặp) mất ~4 phút. Đã vector hoá toàn bộ bằng numpy/pandas: còn ~0.1s/318k
  user (nhanh hơn ~9 lần), Monte Carlo còn ~1 phút, kết quả thống kê giữ
  nguyên (đã đối chiếu với seed cũ).
- **Bug thật (SRM):** `check_srm` chỉ so sánh các nhóm có mặt trong dữ liệu
  quan sát -- nếu một nhóm chưa có user nào (rất hay gặp ngay sau khi tạo
  experiment), nhóm đó biến mất khỏi phép so sánh thay vì hiện ra như một
  cảnh báo SRM rõ ràng, và scipy còn crash vì tổng observed/expected lệch
  nhau. Đã sửa: dùng `expected_split` làm nguồn danh sách nhóm, nhóm vắng mặt
  được tính là 0.
- **Bug thật (API 500):** hệ quả của lỗi trên, `GET /results` sẽ crash với
  `ZeroDivisionError` nếu một nhóm chưa có event nào. Đã thêm guard trả về
  `status: "insufficient_data"` thay vì lỗi 500.
- Ghim version cụ thể trong `requirements.txt` để đảm bảo tái lập được kết
  quả khi cài lại môi trường sau này.
- Thêm `tests/test_api.py` và mở rộng `tests/test_assignment.py` để khoá lại
  các hành vi trên, tổng cộng 16 test.

## Cách chạy

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

# Checkpoint 1: sinh 3 kịch bản mô phỏng
python -m src.simulator

# Checkpoint 4: chứng minh + sửa lỗi peeking (ghi reports/sequential_validation.md)
python -m src.sequential

# Checkpoint 3: validate stats engine trên 3 kịch bản (ghi reports/stats_engine_validation.md)
python -m src.validate_stats_engine

# Checkpoint 5: kiểm chứng power analysis trên kịch bản khác biệt nhỏ
python -m src.power

# Checkpoint 6: phân tích Cookie Cats thật (cần tải dataset trước, xem bên dưới)
python -m src.analyze_cookie_cats

# Checkpoint 3+7+8: chạy pipeline end-to-end + demo guardrail metric
python -m src.pipeline

# Chạy test suite
pytest

# Checkpoint 7: chạy API (in ra API key khi khởi động lần đầu, xem mục "Vận hành như MVP nội bộ")
uvicorn src.api.main:app --reload
```

## Vận hành như MVP nội bộ

API đã được nâng từ prototype lên mức "dùng thử nội bộ được" (không phải
public production-grade), gồm 3 phần:

1. **Persistence (SQLite):** toàn bộ experiment/event được lưu ở
   `data/platform.db` (WAL mode + `synchronous=NORMAL` để ghi nhanh mà vẫn an
   toàn khi crash ứng dụng). Restart/redeploy server không còn mất dữ liệu.
2. **Auth (API key, 2 cấp):** server sinh 1 admin API key ngẫu nhiên ở lần
   khởi động đầu tiên (lưu trong DB, không đổi qua các lần restart, in ra
   log để operator lấy) -- key này bắt buộc để tạo experiment mới. Mỗi
   experiment khi tạo còn được cấp riêng 1 `experiment_key` (trả về đúng 1
   lần trong response của `POST /experiments`), dùng được cho
   events/results/extend **của riêng experiment đó**. Một team chỉ cầm
   `experiment_key` của mình sẽ không đọc/ghi được experiment của team
   khác, dù vẫn cần admin key để tạo experiment mới. Mọi request phải có
   header `X-API-Key: <key>`, thiếu hoặc sai sẽ bị từ chối với 401 (sai
   experiment_id hoặc experiment không tồn tại → 404).
3. **Tự khoá khi sequential test khuyến nghị dừng:** khi `GET /results` nhận
   được khuyến nghị `stop_significant` hoặc `stop_no_effect` (checkpoint 4),
   experiment tự động bị khoá (`locked`): mọi `POST .../events` sau đó bị từ
   chối với 409, và `GET .../results` sau đó luôn trả về đúng kết luận đã
   chốt — không cho phép "peek" tiếp sau khi đã có quyết định dừng, đúng tinh
   thần sequential testing.
4. **Gia hạn `max_looks` (`POST /experiments/{id}/extend`):** nếu một
   experiment chưa bị khoá cần check nhiều lần hơn dự kiến ban đầu (launch
   bị hoãn, ramp-up chậm hơn kế hoạch), team có thể nâng `max_looks` thay vì
   bị ép kết luận `stop_no_effect` sớm. Đây là escape hatch thực dụng, không
   phải alpha-spending schedule tính lại chính xác — nên ưu tiên đặt
   `max_looks` rộng rãi ngay từ đầu, chỉ dùng endpoint này khi thật sự cần.

Ví dụ dùng thử (Windows PowerShell hoặc bash):

```bash
uvicorn src.api.main:app --reload
# log in-terminal sẽ in: [A/B Testing Platform] API key ...: <key>

curl -X POST http://127.0.0.1:8000/experiments \
  -H "Content-Type: application/json" -H "X-API-Key: <key>" \
  -d '{"name":"demo","metric_type":"proportion","max_looks":20}'
```

Đã kiểm chứng qua HTTP thật: (1) thiếu/sai API key → 401; (2) tạo experiment,
restart server, gọi lại `/results` → experiment vẫn còn nguyên (persistence);
(3) sau khi khoá, gửi thêm event → 409, đọc `/results` nhiều lần → luôn trả
về cùng một kết quả đã chốt (không tính thêm "look" mới).

### Vẫn còn thiếu để chạy production thật (public-facing, nhiều team, SLA)

- `experiment_key` chỉ phân quyền theo *experiment*, chưa theo *team/user* —
  chưa có khái niệm "user nào tạo experiment nào", chưa revoke được riêng 1
  key nếu lộ (phải xoá/tạo lại cả experiment), và admin key vẫn là một
  điểm truy cập toàn cục duy nhất. Đủ để cô lập dữ liệu giữa các team dùng
  chung server, nhưng chưa phải access control đầy đủ.
- Chạy 1 process duy nhất (SQLite không hợp để nhiều worker/pod ghi đồng
  thời) — muốn scale ngang cần chuyển sang PostgreSQL.
- Chưa có logging có cấu trúc / metrics / alerting.
- Chưa có Dockerfile/CI-CD.

## Kết quả trên dữ liệu thật (Cookie Cats)

Đã chạy trên `data/real/cookie_cats.csv` (90,189 user, gate_30=44,700 vs
gate_40=45,489). Kết luận: dời cổng chặn từ level 30 sang 40 **không** cải
thiện retention — retention ngày 7 còn giảm có ý nghĩa thống kê ở nhóm
gate_40 (p=0.0016). Kết quả này khớp với các phân tích công khai đã có
sẵn về dataset này (Kaggle/Medium), là bằng chứng công cụ hoạt động đúng
trên dữ liệu thật, không chỉ trên dữ liệu tự sinh. Chi tiết đầy đủ ở
`reports/cookie_cats_analysis.md`.

## Hạn chế & hướng mở rộng

- O'Brien-Fleming boundary hiện dùng công thức xấp xỉ cổ điển
  (`z_k = z_{alpha/2} * sqrt(K/k)`), không phải alpha-spending calibrate
  chính xác bằng Lan-DeMets — đủ để minh hoạ vấn đề và hướng sửa, nhưng một
  hệ thống production nên dùng thư viện chuyên dụng (VD `sequential` trong R,
  hoặc cài đặt spending function đầy đủ).
- Chưa có Bayesian A/B testing hay multi-armed bandit (tự động phân bổ
  traffic về phương án tốt hơn theo thời gian thực) — hướng mở rộng tự nhiên
  khi cần tối ưu thay vì chỉ suy luận thống kê.
- Nếu 2 team chạy experiment trên cùng một tập người dùng cùng lúc, sẽ có
  interaction effect giữa các experiment mà nền tảng hiện chưa xử lý — cần
  cơ chế namespace/layer để tách traffic (như Google's overlapping
  experiment infrastructure).
- Storage của API hiện là in-memory, mất dữ liệu khi restart — cần thay
  bằng database thật trước khi dùng ở production.
