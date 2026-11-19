# PAVE-Net — Sửa đổi giao thức V2.1 (Protocol Amendment)

**Bản sửa đổi của `docs/RESEARCH_PROTOCOL_V2.md`. Tệp V2 KHÔNG bị sửa và vẫn là giao thức đã đóng băng cho Phase 2.**
Phiên bản 2.1 — đóng băng ngày 19/11/2026, **trước khi** chạy bất kỳ mô hình nào của Phase 3A (B0–B3 lồng nhau),
bất kỳ phép can thiệp phổ nào trên dữ liệu thật, và bất kỳ phép dò yếu tố gây nhiễu nào trong mục 9.
Mã băm SHA-256 ghi trong `docs/PROTOCOL_FREEZE_V2.1.txt`.

---

## 1. Lý do sửa đổi
1. Ở Phase 2, validation của PU-BD5 chỉ có **3 ổ (1 ổ/lớp)**. Chọn epoch cho CNN trở nên gần như ngẫu nhiên (epoch tốt nhất
   0–3), và không thể chọn siêu tham số một cách có cơ sở.
2. Tác giả quyết định (19/11/2026): (i) **P1** là baseline chuẩn chính; (ii) nghiên cứu validation lồng nhau theo nhóm ổ;
   (iii) yếu tố gây nhiễu **nhà sản xuất / năm đo** là **kết quả thực nghiệm trung tâm**.
3. Phase 2 đã xem kết quả trên toàn bộ PU-BD5. Từ nay **Paderborn không còn là bộ kiểm định "chưa thấy"**.

## 2. Điều đã học ở Phase 2 (ảnh hưởng tới thiết kế)
- KAIST bão hoà (macro-F1 ≈ 1,0; chỉ-RMS 0,886): chỉ dùng cho phân tích vật lý và lối tắt, không xếp hạng mô hình.
- PU-BD5: P1 đạt macro-F1 cửa sổ 0,618 ± 0,185 (SD qua 5 fold, ddof = 1) và macro-F1 cấp ổ gộp **0,731**
  (11/15 ổ đúng; ma trận [[3,1,1],[0,4,1],[1,0,4]]). CNN đạt 0,49–0,55 (cửa sổ).
- Khái niệm vật lý: OR-vs-IR tốt (AUROC trung bình theo fold 0,905; nhân tạo→thật 0,893). Hư-vs-khỏe yếu (0,775 trung bình
  theo fold; nhân tạo→thật 0,584). Một số ổ hư có bằng chứng yếu (KA15, KA22, KA30, KI14, KI21) hoặc mâu thuẫn (KI16).
- Healthy ≡ IBU ≡ năm 2015 trong PU-BD5 (5/5); trong 10 ổ hư PU-BD5, chỉ KA22 là IBU, chỉ KA30 và KI04 đo năm 2015.

## 3. Khác biệt so với V2

| Mục | V2 | V2.1 |
|---|---|---|
| Chia PU-BD5 | 5 fold: test 1 hàng, val 1 hàng cố định, train 3 hàng | **Lồng nhau**: ngoài 5 fold (test 1 hàng, train 4 hàng); trong 4 fold leave-one-row-out trên 4 hàng train |
| Chọn siêu tham số | cố định, không HPO | lưới nhỏ khai báo trước, chọn trên vòng trong |
| Chọn số epoch | dừng sớm trên 3 ổ val | số epoch cố định chọn từ đường cong trung bình 4 fold trong, rồi huấn luyện lại trên 12 ổ |
| Hiệu chỉnh xác suất | không | temperature scaling khớp trên dự đoán out-of-fold của vòng trong |
| Endpoint chính | macro-F1 cửa sổ | **macro-F1 cấp ổ gộp trên 15 ổ** |
| Baseline chuẩn | — | **B1 = P1** |
| Yếu tố gây nhiễu | đối chứng metadata hậu nghiệm | phân tích trung tâm, khai báo trước (mục 9) |
| Trạng thái PU | "độc lập" cho thiết kế vật lý | **đã xem**; chỉ còn giá trị kiểm định nội bộ |

Các định nghĩa đặc trưng vật lý, cửa sổ, tiền xử lý và nhãn của V2 giữ **nguyên**.

## 4. Chia dữ liệu theo nhóm ổ (PU-BD5)
Hàng r (Lessmeier 2016, Bảng 10): R1 {K001, KA04, KI04}, R2 {K002, KA15, KI14}, R3 {K003, KA16, KI16},
R4 {K004, KA22, KI18}, R5 {K005, KA30, KI21}.
- **Fold ngoài Ok** (k = 1..5): test = Rk (3 ổ: 1 H, 1 IR, 1 OR); outer-train = 4 hàng còn lại (12 ổ, 4/lớp).
- **Fold trong Ok-Ij**: inner-val = Rj (j ≠ k), inner-train = 3 hàng còn lại. Mỗi ổ outer-train được inner-validate đúng 1 lần.
- Ổ là đơn vị độc lập; mọi cửa sổ và mọi điều kiện của một ổ thuộc cùng một vai trò.
- Ghi chú: bài báo gốc dùng 10 tổ hợp 3 train / 2 test cho mỗi lớp và không có validation. Cách ghép hàng là lựa chọn của chúng
  tôi theo thứ tự liệt kê trong Bảng 10.

### Số lượng chính xác
| | Ổ | H / IR / OR | Cửa sổ 1 s |
|---|---|---|---|
| PU-BD5 toàn bộ | 15 | 5 / 5 / 5 | 4 781 (H 1 598, IR 1 594, OR 1 589) |
| Mỗi fold ngoài: test | 3 | 1 / 1 / 1 | 952–959 |
| Mỗi fold ngoài: train | 12 | 4 / 4 / 4 | 3 822–3 829 |
| Mỗi fold trong: train / val | 9 / 3 | 3/3/3 · 1/1/1 | 2 864–2 874 / 952–959 |

Cửa sổ là đơn vị mô tả, **không** phải đơn vị thống kê độc lập. Danh sách đầy đủ 32 ổ và thành phần từng fold:
`results/phase3a/pu_verification/pu_32_bearings.csv`, `fold_composition.csv`.

## 5. Loại trừ dữ liệu
| Loại trừ | Lý do |
|---|---|
| `N15_M01_F10_KA08_2.mat` | không đọc được (hỏng); KA08 không thuộc PU-BD5 |
| 55 lần đo cho 3 thay vì 4 cửa sổ | tệp ngắn hơn 4 s; giữ lại các cửa sổ đầy đủ |
| KB23, KB24, KB27 | hư hỏng kép IR+OR, không có nhãn 3 lớp; chỉ dùng mô tả |
| K006, KI17 | không có trong Bảng 10 nên không thuộc PU-BD5; có trong A2R |
| 12 ổ nhân tạo | không thuộc PU-BD5; dùng cho A2R và phân tích độ tin cậy khái niệm nhân tạo/thật |

## 6. Tính khả thi của validation lồng nhau
- **Thực hành:** khả thi. 5 × 4 = 20 lần khớp trong mỗi cấu hình; CNN khoảng 20–40 s/lần trên RTX 3060 6 GB.
  Hiệu chỉnh vùng vật lý được làm lại cho từng tập train (25 lần).
- **Thống kê:** vòng trong cung cấp 12 ổ out-of-fold (4/lớp) cho mỗi lần chọn, tốt hơn 3 ổ của V2 nhưng vẫn rất ít.
  Độ chính xác cấp ổ có bước 1/12. Vì vậy:
  1. Lưới siêu tham số **nhỏ** (≤ 4 cấu hình/mô hình) để hạn chế chọn quá khớp.
  2. Tiêu chí chọn: **macro-F1 cấp cửa sổ trung bình trên 4 fold trong** (mịn hơn cấp ổ); hoà (chênh ≤ 0,005) thì chọn
     cấu hình đơn giản / regularize mạnh hơn.
  3. Phân tích độ nhạy: kết quả với siêu tham số mặc định của Phase 2 (không chọn) được báo cáo song song.
- **Phương án thay thế nếu vòng trong không ổn định** (độ lệch giữa fold trong > 0,2 macro-F1 cho mọi cấu hình): dùng mặc định
  cố định và báo cáo vòng trong như chẩn đoán. Không bao giờ chọn trên ổ test ngoài.

## 7. Quy tắc chọn siêu tham số (khai báo trước)
| Mô hình | Lưới | Cố định |
|---|---|---|
| B0 luật vật lý | không | ngưỡng τ trên max(`BPFO_harm`, `BPFI_harm`) khớp trên outer-train (tối đa hoá balanced accuracy khỏe-vs-hư); H nếu < τ, ngược lại OR nếu `contrast` > 0, không thì IR |
| B1 = P1 (LR trên 9 khái niệm) | C ∈ {0,01; 0,1; 1; 10} | StandardScaler, class_weight = balanced |
| B2 MLP trên 9 khái niệm | ẩn ∈ {16, 64} × weight decay ∈ {1e-4, 1e-2} | 2 lớp ẩn ReLU, dropout 0,1, AdamW lr 1e-3, batch 256, số epoch ∈ {5, 10, …, 150} chọn theo đường cong trong |
| B3 CNN-1D thô (không ràng buộc vật lý) | lr ∈ {1e-3, 3e-4} | kiến trúc Phase 2 (≈ 0,23 M tham số), 32 kHz, AdamW wd 1e-2, batch 64, số epoch ∈ {1..20} chọn theo đường cong trong |

- Vòng trong dùng seed 42. Mô hình cuối được huấn luyện lại trên 12 ổ outer-train với seed 42, 43, 44; dự đoán chính dùng
  **trung bình xác suất của 3 seed** (seed-ensemble); từng seed được báo cáo riêng.
- B3 dùng sóng thô vì PU có tốc độ không đổi trong mỗi điều kiện (Phase 2: không có lối tắt tốc độ; D1–D3 khác nhau trong nhiễu),
  và vì can thiệp phổ áp dụng trực tiếp được.
- Mọi khớp (hiệu chỉnh vùng vật lý, z-score, imputation, StandardScaler, τ, siêu tham số, số epoch, nhiệt độ) chỉ dùng dữ liệu
  của tập train ở cấp tương ứng.

## 8. Hiệu chỉnh xác suất
- Temperature scaling (một tham số T > 0, khớp bằng tối thiểu NLL) trên **dự đoán out-of-fold của 4 fold trong** với cấu hình
  đã chọn; áp dụng cho dự đoán ổ test ngoài. Báo cáo ECE/NLL trước và sau hiệu chỉnh. Quyết định lớp dùng argmax (không đổi
  theo T). B0 không có xác suất nên ECE không áp dụng.

## 9. Kế hoạch phân tích thống kê và endpoint

**Đơn vị phân tích:** ổ trục (15 trong PU-BD5). Quyết định cấp ổ = argmax trung bình xác suất trên mọi cửa sổ của ổ.

**Endpoint chính (P-1):** macro-F1 cấp ổ gộp trên 15 ổ test (5 fold ngoài), so sánh B2 và B3 với **B1**.
Ước lượng điểm, khoảng tin cậy 95 % bootstrap phân tầng theo lớp trên ổ (10 000 lần), hiệu số ghép cặp so với B1
(bootstrap ghép cặp), số cặp bất đồng và p của McNemar chính xác. **Do n = 15, nghiên cứu không đủ lực cho kiểm định suy luận**
(McNemar chỉ đạt p < 0,05 khi ≥ 6 cặp bất đồng cùng một phía). Kết luận dựa trên ước lượng và phân tích từng ổ.

**Endpoint phụ:**
- S-1 macro-F1 cấp cửa sổ (mô tả; trung bình ± SD qua fold), balanced accuracy, recall từng lớp, ma trận nhầm lẫn cấp ổ.
- S-2 ECE, NLL (thô và sau temperature scaling).
- S-3 Độ nhạy/độ đặc hiệu khỏe-vs-hư cấp ổ.
- **S-4 Yếu tố gây nhiễu (trung tâm):**
  (a) quy tắc chỉ-metadata (nhà sản xuất, năm đo) khớp trên outer-train, đánh giá cấp ổ như mô hình;
  (b) phân biệt khỏe-vs-hư trong cùng tầng: IBU (K001–K005 vs KA22) và năm 2015 (K001–K005 vs KA30, KI04), cho khái niệm và
      điểm mô hình (mô tả; n rất nhỏ);
  (c) **phép dò danh tính phiên đo**: dự đoán năm đo (2014 vs 2015) trong các ổ IR (thật + nhân tạo, 11 ổ) và IBU-vs-không-IBU
      trong mọi ổ hư (thật + nhân tạo, 23 ổ có nhãn 3 lớp) từ đặc trưng thủ công (RF) và từ 9 khái niệm (LR), leave-one-bearing-out,
      AUROC cấp ổ. Nếu phép dò dự đoán được năm/nhà sản xuất khi lớp hư hỏng cố định, tín hiệu rung mang dấu vết phiên đo/nhà sản xuất.
- S-5 Độ tin cậy khái niệm (mục 10 của báo cáo Phase 3A) theo: khỏe-vs-hư, IR-vs-OR, nhân tạo-vs-thật, điều kiện vận hành, từng ổ.
- S-6 Kiểm tra thao tác của can thiệp phổ (synthetic và dữ liệu thật).

**Thăm dò (không xác nhận):** mọi phân tích KAIST; PU-A2R; ảnh hưởng của can thiệp lên đầu ra mô hình; phân tích theo điều kiện
vận hành; mọi phân tích bổ sung sau thời điểm đóng băng này.

## 10. Phân loại xác nhận / thăm dò
- **Không có kết quả nào trên PU còn là kiểm định ngoài độc lập.** Định nghĩa khái niệm vật lý được đóng băng trước khi xem PU
  (V2), nhưng việc chọn P1 làm chuẩn và thiết kế V2.1 chịu ảnh hưởng từ kết quả Phase 2 trên PU.
- Các so sánh P-1 và S-1…S-4 là **xác định trước, đánh giá nội bộ trên dữ liệu đã xem**. Giá trị B1 không phải xác nhận
  (đã biết từ Phase 2 với C = 1). B2/B3 lồng nhau chưa từng chạy.
- **Kiểm định xác nhận thực sự** cho kiến trúc Phase 3B đòi hỏi một bộ dữ liệu ngoài, nhiều mẫu ổ độc lập, được khoá trước khi
  xem (ứng viên cần xác minh: HUST bearing dataset, University of Ottawa bearing dataset, hoặc bộ khác). Bộ này sẽ được chọn và
  khoá bằng một sửa đổi giao thức trước Phase 4.
- Chọn kiến trúc Phase 3B chỉ dựa trên **vòng trong** (out-of-fold trên outer-train), không dựa trên ổ test ngoài.

## 11. Ngân sách
Phase 3A ≤ 2 giờ GPU; RAM ≈ 3 GB nên dữ liệu được nạp theo fold dạng float16.
