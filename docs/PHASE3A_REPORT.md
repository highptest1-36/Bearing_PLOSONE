# PAVE-Net — Báo cáo Phase 3A (giao thức, khái niệm, baseline, thiết kế kiểm chứng giải thích)

Ngày: 19/11/2026. Giao thức: `docs/RESEARCH_PROTOCOL_V2.1.md` (đóng băng; SHA-256 `15343677…2027`); V2 không đổi
(`60d0c66a…6d15`). **Chưa xây PAVE-Net đầy đủ. Dừng sau Phase 3A theo yêu cầu.**
Mọi số liệu sinh từ: `scripts/pu_verification_phase3a.py`, `run_phase3a.py`, `report_phase3a.py`, `concept_audit_phase3a.py`,
`interventions_phase3a.py`. Kiểm thử: **95/95 đạt** (`python -m pytest -q tests`).

## 0. Tóm tắt

1. **Kiểm chứng Phase 2:** các con số chính đúng, nhưng có ba chỗ cần ghi chính xác hơn và một *erratum* về seed (mục 1).
2. **Giao thức V2.1** đã đóng băng trước mọi mô hình Phase 3A. Validation lồng nhau theo nhóm ổ **khả thi về tính toán nhưng
   gần như không mang thông tin cho việc chọn siêu tham số**: quy tắc dự phòng §6 bị kích hoạt (độ phân tán giữa fold trong
   0,25–0,64).
3. **Baseline (15 ổ, cấp ổ):** B1 = P1 macro-F1 **0,739** [KTC 95 % 0,49–0,93]; B2 (MLP trên khái niệm) **giống hệt B1**
   ở cấp ổ; B0 luật vật lý 0,579; B3 CNN 0,551 (mặc định 0,611). CNN quá tự tin (ECE 0,34; nhiệt độ hiệu chỉnh 4–20).
   Không có khác biệt nào có ý nghĩa thống kê (n = 15).
4. **Yếu tố gây nhiễu (kết quả trung tâm):** quy tắc chỉ-năm-đo hoặc chỉ-nhà-sản-xuất nhận đúng **5/5 ổ khỏe**. Đặc trưng phổ
   thông thường dự đoán được năm đo/nhà sản xuất của ổ chưa thấy (AUROC cấp ổ 0,87–1,00). Khái niệm vật lý thì không (0,43–0,44,
   trừ khi bị nhiễu bởi nguồn gốc nhân tạo/thật).
5. **Khái niệm:** họ BPFI tin cậy cho IR-vs-khỏe (AUROC cấp ổ 1,00); `contrast` cho IR-vs-OR (0,88); họ BPFO chỉ một phần;
   **`BPFO_sb` không hợp lệ**; họ BPFI **không xác định được** ở 900 v/ph với hình học FAG (va chạm hài 5·fr).
6. **Can thiệp phổ:** kiểm tra thao tác đạt (xoá đúng bằng chứng, đối chứng không đổi). Thăm dò: **CNN gần như không phản ứng
   khi vạch lỗi bị xoá** (|Δp| ≤ 0,03), kể cả ở những ổ nó phân loại đúng.
7. **Khuyến nghị: MODIFY** — tiến sang Phase 3B với các sửa đổi ở mục 11.

## 1. Kiểm chứng tuyên bố của Phase 2
| Tuyên bố | Kiểm chứng từ `runs.jsonl` | Ghi chú |
|---|---|---|
| P1 macro-F1 0,618 ± 0,185 | đúng: **cấp cửa sổ**, trung bình 5 fold, SD ddof = 1 (ddof = 0: 0,166) | macro-F1 **cấp ổ** gộp 15 ổ = **0,731**, ma trận [[3,1,1],[0,4,1],[1,0,4]] |
| Độ chính xác cấp ổ 0,733 | đúng (11/15) | |
| ECE 0,139 | đúng (trung bình 5 fold; từng fold 0,08–0,18) | cấp cửa sổ |
| Hư-vs-khỏe AUROC 0,775 | đúng, là **trung bình AUROC từng fold** | không phải AUROC gộp |
| Nhân tạo→thật AUROC 0,893 | đúng **cho OR-vs-IR**; hư-vs-khỏe nhân tạo→thật chỉ **0,584** | |
| 78 kiểm thử | đúng | nay 95 |
| **Erratum** | ở Phase 2, mô hình được khởi tạo *trước* `set_seed`: seed điều khiển thứ tự dữ liệu và dropout nhưng **không** điều khiển trọng số ban đầu | chạy lại theo cùng thứ tự vẫn tái lập được; Phase 3A đã sửa và có kiểm thử. Báo cáo Phase 2 không bị sửa; ghi trong `DECISION_LOG.md` |

## 2. B: Kiểm chứng metadata Paderborn
Bảng đầy đủ 32 ổ: `results/phase3a/pu_verification/pu_32_bearings.csv` (nhóm, nhãn, nguồn gốc, phương pháp/mức hư hỏng,
nhà sản xuất, đường kính vòng chia, ngày đo đầu, số tệp, số cửa sổ, vai trò ở mọi giao thức, số lần được test, lý do loại trừ).
Thành phần từng fold: `fold_composition.csv`.

| Nhóm | Số ổ | Nhà sản xuất | Năm đo |
|---|---|---|---|
| Healthy K001–K006 | 6 | IBU 6 | 2015: 6 |
| OR nhân tạo | 7 | IBU 5, MTK 2 | 2014: 7 |
| IR nhân tạo | 5 | IBU 3, MTK 2 | 2014: 1; 2015: 4 |
| OR thật | 5 | FAG 2, MTK 2, IBU 1 | 2014: 4; 2015: 1 (KA30) |
| IR thật | 6 | MTK 4, FAG 2 | 2014: 5; 2015: 1 (KI04) |
| Hư hỏng kép KB | 3 | MTK 1, FAG 2 | 2014: 1; 2015: 2 |

- **PU-BD5 ↔ Lessmeier 2016 Bảng 10:** đúng 15 ổ và đúng thứ tự hàng (đã kiểm bằng mã). **Khác biệt về giao thức:** bài báo dùng
  10 tổ hợp (3 ổ/lớp train, 2 ổ/lớp test, không validation); chúng tôi dùng 5 fold ngoài theo hàng (V2: + 1 hàng validation;
  V2.1: + vòng trong 4 fold). Việc ghép hàng là lựa chọn của chúng tôi theo thứ tự liệt kê; bài báo không nói hàng có ý nghĩa.
- **PU-A2R ↔ Bảng 8:** A2R train thêm K003, K004, K005, KA08, KA09; A2R test thêm **KI04** (Bảng 8 không có KI04 ở test).
  A2R là thiết kế mở rộng của chúng tôi, không phải tái hiện Bảng 8.
- Mỗi ổ BD5 được test đúng 1 lần ở cả V2 và V2.1; K006 và KI17 chỉ có trong A2R; KB chỉ là thử thách.
- Loại trừ: `N15_M01_F10_KA08_2.mat` (hỏng); 55 lần đo có 3 cửa sổ; KB (không có nhãn 3 lớp).
- Số cửa sổ PU-BD5: 4 781 (H 1 598, IR 1 594, OR 1 589); mỗi fold test 952–959, train 3 822–3 829.

## 3. A: Sửa đổi giao thức V2.1 và tính khả thi của validation lồng nhau
- Thiết kế: 5 fold ngoài (test 1 hàng = 3 ổ) × 4 fold trong (val 1 hàng). Mọi khớp (vùng vật lý, z-score, imputation, τ,
  siêu tham số, số epoch, nhiệt độ) chỉ trên dữ liệu train của cấp tương ứng. Kiểm thử: `tests/test_phase3a.py`.
- **Kết quả khả thi** (`results/phase3a/tables/nested_selection.csv`):
  - Tính toán: khả thi (≈ 10 phút/fold ngoài; tổng ≈ 50 phút GPU).
  - Thống kê: macro-F1 của **cùng một cấu hình** dao động 0,18–0,50 (B1) và 0,56–0,64 (B3, cấu hình được chọn) giữa các fold
    trong. Đó là hệ quả của validation chỉ 3 ổ, nơi một ổ đổi nhãn làm chỉ số nhảy mạnh. **Quy tắc dự phòng §6 bị kích hoạt**
    cho B1 (4/5 fold ngoài, mọi cấu hình > 0,2). Với B3, ít nhất cấu hình được chọn vượt ngưỡng ở mọi fold. Giá trị từng fold
    trong của các cấu hình B3 khác và của B2 **không được lưu** (sai lệch giao thức, được ghi nhận).
  - Lựa chọn thực tế: B1 chọn C = 0,01 ở 4/5 fold (C = 1 ở O2); B2 chọn (64 nút ẩn, wd 1e-2) ở 5/5 fold với 5–65 epoch;
    B3 chọn lr 3e-4 ở 3/5 fold với 1–14 epoch. Số epoch dao động mạnh là dấu hiệu chọn theo nhiễu.
- **Kết luận:** validation lồng nhau là cách duy nhất không dùng ổ test để chọn, nhưng với 12 ổ/fold nó **không đủ để phân biệt
  các cấu hình**. Phương án bảo vệ được: (i) siêu tham số mặc định cố định, khai báo trước; (ii) nếu bắt buộc chọn, dùng lưới
  tối thiểu với quy tắc "đơn giản hơn trừ khi tốt hơn rõ rệt"; (iii) mọi so sánh kiến trúc ở Phase 3B phải có ngưỡng vượt lớn
  và xác nhận trên dữ liệu ngoài.

## 4. Đánh giá rò rỉ
| Nguồn rò rỉ tiềm ẩn | Tình trạng |
|---|---|
| Ổ test ngoài trong chọn siêu tham số / epoch / nhiệt độ / τ / vùng vật lý | **Không** — `check_nested` và kiểm thử; vòng trong chỉ dùng 12 ổ outer-train |
| Cửa sổ cùng ổ ở train và test | **Không** — chia theo ổ trước khi cắt cửa sổ |
| Thống kê chuẩn hoá từ test | **Không** — z-score/StandardScaler/imputation từ train |
| Nhãn test trong hiệu chỉnh tần số | **Không** — `PhysicsCache` chỉ dùng ổ train (có kiểm thử) |
| Rò rỉ ở cấp thiết kế | **Có, đã công bố**: PU đã bị xem ở Phase 2; việc chọn P1 làm chuẩn và thiết kế V2.1 chịu ảnh hưởng |
| Vùng ±3,5 % chọn trên KAIST | **Có (hậu nghiệm), đã công bố**; với PU vẫn đóng băng trước khi xem |
| Kiểm toán khái niệm / phép dò | chỉ mô tả, không dùng để chọn mô hình; phép dò dùng vùng động học không hiệu chỉnh |
| Can thiệp | dùng vùng của fold mà ổ là test (hiệu chỉnh từ train); mô hình B1/B3 của fold đó |

## 5. C: Hiệu lực khái niệm
Chi tiết: `docs/audit/07_PHYSICS_CONCEPT_AUDIT.md`. Tóm tắt (AUROC cấp ổ, 15 ổ BD5, mỗi ổ ngoài mẫu):
- BPFI_snr1/harm/sb: IR > H = **1,00**; BPFI_nharm 0,94. BPFO_snr1/harm 0,76; BPFO_nharm 0,70; **BPFO_sb 0,24 (ngược chiều)**.
- contrast: OR > IR = 0,88. Hư-vs-khỏe với từng khái niệm: 0,20–0,88.
- Nhân tạo: OR nhân tạo rất rõ (BPFO 0,92–0,94; contrast 0,96); IR nhân tạo yếu (0,69–0,74).
- Theo điều kiện: 900 v/ph kém nhất (BPFI_nharm 0,43); lực F04 làm yếu BPFO.
- Va chạm hài trục: BPFI không xác định ở 97,5 % cửa sổ FAG tại 900 v/ph (vạch cách 5·fr 0,81 Hz < Δf = 1 Hz).
- Từng ổ: mạnh KA04, KA16, KI04, KI18; trung bình KI21, KI14, KA30; không phân biệt được KA15, KA22; mâu thuẫn KI16.

## 6. D: Baseline (đơn vị = ổ; 15 ổ; mỗi ổ test 1 lần)
`results/phase3a/tables/baselines_summary.csv`, `baselines_per_bearing_predictions.csv`, `fig_bearing_confusion.png`.

| Mô hình | **Macro-F1 cấp ổ** [KTC 95 %] | Acc ổ | Recall H / IR / OR | Macro-F1 cửa sổ (gộp; fold TB ± SD) | ECE (→ sau TS) | Ổ sai |
|---|---|---|---|---|---|---|
| B0 luật vật lý | 0,579 [0,29; 0,80] | 0,600 | 1,0 / 0,4 / 0,4 | 0,608; 0,551 ± 0,251 | — | KA15 KA22 KA30 KI14 KI16 KI21 |
| **B1 = P1 (C chọn lồng nhau)** | **0,739 [0,49; 0,93]** | 0,733 | 0,8 / 0,8 / 0,6 | 0,640; 0,610 ± 0,189 | 0,081 → 0,063 | K005 KA15 KA22 KI16 |
| B1 mặc định C = 1 (chính theo §6) | 0,739 | 0,733 | 0,8 / 0,8 / 0,6 | 0,646; 0,618 ± 0,194 | 0,097 | như B1 |
| B2 MLP khái niệm (ensemble 3 seed) | 0,739 [0,49; 0,93] | 0,733 | 0,8 / 0,8 / 0,6 | 0,659; 0,629 ± 0,213 | 0,109 → 0,043 | như B1 (cả 3 seed) |
| B3 CNN thô (lồng nhau, ensemble) | 0,551 [0,32; 0,74] | 0,533 | 0,8 / 0,4 / 0,4 | 0,553; 0,491 ± 0,159 | 0,342 → 0,289 | K002 KA15 KA22 KA30 KI04 KI16 KI21 |
| B3 mặc định (lr 1e-3, 25 epoch; chính theo §6) | 0,611 [0,37; 0,81] | 0,600 | 0,8 / 0,4 / 0,6 | 0,603; 0,548 ± 0,220 | 0,354 | K002 KA15 KA30 KI04 KI16 KI21 |
| Chỉ nhà sản xuất (metadata) | 0,470 | 0,533 | 1,0 / 0,6 / 0,0 | — | — | 7 ổ |
| Chỉ năm đo (metadata) | 0,444 | 0,533 | 1,0 / 0,6 / 0,0 | — | — | 7 ổ |

- **So với B1** (hiệu số macro-F1 cấp ổ [KTC bootstrap ghép cặp]; McNemar chính xác): B0 −0,159 [−0,43; +0,05], 1 vs 3 ổ bất đồng,
  p = 0,63; B3 −0,188 [−0,45; +0,05], 1 vs 4, p = 0,38; B3 mặc định −0,128 [−0,43; +0,16], 2 vs 4, p = 0,69; B2: 0 bất đồng.
  **Không có khác biệt nào xác lập được**; mọi KTC đều chứa 0.
- Từng seed CNN (lồng nhau): cấp ổ 0,533–0,551; mặc định 0,551–0,663. Từng seed MLP: 0,739 cả 3.
- Nhiệt độ hiệu chỉnh vòng trong: B1 0,95–1,66; B2 1,14–2,17; **B3 4,2–20,0** (O3 chạm biên trên 20). CNN tự tin quá mức nghiêm trọng
  trên ổ chưa thấy.
- **Kiểu lỗi khác nhau:** mô hình khái niệm nhận đúng **loại** khi có bằng chứng nhưng gọi 2 ổ OR yếu (KA15, KA22) là khỏe và gọi
  KI16 (bằng chứng trái dấu) là OR. CNN **không bao giờ** gọi ổ hư là khỏe nhưng nhầm IR↔OR ở 6/10 ổ hư.
- **Thăm dò (sau khi xem kết quả):** độ tin cậy cấp ổ sau TS của B1 có thông tin về sai số (AUROC 0,77; 4 ổ sai đều ≤ 0,53),
  còn của CNN thì phản thông tin (0,36) (`EXPLORATORY_bearing_confidence.csv`).

## 7. Yếu tố gây nhiễu nhà sản xuất / năm đo (kết quả trung tâm, S-4)
- **S-4a:** quy tắc chỉ dùng metadata, học từ ổ outer-train, đạt độ chính xác cấp ổ 0,533 và nhận đúng 5/5 ổ khỏe. Trong PU-BD5,
  "khỏe" trùng hoàn toàn với "IBU, đo năm 2015".
- **S-4b (trong tầng):** IBU (K001–K005 vs KA22): khái niệm max-family AUROC cấp ổ **0,40**. B1 gọi KA22 là khỏe; B3 lồng nhau gọi
  IR, B3 mặc định gọi đúng OR. Năm 2015 (K001–K005 vs KA30, KI04): AUROC 1,00 (nhờ KI04).
- **S-4c (phép dò danh tính phiên đo):** đặc trưng thủ công dự đoán **năm đo trong 11 ổ IR với AUROC cấp ổ 1,00** và IBU-vs-không-IBU
  trong 23 ổ hư 0,87; thăm dò trong 12 ổ nhân tạo (cùng nguồn gốc hư hỏng) IBU-vs-MTK **0,94**. Khái niệm: 0,43 / 0,87 / 0,44.
  Hai phép dò chính bị nhiễu bởi tỷ lệ nhân tạo (80–89 % ở nhóm dương), nên không tách hẳn được nhà sản xuất/năm khỏi nguồn gốc
  hư hỏng. Phép dò thăm dò trong ổ nhân tạo là bằng chứng sạch hơn rằng tín hiệu phổ thông thường mang dấu vết nhà sản xuất.
- **Hệ quả:** độ chính xác của mô hình học sâu trên PU không thể quy cho vật lý hư hỏng nếu không có kiểm soát. Mọi so sánh trong
  bài phải báo cáo cùng đối chứng metadata, phân tích trong tầng và phép dò.

## 8. E: Thiết kế và kiểm chứng can thiệp giải thích
Mã: `pave/explain/interventions.py`; kiểm thử tổng hợp 9/9 (`tests/test_interventions.py`): đồng nhất khi độ mạnh = 0; xoá họ đích
> 10 dB mà không ảnh hưởng họ kia; IR có dải biên; nhiễu gần như không đổi; đối chứng không chồng vùng cấm; phát hiện chồng lấn
3·BPFO ≈ 2·BPFI (KAIST); vòng đối chứng năng lượng luôn kết thúc.

**Dữ liệu thật** (300 cửa sổ = 15 ổ × 4 điều kiện × 5 lần đo; `results/phase3a/interventions/`):

| Kiểm tra thao tác (S-6, xác định trước) | Đích | Đối chứng cùng độ rộng | Đối chứng cùng năng lượng |
|---|---|---|---|
| Δ bằng chứng họ đích, ổ hư (trung vị theo ổ), độ mạnh 1,0 | **−3,42 dB** (ổ mạnh −10,9…−14,2; ổ yếu −2,0…−3,9) | +0,02 | +0,19 |
| Δ bằng chứng họ còn lại | +0,16 | +0,01 | +0,09 |
| Ổ khỏe, Δ họ đích | −2,06 | +0,06 | +0,08 |
| Độ mạnh 0,5 | −1,71 dB | ≈ 0 | ≈ 0 |

- Méo: SNR trong dải trung vị 32–44 dB; tỷ lệ bao bị cắt ≤ 4,5 %; năng lượng toàn phần giữ 99,9 %; rò ngoài dải ≤ 1,9 % (p99 1,1 %).
- Đối chứng năng lượng đạt tỷ lệ năng lượng loại bỏ trung vị 1,27–1,31 so với đích, nhưng 10 % cửa sổ chỉ đạt < 0,66–0,73
  (thiếu hụt được báo cáo, không che giấu).
- Chồng lấn: đến 3,3 Hz của băng đích rơi vào vùng bảo vệ hài trục (đã loại khỏi can thiệp); chồng lấn với họ kia ≤ 0,07 Hz.

**Phản ứng mô hình (THĂM DÒ):**

| | Δp(lớp đúng) đích | đối chứng độ rộng | đối chứng năng lượng | Số ổ đích < cả hai đối chứng |
|---|---|---|---|---|
| B1 (khái niệm) | **−0,21** (−0,07…−0,71) | 0,00 | 0,00 | 10/10 |
| B3 (CNN) | **0,000** (|Δ| ≤ 0,03) | 0,00 | 0,00 | — |

- B1 phụ thuộc vào vạch lỗi; điều này **gần như tất yếu** vì khái niệm được tính từ chính các vạch, nên đây là kiểm tra chuỗi xử lý
  chứ không phải bằng chứng "giải thích trung thực" theo nghĩa mạnh.
- **CNN không thay đổi quyết định** khi bằng chứng tần số lỗi trong dải bao 1–20 kHz bị xoá, kể cả ở KA04, KA16, KI18 mà nó phân loại
  đúng. CNN đúng vì lý do khác (ví dụ phân bố năng lượng phổ, cấu trúc cộng hưởng). Can thiệp chỉ sửa cấu trúc bao trong dải; nội dung
  ngoài dải và năng lượng sóng mang giữ nguyên. **Không tuyên bố nhân quả.**
- Theo điều kiện: tác động lên B1 mạnh nhất ở N15_M07_F10 (−0,31), yếu nhất ở 900 v/ph (−0,08).

## 9. Các trường hợp thất bại
| Ổ | Thực tế | Nhận định |
|---|---|---|
| KA15 | OR, vết lõm mức 1 | không có vạch (BPFO_harm 3,9 dB ≈ khỏe); mọi mô hình khái niệm gọi khỏe; CNN gọi IR |
| KA22 | OR, rỗ mức 1, **IBU 2014** | không có vạch (4,1 dB); khái niệm gọi khỏe; ổ hư IBU duy nhất |
| KI16 | IR, rỗ **mức 3** | bằng chứng trái dấu (contrast +3,8); mọi mô hình gọi OR; BPFI không xác định ở 900 v/ph (FAG) |
| K005 | khỏe | B1/B2 gọi OR (độ tin cậy thấp 0,41); B0 và CNN đúng |
| KA30, KI04, KI21, K002 | | CNN sai; mô hình khái niệm đúng (KI04, KI21 có bằng chứng rõ / trung bình) |

## 10. Giới hạn
- 15 ổ độc lập; mọi KTC rất rộng; không đủ lực suy luận. Kết quả cửa sổ chỉ mô tả.
- PU đã được xem ở Phase 2: kết quả Phase 3A là **đánh giá nội bộ có công bố việc xem trước**, không phải kiểm định ngoài.
- Healthy ≡ IBU ≡ 2015 không thể tách trong PU-BD5.
- Khái niệm và mô hình đọc cùng tín hiệu; "độ trung thực" khái niệm là tự nhất quán.
- Can thiệp chỉ trong miền bao của dải 1–20 kHz; không loại trừ khả năng CNN dùng thông tin vật lý ở dạng khác.
- Giá trị từng fold trong của B2 và các cấu hình B3 không được lưu (sai lệch nhỏ khỏi giao thức).
- B3 dùng sóng thô (không theo bậc) theo V2.1; kết luận về CNN chỉ áp dụng cho kiến trúc CNN-1D nhỏ đã thử.

## 11. Cổng Phase 3A và khuyến nghị: **MODIFY**
**Không STOP**: có bằng chứng vật lý đo được, chuyển được sang ổ chưa thấy, giải thích được (B1, B2), hiệu chỉnh tốt, độ tin cậy có
thông tin; công cụ can thiệp hoạt động như thiết kế.
**Không GO nguyên trạng**, vì:
1. Không có mô hình học sâu nào vượt B1; CNN kém hơn, quá tự tin, không nhạy với bằng chứng vật lý và có nguy cơ học lối tắt nhà sản xuất.
2. Chọn siêu tham số lồng nhau không đủ thông tin với 12 ổ.
3. Một khái niệm không hợp lệ (`BPFO_sb`) và một chế độ không xác định (900 v/ph, FAG) cần sửa trước.

**Sửa đổi đề xuất trước Phase 3B** (cần tác giả quyết định; chi tiết trong `docs/PHASE3B_DESIGN.md`):
1. Bộ khái niệm V3: bỏ `BPFO_sb`; thêm cờ khả nhận diện (khoảng cách vạch tới hài trục so với Δf) và mặt nạ NaN thay cho imputation;
   tổng hợp khái niệm trên cả lần đo 4 s (Δf = 0,25 Hz) để xử lý 900 v/ph. Đóng băng bằng giao thức V2.2.
2. Siêu tham số: mặc định cố định, khai báo trước; vòng trong chỉ dùng cho nhiệt độ và PE-SD, cùng chẩn đoán độ ổn định.
3. Ưu tiên nghiên cứu **PE-SD trên nền B1** (độ tin cậy B1 đã có thông tin) và **ứng viên B** (neo vật lý, nhánh latent có cổng và bị kiểm
   toán). Ứng viên A (PECB thuần với bộ mã hoá CNN) chỉ thử như so sánh, vì bộ mã hoá CNN hiện không dùng bằng chứng vật lý.
4. Mọi mô hình có nhánh latent phải qua: phép dò nhà sản xuất/năm trên biểu diễn, can thiệp đích-vs-đối chứng, phân tích trong tầng IBU.
5. **Khoá một bộ dữ liệu ngoài** (nhiều ổ độc lập, chưa xem) trước khi tuyên bố bất kỳ ưu thế nào.
