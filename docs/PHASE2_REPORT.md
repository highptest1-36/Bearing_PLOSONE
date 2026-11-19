# PAVE-Net — Báo cáo Phase 2 (baseline có giới hạn, trước khi xây mô hình đầy đủ)

Ngày: 19/11/2026. Giao thức: `docs/RESEARCH_PROTOCOL_V2.md` (đóng băng, SHA-256 `60d0c66a…6d15`).
Mọi số liệu dưới đây được sinh tự động từ `results/phase2/runs/*/runs.jsonl` bởi `scripts/report_phase2.py` và
`scripts/physics_diagnostics_phase2.py`. **Chưa triển khai PAVE-Net đầy đủ.**

## 0. Tóm tắt

- **Cổng V1–V4 đều đạt.** V3 (hiệu lực khái niệm trên ổ chưa thấy) đạt nhưng sát ngưỡng ở tiêu chí hư-vs-khỏe
  (AUROC 0,775, ngưỡng 0,75; fold 2 chỉ đạt 0,451).
- **KAIST bão hoà:** hầu hết mô hình đạt macro-F1 ≈ 1,0 cho cả 3 lớp và 7 lớp; riêng **chỉ-RMS** đã đạt 0,886. KAIST không
  phân biệt được phương pháp và không được dùng làm bằng chứng hiệu năng.
- **Paderborn PU-BD5 (tách theo ổ):** chỉ 9 đặc trưng vật lý + hồi quy logistic (**P1**) là baseline tốt nhất
  (macro-F1 cửa sổ 0,618 ± 0,185; độ chính xác cấp ổ 0,733; ECE 0,139). CNN đạt 0,49–0,55 (cấp ổ 0,56–0,62) và hiệu chỉnh
  xác suất kém (ECE 0,34–0,37). Độ biến thiên giữa fold rất lớn.
- **Lối tắt:** trên PU, chỉ-tốc-độ và chỉ-RMS ở mức ngẫu nhiên. Nhưng PU có yếu tố gây nhiễu mới: **Healthy ≡ nhà sản xuất
  IBU ≡ đo năm 2015**. Quy tắc chỉ dùng năm đo đạt độ chính xác cấp ổ 0,60, ngang CNN.
- **Vật lý:** vùng tần số hiệu chỉnh chỉ từ train tái tạo đúng độ lệch KAIST (BPFO +1,5…+2,0 %, BPFI −1,2 %) và trên PU
  trùng bậc động học (|lệch| ≤ 0,44 %). Bằng chứng rõ ở 4/10 ổ hư test, yếu ở 5/10, mâu thuẫn ở 1/10 (KI16).

## 1. A–B: Blueprint và giao thức V2

| Tài liệu | Nội dung |
|---|---|
| `01_PAVE_NET_RESEARCH_BLUEPRINT_V2.md` | Kiến trúc V2 (không nhánh nhiệt), RQ1–RQ5, ma trận thí nghiệm Phase 2 và ablation A0–A5 |
| `docs/RESEARCH_PROTOCOL_V2.md` + `docs/PROTOCOL_FREEZE.txt` | Giao thức đóng băng **trước** mọi phân tích tín hiệu PU |
| `99_ARCHIVE/blueprint_v1_multimodal/` | Blueprint V1, nguyên vẹn (SHA-256 trùng bản gốc) |
| `docs/DECISION_LOG.md` | Các quyết định Phase 2, kể cả các bổ sung hậu nghiệm |

Kết quả Phase 1 không bị xoá hay ghi đè; toàn bộ đầu ra Phase 2 nằm trong `results/phase2/` và `03_MANIFESTS/paderborn/`.

## 2. C: Paderborn — kiểm toán và chia dữ liệu

Chi tiết: `docs/audit/06_PADERBORN_METADATA_AUDIT.md`. Tóm tắt:
- 32 ổ × 80 lần đo, kích thước khớp máy chủ; 1 tệp hỏng (`N15_M01_F10_KA08_2.mat`) bị loại.
- Hình học theo từng ổ từ phiếu mô tả (FAG 28,55 mm; IBU/MTK 29,05 mm).
- Mô-men đo lệch danh định khoảng +0,55 Nm; 44 % tệp dài hơn 4 s.
- **Gây nhiễu:** 6/6 ổ khỏe là IBU và đo năm 2015; 9/11 ổ hư thật đo năm 2014.

Chia dữ liệu (`03_MANIFESTS/paderborn/splits/`, có mã băm; `tests/test_paderborn.py`):

| Giao thức | Train | Validation | Test |
|---|---|---|---|
| PU-BD5 (5 fold, Lessmeier Bảng 10) | 3 hàng = 9 ổ | 1 hàng = 3 ổ | 1 hàng = 3 ổ (H, OR, IR) |
| PU-A2R | 12 ổ nhân tạo/khỏe | 5 ổ nhân tạo/khỏe | K001 + 11 ổ hư thật |
| Thử thách | — | — | KB23, KB24, KB27 (chỉ mô tả) |

Mọi lần đo và mọi điều kiện của một ổ chỉ thuộc một vai trò. Mỗi ổ PU-BD5 được test đúng một lần.

## 3. D: Tiền xử lý và baseline đã triển khai

- Cửa sổ 1 s không chồng lấn, cắt **sau** khi gán vai trò theo bản ghi/ổ. KAIST: 1 620 cửa sổ (2 kênh housing A).
  PU: 10 181 cửa sổ thô (hạ mẫu ×2 → 32 kHz) và 8 955 cửa sổ theo bậc.
- Chuẩn hoá theo bậc: lấy mẫu lại miền góc với tốc độ đo (KAIST: 512 mẫu/vòng × 48 vòng; PU: 1 024 × 24 vòng).
- Mã nguồn: `pave/data/{paderborn,windowing}.py`, `pave/signal/order.py`, `pave/features/{handcrafted,physics}.py`,
  `pave/models/cnn1d.py` (≈ 0,23 M tham số), `pave/training/engine.py` (AdamW, cosine, dừng sớm theo macro-F1 validation),
  `pave/evaluation/metrics.py`, `pave/experiments/phase2.py`, `scripts/run_phase2.py`.
- Chuẩn hoá z-score, imputation, StandardScaler và hiệu chỉnh vùng vật lý: **chỉ từ train** của từng fold.

## 4. E: Bộ trích tần số vật lý

- Định nghĩa đóng băng (giao thức §6): tìm vạch ±3,5 % (nguồn gốc hậu nghiệm từ KAIST) → hiệu chỉnh độ lệch tương đối từ
  cửa sổ TRAIN của lớp tương ứng (SNR ≥ 10 dB, ≥ 20 cửa sổ; tâm = trung vị, nửa rộng = max(3·MAD, 0,5 %)) →
  9 đặc trưng: SNR vạch, SNR trung bình hài 1–3, số hài ≥ 6 dB (nhất quán họ hài), SNR dải biên ±fr, contrast hai họ.
- **Kiểm thử tổng hợp (10, đều đạt)** trong `tests/test_phase2_physics.py`: phát hiện vạch lệch +2 % và họ hài; dải biên IR;
  từ chối nhiễu trắng; hiệu chỉnh chỉ dùng lớp tương ứng và quay về ±3,5 % khi thiếu dữ liệu; vùng hẹp loại vạch sai;
  bất biến bậc khi đổi tốc độ; kiểm tra lỗi đầu vào.
- Chẩn đoán dữ liệu thật: mục 6.3.

## 5. F: Thí nghiệm đã chạy (có giới hạn)

| Bộ | Giao thức | Bài toán | Lượt chạy |
|---|---|---|---|
| KAIST | LOLO (3 fold) | 3 lớp: 8 mô hình; 7 lớp (thăm dò): 4 mô hình | 66 + 3 chẩn đoán |
| PU | BD5 (5 fold) | 3 lớp: 8 mô hình | 70 + 5 chẩn đoán |
| PU | A2R | 3 lớp: 8 mô hình | 14 + 1 chẩn đoán |

99 lượt CNN (3 seed: 42/43/44), tổng **15,5 phút GPU** (RTX 3060 Laptop 6 GB), thấp hơn nhiều ngân sách 3 giờ. Không HPO.

## 6. G: Kết quả (báo cáo tách biệt)

### 6.1 Phân loại (test; trung bình ± SD qua fold × seed)

**PU-BD5 — giao thức trung tâm, ổ chưa thấy**

| Mô hình | macro-F1 cửa sổ | Fold thấp nhất | Bal. acc | ECE | Acc cấp ổ | Acc ổ × điều kiện |
|---|---|---|---|---|---|---|
| C1 thủ công + RF | 0,443 ± 0,185 | 0,158 | 0,523 | 0,338 | 0,600 | 0,517 |
| C2 thủ công + SVM | 0,518 ± 0,118 | 0,344 | 0,608 | 0,344 | 0,667 | 0,617 |
| **P1 vật lý + LR** | **0,618 ± 0,185** | **0,472** | **0,637** | **0,139** | **0,733** | 0,600 |
| D1 CNN thô | 0,486 ± 0,174 | 0,222 | 0,547 | 0,370 | 0,556 | 0,550 |
| D2 CNN theo bậc | 0,547 ± 0,254 | 0,222 | 0,613 | 0,364 | 0,600 | 0,622 |
| D3 CNN bậc + chuẩn hoá RMS | 0,519 ± 0,178 | 0,222 | 0,587 | 0,343 | 0,622 | 0,600 |

- Recall theo lớp (trung bình): CNN nghiêng về Healthy (H 0,80–0,83; IR 0,31–0,50; OR 0,44–0,63); P1 cân bằng hơn (0,61 / 0,70 / 0,60).
- Mỗi fold chỉ có 3 ổ test, nên độ chính xác cấp ổ của một fold chỉ nhận 0, 1/3, 2/3, 1. Chênh lệch giữa các mô hình
  (≤ 0,18 macro-F1) **nhỏ hơn** SD giữa fold. Không thể khẳng định mô hình nào tốt hơn với ý nghĩa thống kê.
- Ổ khó với hầu hết mô hình: KA15 (chỉ C2 đúng), KA30 (chỉ P1 đúng), KI16 (C1, C2, P1 sai; CNN đúng ở 1/3–2/3 seed);
  KI04 và KI21 đúng với P1 nhưng sai với mọi CNN (`T_classification_per_fold.csv`).

**PU-A2R — học trên hư hỏng nhân tạo, test trên hư hỏng thật (1 split)**

| Mô hình | macro-F1 cửa sổ | Acc cấp ổ (12 ổ) |
|---|---|---|
| C1 / C2 | 0,238 / 0,517 | 0,333 / 0,583 |
| P1 | 0,538 | 0,667 |
| D1 / D2 / D3 | 0,606 ± 0,056 / 0,623 ± 0,079 / 0,657 ± 0,071 | 0,583 / 0,583 / 0,667 |

**KAIST LOLO (chuyển tải trong cùng lần lắp đặt)**

- 3 lớp: C1, C2, P1, D1, D2 có macro-F1 1,000; D3 0,988. 7 lớp (thăm dò): C1, D1, D2 1,000; P1 0,911.
- **7 lớp đạt 100 % là do nhận dạng lần lắp đặt (Phase 1), không phải mức độ hư hỏng.** Không được trình bày như khả năng
  phân loại mức độ.

**Thử thách KB23/24/27 (hư hỏng kép, chỉ mô tả):** các mô hình tín hiệu chia dự đoán chủ yếu giữa IR và OR; một số gán
Healthy cho 10–43 % cửa sổ KB24 (`T_challenge_KB.csv`). Không có nhãn đúng duy nhất.

### 6.2 Kiểm soát lối tắt

| Đối chứng | KAIST 3 lớp (test 0 / 2 / 4 Nm) | PU-BD5 | PU-A2R |
|---|---|---|---|
| S1 chỉ-tốc-độ, macro-F1 cửa sổ | 0,222 / 0,655 / 0,789 (cấp bản ghi 0,43 / 0,57 / **1,00**) | 0,249 (ngẫu nhiên) | 0,154 |
| S2 chỉ-RMS, macro-F1 cửa sổ | 0,886 / 0,886 / 0,886 | 0,224 ± 0,147 | 0,225 |
| D1 → D2 (thô → theo bậc) | 1,000 → 1,000 | 0,486 → 0,547 | 0,606 → 0,623 |
| D2 → D3 (+ chuẩn hoá RMS) | 1,000 → 0,988 | 0,547 → 0,519 | 0,623 → 0,657 |
| Quy tắc chỉ-nhà-sản-xuất (hậu nghiệm), acc cấp ổ | — | 0,533 | — |
| Quy tắc chỉ-năm-đo (hậu nghiệm), acc cấp ổ | — | 0,600 | — |

Diễn giải:
- **KAIST:** lối tắt tốc độ hiện diện đúng như dự đoán (mạnh nhất khi test 4 Nm, nơi Healthy chậm hơn 0,30 Hz). Năng lượng
  RMS gần như đủ để phân loại. Chuẩn hoá theo bậc và RMS không làm giảm hiệu năng vì bài toán đã bão hoà bởi các tín hiệu khác.
- **PU:** không có lối tắt tốc độ/RMS. Khác biệt D1/D2/D3 nằm trong nhiễu giữa fold, nên **không đủ bằng chứng** về
  lợi ích của chuẩn hoá theo bậc.
- **PU (metadata):** một quy tắc chỉ biết năm đo đạt cùng mức với CNN ở cấp ổ. Đánh giá tách theo ổ không loại trừ được
  việc mô hình học "dấu vết phiên đo". P1 vượt các quy tắc này (0,733) nhưng chênh lệch tương ứng chỉ khoảng 2 ổ trên 15.

### 6.3 Đặc trưng vật lý (khái niệm)

**Vùng hiệu chỉnh chỉ từ train** (độ lệch tương đối so với bậc động học; `T_physics_features.csv`):

| | BPFO tâm (nửa rộng) | BPFI tâm (nửa rộng) |
|---|---|---|
| KAIST, 3 fold | +1,50…+1,96 % (1,27–2,00 %) | −1,19…−1,20 % (0,50 %) |
| PU-BD5, 5 fold | −0,44…+0,33 % (0,50–1,50 %) | −0,17…−0,18 % (0,50–1,26 %) |
| PU-A2R (train nhân tạo) | 0,00 % (3,50 %: vạch OR nhân tạo phân tán) | −0,17 % (0,50 %) |

- Trên KAIST, quy trình hiệu chỉnh tự tìm lại độ lệch đã thấy ở Phase 1. Đây **không phải** kiểm định độc lập, vì vùng
  tìm kiếm ±3,5 % được chọn sau khi xem KAIST.
- Trên PU (đóng băng trước khi xem), hình học theo phiếu mô tả dự đoán vị trí vạch trong phạm vi ±0,5 %. Vùng ±3,5 % không
  cần thiết nhưng cũng không gây hại.

**Cổng V3 (PU-BD5, ổ test chưa thấy):**

| Fold (ổ test) | AUROC contrast OR vs IR | AUROC họ max: hư vs khỏe |
|---|---|---|
| f1 (K001, KA04, KI04) | 0,999 | 0,991 |
| f2 (K002, KA15, KI14) | 0,717 | **0,451** |
| f3 (K003, KA16, KI16) | 0,960 | 0,856 |
| f4 (K004, KA22, KI18) | 0,988 | 0,770 |
| f5 (K005, KA30, KI21) | 0,861 | 0,810 |
| **Trung bình** | **0,905** (ngưỡng 0,75: đạt) | **0,775** (ngưỡng 0,75: đạt sát) |

**Theo từng ổ test** (`T_physics_pu_bd5_per_bearing.csv`, `fig_p2_pu_physics_per_bearing.png`):
- **Bằng chứng mạnh, đúng hướng:** KA04, KA16 (OR, rỗ mỏi; điểm họ 20–21 dB, đủ 3/3 hài); KI04, KI18 (IR; dải biên trục
  11,6 / 21,1 dB). Đúng dấu contrast ở 75–100 % cửa sổ.
- **Yếu, chồng lấn với ổ khỏe** (điểm họ 4–7 dB so với 3,7–5,7 dB ở ổ khỏe): KA15, KA30 (vết lõm), KA22, KI14, KI21 (mức 1).
- **Mâu thuẫn:** KI16 (rỗ mỏi **mức 3**) có contrast mang dấu OR ở 86 % cửa sổ. Bằng chứng kỳ vọng vắng mặt hoặc trái
  ngược; chưa rõ nguyên nhân.
- **Theo điều kiện:** AUROC hư-vs-khỏe 0,63 ở 900 v/ph so với 0,79–0,82 ở 1500 v/ph; contrast 0,85–0,90 ở mọi điều kiện.
- **A2R:** vùng hiệu chỉnh trên ổ nhân tạo vẫn phân biệt OR/IR trên ổ thật (AUROC 0,893), nhưng **không** phân biệt hư/khỏe
  (0,584).
- **Kết luận vật lý:** khái niệm "loại hư hỏng" (OR vs IR) chuyển được sang ổ chưa thấy và từ nhân tạo sang thật. Khái niệm
  "có hư hỏng hay không" chỉ đáng tin với hư hỏng rỗ mỏi rõ; yếu với vết lõm, hư hỏng mức 1 và tốc độ thấp.

## 7. H: Giả định và giới hạn

**Giả định**
- KAIST: ánh xạ Point1/2 → housing A x/y; đơn vị m/s² theo metadata; tốc độ ước lượng từ đỉnh 2X trong từng cửa sổ.
- PU: tốc độ không đổi trong một lần đo 4 s (độ lệch chuẩn ≈ 0,1 v/ph); hình học theo phiếu mô tả; nhãn theo Bảng 4–5,10.
- Tần số đặc trưng là tham chiếu động học; độ lệch thực tế được hiệu chỉnh từ train, không phải hằng số vật lý.

**Giới hạn dữ liệu**
- KAIST: không lặp mẫu ổ; lớp ≡ ngày đo; mức độ ≡ lần lắp đặt; tốc độ Healthy chưa hiệu chỉnh; vùng ±3,5 % hậu nghiệm.
- PU: 15 ổ trong PU-BD5 (3 ổ test/fold, 3 ổ validation/fold) nên ước lượng rất nhiễu; Healthy ≡ IBU ≡ năm 2015; phần lớn
  hư hỏng thật ở mức 1; 1 tệp hỏng; đơn vị rung không xác định; mô-men đo lệch danh định.
- Validation 3 ổ khiến chọn epoch của CNN gần như ngẫu nhiên (epoch tốt nhất thường là 0–3).

**Giới hạn phương pháp**
- CNN dùng siêu tham số cố định, không HPO. Kết quả CNN là baseline tham chiếu, không phải giới hạn trên.
- Đối chứng metadata và việc xem kết quả chạy thử fold 1 xảy ra trước khi bổ sung đối chứng đó: được ghi là **hậu nghiệm**.
- Không có p-value: đơn vị độc lập là ổ trục (15 trong PU-BD5).

## 8. Kiểm định cổng và đề xuất

| Cổng | Kết quả | Bằng chứng |
|---|---|---|
| V1 Giao thức | **Đạt** | 78 kiểm thử (kể cả manifest thật KAIST và PU); tách theo ổ/bản ghi; mã băm khớp |
| V2 Bộ trích vật lý | **Đạt** | 10/10 kiểm thử tổng hợp |
| V3 Hiệu lực khái niệm (PU, độc lập) | **Đạt (sát ngưỡng)** | AUROC contrast 0,905; hư-vs-khỏe 0,775 (fold 2: 0,451) |
| V4 Thiết kế đánh giá | **Đạt** | S1 chỉ-tốc-độ PU = 0,249 ≤ 0,45; mọi kết quả có cấp ổ |

**Đề xuất cho Phase 3 (cần tác giả quyết định):**
1. Được phép bắt đầu PECB theo cổng, nhưng **mốc so sánh chính phải là P1** (khái niệm vật lý + LR), không chỉ CNN. PECB
   chỉ có giá trị nếu vượt P1 về hiệu năng, độ hiệu chỉnh hoặc khả năng hoãn có căn cứ.
2. Điểm yếu thực sự là **phát hiện hư hỏng nhẹ** (mức 1, vết lõm, 900 v/ph). Đây là nơi PE-SD (hoãn khi bằng chứng yếu)
   có thể có ích và cần được đo bằng risk–coverage.
3. Báo cáo bắt buộc yếu tố gây nhiễu nhà sản xuất/năm đo của PU cùng mọi kết quả, kèm đối chứng metadata.
4. Cân nhắc sửa giao thức validation (ví dụ validation lồng nhau leave-one-row-out trên 4 hàng train), vì 3 ổ validation
   quá nhiễu. Việc này cần ghi là sửa đổi giao thức.
5. KAIST chỉ dùng cho minh hoạ vật lý và phân tích lối tắt, không dùng để xếp hạng mô hình.
