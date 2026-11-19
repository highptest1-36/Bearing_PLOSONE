# PAVE-Net — Báo cáo Phase 3B (phát triển trên Paderborn PU-BD5)

Ngày 19/11/2026. Giao thức: V2.2 (`078d726b…`) + V2.3 (`6b4e4fdd…`) + V2.4 (`dd758b7f…`), cả ba đóng băng **trước** mọi
kết quả phân loại Phase 3B. Paderborn đã được xem ở Phase 2–3A, nên kết quả dưới đây là **đánh giá phát triển**, không phải
xác nhận mù. Đơn vị độc lập là **ổ trục** (15 ổ: 5 H, 5 IR, 5 OR); cửa sổ chỉ là quan sát mô tả.
Nguồn số liệu: `results/phase3b/tables/*` (sinh bởi `scripts/report_phase3b.py`, `pev_report_phase3b.py`,
`efficiency_phase3b.py`). Kiểm thử: **110/110 đạt**.

## 1. Kiến trúc đã triển khai
| ID | Mô hình | Đầu vào | Ghi chú |
|---|---|---|---|
| B0 | luật vật lý V3 | khái niệm 4 s | ngưỡng τ trên f_max (train), lớp theo dấu contrast |
| B1-V2 | P1 (LR, 9 khái niệm V2) | 1 s | baseline chuẩn do tác giả chỉ định |
| B1-V3-1s / B1-V3-4s | LR trên [z, m] V3 | 1 s / 4 s | tách hiệu ứng độ dài ngữ cảnh |
| B2 | MLP 2×64 trên [z, m] | 1 s | theo V2.2 (V2.3 không chuyển B2 sang 4 s) |
| B3 | CNN-1D (0,23 M tham số) | sóng thô 1 s | không ràng buộc vật lý |
| A1 = B4 | đầu vật lý tuyến tính | khái niệm 4 s | không latent (γ = 0) |
| A2 | + đặc trưng độ tin cậy q | 4 s | |
| A3 | + nhánh latent bị chặn (B = 3), không cổng | 4 s + 1 s thô | |
| **A4 PAVE-Net** | hợp nhất có cổng tin cậy bằng chứng | 4 s + 1 s thô | ℓ = ℓ_phys + (1 − g)·3·tanh(ℓ_lat/3), g = σ(uᵀq̃ + b) |
| A6 | concept bottleneck (CNN → 8 khái niệm → tuyến tính) | 1 s thô | mất mát khái niệm có mặt nạ |
Toán học và cấu hình: `docs/RESEARCH_PROTOCOL_V2.2.md` §4–§6; mã: `pave/features/physics_v3.py`, `pave/models/pavenet.py`,
`pave/training/multi.py`, `scripts/run_phase3b.py`. AdamW, siêu tham số cố định, seed 42/43/44 (đặt trước khi khởi tạo mô
hình), checkpoint + SHA-256 (`results/phase3b/checkpoints.csv`, 126 tệp: 105 fold + 21 FINAL).

## 2. Ba phát hiện về khái niệm V3 (trước mọi kết quả phân loại)
1. **1 s không đủ phân giải** (V2.3): với hình học 6203, BPFO ≈ 3,05·fr và BPFI ≈ 4,93–4,95·fr cách 3·fr / 5·fr chỉ 0,8–1,8 Hz.
   Ở Δf = 1 Hz không họ nào nhận diện được (δ −0,24…−1,19 bin). Ở 4 s tất cả nhận diện được, trừ BPFI của FAG ở 900 v/ph
   (δ = 0,77). **Ghi dài hơn không tự động giải quyết mọi va chạm.**
2. **Tiêu chí theo vùng che toàn bộ bằng chứng** (V2.4): vùng hiệu chỉnh từ train rộng ±1,3–1,7 % (bất định vị trí vạch giữa các ổ),
   nên 0 % họ hợp lệ, kể cả ở 4 s. Sửa thành kiểm tra trên **vạch đo được** (δ ≥ 1 và là cực đại cục bộ). Độ hợp lệ khi đó
   55–98 % tuỳ điều kiện và nhà sản xuất (thấp nhất ở IBU và lực F04).
3. Độ bất định tốc độ σ_r trung vị 0,009 Hz; một lần đo đạt 2,57 Hz và bị mặt nạ xử lý.

## 3. Kết quả baseline (cấp ổ; ensemble 3 seed; `main_results.csv`, `per_bearing_predictions.csv`)
| Mô hình | Macro-F1 [KTC 95 %] | Bal. acc | Recall H/IR/OR | Báo động giả H | Macro-F1 cửa sổ | NLL | ECE |
|---|---|---|---|---|---|---|---|
| B0 luật vật lý | 0,655 [0,40; 0,87] | 0,667 | 1,0/0,6/0,4 | 0,0 | 0,658 | — | — |
| B1-V2 (P1) | 0,739 [0,49; 0,93] | 0,733 | 0,8/0,8/0,6 | 0,2 | 0,646 | 0,757 | 0,064 |
| B1-V3-1s | 0,648 [0,41; 0,87] | 0,667 | 0,6/1,0/0,4 | 0,4 | 0,362 | 1,063 | 0,032 |
| **B1-V3-4s** | **0,780 [0,56; 0,93]** | **0,800** | 1,0/1,0/0,4 | **0,0** | **0,729** | **0,625** | 0,064 |
| B2-V3 (1 s) | 0,591 [0,33; 0,80] | 0,600 | 0,6/0,8/0,4 | 0,4 | 0,354 | 1,069 | 0,034 |
| B3 CNN | 0,611 [0,37; 0,81] | 0,600 | 0,8/0,4/0,6 | 0,2 | 0,605 | 1,117 | 0,311 |
| A1/B4 | 0,780 [0,56; 0,93] | 0,800 | 1,0/1,0/0,4 | 0,0 | 0,718 | 0,642 | 0,037 |
| A2 | 0,780 [0,56; 0,93] | 0,800 | 1,0/1,0/0,4 | 0,0 | 0,682 | 0,704 | 0,075 |
| A3 | 0,611 [0,37; 0,81] | 0,600 | 0,8/0,4/0,6 | 0,2 | 0,605 | 1,025 | 0,280 |
| A4 PAVE-Net | 0,611 [0,37; 0,81] | 0,600 | 0,8/0,4/0,6 | 0,2 | 0,619 | 0,946 | 0,199 |
| A6 concept bottleneck | 0,733 [0,52; 0,93] | 0,733 | 1,0/0,6/0,6 | 0,0 | 0,650 | 0,966 | 0,282 |

- **So với B1-V2:** B1-V3-4s +0,041 [−0,18; +0,25], 1 vs 2 ổ bất đồng, McNemar p = 1,0. **Không có khác biệt nào xác lập được**
  (mọi KTC chứa 0).
- **Hiệu ứng ngữ cảnh** (cùng LR, cùng khái niệm V3): 1 s → 4 s tăng 0,648 → 0,780; cửa sổ 0,362 → 0,729. Hiệu ứng **kiến trúc**
  (cùng đầu vào 4 s): A1, A2 = B1-V3-4s; A3, A4 kém hơn. Cải thiện so với P1 đến từ ngữ cảnh 4 s, **không** từ kiến trúc nơ-ron.
- Từng seed (`per_seed.csv`): A1, A2 ổn định 0,780 ở cả 3 seed; A4 0,611–0,678; B3 0,611–0,663; A6 0,663–0,678.
- Từng fold (cửa sổ): B1-V3-4s 0,97 / 0,54 / 0,84 / 0,68 / 0,57; fold O2 (K002, KA15, KI14) khó nhất với mọi mô hình.
- **Lỗi chung của mô hình vật lý:** KA15, KA22, KA30 (OR, mức 1; hai ổ lõm do mảnh vụn) bị gọi khỏe. Cả ba có vạch BPFO ≈ mức khỏe
  (Phase 3A). B1-V3-4s **sửa được** K005 và KI16 so với B1-V2.

## 4. Ablation (`Table4_ablation.csv`)
| ID | Thành phần | Macro-F1 ngoài | Ổ yếu đúng (/6) | Ổ mạnh đúng (/4) | Macro-F1 OOF trong | NLL OOF trong |
|---|---|---|---|---|---|---|
| A0 | luật vật lý | 0,655 | 1 | 4 | — | — |
| A1 | khái niệm V3 (đầu tuyến tính) | 0,780 | 3 | 4 | 0,762 | 0,709 |
| A2 | + độ tin cậy | 0,780 | 3 | 4 | 0,687 | 0,774 |
| A3 | + latent, không cổng | 0,611 | 2 | 3 | 0,542 | 2,352 |
| A4 | hợp nhất có cổng | 0,611 | 2 | 3 | 0,541 | 1,389 |
| A5 | A4 + PE-SD | xem §6 | | | | |
| A6 | concept bottleneck | 0,733 | 3 | 3 | 0,548 | 2,565 |
- **Quy tắc chọn định trước (chỉ OOF vòng trong, 60 dự đoán ổ):** A4 − A2 = −0,146 (< +0,05) và NLL kém hơn ⇒ **loại nhánh latent**.
  Mô hình vật lý cuối: B1-V3-4s (OOF 0,704 > B1-V3-1s 0,600). **Mô hình đề xuất cuối cùng = B1-V3-4s.**
- **RQ2 (H2) không được ủng hộ:** cổng hoạt động đúng định tính (g trung bình 0,52–0,63 ở ổ có bằng chứng mạnh; 0,36–0,43 ở ổ yếu
  và ổ khỏe; `a4_gate_by_bearing.csv`), nhưng chính ở các ổ yếu, nhánh CNN lại sai. Nhường quyền cho nhánh học làm **mất** 1 ổ yếu
  và 1 ổ mạnh. Thêm đặc trưng độ tin cậy vào đầu vật lý (A2) không đổi kết quả cấp ổ và làm NLL OOF xấu đi.

## 5. PEV — kiểm chứng phụ thuộc vào bằng chứng (`pev_targeted_minus_control.csv`, 300 bản ghi 4 s, 10 ổ hư)
| Đại lượng (họ của chính ổ, độ mạnh 1,0) | Đích − đối chứng năng lượng [KTC 95 %] | Số ổ đích < đối chứng |
|---|---|---|
| Bằng chứng họ đích (dB) | −9,29 [−12,90; −5,74] | 10/10 |
| Δp(lớp đúng) B1-V3-4s | **−0,409 [−0,623; −0,206]** | 10/10 |
| Δp A2 | −0,385 [−0,585; −0,205] | 10/10 |
| Δp A4 PAVE-Net | −0,103 [−0,183; −0,042] | 10/10 |
| Δp A6 concept bottleneck | −0,008 [−0,018; −0,001] | 9/10 |
| Δp B3 CNN | +0,005 [0,000; +0,011] | 3/10 |
| Logit vật lý của A4 / phần latent của A4 | −1,11 [−1,50; −0,70] / +0,13 [−0,17; +0,42] | 10/10 / 4/10 |
- Tỷ lệ bản ghi đổi dự đoán dưới can thiệp đích (đối chứng năng lượng): B1-V3-4s 45,5 % (2,0 %), A2 48,0 % (2,0 %),
  A4 4,5 % (0,5 %), A6 0 % (0 %), B3 0 % (0 %).
- Méo: SNR trong dải trung vị 31–42 dB; cắt bao ≤ 7,7 %; rò ngoài dải ≤ 1,5 %; năng lượng toàn phần giữ ≥ 99,8 %. Đối chứng năng
  lượng đạt tỷ lệ năng lượng loại bỏ trung vị 1,19, nhưng 10 % bản ghi chỉ đạt < 0,50 (thiếu hụt được báo cáo, không che giấu).
  Chồng lấn đã loại: tới 8,5 Hz với vùng bảo vệ hài trục, tới 1,4 Hz với họ còn lại.
- **Diễn giải:** (i) mô hình vật lý *phụ thuộc* vào chính bằng chứng mà giải thích của nó nêu tên, nên giải thích **trung thực**
  theo nghĩa phụ thuộc mô hình; (ii) A6 dự đoán khái niệm nhưng các khái niệm dự đoán gần như **không** đổi khi bằng chứng phổ bị xoá,
  tức "khái niệm học được" không trung thực với bằng chứng đo; (iii) CNN và nhánh latent không phụ thuộc vào vạch lỗi trong miền bao.
  Điều này **không** chứng minh CNN không dùng vật lý ở dạng khác. Không có tuyên bố nhân quả về vật lý ổ trục.

## 6. PE-SD — chẩn đoán chọn lọc (`selective_diagnosis.csv`, cấp ổ, 15 ổ)
| Mô hình nền | Điểm | AURC | Rủi ro @12/15 (lỗi) | Rủi ro @9/15 (lỗi) | AUROC phát hiện lỗi |
|---|---|---|---|---|---|
| B1-V3-4s (3 lỗi) | S1 max-prob | **0,075** | 0,167 (2) | 0,111 (1) | 0,78 |
| | S2 entropy | 0,113 | 0,167 (2) | 0,222 (2) | 0,67 |
| | S3 độ rõ bằng chứng | 0,146 | 0,250 (3) | 0,333 (3) | 0,50 |
| | S4 vật lý-nhận thức | 0,114 | 0,250 (3) | 0,222 (2) | 0,61 |
| A4 (6 lỗi) | S1 | 0,367 | 0,417 (5) | 0,444 (4) | 0,50 |
| | S4 | **0,148** | 0,333 (4) | 0,222 (2) | 0,89 |
- **H4 không được ủng hộ cho mô hình cuối:** trên B1-V3-4s, độ tin cậy thông thường (S1) đã xếp lỗi tốt nhất. Điểm vật lý-nhận thức
  chỉ có ích khi mô hình nền là mô hình có nhánh học (A4: AURC 0,367 → 0,148). Với 3 lỗi, mọi so sánh chỉ mang tính mô tả.

## 7. Phân tích lối tắt (`shortcut_probes_summary.csv`; AUROC cấp ổ, trung bình 5 fold [min–max]; 17 ổ chưa thấy mỗi fold)
| Biểu diễn | Nhà sản xuất IBU | Năm 2015 | Nguồn gốc nhân tạo (ổ hư) | Điều kiện (acc 4 lớp, cửa sổ) | Tốc độ 900 v/ph (cửa sổ) |
|---|---|---|---|---|---|
| Embedding B3 | 0,94 [0,91–0,97] | 0,82 | 0,75 | 0,49 | 0,93 |
| Latent A4 | 0,87 [0,82–0,91] | 0,84 | 0,84 | 0,45 | 0,90 |
| Khái niệm dự đoán A6 | 0,93 [0,88–0,95] | 0,76 | 0,59 | 0,46 | 0,84 |
| **Khái niệm V3 4 s đo trực tiếp** | **0,71 [0,58–0,85]** | **0,90** | 0,46 | 0,30 | 0,62 |
- Khái niệm đo trực tiếp mang **ít** thông tin nhà sản xuất, nguồn gốc và điều kiện vận hành hơn các biểu diễn học, nhưng **không bất
  biến**. Chúng đoán năm đo tốt (0,90), có thể vì năm đo trùng với tình trạng khỏe (ổ khỏe đều đo năm 2015) và với nguồn gốc. Phép dò
  cho thấy thông tin *có mặt*, không chứng minh bộ phân loại *dùng* nó.
- Trong tầng IBU (K001–K005 vs KA22): mọi mô hình vật lý gọi KA22 là khỏe; B3, A3, A4 và A6 gọi đúng OR. Đây là trường hợp duy nhất
  mô hình học thắng trong tầng, và không loại trừ được việc chúng nhận dạng "IBU năm 2014".

## 8. Hiệu năng tính toán (`efficiency.csv`; RTX 3060 Laptop 6 GB)
| Mô hình | Tham số | Trễ GPU / CPU (ms, batch 1) | Huấn luyện (s/seed) | GPU đỉnh (MB) |
|---|---|---|---|---|
| A1/B4, A2 | 51 / 72 | 0,19–0,23 / 0,05–0,06 | 2,7–3,1 | 44 |
| B3, A3, A4, A6 | 0,23 M | 1,0–1,6 / 1,6–1,9 | 35–36 | 247 |
| Trích khái niệm V3 (4 s, CPU) | — | 30 ms | — | — |
Tổng thời gian Phase 3B: 5 fold ≈ 78 phút + FINAL 11 phút + PEV ≈ 18 phút.

## 9. Đánh giá tái lập
- Mọi số liệu sinh lại được từ `results/phase3b/folds/*.npz` bằng các kịch bản trên; checkpoint có băm; seed trước khi khởi tạo (có kiểm thử).
- Hạn chế: B1 là hồi quy logistic tất định; mô hình nơ-ron chạy với `cudnn.deterministic = True` nhưng AMP có thể gây khác biệt nhỏ giữa máy.
- Đầu ra smoke giữ ở `results/phase3b/_smoke*`, không dùng cho kết quả.

## 10. Phát hiện âm (giữ nguyên)
1. PAVE-Net có nhánh latent (A4) **không** tốt hơn mô hình vật lý; nó kém hơn 0,169 macro-F1 và hiệu chỉnh xác suất kém hơn.
2. Concept bottleneck học (A6) không trung thực với bằng chứng phổ (Δp −0,008).
3. Điểm chọn lọc vật lý-nhận thức (S4) không tốt hơn max-softmax cho mô hình vật lý.
4. Đặc trưng độ tin cậy (A2) không cải thiện kết quả cấp ổ.
5. Ba ổ OR bằng chứng yếu vẫn bị mọi mô hình vật lý bỏ sót.

## 11. Hạn chế còn lại
15 ổ, KTC rộng, không đủ lực suy luận. PU đã bị xem trước. Healthy ≡ IBU ≡ 2015. Ba sửa đổi giao thức trong phát triển (đều trước kết
quả phân loại, đều có ghi nhận). Khái niệm và mô hình đọc cùng tín hiệu. Can thiệp chỉ trong miền bao 1–20 kHz.

## 12. Quyết định cổng Phase 3B: **GO** (cho kiểm định ngoài, với mô hình vật lý làm ứng viên cuối)
| Điều kiện GO | Trạng thái |
|---|---|
| Triển khai đúng; kiểm thử đạt | 110/110 |
| Toàn vẹn dữ liệu | dữ liệu thô chỉ đọc; băm Paderborn có từ Phase 2 |
| Không có rò rỉ đã biết | kiểm tra lồng nhau ở mọi fold; OOF vòng trong chỉ dùng cho nhiệt độ, S4, quy tắc chọn |
| Mô hình và quy trình đã cố định | đúng; mô hình cuối chọn bằng quy tắc định trước trên OOF vòng trong |
| Đánh giá ngoài thực hiện được mà không cần chọn thêm | đúng: mô hình FINAL đã huấn luyện (21 checkpoint), mô hình chính = B1-V3-4s, mọi mô hình định trước đều được đánh giá |
Kiến trúc nơ-ron PAVE-Net (A4) không thành công; theo hướng dẫn, **mô hình vật lý đơn giản là ứng viên cuối**. PAVE-Net,
B3 và A6 vẫn được đánh giá ngoài như các mô hình định trước để kiểm tra RQ1/RQ5.
