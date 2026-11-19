# PAVE-Net — Giao thức nghiên cứu V2 (ĐÓNG BĂNG)

**PAVE-Net: Physics-Audited Vibration Evidence Network.**
Phiên bản: V2.0 — đóng băng ngày 2026-11-18, **trước khi** phân tích bất kỳ tín hiệu rung nào của bộ dữ liệu
Paderborn (PU). Tại thời điểm đóng băng, đối với PU mới chỉ đọc: bài báo Lessmeier et al. (2016), danh sách tệp,
cấu trúc biến của một tệp `.mat` (K001, N15_M07_F10, lần đo 1: tên kênh, độ dài, tần số lấy mẫu, trường mô tả) và
phiếu mô tả/nhật ký đo của K001. Chưa tính phổ, đặc trưng hay thống kê tín hiệu nào của PU.

Mã băm SHA-256 của tệp này được ghi trong `docs/PROTOCOL_FREEZE.txt`. Mọi thay đổi sau thời điểm này phải ghi vào
`docs/DECISION_LOG.md` kèm thời điểm, lý do và việc dữ liệu test đã được xem hay chưa.

Giao thức V1 (đa phương thức rung–nhiệt) được lưu trữ tại `99_ARCHIVE/blueprint_v1_multimodal/`.

---

## 1. Quyết định khoa học đã chốt (do tác giả phê duyệt sau Phase 1)

1. Loại nhiệt độ khỏi kiến trúc chính và mọi tuyên bố chính; phân tích nhiệt KAIST chỉ là phụ lục về yếu tố gây nhiễu.
2. Tên khung: **PAVE-Net — Physics-Audited Vibration Evidence Network**.
3. Bài toán chính: **3 lớp** Healthy / Inner race (IR) / Outer race (OR).
4. Bài toán 7 lớp (mức độ) trên KAIST: **thăm dò, thứ cấp**, công bố rõ yếu tố gây nhiễu theo lần lắp đặt.
5. Bộ thứ hai: **Paderborn University Bearing DataCenter**; đánh giá **tách theo ổ trục (bearing-disjoint)** là giao
   thức kiểm định trung tâm.
6. KAIST giữ cho phân tích vật lý rung và thí nghiệm chuyển tải **trong cùng một lần lắp đặt** (LOLO).
7. Đặc trưng vật lý: dùng tốc độ trục đo được và **vùng tần số lỗi hiệu chỉnh từ dữ liệu nguồn** (source-calibrated),
   gồm tính nhất quán họ hài và dải biên.
8. Vùng tìm kiếm ±3,5 % là **hậu nghiệm** (chọn sau khi xem KAIST). Kết quả trên KAIST **không** được trình bày như
   kiểm định độc lập của lựa chọn này. PU là phép kiểm tra độc lập vì thiết kế được đóng băng trước khi xem tín hiệu PU.
9. Baseline kiểm soát lối tắt: **chỉ-tốc-độ**, **chỉ-RMS**, và **chuẩn hoá theo bậc (order-normalized)**.
10. Huấn luyện bằng **AdamW**; thuật toán tối ưu không phải đóng góp khoa học.

## 2. Dữ liệu

### 2.1 KAIST (đã kiểm toán ở Phase 1)
- Chỉ rung, kênh housing A x/y (`x_housing_A`, `y_housing_A`), fs = 25 600 Hz; ánh xạ kênh là GIẢ ĐỊNH (Phase 1).
- 21 bản ghi trong phạm vi (7 điều kiện × 3 tải). 24 bản ghi lệch trục/mất cân bằng: tập thử thách ngoài phạm vi.
- Hình học NSK 6205 DDU (bài báo): N = 9, d = 7,90 mm, D = 38,5 mm, θ = 0°.
- Tốc độ trục: ước lượng từ phổ rung từng cửa sổ (đỉnh 2X nội suy, chia 2).

### 2.2 Paderborn (PU)
- 32 ổ 6203, mỗi ổ 4 điều kiện vận hành × 20 lần đo × 4 s; rung `vibration_1` 64 kHz; tốc độ `speed` 4 kHz (vòng/phút).
- Điều kiện (Lessmeier 2016, Bảng 6): N15_M07_F10 (1500 v/ph, 0,7 Nm, 1000 N), N09_M07_F10 (900), N15_M01_F10 (0,1 Nm),
  N15_M07_F04 (400 N).
- Hình học **theo từng ổ** lấy từ phiếu mô tả `<mã>.pdf` trong mỗi tệp nén (đường kính vòng chia khác nhau theo nhà sản
  xuất, ví dụ K001/IBU 29,05 mm; ví dụ FAG trong bài báo 28,55 mm). Nếu phiếu thiếu, dùng 28,55 mm / 6,75 mm / 8 bi
  và gắn cờ.
- Nhãn và nhóm ổ (Lessmeier 2016, Bảng 4, 5, 7, 10):

| Nhóm | Mã ổ |
|---|---|
| Healthy | K001, K002, K003, K004, K005, K006 |
| OR nhân tạo | KA01, KA03, KA05, KA06, KA07, KA08, KA09 |
| IR nhân tạo | KI01, KI03, KI05, KI07, KI08 |
| OR thật (mỏi/lõm) | KA04, KA15, KA16, KA22, KA30 |
| IR thật | KI04, KI14, KI16, KI17, KI18, KI21 |
| Hư hỏng kép (IR+OR) | KB23, KB24, KB27 — **không dùng huấn luyện**; chỉ mô tả thử thách |

## 3. Bài toán và nhãn
- Chính: 3 lớp {H, IR, OR} trên KAIST (21 bản ghi) và PU.
- Thứ cấp (thăm dò): 7 lớp KAIST {H, IR_0p3, IR_1p0, IR_3p0, OR_0p3, OR_1p0, OR_3p0}. Không có bài toán mức độ trên PU.

## 4. Chia dữ liệu (tạo ở cấp bản ghi/ổ trục, TRƯỚC khi cắt cửa sổ)

### 4.1 KAIST
- **Protocol A (LOLO)**: giữ nguyên manifest Phase 1 (`03_MANIFESTS/splits/split_A_test*.json`, mã băm đã ghi). Validation
  = 20 % cuối mỗi bản ghi nguồn, khoảng đệm 2 s. Đây là chuyển tải **trong cùng lần lắp đặt**, không phải giữa các ổ.
- Protocol B (6 hoán vị) và C (trong bản ghi) giữ nguyên, chỉ dùng bổ sung.

### 4.2 PU — giao thức trung tâm `PU-BD5` (tách theo ổ trục)
Dựa trên Lessmeier 2016 Bảng 10. Hàng r = 1..5: {K00r, OR_r, IR_r} với OR = (KA04, KA15, KA16, KA22, KA30),
IR = (KI04, KI14, KI16, KI18, KI21).
- Fold k (k = 1..5): **test** = hàng k; **validation** = hàng (k mod 5) + 1; **train** = 3 hàng còn lại.
- Cả 4 điều kiện vận hành; mọi lần đo và mọi cửa sổ của một ổ chỉ thuộc một vai trò.
- Mỗi fold: 9 ổ train, 3 ổ validation, 3 ổ test (1 ổ/lớp ở test và validation).

### 4.3 PU — `PU-A2R` (thứ cấp: nhân tạo → thật)
- Train: OR nhân tạo {KA01, KA05, KA07, KA08, KA09}, IR nhân tạo {KI01, KI05, KI07}, Healthy {K002, K003, K004, K005}.
- Validation (vẫn là nhân tạo/khỏe, tách theo ổ): {KA03, KA06, KI03, KI08, K006}.
- Test: {K001} ∪ OR thật ∪ IR thật (12 ổ thật + 1 khỏe).

### 4.4 Tập thử thách
- KAIST: 24 bản ghi lệch trục/mất cân bằng; PU: KB23, KB24, KB27. Không dùng để chỉnh tham số hay ngưỡng.

## 5. Tiền xử lý
- Cửa sổ không chồng lấn. KAIST: 1,0 s (25 600 mẫu, 2 kênh). PU: 1,0 s (64 000 mẫu) → 4 cửa sổ/lần đo.
- Đầu vào mạng sâu PU: hạ mẫu ×2 (bộ lọc FIR chống chồng phổ) → 32 kHz, 32 000 mẫu/cửa sổ.
- Chuẩn hoá: z-score theo kênh với thống kê **chỉ từ train** của từng fold.
- **Chuẩn hoá theo bậc (order-normalized)**: lấy mẫu lại miền góc với tốc độ đo được của từng cửa sổ (coi là hằng số trong
  cửa sổ), S = 512 mẫu/vòng (KAIST) và S = 1024 mẫu/vòng (PU); độ dài cố định theo số vòng: KAIST 48 vòng
  (24 576 mẫu), PU 24 vòng (24 576 mẫu). Có lọc chống chồng phổ trước khi lấy mẫu lại.
- Biến thể "order + chuẩn hoá RMS từng cửa sổ" để loại lối tắt năng lượng.

## 6. Đặc trưng vật lý (định nghĩa đóng băng)
1. Phổ bao: lọc thông dải Butterworth bậc 4 pha-không → Hilbert → |·| → bỏ trung bình → phổ biên độ cửa sổ Hann, độ phân giải
   1 Hz. Dải cố định: **KAIST 1–10 kHz** (như Phase 1); **PU 1–20 kHz** (chọn trước, dưới bộ lọc 30 kHz của khuếch đại).
2. Bậc động học từ hình học (KAIST: bài báo; PU: phiếu từng ổ).
3. Tìm vạch: đỉnh mạnh nhất trong **±3,5 %** quanh bậc động học, loại các bin trong ±1 Hz của hài trục (nguồn gốc: hậu nghiệm KAIST).
4. **Hiệu chỉnh từ nguồn (theo fold)**: lấy các cửa sổ TRAIN có nhãn OR (cho họ BPFO) hoặc IR (cho họ BPFI) có SNR vạch ≥ 10 dB;
   tính độ lệch tương đối r = f0/f_động-học − 1. Tâm vùng = trung vị r; nửa độ rộng = max(3 × 1,4826 × MAD(r), 0,5 %), cắt trong
   ±3,5 %. Nếu < 20 cửa sổ đạt chuẩn → dùng ±3,5 % và gắn cờ `fallback`. Hiệu chỉnh theo **độ lệch tương đối** để chuyển được giữa
   các ổ có hình học khác nhau.
5. Đặc trưng mỗi cửa sổ, cho mỗi họ F ∈ {BPFO, BPFI} (KAIST: lấy max qua 2 kênh A):
   - `F_snr1`: SNR (dB) của vạch mạnh nhất trong vùng hiệu chỉnh;
   - `F_harm`: trung bình SNR tại h·f0 (h = 1, 2, 3; dung sai max(1 bin, 0,5 %));
   - `F_nharm`: số hài h ∈ {1,2,3} có SNR ≥ 6 dB (tính nhất quán họ hài);
   - `F_sb`: trung bình SNR tại f0 ± fr (dải biên trục);
   - `contrast` = `BPFO_harm` − `BPFI_harm`.
   SNR = đỉnh / trung vị vòng ±20 Hz (loại bin hài trục và ±2 bin quanh đỉnh).
6. Kiểm thử tổng hợp bắt buộc: phát hiện vạch lệch 2 %, họ hài, dải biên; từ chối nhiễu trắng; bất biến bậc khi đổi tốc độ.

## 7. Mô hình Phase 2 (siêu tham số cố định, không HPO)
| ID | Mô hình | Đầu vào |
|---|---|---|
| S1 | Hồi quy logistic | tốc độ trục của cửa sổ (chỉ-tốc-độ) |
| S2 | Hồi quy logistic | log-RMS từng kênh (chỉ-RMS) |
| C1 | Random Forest (500 cây, min_samples_leaf = 2, class_weight = balanced) | đặc trưng thủ công thời gian + 16 dải năng lượng phổ log |
| C2 | SVM-RBF (C = 10, gamma = scale, balanced) | như C1 |
| P1 | Hồi quy logistic (L2, C = 1, balanced) | chỉ 9 đặc trưng vật lý (mục 6.5) |
| D1 | CNN-1D | sóng thô (chuẩn hoá toàn cục) |
| D2 | CNN-1D | sóng chuẩn hoá theo bậc |
| D3 | CNN-1D | sóng theo bậc + chuẩn hoá RMS từng cửa sổ |

- CNN-1D: 6 khối Conv1d–BatchNorm–ReLU (kernel 64 bước 8 ở khối đầu, sau đó kernel 7 / MaxPool 2), GAP, dropout 0,3, tuyến tính.
- AdamW, lr 1e-3, weight decay 1e-2, batch 64, tối đa 25 epoch, cosine; chọn epoch theo macro-F1 validation; dừng sớm patience 6.
- Seed 42, 43, 44 cho mô hình sâu; mô hình cổ điển một seed (42).
- 7 lớp KAIST: chỉ C1, P1, D1, D2 (thăm dò).

## 8. Đánh giá
- Cấp cửa sổ (mô tả): accuracy, **macro-F1 (chính)**, balanced accuracy, recall từng lớp, ECE (15 bin).
- Cấp bản ghi (KAIST: trung bình xác suất theo bản ghi) và **cấp ổ trục (PU: trung bình xác suất theo ổ × điều kiện và theo ổ)**.
- Báo cáo từng fold, trung bình ± độ lệch chuẩn qua fold và seed; không tính p-value (đơn vị độc lập quá ít).
- Kết quả phân loại, kiểm soát lối tắt và đặc trưng vật lý được báo cáo **tách biệt**.

## 9. Tiêu chí kiểm định trước khi xây PAVE-Net đầy đủ (đóng băng)
- **V1 Giao thức**: mọi manifest chia dữ liệu qua kiểm thử rò rỉ (không trùng ổ/bản ghi giữa vai trò, mã băm khớp).
- **V2 Bộ trích vật lý**: toàn bộ kiểm thử tổng hợp đạt.
- **V3 Hiệu lực khái niệm trên PU (độc lập)**: trên các ổ test của PU-BD5, trung bình qua 5 fold:
  AUROC(`contrast`; OR vs IR, cửa sổ ổ hư) ≥ 0,75 **và** AUROC(max(`BPFO_harm`,`BPFI_harm`); hư vs khỏe) ≥ 0,75.
- **V4 Thiết kế đánh giá**: trên PU-BD5, S1 (chỉ-tốc-độ) có macro-F1 cấp cửa sổ ≤ 0,45 (không có lối tắt tốc độ); mọi kết quả
  có cấp ổ trục.
- Nếu V3 không đạt: PECB phải được thiết kế lại trước khi triển khai mô hình đầy đủ; báo cáo trung thực.

## 10. Ngân sách tính toán Phase 2
≤ 3 giờ GPU (RTX 3060 Laptop 6 GB); RAM khả dụng ≈ 3 GB → cache cửa sổ dạng memmap trên đĩa.
