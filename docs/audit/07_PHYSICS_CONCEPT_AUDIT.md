# Kiểm toán 9 khái niệm vật lý (Phase 3A, phần C)

Mã nguồn: `pave/features/physics.py` (định nghĩa đóng băng, giao thức V2 §6). Số liệu: `results/phase3a/concepts/`
(`python -I scripts/concept_audit_phase3a.py`). Khái niệm luôn được tính với vùng hiệu chỉnh khớp trên **12 ổ outer-train** của
fold, và đánh giá trên các ổ **ngoài** tập đó: ổ test BD5 của fold, cùng 12 ổ nhân tạo, K006 và KI17.

## 1. Chuỗi tính chung
1. Cửa sổ 1 s, rung thô 64 kHz (PU) / 25,6 kHz (KAIST). Lọc thông dải Butterworth bậc 4 pha-không: PU 1–20 kHz, KAIST 1–10 kHz.
2. Bao = |Hilbert(x_b)| − trung bình; phổ biên độ cửa sổ Hann; độ phân giải Δf = 1 Hz; giữ 0–1000 Hz.
3. Bậc động học từ hình học: BPFO = (N/2)(1 − (d/D)cos θ), BPFI = (N/2)(1 + (d/D)cos θ), nhân với fr.
   - **Siêu dữ liệu cần thiết:** tốc độ trục fr (PU: kênh `speed`; KAIST: đỉnh 2X), hình học ổ (N, d, D, θ; PU theo phiếu từng ổ),
     dải lọc. Tất cả đều có sẵn khi suy luận, không cần nhãn.
4. **Tham số chỉ khớp trên train:** vùng hiệu chỉnh cho mỗi họ F: tâm r_F = trung vị độ lệch tương đối của vạch mạnh nhất trong
   ±3,5 % (chỉ trên cửa sổ train có nhãn tương ứng và SNR ≥ 10 dB), nửa rộng w_F = max(3·1,4826·MAD, 0,5 %).
5. SNR(f_c, w) = 20·log10(max E trong [f_c ± w] / trung vị E trong vòng ±20 Hz), loại các bin trong ±1 Hz quanh k·fr và ±2 bin quanh đỉnh.

## 2. Định nghĩa từng khái niệm

| Khái niệm | Ý nghĩa vật lý | Định nghĩa | Đơn vị | Đáp ứng kỳ vọng |
|---|---|---|---|---|
| `BPFO_snr1` | độ nổi của vạch BPFO cơ bản | SNR tại f_O·(1 + r_O), nửa rộng w_O | dB | tăng khi hư vòng ngoài |
| `BPFO_harm` | cường độ họ BPFO | trung bình SNR tại h·f0, h = 1..3 (f0 = đỉnh tìm thấy; dung sai max(1 bin; 0,5 %)) | dB | tăng khi hư vòng ngoài |
| `BPFO_nharm` | nhất quán họ hài | số h ∈ {1,2,3} có SNR ≥ 6 dB | số đếm 0–3 | tăng khi hư vòng ngoài |
| `BPFO_sb` | dải biên ±fr quanh BPFO | trung bình SNR tại f0 ± fr | dB | **không có cơ sở**: vòng ngoài đứng yên |
| `BPFI_snr1` | độ nổi vạch BPFI | như trên cho họ BPFI | dB | tăng khi hư vòng trong |
| `BPFI_harm` | cường độ họ BPFI | như trên | dB | tăng khi hư vòng trong |
| `BPFI_nharm` | nhất quán họ BPFI | như trên | 0–3 | tăng khi hư vòng trong |
| `BPFI_sb` | điều chế theo trục của vòng trong quay | trung bình SNR tại f0 ± fr | dB | tăng khi hư vòng trong (vùng tải quay theo vòng trong) |
| `contrast` | họ nào chiếm ưu thế | `BPFO_harm` − `BPFI_harm` | dB | > 0 cho OR, < 0 cho IR, ≈ 0 cho khỏe |

Mọi khái niệm **đo trực tiếp được khi suy luận** từ một cửa sổ rung, fr và hình học; không cần nhãn hay mô hình.

## 3. Độ tin cậy (AUROC; hướng theo kỳ vọng; ổ ngoài mẫu)

### 3.1 Ổ thật PU-BD5 (15 ổ, mỗi ổ ở fold mà nó là test)
| Khái niệm | Kỳ vọng | AUROC cửa sổ | **AUROC cấp ổ** (5 vs 5) | Hư-vs-khỏe (ổ, 10 vs 5) | OR-vs-IR (ổ) |
|---|---|---|---|---|---|
| BPFO_snr1 | OR > H | 0,740 | 0,76 | 0,82 | 0,68 |
| BPFO_harm | OR > H | 0,702 | 0,76 | 0,58 | 0,76 |
| BPFO_nharm | OR > H | 0,702 | 0,70 | 0,65 | 0,64 |
| BPFO_sb | OR > H | 0,362 | **0,24 (ngược)** | 0,20 | 0,68 |
| BPFI_snr1 | IR > H | 0,907 | **1,00** | 0,88 | 0,00 (đúng chiều IR>OR) |
| BPFI_harm | IR > H | 0,873 | **1,00** | 0,64 | 0,00 |
| BPFI_nharm | IR > H | 0,768 | 0,94 | 0,72 | 0,06 |
| BPFI_sb | IR > H | 0,781 | **1,00** | 0,68 | 0,00 |
| contrast | OR > IR | 0,883 | 0,88 | — | 0,88 |

(OR-vs-IR = 0 cho khái niệm BPFI nghĩa là mọi ổ IR có giá trị cao hơn mọi ổ OR: phân tách hoàn hảo theo chiều IR > OR.)
Với n = 5 vs 5 ổ, AUROC cấp ổ có bước 0,04; một ổ thay đổi có thể đổi AUROC tới 0,2.

### 3.2 Ổ hư nhân tạo (12 ổ; so với ổ khỏe test của fold; trung bình trên 5 lần hiệu chỉnh từ ổ thật)
| Khái niệm | AUROC kỳ vọng (cửa sổ) | OR-vs-IR |
|---|---|---|
| BPFO_snr1 / harm / nharm | 0,93 / 0,94 / 0,92 | 0,90 / 0,94 / 0,91 |
| BPFO_sb | 0,43 | 0,49 |
| BPFI_snr1 / harm / nharm / sb | 0,74 / 0,72 / 0,69 / 0,68 | 0,24 / 0,25 / 0,30 / 0,23 |
| contrast | 0,96 | 0,96 |

Hư hỏng OR nhân tạo (EDM, khoan, khắc điện) cho bằng chứng rất rõ: cả 7 ổ có contrast > 0 và họ BPFO vượt ngưỡng khỏe ở 59–100 %
cửa sổ. Hư hỏng IR nhân tạo yếu: chỉ KI01 (EDM) rõ; KI03, KI05, KI07, KI08 (khắc điện, mức 1–2) gần mức khỏe; KI05 sai dấu contrast.
**Vùng hiệu chỉnh từ ổ thật chuyển được sang ổ nhân tạo**, nhưng độ mạnh bằng chứng phụ thuộc phương pháp tạo hư hỏng.

### 3.3 Theo điều kiện vận hành (AUROC kỳ vọng, cửa sổ, BD5)
| Khái niệm | 900 v/ph (N09) | N15 M01 | N15 F04 | N15 M07 F10 |
|---|---|---|---|---|
| BPFI_harm | 0,72 | 0,90 | 0,89 | 0,93 |
| BPFI_nharm | **0,43** | 0,87 | 0,85 | 0,90 |
| BPFO_harm | 0,69 | 0,74 | 0,64 | 0,74 |
| BPFO_sb | 0,74 | 0,31 | 0,10 | 0,24 |
| contrast | 0,85 | 0,90 | 0,87 | 0,90 |
900 v/ph là điều kiện khó nhất. Lực hướng tâm thấp (F04) làm giảm bằng chứng BPFO (KA16: 9,8 dB ở F04 so với 19,7–24,2 dB ở các điều kiện khác).

### 3.4 Theo từng ổ (BD5; `concept_per_bearing.csv`)
- **Mạnh, đúng hướng:** KA04, KA16 (BPFO_harm 20–21 dB; vượt q95 ổ khỏe train ở 98–99 % cửa sổ); KI04, KI18 (BPFI_harm 14–18 dB;
  BPFI_sb 11,6 / 21,1 dB; 90–97 %).
- **Trung bình:** KI21 (8,3 dB; 69 %), KI14 (5,8 dB; 39 %), KA30 (5,8 dB; 21 %).
- **Không phân biệt được với ổ khỏe:** KA15 (3,9 dB; 1 %), KA22 (4,1 dB; 0 %). Ổ khỏe: BPFO_harm 3,2–5,6 dB, BPFI_harm 1,8–4,2 dB.
- **Mâu thuẫn:** KI16 (rỗ mức 3): BPFO_harm 7,4 > BPFI_harm 4,7 dB, contrast +3,8 (dấu OR); 20 % cửa sổ vượt ngưỡng khỏe ở họ BPFI.

## 4. Chế độ thất bại đã biết
1. **Va chạm hài trục (6203):** BPFO ≈ 3,05·fr và BPFI ≈ 4,93–4,95·fr cách 3·fr và 5·fr 1,1–1,8 %. Ở 900 v/ph: 0,81 Hz (FAG) hoặc
   1,06 Hz (IBU/MTK). Với Δf = 1 Hz và vùng loại trừ ±1 Hz, khái niệm BPFI **không xác định ở 97,5 % cửa sổ FAG tại 900 v/ph**
   (KI16, KI21). Ở 1500 v/ph khoảng cách 1,36–1,77 Hz, vừa đủ tách. KAIST (6205): 3·BPFO ≈ 2·BPFI (5,9 Hz).
2. **`BPFO_sb` không hợp lệ** về vật lý và thực nghiệm (ngược chiều).
3. **Hư hỏng nhẹ / vết lõm do mảnh vụn / hư hỏng phân tán** (KA15, KA22, KA30, nhân tạo khắc điện IR) có thể không sinh vạch.
4. **Bằng chứng trái dấu** (KI16): có thể do hư hỏng ở vị trí/hình dạng đặc biệt; không có cách xác minh từ dữ liệu.
5. **Phụ thuộc điều kiện:** tốc độ thấp và lực nhỏ làm yếu bằng chứng.
6. **Không phải ground truth:** khái niệm và bộ phân loại đọc cùng một tín hiệu; độ "trung thực" khái niệm chỉ là tự nhất quán.

## 5. Khái niệm và yếu tố gây nhiễu (S-4b, S-4c)
- Trong tầng **IBU** (K001–K005 vs KA22): AUROC cấp ổ của max family score = **0,40**. Khái niệm **không** phát hiện được ổ hư IBU
  duy nhất trong BD5.
- Trong tầng **năm 2015** (K001–K005 vs KA30, KI04): AUROC cấp ổ = 1,00 (nhờ KI04 rất rõ; KA30 6,3 dB chỉ cao hơn ổ khỏe cao nhất 6,2 dB).
- **Phép dò danh tính phiên đo** (vùng động học không hiệu chỉnh; loại 1 ổ ra mỗi lần; AUROC cấp ổ):

| Phép dò | Đặc trưng thủ công (RF) | 9 khái niệm (LR) | Ghi chú |
|---|---|---|---|
| Năm 2014 vs 2015 trong 11 ổ IR | **1,00** | 0,43 | 80 % ổ IR năm 2015 là nhân tạo (so với 17 % năm 2014) |
| IBU vs không-IBU trong 23 ổ hư | 0,87 | 0,87 | 89 % ổ hư IBU là nhân tạo (29 % ở nhóm còn lại) |
| *Thăm dò:* IBU vs MTK chỉ trong 12 ổ nhân tạo | **0,94** | 0,44 | cùng nguồn gốc hư hỏng |

Diễn giải thận trọng: đặc trưng phổ thông thường (năng lượng dải, thống kê thời gian) **mang thông tin nhận dạng nhà sản xuất/phiên đo**
ngay cả khi loại hư hỏng giống nhau. Khái niệm vật lý thì không (trừ khi bị nhiễu bởi tỷ lệ nhân tạo/thật). Hai phép dò đầu
không tách được nhà sản xuất/năm khỏi nguồn gốc hư hỏng (nhân tạo vs thật).

## 6. Kết luận về hiệu lực khái niệm
- **Hợp lệ và đáng tin:** họ BPFI (snr1, harm, sb) cho IR-vs-khỏe; `contrast` cho IR-vs-OR; họ BPFO cho OR nhân tạo.
- **Hợp lệ một phần:** họ BPFO cho OR thật (phụ thuộc loại hư hỏng); `*_nharm` ở tốc độ thấp.
- **Không hợp lệ:** `BPFO_sb`.
- **Không xác định trong điều kiện cụ thể:** họ BPFI ở 900 v/ph cho hình học FAG với cửa sổ 1 s.
- Tương quan với nhãn **không** đủ để coi là đáng tin: hư-vs-khỏe ở cấp ổ chỉ 0,58–0,88 với từng khái niệm, và thất bại ở tầng IBU.
