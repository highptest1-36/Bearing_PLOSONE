# PHASE 4 — Báo cáo kiểm định ngoài (Ottawa UORED-VAFCLS)

Ngày: 2026-11-19. Giao thức: `docs/EXTERNAL_VALIDATION_PROTOCOL_V1.md` (EV-V1, SHA-256 `bb297d52…`), đóng băng trong
`docs/PROTOCOL_FREEZE_EV_V1.txt` (UTC 2026-11-18T22:29:55Z, 45 tệp có hash: giao thức, mã, fold O1–O5/FINAL, 21 checkpoint FINAL).
Đánh giá **một lần duy nhất**: `results/phase4/EVALUATED.lock` (2026-11-19 05:30:24, giờ máy). Mọi hash đã khớp trước khi chạy.

> Quy ước nhãn phân tích: **[CHÍNH]** = kết quả chính theo EV-V1; **[THỨ CẤP – ĐĂNG KÝ TRƯỚC]** = phân tích nêu trong EV-V1 §5/§8;
> **[POST-HOC]** = phân tích thăm dò thực hiện sau khi đã xem kết quả, không thay thế kết quả chính.

## 1. Tóm tắt điều hành

- Mô hình chính (do cổng Phase 3B chọn, không dựa trên Ottawa): **B1-V3-4s** (hồi quy logistic trên 8 khái niệm vật lý V3 có mặt nạ,
  ngữ cảnh 4 s). **Macro-F1 cấp bản ghi = 0.518 [KTC 95 % cụm theo ổ bi: 0.31, 0.68]** trên 30 bản ghi chính (20 ổ bi vật lý).
  Thấp hơn rõ so với nội bộ PU (0.780 [0.56, 0.93]).
- Recall H/IR/OR = 0.45/1.00/0.40; **tỷ lệ báo động giả trên trạng thái khỏe = 0.55** (11/20); độ nhạy phát hiện lỗi 0.90;
  định vị đúng loại lỗi trong số lỗi được phát hiện 0.78.
- **B0** (quy tắc ngưỡng vật lý không học, ngưỡng τ từ PU) đạt điểm cao nhất: 0.614 [0.45, 0.72]. B4 0.557; A2 0.500.
- **Mọi mô hình dựa trên CNN sụp đổ** khi chuyển miền: B3 0.311, A4 (PAVE-Net đầy đủ) 0.300, A6 0.306, A3 0.277 — tất cả
  dự đoán "có lỗi" cho 20/20 bản ghi khỏe (báo động giả = 1.0). Chuẩn hóa theo bản ghi (thứ cấp) cải thiện A4 lên 0.459 nhưng
  không cứu B3/A6.
- Phân tích độ nhạy đăng ký trước (σ_r = 1 % f_r): B1-V3-4s giảm còn 0.270 (recall IR = 0) — kết quả mong manh với giả định
  độ bất định tốc độ.
- PEV thứ cấp đăng ký trước trên 10 bản ghi lỗi Ottawa: triệt tiêu có mục tiêu dải đặc trưng của họ lỗi đúng làm giảm bằng chứng
  −12.7 dB và p(lớp đúng) của B1-V3-4s −0.31 [−0.68, −0.09] so với đối chứng khớp năng lượng (10/10 bản ghi); A4 −0.22
  [−0.27, −0.16] (qua nhánh vật lý); B3 ≈ 0. Tức là cơ chế "quyết định dựa vào bằng chứng vật lý" **được bảo toàn** trên miền
  mới, dù độ chính xác phân loại không được bảo toàn.
- Kết luận: giả thuyết khả năng tổng quát hóa ngoài miền (H4) **không được ủng hộ** ở mức hiệu năng; được ủng hộ một phần ở
  mức cơ chế (bằng chứng vật lý vẫn chi phối quyết định của mô hình vật lý) và ở mức so sánh tương đối (mô hình dựa vật lý
  ổn định hơn rõ rệt so với CNN).

## 2. Thiết lập (tóm tắt EV-V1)

| Hạng mục | Thiết lập |
|---|---|
| Dữ liệu | 60 tệp .mat (420 000 × 5, fs = 42 kHz, 10 s); ổ bi 6203 (8 bi, d = 6.77 mm, D = 28.50 mm); BPFO = 3.0498 f_r, BPFI = 4.9502 f_r |
| Đơn vị độc lập | 20 ổ bi vật lý; bootstrap cụm (10 000 lần) |
| Nhiệm vụ chính | 3 lớp H/IR/OR trên 30 bản ghi: H_n_0 (n = 1..20), I_n_2 (1..5), O_n_2 (6..10) |
| Thứ cấp | 10 bản ghi "đang phát triển" (I_n_1, O_n_1); 20 bản ghi lỗi bi/vòng cách (B, C) chỉ phân tích OOD |
| Tốc độ | f_r = giá trị kênh tốc độ tại mẫu 0 (kênh bằng 0 ở mọi mẫu khác); σ_r = 0.5 rpm/60 |
| Tiền xử lý | Giống hệt mã phát triển: 2 bản ghi 4 s ([0,4), [4,8) s), 10 cửa sổ 1 s, CNN lấy mẫu lại 42→32 kHz (16/21), chuẩn hóa theo thống kê PU |
| Mô hình | 11 mô hình FINAL (huấn luyện trên 15 ổ bi PU-BD5, 3 seed), không tinh chỉnh trên Ottawa |
| Hiệu chuẩn | Nhiệt độ từ dự đoán outer-test gộp của PU; ngưỡng chọn lọc từ thứ hạng ổ bi PU (12/15 và 9/15) |

Kiểm tra trước đóng băng chỉ gồm: bài báo Ottawa; audit không dùng nhãn (toàn vẹn tệp, mã hóa tốc độ/tải, kiểm tra chéo 1×);
chạy thử mã đánh giá trên tín hiệu ngẫu nhiên tổng hợp. Chính sách tốc độ được sửa (median → mẫu 0) **sau audit không nhãn và trước
đóng băng** (EV-V1 §3), vì median của kênh tốc độ bằng 0.

## 3. Kết quả chính [CHÍNH]

Nguồn: `results/phase4/tables/external_primary.csv/json`.

| Mô hình | Macro-F1 | KTC 95 % (cụm) | Bal. acc | Recall H/IR/OR | Báo động giả (H) | Độ nhạy lỗi | Định vị | NLL cửa sổ | ECE |
|---|---|---|---|---|---|---|---|---|---|
| **B1-V3-4s (chính)** | **0.518** | [0.31, 0.68] | 0.617 | 0.45/1.0/0.4 | 0.55 | 0.90 | 0.78 | 2.11 | 0.35 |
| B0 (quy tắc) | 0.614 | [0.45, 0.72] | 0.717 | 0.55/1.0/0.6 | 0.45 | 0.90 | 0.89 | – | – |
| B4 (=A1) | 0.557 | [0.39, 0.67] | 0.683 | 0.45/1.0/0.6 | 0.55 | | | | |
| A2 | 0.500 | [0.32, 0.64] | 0.617 | 0.45/1.0/0.4 | 0.55 | | | | |
| B1-V2 | 0.417 | [0.26, 0.55] | 0.650 | 0.15/1.0/0.8 | 0.85 | | | | |
| B1-V3-1s | 0.330 | [0.15, 0.49] | 0.467 | 0.20/0.8/0.4 | 0.80 | | | 1.10 | 0.10 |
| B2 | 0.329 | [0.17, 0.48] | 0.467 | 0.20/0.8/0.4 | 0.80 | | | | |
| B3 (CNN) | 0.311 | [0.18, 0.41] | 0.600 | 0.0/0.8/1.0 | 1.00 | | | | |
| A6 (CBM) | 0.306 | [0.18, 0.39] | 0.600 | 0.0/0.8/1.0 | 1.00 | | | | |
| A4 (PAVE-Net) | 0.300 | [0.18, 0.38] | 0.600 | 0.0/0.8/1.0 | 1.00 | | | | |
| A3 | 0.277 | [0.13, 0.39] | 0.533 | 0.0/0.6/1.0 | 1.00 | | | | |

Ma trận nhầm lẫn B1-V3-4s (hàng = thật H/IR/OR, cột = dự đoán): [[9, 10, 1], [0, 5, 0], [1, 2, 2]].
Hiệu chuẩn ngoài miền kém (ECE cửa sổ 0.35, NLL 2.11; PU nội bộ: NLL 0.625).

Thứ hạng 4 mô hình đầu (B0 > B4 > B1-V3-4s ≈ A2) không giống thứ hạng nội bộ PU (B1-V3-4s = B4 = A2 > B1-V2 > B0); với 30
bản ghi/20 ổ bi, các KTC chồng lấn mạnh và không có so sánh cặp nào được đăng ký trước là kiểm định xác nhận — chỉ mô tả.

## 4. Phân tích thứ cấp đăng ký trước [THỨ CẤP – ĐĂNG KÝ TRƯỚC]

### 4.1 Độ nhạy chính sách tốc độ (σ_r = 1 % f_r)
B1-V3-4s_sigma1pct: macro-F1 0.270 [0.15, 0.38]; recall H/IR/OR 0.45/0.0/0.4. Vùng/guard rộng hơn làm khái niệm BPFI không còn
nhận diện được (BPFI cách 5·f_r chỉ ~1.5 Hz ở 30 Hz khi tính guard 3kσ_r) → mất toàn bộ khả năng phát hiện IR. Kết quả chính
phụ thuộc mạnh vào giả định rằng số đọc tốc độ của tệp chính xác.

### 4.2 Chuẩn hóa biên độ theo bản ghi cho CNN
A4_recnorm 0.459 (báo động giả 0.85); B3_recnorm 0.222; A6_recnorm 0.247. Chênh lệch đơn vị/độ khuếch đại cảm biến
(m/s² Ottawa so với PU) chỉ giải thích được một phần sự sụp đổ của CNN.

### 4.3 Lỗi đang phát triển (10 bản ghi; `external_secondary.csv`)
| Mô hình | Phát hiện | Đúng loại |
|---|---|---|
| B1-V3-4s | 0.9 | 0.6 |
| B0 | 0.8 | 0.8 |
| B4 | 0.9 | 0.8 |
| B3 / A4 / A3 | 1.0 | 1.0 |

Lưu ý: CNN đạt 1.0 vì chúng dự đoán "có lỗi" cho mọi đầu vào Ottawa (kể cả bản ghi khỏe) — không phải bằng chứng phát hiện sớm.

### 4.4 Lỗi ngoài phạm vi (bi/vòng cách; 20 bản ghi)
| Mô hình | Lỗi bi/vòng cách (giai đoạn lỗi): dự đoán H/IR/OR | Giai đoạn phát triển: H/IR/OR |
|---|---|---|
| B1-V3-4s | 8/0/2 | 8/0/2 |
| B0 | 10/0/0 | 10/0/0 |
| B3 | 0/2/8 | 0/2/8 |
| A4 | 0/3/7 | 1/2/7 |

Mô hình vật lý gán lỗi bi/vòng cách vào "khỏe" (đúng về mặt thiết kế: không có khái niệm BSF/FTF), tức là **không phát hiện** các lỗi
này; CNN gán chúng vào IR/OR. Không mô hình nào có cơ chế từ chối OOD hiệu quả.

### 4.5 Chẩn đoán có chọn lọc (PE-SD; `external_selective.csv`)
| Cơ sở | Điểm | AURC | Số lỗi | AUROC phát hiện lỗi | Độ phủ thực tế ở ngưỡng PU 0.8 | Rủi ro |
|---|---|---|---|---|---|---|
| B1-V3-4s | S1 | 0.494 | 14 | 0.455 | 0.77 | 0.478 |
| B1-V3-4s | S3 | 0.481 | 14 | 0.460 | | |
| B1-V3-4s | S4 | 0.533 | 14 | 0.433 | | |
| A4 | S1 | 0.590 | 21 | 0.735 | | |
| A4 | S3 | 0.419 | 21 | **0.915** | | |
| A4 | S4 | 0.442 | 21 | 0.862 | | |

Với mô hình chính, không điểm tin cậy nào phân biệt được dự đoán đúng/sai (AUROC < 0.5): báo động giả trên trạng thái khỏe có
bằng chứng BPFI cao và được mô hình tin tưởng. Với A4, điểm độ rõ của bằng chứng (S3) xác định tốt các lỗi của CNN (0.915),
nhất quán với PU (S4 AUROC 0.89) — đây là kết quả tích cực duy nhất của nhánh PE-SD ngoài miền, nhưng chỉ áp dụng cho một mô
hình có độ chính xác thấp.

### 4.6 PEV trên 10 bản ghi lỗi Ottawa (`scripts/pev_ottawa.py`; `results/phase4/pev/`)
Chạy sau khóa đánh giá chính như một script riêng (đã có trong EV-V1 §8; không huấn luyện, không tinh chỉnh; nhãn chỉ dùng để
đọc p(lớp đúng), giống PEV trên PU). Kiểm tra nhất quán: biến thể gốc tái tạo chính xác các khái niệm đã đóng băng.

Họ lỗi đúng (BPFO cho OR, BPFI cho IR), cường độ 1.0, có mục tiêu (T) trừ đối chứng khớp năng lượng (Ce), trung vị qua 10 bản ghi:

| Đại lượng | T − Ce | KTC 95 % (bootstrap) | Số bản ghi giảm |
|---|---|---|---|
| Bằng chứng (dB) | −12.67 | [−18.05, −7.93] | 10/10 |
| Δp(đúng) B1-V3-4s | −0.312 | [−0.678, −0.087] | 10/10 |
| Δp(đúng) A4 | −0.216 | [−0.269, −0.157] | 10/10 |
| Δp(đúng) B3 | 0.000 | [0.000, 0.000] | 3/10 |
| A4 logit vật lý | −1.56 | [−1.76, −0.90] | 10/10 |
| A4 đóng góp latent | +0.09 | [−0.06, +0.30] | 4/10 |

Tỷ lệ đổi dự đoán khi triệt tiêu có mục tiêu: B1-V3-4s 40 %, A4 30 %, B3 5 % (đối chứng khớp năng lượng: 5 %/0 %/5 %).
Độ méo: SNR trong dải trung vị 13.9 dB (thấp hơn PU 31–42 dB), cắt cực đại 11.7 %, thay đổi ngoài dải ≤ 0.6 %. SNR thấp hơn
có nghĩa là can thiệp trên Ottawa thô hơn trên PU; kết quả nên đọc là định hướng.

## 5. Phân tích nguyên nhân [POST-HOC]

Script `scripts/posthoc_ottawa_exploratory.py` → `results/phase4/posthoc/`. Các phát hiện sau là thăm dò, sinh giả thuyết.

1. **Báo động giả tập trung ở các ổ bi về sau phát sinh lỗi IR.** Bản ghi khỏe bị B1-V3-4s gán lỗi: ổ bi IR (1–5) 5/5; ổ bi OR (6–10)
   2/5; ổ bi bi/vòng cách (11–20) 4/10. Trung vị BPFI_harm ở bản ghi "khỏe" của ổ bi IR = 12.2 dB (PU khỏe: trung vị 9.2, p75 11.0,
   max 14.4); BPFO_harm = 12.5 dB (PU khỏe p75 9.9). Có hai cách giải thích không loại trừ nhau: (a) "khỏe" trong Ottawa là
   trạng thái đầu của thí nghiệm chạy-đến-hỏng (lỗi tạo bằng tải gia tốc) và có thể đã có dấu hiệu tiền lỗi; (b) mức nền của
   bằng chứng khác giữa hai giàn thử (cảm biến, kết cấu truyền, cộng hưởng). Không thể phân định với dữ liệu hiện có.
2. **Dịch chuyển phân bố tốc độ/tải:** tốc độ đọc 1700–2190 rpm (trung vị 1790; danh định 1750); lỗi bi đo ở tải 0 N, các nhóm
   khác 400 N. Kiểm tra chéo 1× từ gia tốc kế lệch −8.2 % … +8.6 % (trung vị −0.7 %) — quá nhiễu để dùng, nhưng cho thấy số đọc
   tốc độ một mẫu có thể không đại diện cho cả 10 s; điều này phù hợp với độ nhạy mạnh ở §4.1.
3. **Tỷ lệ hợp lệ của khái niệm** (định nhận được theo V2.4): BPFO 0.78, BPFI 0.90 trên toàn bộ bản ghi 4 s (PU: khác do hình học
   và tốc độ khác).
4. **Cổng A4** trên Ottawa: trung vị 0.75 (OR), 0.51–0.54 (IR), 0.31–0.34 (khỏe, bi, vòng cách) — cùng thứ tự với PU (cổng mở
   về phía vật lý khi bằng chứng mạnh), nhưng nhánh latent vẫn kéo mọi bản ghi khỏe về lớp lỗi.
5. **Sụp đổ CNN**: tất cả mô hình có nhánh CNN (B3, A3, A4, A6) dự đoán lỗi cho 100 % bản ghi khỏe. Probe ở Phase 3B cho thấy biểu
   diễn CNN mã hóa nhà sản xuất ổ bi với AUROC 0.87–0.94 — gợi ý rằng CNN học đặc điểm đặc thù của giàn thử/ổ bi hơn là cơ chế lỗi.

## 6. So sánh với Siddique và cộng sự (Machines 2026)

Siddique và cs. dùng quy tắc thống kê dựa vật lý, đánh giá ngoài trên Ottawa với tốc độ danh định và báo cáo phát hiện IR 0 %,
OR 20 %, báo động giả 5 %; phân tích post-hoc của họ cho thấy tốc độ thực ≈ 1784 rpm. Nghiên cứu này dùng số đọc tốc độ theo tệp
(được đăng ký trước) và thu được phát hiện cao (IR 100 %, OR 40 % đúng loại, 90 % phát hiện lỗi chung) nhưng báo động giả cao (55 %).
Hai nghiên cứu khác nhau về nhiệm vụ (phân loại 3 lớp so với phát hiện), tập bản ghi, ngưỡng và giao thức, nên **không so sánh trực
tiếp các con số**. Điểm hội tụ: cả hai cho thấy chẩn đoán dựa vật lý chuyển từ PU sang Ottawa phụ thuộc nhạy vào tốc độ và mức nền
của bằng chứng — đánh đổi giữa độ nhạy và báo động giả dịch chuyển mạnh khi đổi giàn thử.

## 7. Diễn giải và hệ quả cho bài báo

- Không được tuyên bố PAVE-Net/B1-V3-4s "tổng quát hóa" sang Ottawa. Phát biểu đúng: macro-F1 0.52 [0.31, 0.68], cao hơn rõ các
  mô hình CNN (≤ 0.31) nhưng không hơn quy tắc vật lý không học (0.61), với báo động giả 55 %.
- Kết quả ngoài miền ủng hộ thông điệp trung tâm của bài: bằng chứng vật lý có thể kiểm toán được (PEV dương tính trên cả PU và
  Ottawa), và kiểm toán này giúp giải thích **vì sao** mô hình thất bại (bằng chứng BPFI cao trên bản ghi khỏe), điều không làm được
  với CNN.
- Hạn chế: 20 ổ bi, 10 bản ghi lỗi chính; một số đọc tốc độ/tệp; nhãn "khỏe" của thí nghiệm chạy-đến-hỏng; cảm biến/giàn thử khác;
  không có phân tích thích nghi miền (cố ý — EV-V1 cấm tinh chỉnh trên Ottawa).

## 8. Tệp kết quả

| Tệp | Nội dung |
|---|---|
| `results/phase4/EVALUATED.lock` | thời điểm + toàn bộ nội dung đóng băng |
| `results/phase4/tables/external_primary.{csv,json}` | kết quả chính 15 dòng (11 mô hình + 4 biến thể) |
| `results/phase4/tables/external_recording_predictions.csv` | xác suất/dự đoán theo bản ghi |
| `results/phase4/tables/external_secondary.csv` | lỗi đang phát triển + OOD |
| `results/phase4/tables/external_selective.csv` | PE-SD |
| `results/phase4/tables/external_concept_validity_speed.csv`, `external_a4_gate.csv`, `checkpoints_evaluated.csv` | chẩn đoán |
| `results/phase4/pev/pev_ottawa_*.csv` | PEV thứ cấp |
| `results/phase4/posthoc/*` | phân tích post-hoc |
| `results/phase4/ottawa_audit.csv`, `ottawa_speed_audit.csv`, `ottawa_segments4s.csv` | audit không nhãn |

## 9. Quyết định
Kiểm định ngoài đã hoàn tất theo đúng EV-V1, không vi phạm giao thức, không có rò rỉ. Kết quả âm tính một phần được giữ nguyên và
sẽ được báo cáo trung thực trong bản thảo (Results + Discussion + Limitations). Không có đánh giá Ottawa nào khác sẽ được thực hiện.
