# PAVE-Net — Đặc tả thiết kế Phase 3B (CHỈ THIẾT KẾ, CHƯA TRIỂN KHAI)

Ngày: 19/11/2026. Căn cứ: `docs/PHASE3A_REPORT.md`, `docs/RESEARCH_PROTOCOL_V2.1.md`.
Không kiến trúc nào được giả định là tốt hơn. Kiến trúc cuối cùng chỉ được chọn bằng vòng trong của validation lồng nhau
(mục 6). Không dùng Firefly, PSO, TLBO hay bất kỳ bộ tối ưu heuristic nào; mạng nơ-ron huấn luyện bằng AdamW; chọn siêu tham số
bằng lưới nhỏ khai báo trước. Tương thích RTX 3060 Laptop 6 GB.

## 1. Bài học từ Phase 3A định hình thiết kế
1. **Khái niệm không đồng đều về độ tin cậy** (`results/phase3a/concepts/concept_reliability.csv`): họ BPFI phân biệt IR-vs-H tốt
   ở cấp ổ; họ BPFO yếu hơn; `BPFO_sb` **ngược chiều** (vòng ngoài đứng yên nên không có điều chế theo trục); `contrast` OR-vs-IR tốt.
2. **Va chạm với hài trục**: với ổ 6203, BPFO ≈ 3,05·fr và BPFI ≈ 4,93–4,95·fr chỉ cách 3·fr và 5·fr khoảng 1–2 %. Ở 900 v/ph,
   cửa sổ 1 s (1 Hz) không tách được; khái niệm BPFI không xác định ở 97,5 % cửa sổ FAG tại 900 v/ph.
3. **Đặc trưng phổ thông thường mã hoá nhà sản xuất/phiên đo** (phép dò S-4c), trong khi 9 khái niệm vật lý gần như không.
   Mọi nhánh latent (CNN) có nguy cơ học lối tắt này.
4. **CNN quá tự tin trên ổ chưa thấy** (nhiệt độ hiệu chỉnh vòng trong lên tới ≈ 12).
5. **Hư-vs-khỏe là điểm yếu chính**; IR-vs-OR khi có bằng chứng là điểm mạnh.

## 2. Bộ khái niệm V3 (đề xuất; phải đóng băng bằng sửa đổi giao thức trước khi huấn luyện 3B)
| Khái niệm | Thay đổi so với V2 | Lý do |
|---|---|---|
| `BPFO_snr1`, `BPFO_harm`, `BPFO_nharm` | giữ | hợp lệ một phần (OR>H cấp ổ 0,70–0,76) |
| `BPFO_sb` | **bỏ** | không có cơ sở vật lý; ngược chiều trên dữ liệu |
| `BPFI_snr1`, `BPFI_harm`, `BPFI_nharm`, `BPFI_sb` | giữ | hợp lệ (IR>H cấp ổ 0,94–1,00) |
| `contrast` | giữ | OR-vs-IR 0,88 |
| `ident_O`, `ident_I` (mới) | cờ khả nhận diện: khoảng cách vạch họ tới hài trục gần nhất chia cho độ phân giải | để mô hình biết khi khái niệm không đo được |
| `meas_*` (mới) | tổng hợp khái niệm trên cả lần đo 4 s (độ phân giải 0,25 Hz) | giảm va chạm hài trục ở 900 v/ph |
| mặt nạ NaN | thay cho imputation bằng trung vị | không che giấu khái niệm không xác định |
Mọi tham số (vùng hiệu chỉnh, ngưỡng) chỉ khớp trên train của cấp tương ứng.

## 3. Ứng viên A — PECB thuần
```
rung (1 s, 32 kHz, chuẩn hoá) -> E_v: CNN-1D gọn (≤ 0,3 M) -> ĉ (K khái niệm dự đoán) -> g: tuyến tính/LR-like -> logits
loss = CE(logits, y) + λ_c · Σ_k m_k · Huber(ĉ_k, c_obs,k)        (m_k = mặt nạ hợp lệ của khái niệm)
```
- Bộ phân loại chỉ nhìn ĉ (nút thắt thật sự). Biến thể huấn luyện: tuần tự (khái niệm trước) và đồng thời.
- Lưới: λ_c ∈ {0,3; 1; 3}; lr ∈ {1e-3; 3e-4}; số epoch chọn theo đường cong vòng trong.
- Ưu điểm: giải thích qua ĉ; có thể can thiệp khái niệm (thay ĉ bằng c_obs).
- Rủi ro: E_v có thể dự đoán ĉ bằng lối tắt (nhà sản xuất); hiệu năng bị chặn trên bởi độ tin cậy của khái niệm.

## 4. Ứng viên B — PAVE-Net neo vật lý
```
c_obs (khái niệm đo, tất định) + q (độ tin cậy: cờ khả nhận diện, mặt nạ NaN, SNR nền)  ->  logits_phys = W_p · [c_obs, q]
rung -> E_v: CNN-1D nhẹ (theo bậc + chuẩn hoá RMS cửa sổ) -> h -> logits_lat = W_l · h
cổng tin cậy bằng chứng: g = σ(u · [q, max family score]) ∈ (0, 1)
logits = logits_phys + γ · (1 − g) · logits_lat,     γ ∈ {0; 0,5; 1} (γ = 0 ≡ B1-like), L2 trên W_l
hiệu chỉnh: temperature scaling trên OOF vòng trong
```
- Ý tưởng: khi bằng chứng vật lý mạnh và đo được (g → 1), quyết định do khái niệm đo; khi yếu (g → 0), nhánh latent được phép đóng góp.
  Nhánh latent **luôn bị kiểm toán**: phép dò nhà sản xuất/năm đo trên h, và can thiệp phổ đích-vs-đối chứng.
- Câu hỏi nghiên cứu: B có giữ được hiệu năng của B1 trên ổ có bằng chứng mạnh (KA04, KA16, KI04, KI18), đồng thời cải thiện
  ổ có bằng chứng yếu, **mà không** tăng độ dự đoán nhà sản xuất/năm đo?

## 5. Mô-đun tương lai
**PEV — Kiểm chứng bằng chứng vật lý.**
- Can thiệp phổ trong miền bao (đã triển khai và kiểm thử ở Phase 3A: `pave/explain/interventions.py`): suy giảm vạch họ đích
  (h = 1–3, dải biên cho BPFI) vs đối chứng cùng độ rộng và đối chứng cùng năng lượng bị loại bỏ; loại trừ chồng lấn với họ kia và
  vùng bảo vệ hài trục; báo cáo năng lượng, méo, tỷ lệ cắt bao, rò ngoài dải.
- Can thiệp khái niệm (cho A): thay ĉ_k bằng c_obs,k, hoặc đặt bằng giá trị tham chiếu khỏe; đo Δlogit.
- Chỉ số: Δp(lớp đúng) đích − đối chứng theo ổ (đơn vị = ổ), độ ổn định theo điều kiện và độ mạnh can thiệp.
- Không tuyên bố nhân quả; chỉ là "phụ thuộc có kiểm soát".

**PE-SD — Chẩn đoán chọn lọc có nhận thức bằng chứng.**
- Điểm rủi ro R = hồi quy logistic có regularize trên [max-prob sau TS, entropy, độ lệch khái niệm D (A) hoặc g (B), cờ khả nhận diện,
  max family score], khớp **chỉ** trên OOF vòng trong.
- Ngưỡng cho độ bao phủ 80 %/90 % chọn trên OOF vòng trong, đông cứng trước khi xem ổ test ngoài.
- Bắt buộc so sánh với: max-softmax của B1 (P1, sau TS), entropy, max-softmax của CNN, và R chỉ dựa trên bằng chứng.
- Chỉ số: AURC, rủi ro tại 80/90 %, độ bao phủ thực tế, số cửa sổ/ổ bị hoãn theo lớp, ở cấp cửa sổ và cấp ổ
  (hoãn cả ổ nếu đa số cửa sổ bị hoãn).

## 6. Quy trình chọn kiến trúc (hợp lệ về phương pháp)
1. Sửa đổi giao thức V2.2 (đóng băng) gồm: bộ khái niệm V3, lưới của A và B, tiêu chí chọn, quy tắc quyết định — **trước** khi huấn luyện.
2. Mọi ứng viên (B1, A, B) chạy cùng validation lồng nhau V2.1 (5 fold ngoài × 4 fold trong, cùng manifest).
3. **Chọn chỉ trên OOF vòng trong** (12 ổ/fold): tiêu chí chính = macro-F1 cửa sổ trung bình 4 fold trong; phụ = NLL sau TS.
   Ứng viên chỉ được chấp nhận nếu vượt B1 trên OOF vòng trong ≥ 0,03 macro-F1 ở ≥ 4/5 fold ngoài, và phép dò nhà sản xuất trên
   biểu diễn của nó không tăng so với đặc trưng khái niệm.
4. Kết quả ổ test ngoài chỉ được báo cáo, không dùng để chọn.
5. Vì PU đã bị xem, xác nhận cuối cần **bộ dữ liệu ngoài được khoá trước** (Phase 4).

## 7. Điều chỉnh sau khi có kết quả Phase 3A
- **Siêu tham số:** validation lồng nhau không phân biệt được cấu hình (quy tắc dự phòng §6 bị kích hoạt). Phase 3B dùng siêu tham số
  **mặc định khai báo trước**; vòng trong chỉ dùng để khớp nhiệt độ, khớp bộ chấm điểm PE-SD và báo cáo độ ổn định. Tiêu chí chấp nhận
  ở mục 6.3 vẫn áp dụng, nhưng tính trên OOF vòng trong với cấu hình cố định.
- **Mốc so sánh:** B1 (= P1). B2 không cải thiện cấp ổ, nên phi tuyến trên khái niệm hiện tại không phải hướng ưu tiên.
- **Nhánh latent:** CNN thô hiện tại không phản ứng với việc xoá vạch lỗi (|Δp| ≤ 0,03) và quá tự tin (T tới 20). Trong ứng viên A, bộ
  mã hoá phải được ép học khái niệm (λ_c lớn) và kiểm tra lại bằng can thiệp. Trong ứng viên B, nhánh latent chỉ được phép đóng góp
  khi cổng bằng chứng mở và phải qua phép dò nhà sản xuất.
- **PE-SD ưu tiên:** độ tin cậy cấp ổ của B1 sau TS đã có thông tin (thăm dò: AUROC 0,77 cho đúng/sai), còn CNN thì không (0,36).
- **Dữ liệu ngoài:** chọn và khoá trước Phase 4; ứng viên cần xác minh giấy phép, số ổ độc lập, hình học, tốc độ.

## 8. Ngân sách
Mỗi ứng viên: 5 × 4 fold trong × ≤ 6 cấu hình × ≤ 20 epoch (≈ 30 s/khớp) ≈ 1 giờ GPU; huấn luyện lại 5 × 3 seed ≈ 15 phút.
Tổng Phase 3B ≤ 4 giờ GPU; batch ≤ 64 cửa sổ × 32 000 mẫu, fp16 autocast; bộ nhớ GPU < 2 GB.
