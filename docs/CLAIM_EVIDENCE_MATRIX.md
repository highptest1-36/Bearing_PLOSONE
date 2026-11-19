# CLAIM–EVIDENCE MATRIX (bản thảo v2, 2026-11-19)

**Mức bằng chứng**
- **A** = kiểm chứng trực tiếp, đặc tả trước, nằm trong giao thức đóng băng.
- **B** = kiểm chứng trực tiếp nhưng mô tả, công suất thấp, hoặc là endpoint thứ cấp.
- **C** = post hoc hoặc suy luận.
- **—** = không được hỗ trợ.

Cột "Phải làm yếu?" cho biết câu ở bản v1 có phải sửa hay không, và trạng thái trong v2.

## Tuyên bố chính

| # | Tuyên bố (v2) | Kết quả / tệp hỗ trợ | Mức | Phải làm yếu? |
|---|---|---|---|---|
| C1 | Với hình học 6203, cửa sổ 1 s không tách được BPFO/BPFI khỏi hài trục; bản ghi 4 s thì tách được (trừ BPFI FAG 900 rpm, δ = 0.77) | Bảng định nhận giải tích (V2.3); `pave/features/physics_v3.py`; `results/phase3b/` (phân số định nhận 77 %/87 %) | A (giải tích, không nhãn) | Không |
| C2 | B1-V3-4s có ước lượng điểm macro-F1 cấp ổ cao nhất trên PU-BD5: 0.780 [0.56, 0.93] | `results/phase3b/tables/main_results.csv`; Bảng 4 | B (ước lượng phát triển; 15 ổ; các ổ đã dùng ở Phase 3A) | **Có**: v1 viết "best model", "better than" → v2 "highest point estimate; ranking descriptive" |
| C3 | B1-V3-4s hơn B1-V2 (+0.041) và hơn CNN (+0.169) | S2 Table: [−0.18, +0.25], p = 1.0; [−0.41, +0.07], p = 0.375 | — (không có ý nghĩa thống kê) | **Có**: chỉ nêu là không được hỗ trợ |
| C4 | Quy tắc đặc tả trước loại nhánh latent (A4 0.541 vs A2 0.687; NLL 1.389 vs 0.774) | `inclusion_decision.json`; V2.2 §9 | A (quy tắc đóng băng), nhưng gộp qua các fold | **Có**: nêu rõ việc gộp; bổ sung kết quả fold-local |
| C5 | Chọn cục bộ theo fold cho cùng lựa chọn ở cả 5 fold → cùng macro-F1 0.780 | `results/phase5_audit/foldlocal_selection*.{csv,json}` (S3 File) | C (post hoc, chỉ dùng dự đoán đã lưu) | Mới; dán nhãn post hoc |
| C6 | Nhánh latent không giúp ổ bằng chứng yếu (2/6 vs 3/6) | `per_bearing_predictions.csv`; Bảng 5 | B | Không |
| C7 | Cổng hoạt động như thiết kế (g 0.52–0.63 cho ổ mạnh, 0.36–0.43 cho ổ yếu/khỏe) | `a4_gate_by_bearing.csv` | B (mô tả) | Không |
| C8 | Xóa có mục tiêu làm giảm bằng chứng 9.3 dB so với đối chứng năng lượng; Δp B1-V3-4s −0.409 [−0.62, −0.21], 10/10 ổ | `pev_targeted_minus_control.csv`; Bảng 6 | A (PEV đặc tả trong V2.2) | Không về số liệu |
| C9 | Các mô hình vật lý *phụ thuộc* vào đúng bằng chứng mà giải thích của chúng nêu (dependence + faithfulness) | C8 + đối chứng khớp độ rộng và năng lượng | A trong phạm vi các dải được kiểm định | **Có**: bỏ "exactly"; tách khỏi hợp lệ vật lý và nhân quả |
| C10 | CNN (B3) và A6 không nhạy với việc xóa các dải này | Δp +0.005; −0.008 | B | **Có**: "not at all" → "no measurable change"; không suy ra "không dùng vật lý" |
| C11 | Điểm R4 giúp A4 (AURC 0.367 → 0.148) nhưng không giúp B1-V3-4s | `selective_diagnosis.csv`; Bảng 7 | B (15 ổ, 3–6 lỗi) | Không |
| C12 | Biểu diễn học được mã hóa nhà sản xuất mạnh hơn khái niệm đo; năm đo dễ dự đoán từ khái niệm đo hơn | `shortcut_probes_summary.csv`; Bảng 8 | B (thông tin có mặt ≠ được dùng) | Không (đã có lưu ý) |
| C13 | Ottawa: B1-V3-4s macro-F1 0.518 [0.31, 0.68], bal. acc. 0.617, FA 0.55 | `results/phase4/tables/external_primary.csv`; Bảng 9; khóa EV-V1 | A (endpoint chính, đánh giá một lần) | Không |
| C14 | "Mọi mô hình vật lý vẫn trên mọi mô hình CNN" | Macro-F1 đúng; bal. acc. CNN 0.53–0.60 ≈ B1-V3-4s 0.62 > B1-V3-1s/B2 0.47 | — ở dạng v1 | **Có, sai một phần**: v2 chỉ nói macro-F1, nêu bal. acc. và giải thích tỷ lệ lớp |
| C15 | Chênh lệch 0.78 → 0.52 không chỉ do domain shift (tỷ lệ lớp, đơn vị quyết định, dữ liệu mỗi quyết định, nghĩa của nhãn khỏe) | `results/phase5_audit/prevalence_reweighting.csv` (0.518 → 0.596 khi đổi trọng số về 5/5/5) | C (số học post hoc trên ma trận đã công bố) | Mới; dán nhãn post hoc |
| C16 | Phụ thuộc bằng chứng vẫn giữ trên Ottawa (Δp B1-V3-4s −0.31 [−0.68, −0.09], 10/10 bản ghi) | `results/phase4/pev/*`; S5 Table | B (thứ cấp đặc tả trước; chạy bằng script riêng sau khi khóa; SNR trong dải thấp hơn) | Không |
| C17 | Báo động giả trùng với vạch phổ khoảng 1 % dưới BPFI trong bản ghi khỏe của ổ 1–5 | `results/phase4/posthoc/*`; Fig 8D | C (post hoc, mô tả) | **Có**: "traced to/located" → "coincided with/showed where"; nêu rõ chỉ là liên quan |
| C18 | σ_r = 1 % làm mất toàn bộ nhận dạng IR (0.270) | `external_primary.csv` (B1-V3-4s_sigma1pct) | A (độ nhạy đặc tả trước) | Không |
| C19 | Nghiên cứu "pre-registered" | Chỉ có đóng băng nội bộ bằng hash, timestamp tự ghi | — | **Có**: thay bằng "specified in internally hash-frozen protocols"; nói rõ không có registry |
| C20 | 15 ổ PU-BD5 "naturally damaged" | Lessmeier 2016, Bảng 5/10: 5 ổ khỏe + 10 ổ hư hỏng thật từ thử nghiệm tuổi thọ gia tốc | — | **Có, sai**: đã sửa |
| C21 | Đóng góp là phương pháp kiểm toán, không phải kiến trúc | Toàn bộ kết quả; mô hình cuối là hồi quy logistic | B | Không (đã định vị đúng) |
| C22 | Ngữ cảnh 4 s là yếu tố chính (1 s 0.648 vs 4 s 0.780) | So sánh khớp mô hình B1-V3-1s/4s | B (cùng mô hình; khác biệt không kiểm định cặp) | Thêm lưu ý: so sánh với CNN lẫn cả ngữ cảnh và băng thông |

## Tuyên bố đã bỏ hoặc thay ở v2

- "physics-based models remained well above every CNN-based model" → chỉ đúng với macro-F1. Đã thay bằng mô tả cả hai chỉ số.
- "the outer test confirmed that decision" → "consistent with".
- "the physics audit traced [the false alarms] to BPFI-like lines" → "coincided with envelope lines near the inner-race fault frequency".
- "Physics-audited evidence makes bearing diagnoses verifiable and their failures interpretable" → "made the model's dependence testable and its errors inspectable".
