# PRE-SUBMISSION METHOD AUDIT — bản thảo PLOS ONE (B1-V3-4s, physics-audited)

Ngày: 2026-11-19. Thực hiện theo yêu cầu kiểm toán trước khi nộp. Không huấn luyện lại, không
chạy lại đánh giá Ottawa, không đọc nhãn Ottawa ngoài các bảng kết quả đã đóng băng, không sửa giao thức/dự đoán/hash/checkpoint.
Bản thảo gốc được lưu nguyên tại `manuscript_archive/v1_2026-11-19_pre_audit/` (kèm SHA256SUMS.txt).

Quy ước: **[SỰ THẬT]** = kiểm chứng trực tiếp từ mã/tệp; **[SUY LUẬN]** = diễn giải của người kiểm toán.
Mức độ: **Nghiêm trọng** (chặn nộp) / **Lớn** (reviewer gần như chắc chắn nêu; phải sửa) / **Nhỏ**.
Vị trí "v1" tham chiếu số dòng trong bản gốc đã lưu trữ.

---

## Task 1 — Nested CV và chọn kiến trúc

### M1. Quyết định chọn kiến trúc dùng inner-OOF gộp qua cả 5 fold ngoài — **Lớn (đã xử lý)**

**[SỰ THẬT] Truy vết 60 dự đoán inner-OOF.**
- `pave/experiments/phase3a.py:26-40`: fold ngoài k giữ hàng k (3 ổ) làm test; 4 fold trong của fold k lần lượt giữ một
  trong 4 hàng còn lại làm inner-val. Mỗi fold ngoài cho 12 dự đoán cấp ổ inner-OOF (seed 42).
- `scripts/report_phase3b.py:169-187`: nối inner-OOF của **cả 5 fold ngoài** (khóa `"{fold}|{bearing}"`) thành 60 dự đoán cấp ổ,
  rồi áp **một** quyết định toàn cục: giữ nhánh latent nếu F1(A4) − F1(A2) ≥ 0.05 và NLL(A4) ≤ NLL(A2); chọn 1 s / 4 s theo F1.
- Trong 60 dự đoán, mỗi ổ trong 15 ổ xuất hiện **4 lần** (ở inner-val của 4 fold ngoài mà nó là ổ huấn luyện).
- Quy tắc này đúng như giao thức V2.2 §9 (`docs/RESEARCH_PROTOCOL_V2.2.md:128-132`) và V2.3 §4: "pooled inner OOF (5 outer × 12 bearings = 60)".
  Không có vi phạm giao thức.

**[SỰ THẬT] Có rò rỉ chọn lựa không?** Có, theo nghĩa hẹp. Nhãn của ổ test ngoài thuộc fold k đã đi vào quyết định toàn cục
qua inner fold của các fold j ≠ k. Vì vậy ước lượng outer-test của *mô hình được chọn toàn cục* không hoàn toàn độc lập với
việc chọn. Ước lượng outer-test của *từng mô hình như một quy trình cố định* (B1-V3-4s, A4, B3, …) vẫn hợp lệ, vì không có
dữ liệu test ngoài nào dùng để huấn luyện, hiệu chuẩn hay chọn ngưỡng trong chính fold đó (`check_nested`, phase3a.py:43-50).

**Cơ chế thiên lệch [SUY LUẬN].** Đây là thiên lệch do chọn lựa ("winner's curse") ở cấp quy trình. Một mô hình hợp với các
ổ cụ thể (ví dụ KA15/KA22/KA30) sẽ có inner-OOF tốt hơn, và chính các ổ đó lại được dùng làm outer-test. Mức thiên lệch
có lẽ nhỏ vì chỉ có 2 quyết định nhị phân, ngưỡng cách biệt lớn (0.541 vs 0.687; 0.704 vs 0.600), và mô hình inner là
mô hình 9 ổ khác mô hình outer 12 ổ. Nhưng về nguyên tắc phải khai báo.

**[SỰ THẬT] Phân tích độ nhạy chọn cục bộ theo fold (POST HOC, khám phá).** Script mới `scripts/audit_foldlocal_selection.py`
áp lại đúng quy tắc trong từng fold ngoài, chỉ dùng 12 dự đoán inner-OOF của fold đó. Script chỉ đọc
`results/phase3b/folds/O*.npz` đã lưu và có assert tái tạo đúng các số trong `inclusion_decision.json`. Kết quả nằm ở
`results/phase5_audit/foldlocal_selection{.csv,_summary.json}`:

| Fold | Ổ test | F1 A2 / A4 | Giữ latent | F1 1 s / 4 s | Chọn |
|---|---|---|---|---|---|
| O1 | K001 KA04 KI04 | 0.561 / 0.398 | Không | 0.628 / 0.676 | B1-V3-4s |
| O2 | K002 KA15 KI14 | 0.743 / 0.746 | Không (Δ 0.003 < 0.05) | 0.739 / 0.743 | B1-V3-4s |
| O3 | K003 KA16 KI16 | 0.561 / 0.422 | Không | 0.444 / 0.667 | B1-V3-4s |
| O4 | K004 KA22 KI18 | 0.743 / 0.439 | Không | 0.676 / 0.676 (**hòa** → 4 s theo V2.3) | B1-V3-4s |
| O5 | K005 KA30 KI21 | 0.822 / 0.663 | Không | 0.472 / 0.746 | B1-V3-4s |

Quy trình chọn cục bộ cho **cùng** dự đoán outer-test và cùng macro-F1 **0.780**. Kết luận: cơ chế thiên lệch tồn tại về
nguyên tắc nhưng không làm đổi kết quả thực tế. Lưu ý: ở O4 kết quả phụ thuộc quy tắc hòa. V2.2 quy định hòa → 1 s; V2.3 §4
(đóng băng trước kết quả) đổi thành hòa → 4 s, và mã khớp V2.3.

**Câu trong bản thảo v1 khẳng định sai/quá mức về tính độc lập:**

| Vị trí (v1) | Câu gốc | Sửa |
|---|---|---|
| `02_methods.tex:62` | "Before any outer-test result was produced, the protocol specified… on pooled inner OOF predictions (60 …)" | Nêu rõ: gộp qua 5 fold; mỗi ổ xuất hiện 4 lần; quy tắc hòa V2.3 |
| `02_methods.tex:19` | "Every bearing is tested exactly once" | "Within the development phase…"; thêm đoạn phân biệt ước lượng mô hình cố định với quyết định chọn toàn cục |
| `Table2_splits` (ghi chú; `make_tables_extra.py:40`) | "No window… of a test bearing is used for … model selection." | **Sai**, đã sửa: chỉ đúng trong một fold; quyết định chọn dùng inner-OOF gộp |
| `Table4_ablation` (ghi chú) | "Inner-OOF values (60 bearing predictions) were the only basis…" | Thêm "pooled over the five outer folds; each bearing appears four times" và kết quả fold-local |
| `03_results.tex:16` | "The outer-test results agree with this decision" | "were consistent with"; thêm kết quả fold-local |
| `04_discussion.tex:4,24` | "removed it before the outer-test results were known"; "the outer test confirmed that decision" | Viết lại: quy tắc gộp + fold-local cho cùng lựa chọn; khuyến nghị chọn cục bộ theo fold |

### M2. Tái sử dụng cùng 15 ổ qua nhiều pha phát triển — **Lớn (đã khai báo; không sửa được nếu không có dữ liệu mới)**

**[SỰ THẬT]**
- `docs/PROTOCOL_FREEZE_V2.2.txt`: "Inspected before freeze: all Phase 1-3A outputs".
- Phase 3A (`docs/PHASE3A_REPORT.md:14,100-111`) đã chạy outer-test có nhãn trên **cùng** 15 ổ PU-BD5: B1 = 0.739, B3,
  danh sách ổ sai.
- Sau đó V2.2 mới đặc tả khái niệm V3, A1–A6, cấu hình cố định và quy tắc chọn kiến trúc.

**[SUY LUẬN] Hệ quả.**
- Ước lượng nội bộ 0.780 là *ước lượng phát triển*, không phải ước lượng xác nhận.
- So sánh V3 với V2 (+0.041) đặc biệt dễ lạc quan, vì V3 được thiết kế sau khi đã thấy V2 sai ở đâu, dù thay đổi V3 dựa trên
  phân tích định nhận giải tích chứ không dựa trên ổ cụ thể.
- Chỉ Ottawa là dữ liệu chưa ảnh hưởng tới bất kỳ quyết định thiết kế nào.

**Bản thảo v1** không nói điều này. Câu "Each preceded every classification result of the development phase"
(`02_methods.tex:4`) đúng với Phase 3B nhưng gây hiểu lầm, vì Phase 3A cũng là kết quả phân loại có nhãn trên cùng các ổ.

**Đã sửa:** thêm đoạn thứ hai trong "Study design and protocol control", thêm một ý trong Limitations, và nêu trong phần
Discussion về RQ1.

---

## Task 2 — Tuyên bố về giao thức

**[SỰ THẬT] Dòng thời gian** (từ `docs/PROTOCOL_FREEZE*.txt`; mọi mốc do chính dự án ghi, cùng một máy, không có dấu thời
gian bên thứ ba):

| Giao thức | UTC | Đã thấy trước khi đóng băng | Phân loại |
|---|---|---|---|
| V2 | 2026-11-18 16:41 | chỉ metadata 1 tệp K001 | đóng băng nội bộ bằng hash, trước dữ liệu |
| V2.1 | 17:32 | toàn bộ kết quả Phase 2 | đóng băng nội bộ; thiết kế 3A sau khi thấy Phase 2 |
| V2.2 | 20:24 | toàn bộ Phase 1–3A (có kết quả outer-test có nhãn trên PU-BD5) | đặc tả trước cho các so sánh Phase 3B, nhưng **không** độc lập với dữ liệu PU |
| V2.3 | 20:26 | bảng định nhận giải tích (chỉ hình học + tốc độ danh định) | sửa đổi nội bộ, không dùng nhãn |
| V2.4 | 20:34 | phân số hợp lệ không nhãn + vùng hiệu chuẩn từ smoke run O1 | sửa đổi nội bộ, không dùng nhãn |
| EV-V1 | 22:29 | bài báo Ottawa, audit không nhãn, chạy thử tổng hợp | đặc tả trước thật sự đối với dữ liệu Ottawa; đánh giá một lần có khóa |

**Phân loại theo yêu cầu**
- *Phân tích phát triển khám phá:* Phase 2, Phase 3A, thiết kế V3 và PAVE-Net.
- *Phân tích đóng băng nội bộ bằng hash:* tất cả V2–V2.4 và EV-V1.
- *So sánh đặc tả trước thật sự:* các so sánh Phase 3B so với V2.2–V2.4 (đặc tả trước khi có kết quả 3B, nhưng trên dữ liệu
  đã dùng); toàn bộ phân tích chính và thứ cấp trên Ottawa theo EV-V1.
- *Đăng ký bên ngoài:* **không có** (không OSF, không registry, không timestamp bên thứ ba).

→ Không được dùng "pre-registered" cho nghiên cứu này. Dùng "specified in a protocol frozen (hash-recorded) before…".

**Câu quá mức trong v1 và cách sửa**

| Vị trí (v1) | Câu gốc | Sửa |
|---|---|---|
| `pavenet_main.tex:79` (Abstract) | "In a single pre-registered evaluation" | "under a protocol frozen beforehand" |
| `01_intro_related.tex:11` | "pre-register every analysis in versioned protocols" | "specify each analysis in a versioned internal protocol whose SHA-256 hash was recorded…; not deposited in a public registry" |
| `01_intro_related.tex:9` | Siddique "with a pre-registered external test" | "that the authors describe as pre-registered" (không tự xác minh được) |
| `01_intro_related.tex:20` (v) | "single, pre-specified external evaluation" | "specified in a protocol frozen before any Ottawa signal was analysed" |
| `02_methods.tex:3` | tiêu đề "protocol registration" | "protocol control"; thêm câu thời điểm do tác giả ghi, không có bên thứ ba xác minh |
| `04_discussion.tex:33` (Conclusions) | "single, pre-registered external evaluation" | "specified in a protocol frozen beforehand" |

---

## Task 3 — Công bằng baseline

**[SỰ THẬT]**
- Khái niệm vật lý (cả 1 s và 4 s) tính từ tín hiệu **64 kHz**, dải 1–20 kHz (`pave/experiments/phase3b.py:21-23`,
  `pave/data/windowing.py:92,116`).
- CNN (B3, và nhánh latent của A3/A4/A6) nhận cửa sổ **1 s**, **hạ mẫu ×2 xuống 32 kHz** (`windowing.py:109`), tức băng thông
  ≤ 16 kHz.
- B0, B1-V3-4s, A1–A4 (nhánh vật lý) dùng bản ghi **4 s**.
- Quyết định cấp ổ của mọi mô hình lấy trung bình trên mọi cửa sổ của ổ, nên tổng lượng dữ liệu cho mỗi quyết định như nhau.
  Ngữ cảnh cho *mỗi* dự đoán thì khác.

**[SUY LUẬN]**
- So sánh "physics vs CNN" trộn ba yếu tố: (i) kiến trúc/ràng buộc vật lý, (ii) độ dài ngữ cảnh (4 s vs 1 s, quyết định
  khả năng định nhận BPFO/BPFI), (iii) băng thông (20 vs 16 kHz).
- So sánh có kiểm soát duy nhất là B1-V3-1s vs B1-V3-4s, cùng mô hình nhưng khác ngữ cảnh: 0.648 vs 0.780. Điều này gợi ý
  ngữ cảnh là yếu tố lớn.
- Không thể kết luận CNN kém *vì* thiếu vật lý.
- Khác biệt B3 vs B1-V3-4s (−0.169, KTC −0.41; +0.07; McNemar p = 0.375) không có ý nghĩa thống kê.

**Đã sửa:** Methods (đoạn B3), ghi chú Bảng 4, Discussion RQ1, Limitations, Future work. **Không** khẳng định kết quả mới.

**Đề xuất phân tích tùy chọn (chỉ trên PU, phát triển, phải dán nhãn post hoc).** Chi tiết ở `docs/REVISION_PLAN.md` §B:
- (a) CNN trên 4 s, 32 kHz;
- (b) CNN trên 1 s/4 s, 64 kHz;
- (c) baseline phổ đường bao 4 s (LR/MLP trên E(f) 0–1 kHz, Δf = 0.25 Hz);
- (d) chọn kiến trúc cục bộ theo fold.

---

## Task 4 — Diễn giải kiểm định ngoài

**[SỰ THẬT] Phân bố lớp.**
- PU-BD5: 5 H / 5 IR / 5 OR **ổ**; mỗi quyết định trung bình khoảng 80 bản ghi 4 s (1200 bản ghi / 15 ổ, 4 điều kiện vận hành).
- Ottawa sơ cấp: 20 H / 5 IR / 5 OR **bản ghi**; mỗi quyết định dựa trên 1 bản ghi 10 s, 1 tốc độ.
- Ổ 1–10 đóng góp cả bản ghi khỏe lẫn bản ghi lỗi (cặp cùng ổ vật lý). KTC bootstrap theo cụm 20 ổ đã tính đến điều này.

**[SỰ THẬT] Phân tích số học post hoc** (`scripts/audit_prevalence.py`). Script chỉ dùng ma trận nhầm lẫn đã công bố trong
`results/phase4/tables/external_primary.csv`, không chạy mô hình, không đọc tín hiệu/nhãn Ottawa, và có assert tái tạo
macro-F1 đã báo cáo. Kết quả ở `results/phase5_audit/prevalence_reweighting.csv`:

| Mô hình | Ottawa macro-F1 | Ottawa macro-F1 (đổi trọng số về 5/5/5) | Bal. acc. PU → Ottawa | FA PU → Ottawa |
|---|---|---|---|---|
| B1-V3-4s | 0.518 | 0.596 | 0.800 → 0.617 | 0.0 → 0.55 |
| B0 | 0.614 | 0.706 | 0.667 → 0.717 | 0.0 → 0.45 |
| B1-V2 | 0.417 | 0.589 | 0.733 → 0.650 | 0.2 → 0.85 |
| B3 CNN | 0.311 | 0.484 | 0.600 → 0.600 | 0.2 → 1.00 |
| A4 PAVE-Net | 0.300 | 0.479 | 0.600 → 0.600 | 0.2 → 1.00 |
| A6 | 0.306 | 0.482 | 0.733 → 0.600 | 0.0 → 1.00 |
| B1-V3-1s | 0.330 | 0.434 | 0.667 → 0.467 | 0.4 → 0.80 |

**[SUY LUẬN] Vì sao chênh lệch macro-F1 (0.78 → 0.52) không chỉ do domain shift**
1. **Tỷ lệ lớp:** H chiếm 2/3 tập Ottawa. Mỗi báo động giả kéo precision của IR/OR xuống mạnh hơn so với PU. Riêng việc
   đổi trọng số đã nâng 0.518 lên 0.596.
2. **Đơn vị và lượng dữ liệu mỗi quyết định:** ổ (khoảng 80 bản ghi, 4 điều kiện) so với bản ghi (1 bản ghi, 1 điều kiện).
   Cấp bản ghi nhiễu hơn.
3. **Ngữ nghĩa nhãn "khỏe":** trạng thái khỏe của thí nghiệm chạy-đến-hỏng có thể đã chứa tiền lỗi; báo động giả tập trung
   ở ổ 1–5 (sau này hỏng IR).
4. **Phần còn lại** (bal. acc. 0.80 → 0.62) ứng với FA 0 → 0.55; recall IR/OR giữ nguyên (1.0/0.4).

**Phát hiện mới quan trọng.** Câu v1 "all physics-based models remained above all CNN-based models"
(`04_discussion.tex:15`) và "remained well above every CNN-based model" (`04_discussion.tex:4`) **chỉ đúng với macro-F1**:
- Theo balanced accuracy, CNN-based 0.53–0.60 gần bằng B1-V3-4s 0.62, và **cao hơn** B1-V3-1s/B2 (0.47).
- Lợi thế macro-F1 chủ yếu do CNN sai toàn bộ lớp khỏe, lớp chiếm đa số.
- Balanced accuracy là endpoint thứ cấp đã đặc tả trong EV-V1 (dòng 66) và đã có trong Bảng 9, nên đây không phải phân tích mới.

**Đã sửa:** Abstract, Results (thêm đoạn diễn giải và câu về balanced accuracy của CNN), Discussion (Principal findings, RQ5),
Methods (Statistical analysis), caption Fig 7. Đánh giá Ottawa một lần vẫn giữ nguyên; không chọn hay tinh chỉnh mô hình
bằng nhãn Ottawa.

---

## Task 5 — Tuyên bố về khả năng giải thích

**Phân biệt áp dụng trong bản sửa (Methods, đoạn PEV):**
- *Phụ thuộc mô hình* (output đổi khi can thiệp): **được kiểm định**.
- *Tính trung thực của giải thích* (bằng chứng được nêu = bằng chứng mô hình phụ thuộc; so với đối chứng khớp độ rộng và
  năng lượng): **được kiểm định**.
- *Tính hợp lệ vật lý* (vạch phổ thật sự do khuyết tật sinh ra): **không** kiểm định.
- *Nhân quả cơ học*: **không** kiểm định.

| Vị trí (v1) | Câu gốc | Vấn đề | Sửa |
|---|---|---|---|
| Abstract `pavenet_main.tex:79` | "the false alarms were traced to BPFI-like lines" | "traced" ngụ ý nhân quả; phân tích là post hoc, mô tả | "coincided with BPFI-like envelope lines" |
| Abstract | "made diagnoses verifiable and their failures interpretable" | quá rộng | "made the model's dependence testable and its errors inspectable" |
| Abstract | "the CNN was unaffected" | khẳng định tuyệt đối | "showed no measurable change" |
| `03_results.tex:21` | "depend on exactly the evidence their explanation names" | "exactly" quá mức | "Within the tested bands… depended…; establishes dependence and faithfulness, not that the lines were produced by the defects" |
| `03_results.tex:21` | "B3 not at all" | tuyệt đối | "showed no measurable change (+0.005)" |
| `03_results.tex:21` | "their decisions do not rest on these narrow bands" | suy ra quá | "their outputs are not sensitive to the suppression of these narrow bands" |
| `03_results.tex:48` | "located the false alarms"; "the BPFI evidence comes from a line" | ngụ ý xác định nguồn gốc | "showed where they occurred"; "the BPFI evidence that the model used is a line…"; thêm "association… descriptive" |
| `03_results.tex:50` (Fig 8) | "is read as inner-race evidence" | — | "the model uses… as inner-race evidence" |
| `04_discussion.tex:15` | "the false alarms can be traced to specific envelope lines" | nhân quả | "can be inspected at the level of specific envelope lines" |
| `04_discussion.tex:21` | "This suggests that the learned weights… transfer less well" | giả thuyết chưa kiểm định | "One possible reading, which we did not test, is…" |
| `04_discussion.tex:33` | "verified that its decisions depend…"; "which the physics audit traced to…" | — | "showed that its predictions depend…"; "which coincided with…" |

---

## Task 6 — Sửa lỗi khoa học trong bản thảo

| # | Vị trí (v1) | Vấn đề [SỰ THẬT] | Mức | Sửa |
|---|---|---|---|---|
| S1 | `02_methods.tex:8` | "the 15 bearings with naturally developed damage": sai, vì 5 ổ khỏe không có hư hỏng; 10 ổ còn lại có "real damages caused by accelerated lifetime tests" (Lessmeier 2016, Table 5, Table 10), không phải hư hỏng "tự nhiên". KA15 là biến dạng dẻo (indentation), không phải mỏi. | Lớn | "5 healthy… ten carry real damage produced in accelerated lifetime tests… as opposed to artificially machined damage" |
| S2 | `02_methods.tex:71` | "artificial versus natural" | Nhỏ | "artificial versus accelerated-lifetime damage" |
| S3 | `Table1_datasets` (ghi chú) | không nêu nguồn gốc hư hỏng | Nhỏ | thêm vào ghi chú |
| S4 | `02_methods.tex:4` | "Each preceded every classification result of the development phase": mơ hồ (Phase 3A đã có kết quả phân loại) | Lớn | viết lại (xem M2) |
| S5 | `03_results.tex:9`, `04_discussion.tex:4,33` | "best model", "better than": không có khác biệt cặp nào có ý nghĩa (S2 Table: B3 −0.169 [−0.41, +0.07], A6 −0.047 [−0.25, +0.18]) | Lớn | "highest point estimate…; ranking descriptive" |
| S6 | `Table2_splits` (ghi chú) | khẳng định không dùng ổ test cho model selection | Lớn | sửa (xem M1) |
| S7 | Fig 7A caption | đặt macro-F1 cấp ổ PU và cấp bản ghi Ottawa trên cùng trục mà không cảnh báo | Nhỏ | thêm câu không so sánh trực tiếp |
| S8 | `01_intro_related.tex:20` | "identical bearing-disjoint nested protocols": đúng về split, nhưng đầu vào khác nhau | Nhỏ | "on the same… splits" + ghi chú đầu vào trong Methods |
| S9 | `01_intro_related.tex:31` | "we compare PAVE-Net with the closest prior work": mô hình đề xuất không phải PAVE-Net | Nhỏ | "the present study" |
| S10 | Tác giả/đơn vị/lời cảm ơn | placeholder | Nghiêm trọng (định dạng) | đã điền theo thông tin tác giả cung cấp, đúng định dạng PLOS (không địa chỉ đường phố, không mã bưu chính) |

**Kiểm tra đã thực hiện, không phát hiện lỗi:**
- Công thức BPFO/BPFI.
- Đơn vị của σ_r và g_k (Hz).
- Δf = 1/T.
- SNR 20 log10.
- Số ổ và số bản ghi (1200 = 15 × 80; 30 + 10 + 20 bản ghi Ottawa).
- KTC trong Abstract khớp `main_results`/`external_primary`.
- Số trong Bảng 9 tái tạo từ ma trận nhầm lẫn (assert trong `audit_prevalence.py`).
- 28 tài liệu tham khảo đã được xác minh ở phiên trước (`02_LITERATURE/verified_references_report.md`); không thêm tài liệu mới.

**Chưa sửa (cần tác giả):**
- Tên tiếng Anh chính thức của đơn vị thứ hai. Email vaa.edu.vn thường gắn với "Vietnam Aviation Academy"; người dùng ghi
  "Vietnam Aerospace University". Đã giữ nguyên theo người dùng, **cần tác giả xác nhận**.

---

## Task 7 — Tuân thủ PLOS ONE

Xem bảng chi tiết trong `docs/REVISION_PLAN.md` §C và `FINAL_PLOS_ONE_PACKAGE/review/PLOS_COMPLIANCE_CHECKLIST.md`
(tổng hợp từ hướng dẫn tác giả tải về trong `00_AUTHOR_GUIDELINES/` và các trang chính thức của PLOS ONE).

---

## Vấn đề nghiêm trọng chưa giải quyết

Không có vấn đề phương pháp nghiêm trọng nào chưa giải quyết:
- Cơ chế thiên lệch chọn lựa (M1) đã được kiểm tra bằng phân tích fold-local và không đổi kết quả.
- Tái sử dụng dữ liệu (M2) không sửa được nếu không có dữ liệu mới, nhưng đã khai báo đầy đủ; Ottawa là kiểm định độc lập.

Các mục chặn nộp còn lại đều là **quyết định của tác giả**: kho dữ liệu/mã + DOI, tuyên bố tài trợ,
giấy phép dữ liệu, tên đơn vị.
