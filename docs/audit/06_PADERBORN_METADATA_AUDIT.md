# Kiểm toán metadata bộ dữ liệu Paderborn (Phase 2, phần C)

Nguồn: `03_MANIFESTS/paderborn/pu_files.csv`, `pu_bearings.csv`, `results/phase2/pu_audit/*`
(`python -I scripts/build_paderborn_cache.py`, `python -I scripts/audit_paderborn.py`, 18/11/2026).
Tài liệu gốc: Lessmeier et al. (2016), doi:10.36001/phme.2016.v3i1.1577 (bản lưu `02_LITERATURE/`); phiếu mô tả và nhật ký đo
của từng ổ (giải nén vào `01_RAW_DATA/Paderborn_KAt_BearingDataCenter/docs/<mã>/`).
Ký hiệu: **ĐÃ XÁC MINH** (đo từ tệp), **BÀI BÁO** (nêu trong Lessmeier 2016), **GIẢ ĐỊNH**.

## 1. Nguồn gốc và tính toàn vẹn (ĐÃ XÁC MINH)
- 32 tệp `.rar` tải từ `https://groups.uni-paderborn.de/kat/BearingDataCenter/` (18/11/2026), kích thước khớp tuyệt đối
  với `content-length` của máy chủ; SHA-256 từng tệp nén nằm trong `pu_bearings.csv`. Tệp nén gốc được giữ nguyên.
- Mỗi tệp nén: 80 tệp `.mat` (4 điều kiện × 20 lần đo) + phiếu mô tả `<mã>.pdf` + `measuring_log_<mã>.pdf`. Đủ 32/32.
- 2 560 tệp `.mat`, không có nội dung trùng lặp (SHA-256), không NaN/Inf; chuỗi giá trị hằng dài nhất 6 mẫu (không cắt ngọn).
- **1 tệp hỏng:** `KA08/N15_M01_F10_KA08_2.mat` — header và danh mục biến đọc được nhưng phần dữ liệu không giải mã được
  (`TypeError: Expecting matrix here`). Đã loại khỏi phân tích (KA08 không thuộc PU-BD5; thuộc tập train của PU-A2R).
- Nhật ký thay đổi trên máy chủ chỉ có một mục: hiệu chỉnh mô tả hình học hư hỏng của KI03 (13/07/2016).
- Giấy phép dữ liệu (BÀI BÁO): CC BY-NC 4.0, cho phép dùng học thuật phi thương mại, bắt buộc trích dẫn.

## 2. Cấu trúc tệp (ĐÃ XÁC MINH)
Struct MATLAB mang tên tệp, gồm `Info`, `X` (3 trục thời gian: Mech_4kHz, HostService 64 kHz, Temp_1Hz), `Y`
(`vibration_1`, `phase_current_1/2` @64 kHz; `speed`, `torque`, `force` @4 kHz; `temp_2_bearing_module` @1 Hz) và
`Description.General.DateTime`. **Đơn vị không được lưu trong tệp.** Nhật ký đo: cảm biến PCB 336C04 (100 mV/g), khuếch đại
Kistler 5015 (lọc 30 kHz), gia nhiệt sơ bộ 37 phút tới 45 °C. Trục thời gian 64 kHz có bước lượng tử hoá nhưng
(n−1)/T = 64 000 Hz.

## 3. Độ dài và điều kiện vận hành (ĐÃ XÁC MINH)
- Độ dài rung 256 000–299 038 mẫu (4,00–4,67 s); 44 % số tệp dài hơn 4 s; 55 tệp chỉ cho 3 cửa sổ 1 s thay vì 4.
- Tốc độ trung bình đúng danh định: 899,8 v/ph (N09) và 1499,6 v/ph (N15); một lần đo lẻ ở N09 đạt 930,9 v/ph.
  **Tốc độ không phụ thuộc lớp** ⇒ không có lối tắt tốc độ (khác KAIST).
- Mô-men đo được lệch khỏi danh định: M07 → 1,23–1,26 Nm (danh định 0,7); M01 → 0,62 Nm (danh định 0,1). Cấu trúc tương
  đối giữa các điều kiện vẫn đúng; có thể là lệch không của cảm biến (chưa xác định).
- Lực hướng tâm: F10 → 1 031–1 048 N; F04 → 419 N (đúng danh định). Nhiệt độ module ổ: 25,0–58,5 °C (bài báo nêu "khoảng 45–50 °C").

## 4. Hình học theo từng ổ (ĐÃ XÁC MINH từ phiếu mô tả)
- 3 nhà sản xuất: IBU (15 ổ), MTK (11), FAG (6). Đường kính vòng chia: FAG 28,55 mm; IBU và MTK 29,05 mm; mọi ổ 8 bi, 6,75 mm, góc 0°.
- Bậc động học: BPFO 3,054–3,071 × fr; BPFI 4,929–4,946 × fr (khác nhau ≈ 0,5 % giữa nhà sản xuất). Không ổ nào phải dùng giá trị mặc định.

## 5. Yếu tố gây nhiễu mới phát hiện (ĐÃ XÁC MINH) — quan trọng
1. **Nhà sản xuất ≡ Healthy:** cả 6 ổ khỏe (K001–K006) là IBU. Ổ hư thật: FAG 4, MTK 6, IBU 1 (KA22). Ổ nhân tạo: IBU 8, MTK 4.
2. **Thời điểm đo ≡ Healthy:** cả 6 ổ khỏe được đo trong năm **2015** (25/03 và 17/09–16/10/2015). 9/11 ổ hư thật được đo
   trong năm **2014** (06–08/2014); chỉ KI04 (07/2015) và KA30 (03/2015) đo năm 2015. 8/12 ổ nhân tạo đo năm 2014.
3. Hệ quả: đánh giá tách theo ổ **không** loại trừ được lối tắt "nhận dạng phiên đo / nhà sản xuất". Các quy tắc chỉ dùng
   metadata, học từ ổ train của từng fold (hậu nghiệm, không huấn luyện mô hình), đạt độ chính xác cấp ổ trên PU-BD5:
   **chỉ nhà sản xuất 0,53**, **chỉ năm đo 0,60**, cả hai nhận đúng 5/5 ổ khỏe (`results/phase2/tables/T_metadata_confound_controls.csv`).
4. RMS rung gần như nhau giữa các nhóm (trung vị 0,24–0,28 đơn vị lưu trữ; KB 0,52) ⇒ lối tắt năng lượng yếu (khác KAIST).

## 6. Hư hỏng (BÀI BÁO, Bảng 4–5)
Hư hỏng thật trong PU-BD5 phần lớn ở **mức 1** (KA04, KA15, KA22, KA30, KI04, KI14, KI21); mức 2: KA16, KI18; mức 3: KI16.
KA15, KA30 là vết lõm do mảnh vụn (biến dạng dẻo), không phải rỗ mỏi. KI04, KI14 là "hư hỏng nhiều" (M). Mức độ không được dùng làm nhãn.

## 7. Giới hạn
- Chỉ 5 ổ/lớp trong PU-BD5 (15 ổ); mỗi fold test có 3 ổ ⇒ độ chính xác cấp ổ của một fold chỉ nhận các giá trị 0, 1/3, 2/3, 1.
- K006 và KI17 không nằm trong PU-BD5 (theo Bảng 10 của bài báo); có trong PU-A2R.
- Đơn vị rung (V hay g sau khuếch đại) không xác định được từ tệp.
