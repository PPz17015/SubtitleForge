# SubtitleForge Pro — Hướng Dẫn Sử Dụng

Ứng dụng tạo subtitle tự động từ video bằng AI. Chỉ cần chọn file phim → hệ thống tự động xử lý → thông báo khi hoàn tất.

---

## 1. Cài đặt

```bash
pip install -r requirements.txt
```

> **Yêu cầu:** Python 3.10+, FFmpeg đã cài trong PATH, GPU NVIDIA (tuỳ chọn, để tăng tốc).

---

## 2. Khởi chạy ứng dụng

```bash
python src/gui.py
```

Cửa sổ **SubtitleForge Pro** sẽ hiện ra.

---

## 3. Các bước sử dụng

### Bước 1: Nhập Gemini API Key

Nhập API key vào ô **"Gemini API Key"** ở phần **🔑 Cấu hình API**.

> 💡 Lấy API key miễn phí tại: [Google AI Studio](https://aistudio.google.com/app/apikey)

### Bước 2: Chọn file video

Nhấn nút **📂 Browse** → chọn file phim (MP4, MKV, AVI, MOV, tối đa 20GB).

Thông tin file (tên, dung lượng) sẽ hiển thị bên dưới.

### Bước 3: Cấu hình (tuỳ chọn)

Phần **⚙️ Cài đặt** có các tuỳ chọn sau. Mặc định đã tối ưu, bạn có thể bỏ qua nếu không cần thay đổi:

| Tuỳ chọn | Mặc định | Mô tả |
|----------|---------|-------|
| Ngôn ngữ nguồn | Japanese | Ngôn ngữ gốc của phim |
| Ngôn ngữ đích | Vietnamese | Ngôn ngữ dịch sang |
| Whisper Model | Small | Model nhận dạng giọng nói (Medium cho chất lượng cao hơn) |
| Sử dụng GPU | ✅ Bật | Tăng tốc xử lý nếu có card NVIDIA |
| Dịch theo ngữ cảnh | ✅ Bật | AI phân tích nhân vật, cảnh phim để dịch chính xác hơn |
| Kiểm tra chất lượng | ✅ Bật | Tự động kiểm tra và sửa lỗi dịch |
| Ngữ cảnh video | (trống) | Mô tả nội dung phim để dịch tốt hơn |

**Gợi ý ngữ cảnh video:**
- `Phim anime gia đình, mẹ và con trai nói chuyện`
- `J-Drama văn phòng, đồng nghiệp cùng công ty`
- `Phim hành động, nhóm cảnh sát điều tra`

### Bước 4: Bắt đầu xử lý

Nhấn nút **🚀 Bắt Đầu Xử Lý**. Hệ thống sẽ tự động chạy qua 5 giai đoạn:

```
1. Trích xuất audio    → Tách âm thanh từ video
2. Transcribe          → Nhận dạng giọng nói thành text (Whisper AI)
3. Dịch thuật          → Dịch subtitle sang tiếng Việt (Gemini AI)
4. Kiểm tra chất lượng → Tự sửa lỗi dịch sai nghĩa, xưng hô
5. Lưu file            → Xuất file subtitle (SRT + VTT + ASS)
```

Theo dõi tiến trình qua:
- **Thanh progress** — phần trăm hoàn thành
- **Trạng thái** — giai đoạn đang chạy
- **Log** — chi tiết từng bước

### Bước 5: Nhận kết quả

Khi hoàn tất, ứng dụng sẽ hiện thông báo **"Xử lý hoàn tất!"**.

File subtitle được lưu cùng thư mục với video:

```
📁 Thư mục video/
├── movie.srt    ← SubRip (phổ biến nhất)
├── movie.vtt    ← WebVTT (dùng cho web)
└── movie.ass    ← ASS (có styling)
```

---

## 4. Chọn Whisper Model

| Model | VRAM | Tốc độ | Chất lượng | Khi nào dùng |
|-------|------|--------|-----------|-------------|
| Tiny | ~1GB | Rất nhanh | Thấp | Test nhanh |
| Small | ~2GB | Vừa | Tốt | Mặc định, đủ dùng |
| Medium | ~5GB | Chậm | Rất tốt | Phim Nhật phức tạp |
| Large | ~10GB | Rất chậm | Xuất sắc | Chất lượng cao nhất |

---

## 5. Cơ chế kiểm tra chất lượng (tự động)

Khi bật **"Kiểm tra chất lượng bản dịch"**, hệ thống tự động:

1. **Phân tích nhân vật** — nhận diện speakers, xác định giọng điệu
2. **Phát hiện cảnh** — chia phim thành các cảnh dựa trên khoảng lặng
3. **Dịch thông minh** — prompt kèm thông tin nhân vật + cảnh
4. **Kiểm tra batch** — check 20 câu/lần (nhanh, tiết kiệm API)
5. **Kiểm tra nhất quán** — xưng hô, tên riêng nhất quán cả phim
6. **Tự sửa lỗi** — re-translate câu lỗi kèm feedback cụ thể

---

## 6. Xử lý sự cố

| Vấn đề | Cách xử lý |
|--------|-----------|
| Báo thiếu FFmpeg | Cài FFmpeg và thêm vào PATH |
| Lỗi GPU / CUDA | Bỏ tick "Sử dụng GPU", dùng CPU |
| Dịch chậm | Chuyển Whisper Model về Small |
| Lỗi API key | Kiểm tra key tại [Google AI Studio](https://aistudio.google.com/app/apikey) |
| File quá lớn | Giới hạn 20GB, nên dùng file dưới 10GB |
| Muốn hủy giữa chừng | Nhấn nút **⏹ Hủy** |
