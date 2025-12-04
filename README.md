Họ và tên: Nguyễn Công Hiếu
MSSV: 22110124
Ứng dụng Web cho Computer Vision

# 🖼️ Digital Imaging Processing App (Streamlit)

Ứng dụng web xử lý ảnh với Streamlit, triển khai các kỹ thuật nền tảng của Digital Image Processing: biến đổi cường độ, histogram, tương quan (NCC), lọc không gian, Fourier 1-D/2-D, PCA Face Detection, khôi phục ảnh (restoration), và hình thái học (morphology).

---

## 🚀 Cài đặt nhanh

Yêu cầu: Python 3.8+ (khuyến nghị 3.10–3.11), pip, Internet để cài thư viện.

Windows (PowerShell):

```powershell
cd "c:\Users\HIEU\OneDrive\Documents\Gki_CV\digital_imaging"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Khởi động ứng dụng:

```powershell
python -m streamlit run app.py
```

Mặc định app chạy tại: http://localhost:8501

Lưu ý PowerShell: Nếu gặp lỗi thực thi script, mở PowerShell với quyền Admin và chạy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## 📦 Thư viện chính (requirements)

- streamlit==1.28.0
- opencv-python==4.8.1.78
- numpy==1.24.3
- Pillow==10.0.1
- scikit-image==0.21.0
- matplotlib==3.7.2
- scipy==1.11.2
- scikit-learn==1.3.0 (PCA)

File `requirements.txt` đã kèm đủ các phiên bản trên.

---

## 📁 Cấu trúc thư mục

```
digital_imaging/
├── app.py                     # Ứng dụng Streamlit chính (UI & định tuyến)
├── requirements.txt
├── README.md
├── models/                    # Tuỳ chọn: lưu mô hình PCA 'pca_face_model.npz'
├── data/                      # Tuỳ chọn: dữ liệu mẫu
├── utils/                     # Tuỳ chọn
└── src/
		├── image_processing.py    # Hàm tiện ích cơ bản
		├── processing_functions.py# Biến đổi cường độ (resolution, quantization, ...)
		├── histogram.py           # Equalization & Matching
		├── correlation.py         # Normalized Cross-Correlation (NCC)
		├── filtering.py           # Convolution & các bộ lọc không gian
		├── fourier.py             # Fourier 1-D/2-D
		├── pca_face.py            # PCA training/detection + Haar fallback
		├── restoration.py         # Noise models, denoise, periodic reduction, inverse
		├── morphology.py          # Erosion, dilation, opening, closing, ...
		└── ui_helpers.py          # Các control trong sidebar
```

---

## 💻 Chạy ứng dụng

1) Kích hoạt môi trường ảo và cài thư viện (xem mục “Cài đặt nhanh”).
2) Chạy app:

```powershell
python -m streamlit run app.py
```

Nếu cổng 8501 bận, đổi cổng khác:

```powershell
python -m streamlit run app.py --server.port 8502
```

---

## 🧭 Hướng dẫn sử dụng nhanh

- Upload ảnh (PNG/JPG/JPEG/BMP) ở khu vực trung tâm.
- Chọn “Chọn loại xử lý” trong sidebar và điều chỉnh tham số.
- So sánh ảnh gốc và ảnh đã xử lý ở hai cột.
- Có thể bật Histogram cho ảnh gốc/ảnh xử lý.
- Nhấn “Tải ảnh đã xử lý” trong sidebar để lưu kết quả PNG.

---

## 🧩 Các chức năng chính theo danh mục

### 1) Intensity Transformations
- Resolution: thay đổi tỷ lệ ảnh (10%–200%).
- Quantization: lượng tử hoá mức xám/màu (4, 8, 16, 256 levels).
- RGB: bật/tắt kênh Red/Green/Blue.
- Negative Images: s = 255 − r.
- Thresholding: nhị phân hoá với ngưỡng tuỳ chọn.
- Logarithmic Transformations: s = c·log(1+r) (mở rộng vùng tối).
- Power-law (Gamma): s = r^γ (γ<1 sáng hơn, γ>1 tối hơn).
- Contrast Stretching: Min-Max hoặc Percentile.
- Piecewise Linear: biến đổi tuyến tính từng đoạn.
- Gray-level Slicing: làm nổi bật dải mức xám.
- Bit-plane Slicing: hiển thị/tái tạo theo bit-plane.

### 2) Histogram
- Histogram Equalization: cân bằng histogram để tăng tương phản.
- Histogram Matching: khớp Uniform/Gaussian/ảnh tham chiếu.

### 3) Correlation (NCC)
- Auto Detect Mask: trích template từ ảnh gốc (toạ độ/size theo %).
- Upload Template: dùng ảnh mẫu tải lên.
- Custom Kernel: tương quan với kernel tuỳ chỉnh.
Kết quả: hiển thị vùng tương quan cao; có thể xem mask đã trích.

### 4) Filtering (Không gian)
- Convolution: nhân chập với kernel tuỳ chỉnh (đảo kernel chuẩn).
- Smoothing Linear Filter: Average/Gaussian/Box (kernel, sigma).
- Median Filter: lọc nhiễu muối tiêu.
- Sharpening: Laplacian, Unsharp, High-boost.
- Spatial Filter: nhóm tuỳ chọn (Smoothing/Order-Statistic/Sharpening/High-Pass).

### 5) Fourier Transform
- 2-D: hiển thị phổ biên độ ảnh (đã chuẩn hoá để xem).
- 1-D: chọn Row/Column index, hiển thị tín hiệu và magnitude.

### 6) PCA Face Detection
- Mặc định: cố gắng tải mô hình PCA: `models/pca_face_model.npz`.
	- Nếu không có, dùng Haar cascade làm fallback.
- Tuỳ chọn nâng cao: “Train from uploaded faces” để huấn luyện PCA từ ảnh mặt tải lên (dạng grayscale, thống nhất kích thước).
- Kết quả: khung phát hiện và đếm số vùng.

### 7) Restoration (Khôi phục/Mô phỏng suy giảm)
- Noise Models: Gaussian (mean/var), Salt & Pepper (amount), Periodic (amplitude, tần số u/v).
- Spatial Denoising: Median/Gaussian/Average (kernel, sigma).
- Periodic Noise Reduction: tự phát hiện đỉnh nhiễu theo phổ và tạo notch filter (Top-K, bán kính notch).
- Linear Degradation (simulate): Gaussian blur (sigma) hoặc Motion blur (length/angle).
- Inverse Filtering: lọc nghịch có điều chuẩn (epsilon) với PSF Gaussian/Motion.

### 8) Morphology
- Erosion, Dilation, Opening, Closing, Gradient, Top-hat, Black-hat.
- Kernel Shape: Rect/Ellipse/Cross, Kernel Size: lẻ (3–31), Iterations cho các phép cần lặp.

---

## 🔧 Tuỳ chỉnh & mở rộng

- Thêm thuật toán mới vào các file trong `src/` rồi nối UI ở `app.py` và `src/ui_helpers.py`.
- Có thể bổ sung mô hình PCA tiền huấn luyện vào `models/pca_face_model.npz` để tăng tốc.

---

## ❗ Troubleshooting

- Streamlit báo cổng bận: thêm `--server.port 8502`.
- PowerShell không chạy được Activate.ps1: cấp quyền với `Set-ExecutionPolicy` (xem phần Cài đặt nhanh).
- Lỗi thiếu thư viện: đảm bảo đã kích hoạt đúng venv và `pip install -r requirements.txt` thành công.