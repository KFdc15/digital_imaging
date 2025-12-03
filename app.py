import streamlit as st
import cv2
import io
from PIL import Image

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from image_processing import ImageProcessor
from ui_helpers import load_image, display_histogram, setup_sidebar_controls
from processing_functions import (
    apply_resolution, apply_quantization, apply_rgb_channels, apply_negative,
    apply_thresholding, apply_logarithmic, apply_gamma, apply_contrast_stretching,
    apply_piecewise_linear, apply_gray_level_slicing, apply_bit_plane_slicing
)
from histogram import apply_histogram_equalization, apply_histogram_matching
from correlation import (
    apply_correlation_auto_detect, apply_correlation_template, apply_correlation_custom_kernel
)
from filtering import (
    apply_convolution, apply_smoothing_linear_filter, apply_median_filter,
    apply_sharpening, apply_spatial_filter
)
from fourier import fourier_1d, fourier_2d
from pca_face import build_pca_model, detect_faces_pca, load_pca_model, haar_detect_faces
from restoration import (
    add_gaussian_noise, add_salt_pepper_noise, add_periodic_noise,
    spatial_denoise, periodic_noise_reduction,
    apply_linear_degradation, inverse_filtering
)
from morphology import (
    morph_erosion, morph_dilation, morph_open, morph_close,
    morph_gradient, morph_tophat, morph_blackhat
)


def main():
    st.set_page_config(
        page_title="Digital Imaging Processing",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("Digital Imaging Processing Application")
    st.markdown("### Ứng dụng xử lý ảnh với các kỹ thuật Digital Imaging")
    
    st.sidebar.header("Tùy chọn xử lý")
    
    processing_category = st.sidebar.selectbox(
        "Chọn loại xử lý",
        ["Không xử lý", "Resolution", "Quantization", "RGB", "Negative Images", 
         "Thresholding", "Logarithmic Transformations", "Power-law (Gamma)", 
         "Contrast Stretching", "Piecewise Linear", "Gray-level Slicing", 
         "Bit-plane Slicing", "Histogram Equalization", "Histogram Matching",
            "Correlation", "Convolution", "Smoothing Linear Filter", "Median Filter", "Sharpening",
                "Spatial Filter", "Fourier Transform", "PCA Face Detection", "Morphology", "Restoration"]
    )
    
    params = setup_sidebar_controls(processing_category)
    
    uploaded_file = st.file_uploader("Chọn ảnh để xử lý", type=['png', 'jpg', 'jpeg', 'bmp'])
    
    if uploaded_file is not None:
        original_image = load_image(uploaded_file)
        processed_image = original_image.copy()
        processor = ImageProcessor()
        
        # Áp dụng xử lý dựa trên loại được chọn
        if processing_category == "Resolution":
            processed_image = apply_resolution(original_image, processor, params['scale'])
        
        elif processing_category == "Quantization":
            processed_image = apply_quantization(original_image, params['levels'])
        
        elif processing_category == "RGB":
            processed_image = apply_rgb_channels(original_image, params['show_red'], params['show_green'], params['show_blue'])
        
        elif processing_category == "Negative Images":
            processed_image = apply_negative(original_image)
        
        elif processing_category == "Thresholding":
            processed_image = apply_thresholding(original_image, processor, params['threshold_value'])
        
        elif processing_category == "Logarithmic Transformations":
            processed_image = apply_logarithmic(original_image, processor, params['c_log'])
        
        elif processing_category == "Power-law (Gamma)":
            processed_image = apply_gamma(original_image, params['gamma'])
        
        elif processing_category == "Contrast Stretching":
            processed_image = apply_contrast_stretching(
                original_image, 
                params['contrast_method'],
                params.get('percentile_low', 2),
                params.get('percentile_high', 98)
            )
        
        elif processing_category == "Piecewise Linear":
            processed_image = apply_piecewise_linear(
                original_image, processor, 
                params['r1'], params['s1'], params['r2'], params['s2']
            )
        
        elif processing_category == "Gray-level Slicing":
            processed_image = apply_gray_level_slicing(original_image, processor, params['slice_min'], params['slice_max'])
        
        elif processing_category == "Bit-plane Slicing":
            processed_image = apply_bit_plane_slicing(
                original_image, processor, 
                params['bit_plane'], 
                params.get('reconstruct_planes', [])
            )
        
        elif processing_category == "Histogram Equalization":
            processed_image = apply_histogram_equalization(original_image, processor)
        
        elif processing_category == "Histogram Matching":
            processed_image = apply_histogram_matching(
                original_image, processor,
                params['matching_method'],
                params.get('reference_image'),
                params.get('gaussian_mean', 128),
                params.get('gaussian_std', 30)
            )
            if processed_image is None:
                processed_image = original_image.copy()
                st.warning("Please upload a reference image for histogram matching")
        
        elif processing_category == "Correlation":
            if params['correlation_method'] == "Auto Detect Mask":
                processed_image = apply_correlation_auto_detect(
                    original_image, processor,
                    params['mask_size'], params['mask_x'], params['mask_y']
                )
            elif params['correlation_method'] == "Upload Template":
                result = apply_correlation_template(
                    original_image, 
                    params.get('template_image'), 
                    processor
                )
                if result is None:
                    processed_image = original_image.copy()
                    st.warning("Please upload a template image for normalized correlation")
                else:
                    processed_image = result
            else:
                processed_image = apply_correlation_custom_kernel(
                    original_image, processor, params['custom_kernel']
                )
        
        elif processing_category == "Convolution":
            processed_image = apply_convolution(original_image, params['custom_kernel'])
        
        elif processing_category == "Smoothing Linear Filter":
            processed_image = apply_smoothing_linear_filter(
                original_image, 
                params['filter_type'], 
                params['kernel_size_filter'],
                params.get('sigma', 1.0)
            )
        
        elif processing_category == "Median Filter":
            processed_image = apply_median_filter(original_image, params['median_kernel_size'])
        
        elif processing_category == "Sharpening":
            processed_image = apply_sharpening(
                original_image, 
                params['sharpen_method'], 
                params['sharpen_strength']
            )
        
        elif processing_category == "Spatial Filter":
            processed_image = apply_spatial_filter(
                original_image, 
                params['spatial_filter_type'], 
                params['spatial_kernel_size'],
                params.get('spatial_sigma', 1.0)
            )
        
        # FOURIER TRANSFORM
        elif processing_category == "Fourier Transform":
            mode = params.get('fourier_mode', '2-D')
            if mode == '2-D':
                processed_image = fourier_2d(original_image)
            else:
                axis = params.get('fourier_axis', 'Row').lower()
                idx = int(params.get('fourier_index', 0))
                signal, magnitude = fourier_1d(original_image, axis=axis, index=idx)
                # Draw a guide line on the original for visualization
                disp = original_image.copy()
                if len(disp.shape) == 2:
                    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
                if axis == 'row':
                    y = max(0, min(disp.shape[0]-1, idx))
                    disp[y:y+1, :, :] = [255, 0, 0]
                else:
                    x = max(0, min(disp.shape[1]-1, idx))
                    disp[:, x:x+1, :] = [255, 0, 0]
                processed_image = disp
                # Show plots in the right column after images
                st.session_state['fourier_1d_signal'] = signal
                st.session_state['fourier_1d_magnitude'] = magnitude
                st.session_state['fourier_1d_axis'] = axis
                st.session_state['fourier_1d_index'] = idx
        
        # PCA FACE DETECTION
        elif processing_category == "PCA Face Detection":
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'pca_face_model.npz')
            model = None
            # Option A: user wants to train on uploaded faces now
            if params.get('pca_use_uploaded_training'):
                face_files = params.get('pca_training_images', [])
                faces = []
                for f in face_files:
                    try:
                        faces.append(load_image(f))
                    except Exception:
                        pass
                model = build_pca_model(
                    faces,
                    window_size=(params.get('pca_window_h', 64), params.get('pca_window_w', 64)),
                    n_components=params.get('pca_components', 20)
                )
                if model is None:
                    st.warning("Không có ảnh huấn luyện PCA được cung cấp. Sẽ thử dùng mô hình có sẵn hoặc Haar fallback.")
            # Option B: load pre-trained model from disk
            if model is None:
                model = load_pca_model(model_path)
            if model is not None:
                vis, boxes = detect_faces_pca(
                    original_image,
                    model,
                    stride=int(params.get('pca_stride', 16)),
                    threshold=float(params.get('pca_threshold', 1500.0))
                )
                processed_image = vis
                st.info(f"Số vùng phát hiện (PCA): {len(boxes)}")
            else:
                # Fallback to Haar if no PCA model is available
                vis, boxes = haar_detect_faces(original_image)
                processed_image = vis
                st.info(f"Số vùng phát hiện (Haar fallback): {len(boxes)}")

        # MORPHOLOGY
        elif processing_category == "Morphology":
            op = params.get('morph_operation', 'Erosion')
            shape = params.get('morph_shape', 'Rect')
            ksize = int(params.get('morph_ksize', 5))
            iters = int(params.get('morph_iterations', 1))
            if op == "Erosion":
                processed_image = morph_erosion(original_image, shape=shape, ksize=ksize, iterations=iters)
            elif op == "Dilation":
                processed_image = morph_dilation(original_image, shape=shape, ksize=ksize, iterations=iters)
            elif op == "Opening":
                processed_image = morph_open(original_image, shape=shape, ksize=ksize, iterations=iters)
            elif op == "Closing":
                processed_image = morph_close(original_image, shape=shape, ksize=ksize, iterations=iters)
            elif op == "Gradient":
                processed_image = morph_gradient(original_image, shape=shape, ksize=ksize)
            elif op == "Top-hat":
                processed_image = morph_tophat(original_image, shape=shape, ksize=ksize)
            else:
                processed_image = morph_blackhat(original_image, shape=shape, ksize=ksize)

        # RESTORATION
        elif processing_category == "Restoration":
            task = params.get('restoration_task')
            if task == "Noise Models":
                ntype = params.get('noise_type')
                if ntype == "Gaussian":
                    processed_image = add_gaussian_noise(
                        original_image,
                        mean=float(params.get('gauss_mean', 0.0)),
                        var=float(params.get('gauss_var', 0.01))
                    )
                elif ntype == "Salt & Pepper":
                    processed_image = add_salt_pepper_noise(
                        original_image,
                        amount=float(params.get('sp_amount', 0.02))
                    )
                else:
                    processed_image = add_periodic_noise(
                        original_image,
                        amplitude=float(params.get('per_amp', 30.0)),
                        freq_u=int(params.get('per_fu', 5)),
                        freq_v=int(params.get('per_fv', 5))
                    )
            elif task == "Spatial Denoising":
                processed_image = spatial_denoise(
                    original_image,
                    method=params.get('denoise_method', 'Median'),
                    kernel_size=int(params.get('denoise_kernel', 5)),
                    sigma=float(params.get('denoise_sigma', 1.0))
                )
            elif task == "Periodic Noise Reduction":
                processed_image = periodic_noise_reduction(
                    original_image,
                    k_peaks=int(params.get('notch_k', 10)),
                    notch_radius=int(params.get('notch_radius', 3))
                )
            elif task == "Linear Degradation (simulate)":
                dtyp = params.get('deg_type', 'Gaussian Blur')
                if dtyp == "Gaussian Blur":
                    processed_image = apply_linear_degradation(
                        original_image, method="Gaussian", sigma=float(params.get('deg_sigma', 2.0))
                    )
                else:
                    processed_image = apply_linear_degradation(
                        original_image, method="Motion",
                        length=int(params.get('deg_length', 15)),
                        angle=int(params.get('deg_angle', 0))
                    )
            else:  # Inverse Filtering
                ityp = params.get('inv_type', 'Gaussian')
                if ityp == "Gaussian":
                    processed_image = inverse_filtering(
                        original_image, method="Gaussian", sigma=float(params.get('inv_sigma', 2.0)),
                        epsilon=float(params.get('inv_epsilon', 1e-3))
                    )
                else:
                    processed_image = inverse_filtering(
                        original_image, method="Motion",
                        length=int(params.get('inv_length', 15)),
                        angle=int(params.get('inv_angle', 0)),
                        epsilon=float(params.get('inv_epsilon', 1e-3))
                    )

        # Hiển thị ảnh gốc và ảnh đã xử lý
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh gốc")
            st.image(original_image, channels="RGB" if len(original_image.shape) == 3 else "GRAY", use_column_width=True)
            
            if st.checkbox("Hiển thị Histogram (Ảnh gốc)"):
                fig = display_histogram(original_image, "Histogram - Ảnh gốc")
                st.pyplot(fig)
        
        with col2:
            st.subheader("Ảnh đã xử lý")
            st.image(processed_image, channels="RGB" if len(processed_image.shape) == 3 else "GRAY", use_column_width=True)
            
            if st.checkbox("Hiển thị Histogram (Ảnh xử lý)"):
                fig = display_histogram(processed_image, "Histogram - Ảnh xử lý")
                st.pyplot(fig)
            
            # Extra plots for 1-D Fourier
            if processing_category == "Fourier Transform" and st.session_state.get('fourier_1d_signal') is not None:
                import matplotlib.pyplot as plt
                fig_sig, ax = plt.subplots(1, 2, figsize=(10, 3))
                ax[0].plot(st.session_state['fourier_1d_signal'])
                ax[0].set_title(f"1-D Signal ({st.session_state['fourier_1d_axis']} {st.session_state['fourier_1d_index']})")
                ax[0].grid(True, alpha=0.3)
                ax[1].plot(st.session_state['fourier_1d_magnitude'])
                ax[1].set_title("Magnitude Spectrum")
                ax[1].grid(True, alpha=0.3)
                st.pyplot(fig_sig)
        
        # Hiển thị mask nếu dùng Auto Detect Mask
        if processing_category == "Correlation" and params.get('correlation_method') == "Auto Detect Mask":
            if 'extracted_mask' in st.session_state:
                st.markdown("---")
                st.subheader("Mask được trích xuất từ ảnh gốc")
                mask_img = st.session_state['extracted_mask']
                st.image(mask_img, channels="RGB" if len(mask_img.shape) == 3 else "GRAY", use_column_width=True)
        
        # Thông tin ảnh và nút tải xuống
        st.sidebar.markdown("---")
        st.sidebar.subheader("Thông tin ảnh")
        st.sidebar.write(f"**Kích thước gốc:** {original_image.shape[1]} x {original_image.shape[0]}")
        if len(original_image.shape) == 3:
            st.sidebar.write(f"**Số kênh:** {original_image.shape[2]}")
        st.sidebar.write(f"**Kiểu dữ liệu:** {original_image.dtype}")
        
        if processing_category != "Không xử lý":
            st.sidebar.markdown("---")
            
            if len(processed_image.shape) == 2:
                pil_img = Image.fromarray(processed_image)
            else:
                pil_img = Image.fromarray(processed_image)
            
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.sidebar.download_button(
                label="Tải ảnh đã xử lý",
                data=byte_im,
                file_name="processed_image.png",
                mime="image/png"
            )
    
    else:
        st.info("Vui lòng upload một ảnh để bắt đầu xử lý")
        
        with st.expander("Hướng dẫn sử dụng"):
            st.markdown("""
            ### Cách sử dụng ứng dụng:
            
            1. **Chọn loại xử lý**: Sử dụng dropdown trong sidebar
            2. **Điều chỉnh tham số**: Thay đổi các slider/checkbox
            3. **Upload ảnh**: Click vào nút "Browse files" và chọn ảnh từ máy tính
            4. **Xem kết quả**: So sánh ảnh gốc và ảnh đã xử lý
            5. **Tải xuống**: Click nút "Tải ảnh đã xử lý" để lưu kết quả
            
            ### Các loại xử lý có sẵn:
            
            - **Resolution**: Thay đổi độ phân giải của ảnh (10% - 200%)
            - **Quantization**: Giảm số mức màu (4, 8, 16, 256 levels)
            - **RGB**: Hiển thị/ẩn các kênh màu Red, Green, Blue
            - **Negative Images**: Đảo ngược giá trị pixel (s = 255 - r)
            - **Thresholding**: Chuyển ảnh sang nhị phân với ngưỡng tùy chỉnh
            - **Logarithmic Transformations**: s = c * log(1 + r) - mở rộng vùng tối
            - **Power-law (Gamma)**: s = r^γ - hiệu chỉnh độ sáng (γ<1: sáng hơn, γ>1: tối hơn)
            - **Contrast Stretching**: Min-Max hoặc Percentile stretching
            - **Piecewise Linear**: Biến đổi tuyến tính từng phần với các điểm tùy chỉnh
            - **Gray-level Slicing**: Làm nổi bật một dải mức xám cụ thể
            - **Bit-plane Slicing**: Hiển thị từng bit-plane riêng lẻ hoặc tái tạo từ các bit-plane đã chọn
            - **Histogram Equalization**: Cân bằng histogram để tăng độ tương phản
            - **Histogram Matching**: Khớp histogram với Uniform, Gaussian hoặc ảnh tham chiếu
            - **Normalized Correlation**: Đo lường sự tương đồng giữa ảnh và template (Auto Detect, Upload Template, Custom Kernel)
            - **Convolution**: Áp dụng convolution với kernel tùy chỉnh (flipped)
            - **Smoothing Linear Filter**: Làm mượt ảnh với Average, Gaussian hoặc Box filter
            - **Median Filter**: Lọc nhiễu muối tiêu với median filter
            - **Sharpening**: Làm sắc nét với Laplacian, Unsharp Masking hoặc High-boost
            - **Spatial Filter**: Các bộ lọc không gian (Smoothing, Order-Statistic, Sharpening, High-Pass)
            - **Fourier Transform**: Phân tích phổ 1-D (hàng/cột) và 2-D
            - **PCA Face Detection**: Dò khuôn mặt bằng PCA (cửa sổ trượt + lỗi tái tạo)
            - **Morphology**: Co, giãn, mở, đóng, gradient, top-hat, black-hat với kernel Rect/Ellipse/Cross
            - **Restoration**: Nhiễu/khôi phục (Gaussian, Salt & Pepper, Periodic); Denoising; Notch filtering; Mô phỏng suy giảm Gaussian/Motion; Inverse filtering
            """)


if __name__ == "__main__":
    main()
