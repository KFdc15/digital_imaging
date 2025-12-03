import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def load_image(image_file):
    img = Image.open(image_file)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB' and img.mode != 'L':
        img = img.convert('RGB')
    img_array = np.array(img)
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    return img_array


def display_histogram(image, title="Histogram"):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if len(image.shape) == 3:
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, label=color.upper())
        ax.legend()
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(hist, color='black')
    
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig
    
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig


# SIDEBAR CONTROLS
def setup_sidebar_controls(processing_category):
    params = {}
    
    # RESOLUTION
    if processing_category == "Resolution":
        params['scale'] = st.sidebar.slider("Scale (%)", 10, 200, 100, 10)
    
    # QUANTIZATION
    elif processing_category == "Quantization":
        params['levels'] = st.sidebar.selectbox("Levels", [4, 8, 16, 256])
    
    # RGB CHANNELS
    elif processing_category == "RGB":
        params['show_red'] = st.sidebar.checkbox("Red Channel", value=True)
        params['show_green'] = st.sidebar.checkbox("Green Channel", value=True)
        params['show_blue'] = st.sidebar.checkbox("Blue Channel", value=True)
    
    # THRESHOLDING
    elif processing_category == "Thresholding":
        params['threshold_value'] = st.sidebar.slider("Threshold Value", 0, 255, 127)
    
    # LOGARITHMIC TRANSFORMATIONS
    elif processing_category == "Logarithmic Transformations":
        params['c_log'] = st.sidebar.slider("C constant", 0.1, 5.0, 1.0, 0.1)
    
    # POWER-LAW (GAMMA)
    elif processing_category == "Power-law (Gamma)":
        params['gamma'] = st.sidebar.slider("Gamma (γ)", 0.1, 5.0, 1.0, 0.1)
        st.sidebar.info(f"γ < 1: Brighten dark regions\nγ = 1: No change\nγ > 1: Darken bright regions")
    
    # CONTRAST STRETCHING
    elif processing_category == "Contrast Stretching":
        params['contrast_method'] = st.sidebar.selectbox("Method", ["Min-Max", "Percentile Stretching"])
        if params['contrast_method'] == "Percentile Stretching":
            params['percentile_low'] = st.sidebar.slider("Lower Percentile", 0, 10, 2)
            params['percentile_high'] = st.sidebar.slider("Upper Percentile", 90, 100, 98)
    
    # PIECEWISE LINEAR
    elif processing_category == "Piecewise Linear":
        st.sidebar.markdown("**Define transformation points:**")
        params['r1'] = st.sidebar.slider("r1 (input)", 0, 255, 85)
        params['s1'] = st.sidebar.slider("s1 (output)", 0, 255, 0)
        params['r2'] = st.sidebar.slider("r2 (input)", 0, 255, 170)
        params['s2'] = st.sidebar.slider("s2 (output)", 0, 255, 255)
    
    # GRAY-LEVEL SLICING
    elif processing_category == "Gray-level Slicing":
        params['slice_min'] = st.sidebar.slider("Min gray level", 0, 255, 100)
        params['slice_max'] = st.sidebar.slider("Max gray level", 0, 255, 200)
    
    # BIT-PLANE SLICING
    elif processing_category == "Bit-plane Slicing":
        params['bit_plane'] = st.sidebar.slider("Bit Plane (0=LSB, 7=MSB)", 0, 7, 7)
        if st.sidebar.checkbox("Reconstruction Mode", value=False):
            params['reconstruct_planes'] = st.sidebar.multiselect(
                "Select planes to reconstruct",
                options=[7, 6, 5, 4, 3, 2, 1, 0],
                default=[7, 6, 5, 4]
            )
        else:
            params['reconstruct_planes'] = []
    
    # HISTOGRAM MATCHING
    elif processing_category == "Histogram Matching":
        params['matching_method'] = st.sidebar.selectbox("Reference Type", ["Uniform", "Gaussian", "Custom Image"])
        if params['matching_method'] == "Custom Image":
            st.sidebar.markdown("**Upload reference image:**")
            reference_file = st.sidebar.file_uploader("Reference Image", type=['png', 'jpg', 'jpeg', 'bmp'], key="reference")
            params['reference_image'] = load_image(reference_file) if reference_file is not None else None
        elif params['matching_method'] == "Gaussian":
            params['gaussian_mean'] = st.sidebar.slider("Mean", 0, 255, 128)
            params['gaussian_std'] = st.sidebar.slider("Standard Deviation", 1, 100, 30)
    
    # NORMALIZED CORRELATION
    elif processing_category == "Correlation":
        st.sidebar.markdown("**Normalized Correlation**")
        params['correlation_method'] = st.sidebar.selectbox("Method", ["Auto Detect Mask", "Upload Template", "Custom Kernel"])
        
        if params['correlation_method'] == "Auto Detect Mask":
            st.sidebar.markdown("**Automatic mask detection from image**")
            params['mask_size'] = st.sidebar.slider("Mask Size", 20, 200, 50, 10)
            params['mask_x'] = st.sidebar.slider("Mask Center X (%)", 0, 100, 50, 5)
            params['mask_y'] = st.sidebar.slider("Mask Center Y (%)", 0, 100, 50, 5)
        elif params['correlation_method'] == "Upload Template":
            st.sidebar.markdown("**Upload template/mask image:**")
            template_file = st.sidebar.file_uploader("Template Image", type=['png', 'jpg', 'jpeg', 'bmp'], key="template")
            params['template_image'] = load_image(template_file) if template_file is not None else None
        else:
            st.sidebar.markdown("**Custom Kernel (3x3):**")
            col1, col2, col3 = st.sidebar.columns(3)
            k = []
            for i in range(3):
                with [col1, col2, col3][i]:
                    for j in range(3):
                        val = st.number_input(f"[{j},{i}]", value=1.0 if i==1 and j==1 else 0.0, key=f"corr_{i}_{j}", label_visibility="collapsed")
                        k.append(val)
            params['custom_kernel'] = np.array(k).reshape(3, 3)
    
    # CONVOLUTION
    elif processing_category == "Convolution":
        st.sidebar.markdown("**Custom Kernel (3x3):**")
        col1, col2, col3 = st.sidebar.columns(3)
        k = []
        for i in range(3):
            with [col1, col2, col3][i]:
                for j in range(3):
                    val = st.number_input(f"[{j},{i}]", value=1.0/9.0, key=f"conv_{i}_{j}", label_visibility="collapsed")
                    k.append(val)
        params['custom_kernel'] = np.array(k).reshape(3, 3)
    
    # SMOOTHING LINEAR FILTER
    elif processing_category == "Smoothing Linear Filter":
        params['filter_type'] = st.sidebar.selectbox("Filter Type", ["Average", "Gaussian", "Box"])
        params['kernel_size_filter'] = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)
        if params['filter_type'] == "Gaussian":
            params['sigma'] = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
    
    # MEDIAN FILTER
    elif processing_category == "Median Filter":
        params['median_kernel_size'] = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)
    
    # SHARPENING
    elif processing_category == "Sharpening":
        params['sharpen_method'] = st.sidebar.selectbox("Method", ["Laplacian", "Unsharp Masking", "High-boost"])
        params['sharpen_strength'] = st.sidebar.slider("Strength", 0.1, 3.0, 1.0, 0.1)
    
    # SPATIAL FILTER
    elif processing_category == "Spatial Filter":
        params['spatial_filter_type'] = st.sidebar.selectbox(
            "Filter Type", 
            ["Smoothing (Average)", "Smoothing (Weighted)", "Smoothing (Gaussian)", 
             "Order-Statistic (Median)", "Order-Statistic (Max)", "Order-Statistic (Min)",
             "Sharpening (Laplacian)", "Sharpening (Gradient)", "High-Pass"]
        )
        params['spatial_kernel_size'] = st.sidebar.slider("Kernel Size", 3, 11, 3, step=2)
        if "Gaussian" in params.get('spatial_filter_type', ''):
            params['spatial_sigma'] = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
    
    # FOURIER TRANSFORM
    elif processing_category == "Fourier Transform":
        params['fourier_mode'] = st.sidebar.selectbox("Mode", ["2-D", "1-D"])  # default 2-D
        if params['fourier_mode'] == "1-D":
            params['fourier_axis'] = st.sidebar.selectbox("Axis", ["Row", "Column"])  # Row or Column
            params['fourier_index'] = st.sidebar.number_input("Row/Column Index", min_value=0, value=0, step=1)
        st.sidebar.info("1-D: phân tích 1 hàng/cột. 2-D: phổ biên độ toàn ảnh.")
    
    # PCA FACE DETECTION
    elif processing_category == "PCA Face Detection":
        params['pca_use_uploaded_training'] = st.sidebar.checkbox("Advanced: Train from uploaded faces", value=False)
        if params['pca_use_uploaded_training']:
            st.sidebar.markdown("**Upload tập ảnh khuôn mặt để huấn luyện PCA**")
            files = st.sidebar.file_uploader(
                "Training Face Images", type=['png', 'jpg', 'jpeg', 'bmp'], accept_multiple_files=True, key="pca_faces"
            )
            params['pca_training_images'] = files or []
        st.sidebar.caption("Sử dụng tham số mặc định cho PCA")
    
    # MORPHOLOGY
    elif processing_category == "Morphology":
        params['morph_operation'] = st.sidebar.selectbox(
            "Operation",
            [
                "Erosion",
                "Dilation",
                "Opening",
                "Closing",
                "Gradient",
                "Top-hat",
                "Black-hat",
            ]
        )
        params['morph_shape'] = st.sidebar.selectbox("Kernel Shape", ["Rect", "Ellipse", "Cross"]) 
        params['morph_ksize'] = st.sidebar.slider("Kernel Size", 3, 31, 5, 2)
        if params['morph_operation'] in ["Erosion", "Dilation", "Opening", "Closing"]:
            params['morph_iterations'] = st.sidebar.slider("Iterations", 1, 10, 1, 1)

    # RESTORATION
    elif processing_category == "Restoration":
        params['restoration_task'] = st.sidebar.selectbox(
            "Task",
            [
                "Noise Models",
                "Spatial Denoising",
                "Periodic Noise Reduction",
                "Linear Degradation (simulate)",
                "Inverse Filtering",
            ]
        )
        if params['restoration_task'] == "Noise Models":
            params['noise_type'] = st.sidebar.selectbox("Noise Type", ["Gaussian", "Salt & Pepper", "Periodic"])
            if params['noise_type'] == "Gaussian":
                params['gauss_mean'] = st.sidebar.slider("Mean", -0.1, 0.1, 0.0, 0.01)
                params['gauss_var'] = st.sidebar.slider("Variance", 0.0, 0.05, 0.01, 0.005)
            elif params['noise_type'] == "Salt & Pepper":
                params['sp_amount'] = st.sidebar.slider("Amount", 0.0, 0.2, 0.02, 0.01)
            else:
                params['per_amp'] = st.sidebar.slider("Amplitude", 0.0, 100.0, 30.0, 5.0)
                params['per_fu'] = st.sidebar.slider("Freq U", 1, 50, 5, 1)
                params['per_fv'] = st.sidebar.slider("Freq V", 1, 50, 5, 1)
        elif params['restoration_task'] == "Spatial Denoising":
            params['denoise_method'] = st.sidebar.selectbox("Method", ["Median", "Gaussian", "Average"])
            params['denoise_kernel'] = st.sidebar.slider("Kernel Size", 3, 15, 5, 2)
            if params['denoise_method'] == "Gaussian":
                params['denoise_sigma'] = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
        elif params['restoration_task'] == "Periodic Noise Reduction":
            params['notch_k'] = st.sidebar.slider("Top-K peaks", 2, 100, 10, 2)
            params['notch_radius'] = st.sidebar.slider("Notch radius", 1, 15, 3, 1)
        elif params['restoration_task'] == "Linear Degradation (simulate)":
            params['deg_type'] = st.sidebar.selectbox("Type", ["Gaussian Blur", "Motion Blur"])
            if params['deg_type'] == "Gaussian Blur":
                params['deg_sigma'] = st.sidebar.slider("Sigma", 0.5, 10.0, 2.0, 0.5)
            else:
                params['deg_length'] = st.sidebar.slider("Length", 3, 51, 15, 2)
                params['deg_angle'] = st.sidebar.slider("Angle (deg)", 0, 180, 0, 5)
        else:  # Inverse Filtering
            params['inv_type'] = st.sidebar.selectbox("PSF Type", ["Gaussian", "Motion"])
            if params['inv_type'] == "Gaussian":
                params['inv_sigma'] = st.sidebar.slider("Sigma", 0.5, 10.0, 2.0, 0.5)
            else:
                params['inv_length'] = st.sidebar.slider("Length", 3, 51, 15, 2)
                params['inv_angle'] = st.sidebar.slider("Angle (deg)", 0, 180, 0, 5)
            params['inv_epsilon'] = st.sidebar.slider("Epsilon", 1e-5, 1e-1, 1e-3, format="%e")
    
    return params
