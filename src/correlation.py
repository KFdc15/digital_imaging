import numpy as np
import cv2
import streamlit as st


# NORMALIZED CORRELATION (MANUAL IMPLEMENTATION)
def normalized_correlation_manual(image, template):
    H, W = image.shape
    h, w = template.shape
    result = np.zeros((H - h + 1, W - w + 1), dtype=np.float32)
    
    template_mean = np.mean(template)
    template_std = np.std(template)
    
    for y in range(H - h + 1):
        for x in range(W - w + 1):
            region = image[y:y+h, x:x+w]
            region_mean = np.mean(region)
            region_std = np.std(region)
            
            if region_std == 0 or template_std == 0:
                result[y, x] = 0
            else:
                numerator = np.sum((region - region_mean) * (template - template_mean))
                denominator = h * w * region_std * template_std
                result[y, x] = numerator / denominator
    
    return result


# CORRELATION AUTO DETECT MASK
def apply_correlation_auto_detect(original_image, processor, mask_size, mask_x, mask_y):
    gray_input = processor.convert_to_grayscale(original_image).astype(np.float32)
    h, w = gray_input.shape
    
    center_x = int(w * mask_x / 100)
    center_y = int(h * mask_y / 100)
    half_size = mask_size // 2
    
    y1 = max(0, center_y - half_size)
    y2 = min(h, center_y + half_size)
    x1 = max(0, center_x - half_size)
    x2 = min(w, center_x + half_size)
    
    if len(original_image.shape) == 3:
        mask_template = original_image[y1:y2, x1:x2].copy()
    else:
        mask_template = original_image[y1:y2, x1:x2].copy()
    st.session_state['extracted_mask'] = mask_template
    
    gray_template = gray_input[y1:y2, x1:x2].copy()
    
    ncc_map = normalized_correlation_manual(gray_input, gray_template)
    
    result_normalized = ((ncc_map - ncc_map.min()) / (ncc_map.max() - ncc_map.min() + 1e-5) * 255).astype(np.uint8)
    return result_normalized


# CORRELATION UPLOAD TEMPLATE
def apply_correlation_template(original_image, template_image, processor):
    if template_image is None:
        return None
    
    gray_input = processor.convert_to_grayscale(original_image).astype(np.float32)
    gray_template = processor.convert_to_grayscale(template_image).astype(np.float32)
    
    ncc_map = normalized_correlation_manual(gray_input, gray_template)
    
    result_normalized = ((ncc_map - ncc_map.min()) / (ncc_map.max() - ncc_map.min() + 1e-5) * 255).astype(np.uint8)
    return result_normalized


# CORRELATION CUSTOM KERNEL
def apply_correlation_custom_kernel(original_image, processor, custom_kernel):
    mean = np.mean(custom_kernel)
    std = np.std(custom_kernel)
    if std > 0:
        kernel_normalized = (custom_kernel - mean) / std
    else:
        kernel_normalized = custom_kernel
    
    if len(original_image.shape) == 3:
        gray = processor.convert_to_grayscale(original_image)
    else:
        gray = original_image
    
    img_mean = cv2.boxFilter(gray.astype(np.float32), -1, custom_kernel.shape, normalize=True)
    img_sqr_mean = cv2.boxFilter((gray.astype(np.float32)**2), -1, custom_kernel.shape, normalize=True)
    img_std = np.sqrt(np.maximum(img_sqr_mean - img_mean**2, 0))
    
    result = cv2.filter2D(gray.astype(np.float32), -1, kernel_normalized, borderType=cv2.BORDER_REPLICATE)
    
    normalized_corr = result / (img_std + 1e-5)
    return np.clip((normalized_corr - normalized_corr.min()) / (normalized_corr.max() - normalized_corr.min() + 1e-5) * 255, 0, 255).astype(np.uint8)
