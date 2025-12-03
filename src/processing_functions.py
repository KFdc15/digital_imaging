import numpy as np
import cv2


# RESOLUTION
def apply_resolution(image, processor, scale):
    return processor.resize_image(image, scale)


# QUANTIZATION
def apply_quantization(image, levels):
    processed = (image // (256 // levels)) * (256 // levels)
    return np.clip(processed, 0, 255).astype(np.uint8)


# RGB CHANNELS
def apply_rgb_channels(image, show_red, show_green, show_blue):
    processed = image.copy()
    if len(processed.shape) == 3:
        if not show_red:
            processed[:, :, 0] = 0
        if not show_green:
            processed[:, :, 1] = 0
        if not show_blue:
            processed[:, :, 2] = 0
    return processed


# NEGATIVE IMAGES
def apply_negative(image):
    return 255 - image


# THRESHOLDING
def apply_thresholding(image, processor, threshold_value):
    gray = processor.convert_to_grayscale(image)
    return np.where(gray > threshold_value, 255, 0).astype(np.uint8)


# LOGARITHMIC TRANSFORMATIONS
def apply_logarithmic(image, processor, c_log):
    gray = processor.convert_to_grayscale(image).astype(np.float64)
    processed = c_log * np.log(1 + gray)
    return np.clip(processed, 0, 255).astype(np.uint8)


# POWER-LAW (GAMMA)
def apply_gamma(image, gamma):
    normalized = image.astype(np.float64) / 255.0
    processed = np.power(normalized, gamma) * 255.0
    return np.clip(processed, 0, 255).astype(np.uint8)


# CONTRAST STRETCHING
def apply_contrast_stretching(image, method, percentile_low=2, percentile_high=98):
    if method == "Min-Max":
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            return image.copy()
    else:
        low_val = np.percentile(image, percentile_low)
        high_val = np.percentile(image, percentile_high)
        if high_val > low_val:
            return np.clip((image - low_val) / (high_val - low_val) * 255, 0, 255).astype(np.uint8)
        else:
            return image.copy()


# PIECEWISE LINEAR
def apply_piecewise_linear(image, processor, r1, s1, r2, s2):
    gray = processor.convert_to_grayscale(image).astype(np.float64)
    processed = np.zeros_like(gray)
    
    mask1 = gray <= r1
    processed[mask1] = (s1 / r1) * gray[mask1] if r1 > 0 else 0
    
    mask2 = (gray > r1) & (gray <= r2)
    if r2 > r1:
        processed[mask2] = ((s2 - s1) / (r2 - r1)) * (gray[mask2] - r1) + s1
    else:
        processed[mask2] = s1
    
    mask3 = gray > r2
    if 255 > r2:
        processed[mask3] = ((255 - s2) / (255 - r2)) * (gray[mask3] - r2) + s2
    else:
        processed[mask3] = s2
    
    return np.clip(processed, 0, 255).astype(np.uint8)


# GRAY-LEVEL SLICING
def apply_gray_level_slicing(image, processor, slice_min, slice_max):
    gray = processor.convert_to_grayscale(image)
    return np.where((gray >= slice_min) & (gray <= slice_max), 255, gray).astype(np.uint8)


# BIT-PLANE SLICING
def apply_bit_plane_slicing(image, processor, bit_plane, reconstruct_planes):
    gray = processor.convert_to_grayscale(image)
    if len(reconstruct_planes) > 0 and reconstruct_planes != [7, 6, 5, 4, 3, 2, 1, 0]:
        processed = np.zeros_like(gray)
        for plane in reconstruct_planes:
            bit_image = (gray >> plane) & 1
            processed = processed | (bit_image << plane)
        return processed
    else:
        bit_image = (gray >> bit_plane) & 1
        return bit_image * 255
