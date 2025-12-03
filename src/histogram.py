import numpy as np
import cv2
from scipy import stats


# HISTOGRAM EQUALIZATION
def apply_histogram_equalization(image, processor):
    return processor.histogram_equalization(image)


# HISTOGRAM MATCHING
def apply_histogram_matching(image, processor, method, reference_image=None, gaussian_mean=128, gaussian_std=30):
    gray_input = processor.convert_to_grayscale(image)
    hist_input, _ = np.histogram(gray_input.flatten(), 256, [0, 256])
    cdf_input = hist_input.cumsum()
    cdf_input = cdf_input / cdf_input[-1]
    
    if method == "Uniform":
        cdf_reference = np.linspace(0, 1, 256)
    
    elif method == "Gaussian":
        x = np.arange(256)
        hist_reference = stats.norm.pdf(x, gaussian_mean, gaussian_std)
        hist_reference = hist_reference / hist_reference.sum()
        cdf_reference = np.cumsum(hist_reference)
    
    elif method == "Custom Image":
        if reference_image is None:
            return None
        gray_reference = processor.convert_to_grayscale(reference_image)
        hist_reference, _ = np.histogram(gray_reference.flatten(), 256, [0, 256])
        cdf_reference = hist_reference.cumsum()
        cdf_reference = cdf_reference / cdf_reference[-1]
    else:
        return None
    
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = np.argmin(np.abs(cdf_reference - cdf_input[i]))
        lookup_table[i] = j
    
    return lookup_table[gray_input]
