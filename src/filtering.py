import numpy as np
import cv2


# CONVOLUTION
def apply_convolution(original_image, custom_kernel):
    kernel_flipped = np.flip(np.flip(custom_kernel, 0), 1)
    if len(original_image.shape) == 3:
        processed = np.zeros_like(original_image)
        for i in range(3):
            processed[:,:,i] = cv2.filter2D(original_image[:,:,i], -1, kernel_flipped, borderType=cv2.BORDER_REPLICATE)
    else:
        processed = cv2.filter2D(original_image, -1, kernel_flipped, borderType=cv2.BORDER_REPLICATE)
    return processed


# SMOOTHING LINEAR FILTER
def apply_smoothing_linear_filter(original_image, filter_type, kernel_size, sigma=1.0):
    if filter_type == "Average":
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        return cv2.filter2D(original_image, -1, kernel)
    elif filter_type == "Gaussian":
        return cv2.GaussianBlur(original_image, (kernel_size, kernel_size), sigma)
    elif filter_type == "Box":
        return cv2.boxFilter(original_image, -1, (kernel_size, kernel_size), normalize=True)


# MEDIAN FILTER
def apply_median_filter(original_image, kernel_size):
    return cv2.medianBlur(original_image, kernel_size)


# SHARPENING
def apply_sharpening(original_image, method, strength):
    if method == "Laplacian":
        laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        if len(original_image.shape) == 3:
            laplacian = np.zeros_like(original_image, dtype=np.float32)
            for i in range(3):
                laplacian[:,:,i] = cv2.filter2D(original_image[:,:,i].astype(np.float32), -1, laplacian_kernel)
        else:
            laplacian = cv2.filter2D(original_image.astype(np.float32), -1, laplacian_kernel)
        return np.clip(original_image.astype(np.float32) + strength * laplacian, 0, 255).astype(np.uint8)
    
    elif method == "Unsharp Masking":
        blurred = cv2.GaussianBlur(original_image, (5, 5), 1.0)
        return cv2.addWeighted(original_image, 1.0 + strength, blurred, -strength, 0)
    
    elif method == "High-boost":
        blurred = cv2.GaussianBlur(original_image, (5, 5), 1.0)
        mask = original_image.astype(np.float32) - blurred.astype(np.float32)
        return np.clip(original_image.astype(np.float32) + strength * mask, 0, 255).astype(np.uint8)


# SPATIAL FILTER
def apply_spatial_filter(original_image, filter_type, kernel_size, sigma=1.0):
    if filter_type == "Smoothing (Average)":
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        return cv2.filter2D(original_image, -1, kernel)
    
    elif filter_type == "Smoothing (Weighted)":
        if kernel_size == 3:
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16
        else:
            center = kernel_size // 2
            kernel = np.zeros((kernel_size, kernel_size), np.float32)
            for i in range(kernel_size):
                for j in range(kernel_size):
                    dist = abs(i - center) + abs(j - center)
                    kernel[i, j] = kernel_size - dist
            kernel = kernel / kernel.sum()
        return cv2.filter2D(original_image, -1, kernel)
    
    elif filter_type == "Smoothing (Gaussian)":
        return cv2.GaussianBlur(original_image, (kernel_size, kernel_size), sigma)
    
    elif filter_type == "Order-Statistic (Median)":
        return cv2.medianBlur(original_image, kernel_size)
    
    elif filter_type == "Order-Statistic (Max)":
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(original_image, kernel)
    
    elif filter_type == "Order-Statistic (Min)":
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(original_image, kernel)
    
    elif filter_type == "Sharpening (Laplacian)":
        laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        if len(original_image.shape) == 3:
            laplacian = np.zeros_like(original_image, dtype=np.float32)
            for i in range(3):
                laplacian[:,:,i] = cv2.filter2D(original_image[:,:,i].astype(np.float32), -1, laplacian_kernel)
        else:
            laplacian = cv2.filter2D(original_image.astype(np.float32), -1, laplacian_kernel)
        return np.clip(original_image.astype(np.float32) + laplacian, 0, 255).astype(np.uint8)
    
    elif filter_type == "Sharpening (Gradient)":
        if len(original_image.shape) == 3:
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = original_image
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        return np.clip(gradient, 0, 255).astype(np.uint8)
    
    elif filter_type == "High-Pass":
        blurred = cv2.GaussianBlur(original_image, (kernel_size, kernel_size), 0)
        high_pass = original_image.astype(np.float32) - blurred.astype(np.float32)
        return np.clip(high_pass + 128, 0, 255).astype(np.uint8)
