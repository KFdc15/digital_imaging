"""
Digital Imaging Fundamentals - Image Processing Module
Các chức năng xử lý ảnh cơ bản cho ứng dụng
"""

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import filters, exposure, transform, util


class ImageProcessor:
    """Lớp xử lý các thao tác xử lý ảnh"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def convert_to_grayscale(image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    @staticmethod
    def apply_gaussian_blur(image, kernel_size=5):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def apply_median_blur(image, kernel_size=5):
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def sharpen_image(image):
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def detect_edges_canny(image, threshold1=100, threshold2=200):
        gray = ImageProcessor.convert_to_grayscale(image)
        edges = cv2.Canny(gray, threshold1, threshold2)
        return edges
    
    @staticmethod
    def detect_edges_sobel(image):
        gray = ImageProcessor.convert_to_grayscale(image)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel / sobel.max() * 255)
        return sobel
    
    @staticmethod
    def detect_edges_laplacian(image):
        gray = ImageProcessor.convert_to_grayscale(image)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        return laplacian
    
    @staticmethod
    def adjust_brightness(image, beta=50):
        return cv2.convertScaleAbs(image, alpha=1, beta=beta)
    
    @staticmethod
    def adjust_contrast(image, alpha=1.5):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    
    @staticmethod
    def histogram_equalization(image):
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            return cv2.equalizeHist(image)
    
    @staticmethod
    def adaptive_histogram_equalization(image, clip_limit=2.0, tile_size=8):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            return clahe.apply(image)
    
    @staticmethod
    def threshold_binary(image, threshold_value=127):
        gray = ImageProcessor.convert_to_grayscale(image)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        return binary
    
    @staticmethod
    def threshold_otsu(image):
        gray = ImageProcessor.convert_to_grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    @staticmethod
    def threshold_adaptive(image, block_size=11, c=2):
        gray = ImageProcessor.convert_to_grayscale(image)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, block_size, c)
        return adaptive
    
    @staticmethod
    def rotate_image(image, angle):
        if len(image.shape) == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        return rotated
    
    @staticmethod
    def flip_image(image, flip_code):
        return cv2.flip(image, flip_code)
    
    @staticmethod
    def resize_image(image, scale_percent):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def add_noise_gaussian(image, mean=0, sigma=25):
        noise = np.random.normal(mean, sigma, image.shape)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def add_noise_salt_pepper(image, amount=0.05):
        noisy = util.random_noise(image, mode='s&p', amount=amount)
        noisy = (noisy * 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def morphology_erode(image, kernel_size=5, iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(image, kernel, iterations=iterations)
    
    @staticmethod
    def morphology_dilate(image, kernel_size=5, iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(image, kernel, iterations=iterations)
    
    @staticmethod
    def morphology_opening(image, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    @staticmethod
    def morphology_closing(image, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    @staticmethod
    def get_histogram(image):
        if len(image.shape) == 3:
            histograms = []
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                histograms.append((hist, color))
            return histograms
        else:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            return [(hist, 'gray')]
    
    @staticmethod
    def apply_custom_filter(image, kernel):
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def emboss_filter(image):
        kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])
        gray = ImageProcessor.convert_to_grayscale(image)
        embossed = cv2.filter2D(gray, -1, kernel)
        return embossed
    
    @staticmethod
    def invert_colors(image):
        return cv2.bitwise_not(image)
    
    @staticmethod
    def sepia_tone(image):
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        sepia = cv2.transform(image, kernel)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        return sepia
    
    @staticmethod
    def cartoon_effect(image, d=9, sigma_color=75, sigma_space=75):
        smooth = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        cartoon = cv2.bitwise_and(smooth, smooth, mask=edges)
        return cartoon
