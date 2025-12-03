import cv2
import numpy as np


def to_gray(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def get_structuring_element(shape: str, ksize: int):
    k = int(ksize) | 1
    if shape == "Ellipse":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    elif shape == "Cross":
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
    else:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))


# BASIC OPERATIONS
def morph_erosion(image, shape="Rect", ksize=3, iterations=1):
    kernel = get_structuring_element(shape, ksize)
    return cv2.erode(image, kernel, iterations=int(iterations))


def morph_dilation(image, shape="Rect", ksize=3, iterations=1):
    kernel = get_structuring_element(shape, ksize)
    return cv2.dilate(image, kernel, iterations=int(iterations))


# COMPOUND OPERATIONS
def morph_open(image, shape="Rect", ksize=3, iterations=1):
    kernel = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=int(iterations))


def morph_close(image, shape="Rect", ksize=3, iterations=1):
    kernel = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=int(iterations))


def morph_gradient(image, shape="Rect", ksize=3):
    kernel = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def morph_tophat(image, shape="Rect", ksize=3):
    kernel = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def morph_blackhat(image, shape="Rect", ksize=3):
    kernel = get_structuring_element(shape, ksize)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
