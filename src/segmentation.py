import numpy as np
import cv2


def to_gray(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def global_threshold_mean(image):
    gray = to_gray(image)
    t = float(np.mean(gray))
    _, mask = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return mask


def otsu_threshold(image):
    gray = to_gray(image)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def kmeans_segmentation(image, k=2, attempts=10, seed=0, output="binary-max"):
    gray = to_gray(image)
    Z = gray.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    if seed is not None:
        np.random.seed(int(seed))
    _, labels, centers = cv2.kmeans(
        Z, int(k), None, criteria, int(attempts), cv2.KMEANS_RANDOM_CENTERS
    )
    centers = centers.flatten()
    seg = centers[labels.flatten()].reshape(gray.shape).astype(np.uint8)
    if output == "labels":
        # Return label map normalized to 0..k-1 (uint8)
        return labels.reshape(gray.shape).astype(np.uint8)
    else:
        # Binary using brightest cluster as foreground
        max_center = np.argmax(centers)
        mask = (labels.reshape(gray.shape) == max_center).astype(np.uint8) * 255
        return mask


def colorize_labels(label_map, k=None):
    lbl = label_map.astype(np.int32)
    if k is None:
        k = int(lbl.max()) + 1
    # Map each label to a color via COLORMAP
    norm = (lbl.astype(np.float32) / max(1, k - 1) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
