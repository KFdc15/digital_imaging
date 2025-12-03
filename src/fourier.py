import numpy as np
import cv2
import matplotlib.pyplot as plt


# FOURIER 1-D
def fourier_1d(image, axis="row", index=0):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    h, w = gray.shape
    index = max(0, min(index, h-1 if axis == "row" else w-1))
    signal = gray[index, :] if axis == "row" else gray[:, index]
    signal = signal.astype(np.float32)
    fft_vals = np.fft.fft(signal)
    fft_shift = np.fft.fftshift(fft_vals)
    magnitude = 20 * np.log1p(np.abs(fft_shift))
    return signal, magnitude


# FOURIER 2-D
def fourier_2d(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log1p(np.abs(fshift))
    # Normalize to 0-255 for display
    mag_norm = (magnitude_spectrum - magnitude_spectrum.min())
    mag_norm = (mag_norm / (mag_norm.max() + 1e-8) * 255).astype(np.uint8)
    return mag_norm
