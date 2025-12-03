import numpy as np
import cv2


def to_gray(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


# NOISE MODELS
def add_gaussian_noise(image, mean=0.0, var=0.01):
    img = image.astype(np.float32)
    if img.ndim == 2:
        noise = np.random.normal(mean, np.sqrt(var) * 255.0, img.shape).astype(np.float32)
        out = img + noise
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        noise = np.random.normal(mean, np.sqrt(var) * 255.0, img.shape).astype(np.float32)
        out = img + noise
        return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.02):
    out = image.copy()
    num_salt = int(np.ceil(amount * out.size * 0.5))
    num_pepper = int(np.ceil(amount * out.size * 0.5))
    # Salt
    coords = tuple([np.random.randint(0, i - 1, num_salt) for i in out.shape[:2]])
    if out.ndim == 2:
        out[coords] = 255
    else:
        out[coords[0], coords[1], :] = 255
    # Pepper
    coords = tuple([np.random.randint(0, i - 1, num_pepper) for i in out.shape[:2]])
    if out.ndim == 2:
        out[coords] = 0
    else:
        out[coords[0], coords[1], :] = 0
    return out


def add_periodic_noise(image, amplitude=30.0, freq_u=5, freq_v=5):
    gray = to_gray(image).astype(np.float32)
    H, W = gray.shape
    y = np.arange(H).reshape(-1, 1)
    x = np.arange(W).reshape(1, -1)
    pattern = amplitude * (np.sin(2 * np.pi * freq_u * y / H) + np.sin(2 * np.pi * freq_v * x / W))
    noisy = gray + pattern
    return np.clip(noisy, 0, 255).astype(np.uint8)


# SPATIAL DENOISING
def spatial_denoise(image, method="Median", kernel_size=5, sigma=1.0):
    if method == "Median":
        return cv2.medianBlur(image, int(kernel_size) | 1)
    elif method == "Gaussian":
        k = int(kernel_size) | 1
        return cv2.GaussianBlur(image, (k, k), sigma)
    else:
        # Average
        k = int(kernel_size) | 1
        kernel = np.ones((k, k), np.float32) / (k * k)
        return cv2.filter2D(image, -1, kernel)


# PERIODIC NOISE REDUCTION (AUTO NOTCH)
def periodic_noise_reduction(image, k_peaks=10, notch_radius=3):
    gray = to_gray(image).astype(np.float32)
    H, W = gray.shape
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    mag = np.log1p(np.abs(Fshift))
    # mask out low frequencies around center
    cy, cx = H // 2, W // 2
    rr = notch_radius * 2 + 6
    yy, xx = np.ogrid[:H, :W]
    center_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= rr ** 2
    mag_centered = mag.copy()
    mag_centered[center_mask] = 0

    flat_idx = np.argpartition(mag_centered.ravel(), -k_peaks)[-k_peaks:]
    coords = np.column_stack(np.unravel_index(flat_idx, (H, W)))

    notch = np.ones((H, W), np.float32)
    for (y, x) in coords:
        # skip near center
        if (y - cy) ** 2 + (x - cx) ** 2 < rr ** 2:
            continue
        # symmetric coordinate
        ys = (2 * cy - y) % H
        xs = (2 * cx - x) % W
        # zero out small disks around peaks
        mask1 = (yy - y) ** 2 + (xx - x) ** 2 <= notch_radius ** 2
        mask2 = (yy - ys) ** 2 + (xx - xs) ** 2 <= notch_radius ** 2
        notch[mask1] = 0.0
        notch[mask2] = 0.0

    Fnotch = Fshift * notch
    Finv = np.fft.ifft2(np.fft.ifftshift(Fnotch))
    rec = np.real(Finv)
    return np.clip(rec, 0, 255).astype(np.uint8)


# LINEAR, POSITION-INVARIANT DEGRADATIONS (SIMULATE)
def gaussian_psf(size, sigma):
    h, w = size
    y = np.arange(h) - h // 2
    x = np.arange(w) - w // 2
    X, Y = np.meshgrid(x, y)
    g = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    g /= g.sum() + 1e-8
    return g.astype(np.float32)


def motion_psf(size, length=9, angle_deg=0):
    h, w = size
    psf = np.zeros((h, w), np.float32)
    center = (w // 2, h // 2)
    angle = np.deg2rad(angle_deg)
    dx = np.cos(angle)
    dy = np.sin(angle)
    for i in range(-length // 2, length // 2 + 1):
        x = int(center[0] + i * dx)
        y = int(center[1] + i * dy)
        if 0 <= x < w and 0 <= y < h:
            psf[y, x] = 1.0
    s = psf.sum()
    if s > 0:
        psf /= s
    return psf


def pad_and_shift_psf(psf, img_shape):
    H, W = img_shape
    ph, pw = psf.shape
    pad = np.zeros((H, W), np.float32)
    pad[:ph, :pw] = psf
    cy, cx = ph // 2, pw // 2
    pad = np.roll(pad, -cy, axis=0)
    pad = np.roll(pad, -cx, axis=1)
    return pad


def apply_linear_degradation(image, method="Gaussian", sigma=2.0, length=9, angle=0):
    if method == "Gaussian":
        ksize = int(6 * sigma + 1) | 1
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)
    else:
        # Motion blur via PSF convolution
        gray = to_gray(image)
        psf = motion_psf((21, 21), length=length, angle_deg=angle)
        psf = psf / (psf.sum() + 1e-8)
        degraded = cv2.filter2D(gray.astype(np.float32), -1, psf)
        return np.clip(degraded, 0, 255).astype(np.uint8)


# INVERSE FILTERING
def inverse_filtering(observed_image, method="Gaussian", sigma=2.0, length=9, angle=0, epsilon=1e-3):
    gray = to_gray(observed_image).astype(np.float32)
    Hh, Hw = gray.shape
    if method == "Gaussian":
        psf_small = gaussian_psf((21, 21), sigma)
    else:
        psf_small = motion_psf((21, 21), length=length, angle_deg=angle)
    psf = pad_and_shift_psf(psf_small, (Hh, Hw))
    Hf = np.fft.fft2(psf)
    G = np.fft.fft2(gray)
    denom = Hf.copy()
    # regularize to avoid division by zero
    denom[np.abs(denom) < epsilon] = epsilon
    F_hat = G / denom
    f_rec = np.real(np.fft.ifft2(F_hat))
    return np.clip(f_rec, 0, 255).astype(np.uint8)
