import numpy as np
import cv2
from sklearn.decomposition import PCA
import os


# PCA MODEL
def build_pca_model(face_images, window_size=(64, 64), n_components=20):
    X = []
    for img in face_images:
        if img is None:
            continue
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        resized = cv2.resize(gray, window_size, interpolation=cv2.INTER_AREA)
        X.append(resized.flatten().astype(np.float32))
    if len(X) == 0:
        return None
    X = np.array(X)
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec
    pca = PCA(n_components=min(n_components, X_centered.shape[1], X_centered.shape[0]))
    pca.fit(X_centered)
    model = {"pca": pca, "mean": mean_vec, "window_size": window_size}
    return model


def reconstruction_error(model, patch_vec):
    mean_vec = model["mean"]
    centered = patch_vec - mean_vec
    if "pca" in model and model["pca"] is not None:
        pca = model["pca"]
        coeffs = pca.transform([centered])[0]
        recon = pca.inverse_transform([coeffs])[0] + mean_vec
    else:
        comps = model["components"]  # shape: (k, D)
        coeffs = centered @ comps.T
        recon = coeffs @ comps + mean_vec
    err = np.mean((patch_vec - recon) ** 2)
    return err


# DETECTION
def detect_faces_pca(original_image, model, stride=16, threshold=1500.0):
    if model is None:
        return original_image, []
    ws = model["window_size"]
    if len(original_image.shape) == 3:
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = original_image
    H, W = gray.shape
    h, w = ws
    boxes = []
    vis = original_image.copy()
    # Optional: resize large images for performance
    # Sliding window
    for y in range(0, max(1, H - h + 1), stride):
        for x in range(0, max(1, W - w + 1), stride):
            patch = gray[y:y+h, x:x+w]
            if patch.shape[0] != h or patch.shape[1] != w:
                continue
            vec = patch.flatten().astype(np.float32)
            err = reconstruction_error(model, vec)
            if err < threshold:
                boxes.append((x, y, w, h, err))
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return vis, boxes


# SAVE/LOAD MODEL
def save_pca_model(model, path):
    comps = model.get("components")
    if comps is None and "pca" in model and model["pca"] is not None:
        comps = model["pca"].components_
    mean = model["mean"]
    h, w = model["window_size"]
    np.savez_compressed(path, components=comps, mean=mean, window_h=h, window_w=w)


def load_pca_model(path):
    if not os.path.exists(path):
        return None
    data = np.load(path)
    comps = data["components"]
    mean = data["mean"]
    window_size = (int(data["window_h"]), int(data["window_w"]))
    model = {"components": comps, "mean": mean, "window_size": window_size, "pca": None}
    return model


# HAAR FALLBACK DETECTOR
def haar_detect_faces(original_image):
    if len(original_image.shape) == 3:
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        vis = original_image.copy()
    else:
        gray = original_image
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    boxes = []
    for (x, y, w, h) in faces:
        boxes.append((x, y, w, h, 0.0))
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return vis, boxes
