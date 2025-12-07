import cv2
import numpy as np
from typing import Tuple
from src.metrics.ita import compute_ita_from_patch


def read_image_bgr(path: str) -> np.ndarray:
    """
    Read an image from disk in BGR format (as used by OpenCV).
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return img


def bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR uint8 image (0-255) to CIELAB (float32).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab


def center_face_patch(img_bgr: np.ndarray, patch_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    TEMPORARY: take a center patch as a proxy for cheek/skin region.
    We'll later swap this out for a MediaPipe-based cheek detector.
    """
    h, w, _ = img_bgr.shape
    ph, pw = patch_size

    cy, cx = h // 2, w // 2
    y1 = max(cy - ph // 2, 0)
    y2 = min(cy + ph // 2, h)
    x1 = max(cx - pw // 2, 0)
    x2 = min(cx + pw // 2, w)

    patch = img_bgr[y1:y2, x1:x2, :]
    return patch


def compute_skin_ita_from_image(path: str) -> float:
    """
    High-level helper:
    1. Read image from disk
    2. Extract a center patch
    3. Convert to LAB
    4. Compute ITA from that patch
    """
    img_bgr = read_image_bgr(path)
    patch_bgr = center_face_patch(img_bgr)
    patch_lab = bgr_to_lab(patch_bgr)
    ita = compute_ita_from_patch(patch_lab)
    return ita
