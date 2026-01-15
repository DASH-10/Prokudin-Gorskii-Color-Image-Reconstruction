from typing import Tuple
import numpy as np
import cv2

def load_image(path: str) -> np.ndarray:
    """Load stacked B/G/R plate as grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def split_image(stacked_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split stacked grayscale image into B, G, R (top->bottom)."""
    assert stacked_img.ndim == 2, "Expected a single-channel image."
    h, w = stacked_img.shape
    h3 = h // 3
    b = stacked_img[0:h3, :]
    g = stacked_img[h3:2 * h3, :]
    r = stacked_img[2 * h3:3 * h3, :]
    return b, g, r

def apply_alignment(channel: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift channel by integer (dx, dy). +dx→right, +dy→down."""
    return np.roll(channel, shift=(dy, dx), axis=(0, 1))

def create_color_image(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Normalize each channel to [0..255] and stack RGB."""
    def norm(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return (x * 255.0).astype(np.uint8)
    B = norm(b); G = norm(g); R = norm(r)
    return np.dstack([R, G, B])  # we work in RGB (cv2.imwrite needs BGR)

def auto_crop(img_rgb: np.ndarray, thresh: float = 12, max_crop: int = 120):
    """
    Scan inward from each edge until mean intensity passes 'thresh'.
    Returns: (cropped_img, (top, bottom, left, right))
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    top = 0
    for i in range(min(max_crop, h // 3)):
        if gray[i, :].mean() > thresh:
            top = i; break

    bottom = h - 1
    for i in range(h - 1, max(h - 1 - max_crop, h * 2 // 3), -1):
        if gray[i, :].mean() > thresh:
            bottom = i; break

    left = 0
    for j in range(min(max_crop, w // 3)):
        if gray[:, j].mean() > thresh:
            left = j; break

    right = w - 1
    for j in range(w - 1, max(w - 1 - max_crop, w * 2 // 3), -1):
        if gray[:, j].mean() > thresh:
            right = j; break

    pad = 2
    top = max(0, top + pad)
    left = max(0, left + pad)
    bottom = min(h - 1, bottom - pad)
    right = min(w - 1, right - pad)

    if bottom <= top or right <= left:
        return img_rgb, (0, h - 1, 0, w - 1)

    cropped = img_rgb[top:bottom + 1, left:right + 1, :]
    return cropped, (top, bottom, left, right)
