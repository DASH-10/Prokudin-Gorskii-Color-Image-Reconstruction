import numpy as np
import cv2

def _hist_eq_color(img_rgb: np.ndarray) -> np.ndarray:
    """Equalize luminance (Y in YCrCb) to boost contrast while keeping colors natural."""
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)
    y = cv2.equalizeHist(y)
    img_eq = cv2.merge([y, cr, cb])
    return cv2.cvtColor(img_eq, cv2.COLOR_YCrCb2RGB)

def _gamma(img_rgb: np.ndarray, gamma: float = 1.1) -> np.ndarray:
    """Gamma correction with LUT. gamma>1 brightens a bit."""
    inv = 1.0 / max(gamma, 1e-6)
    lut = (np.arange(256) / 255.0) ** inv
    lut = np.clip(lut * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img_rgb, lut)

def _unsharp(img_rgb: np.ndarray, amount: float = 0.6, radius: float = 1.2) -> np.ndarray:
    """Unsharp masking for crispness. Raise amount to sharpen more."""
    blur = cv2.GaussianBlur(img_rgb, (0, 0), radius)
    sharp = cv2.addWeighted(img_rgb, 1 + amount, blur, -amount, 0)
    return sharp

def enhance_image(img_rgb: np.ndarray, gamma_value: float = 1.1, do_unsharp: bool = True) -> np.ndarray:
    """Hist-eq (luma) -> gamma -> optional unsharp."""
    out = _hist_eq_color(img_rgb)
    out = _gamma(out, gamma=gamma_value)
    if do_unsharp:
        out = _unsharp(out, amount=0.6, radius=1.2)
    return out
