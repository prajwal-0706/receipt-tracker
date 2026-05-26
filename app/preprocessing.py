from __future__ import annotations
import io
import cv2
import numpy as np
from PIL import Image

MAX_LONG_EDGE = 1280  # tuned for A10G/V100; drop to 896 on T4-class GPUs


def preprocess_for_vlm(image_bytes: bytes) -> Image.Image:
    # Load via PIL (handles webp/gif reliably; some OpenCV builds don't ship libwebp),
    # then convert to numpy for the OpenCV cleanup, then back to PIL.
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil = _resize_pil(pil, MAX_LONG_EDGE)
    try:
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        bgr = _deskew(bgr)
        bgr = _boost_contrast(bgr)
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    except Exception:
        return pil


def _resize_pil(img: Image.Image, max_edge: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_edge:
        return img
    scale = max_edge / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _deskew(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=80, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return bgr
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < angle < 45:
            angles.append(angle)
    if not angles:
        return bgr
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return bgr
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _boost_contrast(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
