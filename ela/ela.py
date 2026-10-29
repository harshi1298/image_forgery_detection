"""
Error Level Analysis (ELA) implementation for detecting compression inconsistencies.
"""

from typing import Tuple, Dict, Any, Optional
import io
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def compute_ela(
    image_rgb: np.ndarray,
    quality: int = 90,
    scale: Optional[float] = 15.0,
    auto_scale: bool = False,
) -> Dict[str, Any]:
    """
    Perform Error Level Analysis (ELA) on an RGB image.

    Steps:
    1. Recompress the image to JPEG at the specified quality factor in memory.
    2. Compute pixel-wise absolute difference between original and recompressed image.
    3. Amplify the difference for human visual inspection.
    4. Compute error distribution metrics.

    Args:
        image_rgb: RGB uint8 numpy array
        quality: JPEG compression quality (1-100, typically 90 or 95)
        scale: Multiplier for difference amplification (typically 10-25)
        auto_scale: If True, scale dynamically so the maximum difference maps to 255

    Returns:
        Dict containing:
            ela_image: amplified difference as RGB uint8 array
            raw_diff: unscaled floating difference array
            recompressed_rgb: the recompressed JPEG image
            mean_error: mean absolute error across all pixels
            max_error: maximum pixel difference
            std_error: standard deviation of error
            suspicious_score: normalized anomaly score in [0.0, 1.0]
    """
    orig_pil = Image.fromarray(image_rgb)

    # Recompress to in-memory JPEG buffer
    buffer = io.BytesIO()
    orig_pil.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    recompressed_pil = Image.open(buffer).convert("RGB")
    recompressed_rgb = np.array(recompressed_pil)

    # Compute difference
    orig_f = image_rgb.astype(np.float32)
    recomp_f = recompressed_rgb.astype(np.float32)
    diff = np.abs(orig_f - recomp_f)

    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff))

    if auto_scale:
        multiplier = 255.0 / max(max_diff, 1.0)
    else:
        multiplier = float(scale) if scale is not None else 15.0

    ela_amplified = np.clip(diff * multiplier, 0, 255).astype(np.uint8)

    # Anomaly score based on standard deviation and high-error percentile
    p95 = float(np.percentile(diff, 95))
    # Normal JPEG error has narrow standard deviation; spliced uncompressed or different-Q regions have elevated tails
    anomaly_score = float(np.clip((std_diff * 0.4 + p95 * 0.6) / 20.0, 0.0, 1.0))

    return {
        "ela_image": ela_amplified,
        "raw_diff": diff,
        "recompressed_rgb": recompressed_rgb,
        "mean_error": mean_diff,
        "max_error": max_diff,
        "std_error": std_diff,
        "p95_error": p95,
        "scale_used": multiplier,
        "suspicious_score": anomaly_score,
    }


def generate_ela_mask(
    ela_dict: Dict[str, Any],
    percentile_threshold: float = 95.0,
    min_component_area: int = 150,
) -> np.ndarray:
    """
    Generate a binary mask of localized high-error ELA regions.
    """
    raw_diff = ela_dict["raw_diff"]
    # Gray difference
    gray_diff = np.mean(raw_diff, axis=2).astype(np.float32)

    # Threshold based on high percentile
    thresh_val = float(np.percentile(gray_diff, percentile_threshold))
    # Must also be above a minimum baseline difference
    thresh_val = max(thresh_val, ela_dict["mean_error"] + 2.0 * ela_dict["std_error"])

    binary_mask = (gray_diff >= thresh_val).astype(np.uint8) * 255

    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # Filter out tiny specks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    clean_mask = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_area:
            clean_mask[labels == i] = 255

    return clean_mask


def visualize_ela_heatmap(ela_dict: Dict[str, Any]) -> np.ndarray:
    """
    Produce a color heatmap (JET colormap) of compression discrepancy.
    """
    raw_diff = ela_dict["raw_diff"]
    gray_diff = np.mean(raw_diff, axis=2)

    # Normalize to [0, 255]
    p98 = max(float(np.percentile(gray_diff, 98)), 1.0)
    norm = np.clip((gray_diff / p98) * 255.0, 0, 255).astype(np.uint8)

    # Smooth slightly for clean heatmap visualization
    smoothed = cv2.GaussianBlur(norm, (9, 9), 2.0)
    heatmap_bgr = cv2.applyColorMap(smoothed, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

