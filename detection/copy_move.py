"""
Copy-Move detection pipeline combining ORB, intra-image matching, RANSAC, and visual output.
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

from .orb_detector import detect_keypoints_and_descriptors
from .matcher import match_keypoints_intra_image, apply_ransac_filtering


def detect_copy_move(
    gray_image: np.ndarray,
    feature_method: str = "sift",
    n_features: int = 3500,
    ratio_threshold: float = 0.78,
    min_spatial_distance: float = 30.0,
    max_hamming_distance: int = 58,
    reproj_threshold: float = 6.0,
    min_inliers_for_forgery: int = 10,
    min_patch_ncc: float = 0.35,
) -> Dict[str, Any]:
    """
    Run full Copy-Move detection pipeline on preprocessed grayscale image.
    Supports both SIFT (scale & rotation invariant) and multi-scale ORB.

    Returns dict containing:
        is_detected: bool (True if suspected copy-move)
        confidence: float in [0.0, 1.0] (forgery probability)
        keypoints: list of cv2.KeyPoint
        descriptors: np.ndarray of descriptors
        all_matches: list of candidate match tuples
        inlier_matches: list of geometrically verified match tuples
        transform_matrix: estimated geometric transform
        matched_points_src: list of (x, y) coords
        matched_points_dst: list of (x, y) coords
        mean_ncc: float average NCC score of inliers
        estimated_scale: estimated zoom/scale ratio of manipulation
        scale_description: human-readable zoom/scale interpretation
    """
    keypoints, descriptors = detect_keypoints_and_descriptors(
        gray_image,
        n_features=n_features,
        method=feature_method,
    )

    if not keypoints or descriptors is None or len(keypoints) < 4:
        return {
            "is_detected": False,
            "confidence": 0.0,
            "keypoints": keypoints or [],
            "descriptors": descriptors,
            "all_matches": [],
            "inlier_matches": [],
            "transform_matrix": None,
            "matched_points_src": [],
            "matched_points_dst": [],
            "mean_ncc": 0.0,
            "estimated_scale": 1.0,
            "scale_description": "N/A",
        }

    raw_matches = match_keypoints_intra_image(
        keypoints=keypoints,
        descriptors=descriptors,
        gray_image=gray_image,
        ratio_threshold=ratio_threshold,
        min_spatial_distance=min_spatial_distance,
        max_hamming_distance=max_hamming_distance,
        min_patch_ncc=min_patch_ncc,
    )

    inlier_matches, transform_matrix, _, estimated_scale = apply_ransac_filtering(
        keypoints=keypoints,
        matches=raw_matches,
        reproj_threshold=reproj_threshold,
        min_inliers=min_inliers_for_forgery,
        min_spatial_distance=min_spatial_distance,
    )

    src_pts = []
    dst_pts = []
    ncc_scores = []
    for m in inlier_matches:
        idx1, idx2 = m[0], m[1]
        src_pts.append(keypoints[idx1].pt)
        dst_pts.append(keypoints[idx2].pt)
        if len(m) >= 5:
            ncc_scores.append(m[4])

    inlier_count = len(inlier_matches)
    is_detected = inlier_count >= min_inliers_for_forgery
    mean_ncc = float(np.mean(ncc_scores)) if ncc_scores else 0.0

    # Describe zoom / scaling
    if is_detected and transform_matrix is not None:
        s = estimated_scale
        if 0.92 <= s <= 1.08:
            scale_desc = f"1:1 Scale (Unscaled, s={s:.2f})"
        elif s < 0.92:
            pct_out = int(round((1.0 - s) * 100))
            reciprocal = 1.0 / max(s, 1e-4)
            pct_in = int(round((reciprocal - 1.0) * 100))
            scale_desc = f"{s:.2f}x (Zoom Out ~{pct_out}% / In ~{pct_in}%)"
        else:
            pct_in = int(round((s - 1.0) * 100))
            reciprocal = 1.0 / s
            pct_out = int(round((1.0 - reciprocal) * 100))
            scale_desc = f"{s:.2f}x (Zoom In ~{pct_in}% / Out ~{pct_out}%)"
    else:
        scale_desc = "1:1 / None detected"

    # Calibrate confidence (forgery probability)
    if is_detected:
        inlier_factor = min(1.0, inlier_count / 30.0)
        confidence = float(np.clip(0.72 + 0.20 * inlier_factor + 0.08 * mean_ncc, 0.72, 0.99))
    else:
        confidence = float(np.clip(inlier_count / (min_inliers_for_forgery * 2.5), 0.0, 0.35))

    return {
        "is_detected": is_detected,
        "confidence": confidence,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "all_matches": raw_matches,
        "inlier_matches": inlier_matches,
        "transform_matrix": transform_matrix,
        "matched_points_src": src_pts,
        "matched_points_dst": dst_pts,
        "mean_ncc": mean_ncc,
        "estimated_scale": estimated_scale,
        "scale_description": scale_desc,
    }


def visualize_copy_move_matches(
    image_rgb: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    inlier_matches: List[Tuple],
    raw_matches: Optional[List[Tuple]] = None,
    show_outliers: bool = False,
) -> np.ndarray:
    """
    Render copy-move matches on the image:
    Draws points and connecting lines between matched source & destination keypoints.
    """
    canvas = image_rgb.copy()

    # Draw non-inlier matches faintly if requested
    if show_outliers and raw_matches:
        inlier_set = set(inlier_matches)
        for m in raw_matches:
            if m not in inlier_set:
                p1 = tuple(map(int, np.round(keypoints[m[0]].pt)))
                p2 = tuple(map(int, np.round(keypoints[m[1]].pt)))
                cv2.line(canvas, p1, p2, (180, 180, 180), 1, cv2.LINE_AA)

    colors = [
        (0, 255, 255),    # Cyan
        (255, 105, 180),  # Pink
        (50, 205, 50),    # Lime
        (255, 215, 0),    # Gold
        (255, 69, 0),     # OrangeRed
        (138, 43, 226),   # Violet
        (0, 191, 255),    # Deep Sky Blue
    ]

    for idx, m in enumerate(inlier_matches):
        p1 = tuple(map(int, np.round(keypoints[m[0]].pt)))
        p2 = tuple(map(int, np.round(keypoints[m[1]].pt)))
        color = colors[idx % len(colors)]

        cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 6, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas
