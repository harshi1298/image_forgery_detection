"""
Suspicious region mask generation from matched keypoints with compact morphology and boundary refinement.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


def generate_suspicious_mask(
    image_shape: Tuple[int, int],
    matched_points_src: List[Tuple[float, float]],
    matched_points_dst: List[Tuple[float, float]],
    is_detected: bool = True,
    point_radius: int = 6,
    dilation_kernel_size: int = 7,
    closing_kernel_size: int = 7,
    min_component_area: int = 100,
    use_convex_hull_clusters: bool = False,
) -> np.ndarray:
    """
    Convert sparse matched keypoint pairs into a tight, accurate binary mask of suspected forged regions.

    Args:
        image_shape: (H, W) or (H, W, C)
        matched_points_src: list of (x, y) coords for source points
        matched_points_dst: list of (x, y) coords for destination points
        is_detected: whether forgery was detected (if False, returns clean blank mask)
        point_radius: radius drawn around each matched keypoint (default 6 px)
        dilation_kernel_size: structuring element size for dilation (default 7 px)
        closing_kernel_size: structuring element size for closing (default 7 px)
        min_component_area: minimum area (in pixels) to keep a detected region
        use_convex_hull_clusters: whether to draw convex hulls over tight local clusters

    Returns:
        binary mask of shape (H, W), dtype uint8 with values 0 (authentic) and 255 (suspicious)
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # If the detector did not confirm forgery, return a completely clean blank mask
    if not is_detected:
        return mask

    all_points = list(matched_points_src) + list(matched_points_dst)
    if not all_points:
        return mask

    # 1. Draw solid circles around each verified keypoint
    for pt in all_points:
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(mask, (cx, cy), point_radius, 255, -1)

    # 2. Local spatial clustering: compute convex hull only for tightly localized sub-clusters
    if use_convex_hull_clusters and len(all_points) >= 6:
        from sklearn.cluster import DBSCAN
        pts_np = np.array(all_points, dtype=np.float32)
        # Tightly localized cluster proximity (25 px)
        db = DBSCAN(eps=25.0, min_samples=4).fit(pts_np)
        for label in set(db.labels_):
            if label == -1:
                continue
            cluster_pts = pts_np[db.labels_ == label]
            if len(cluster_pts) >= 4:
                hull = cv2.convexHull(cluster_pts.astype(np.int32).reshape(-1, 1, 2))
                cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)

    # 3. Compact elliptical dilation to bridge immediate point gaps
    if dilation_kernel_size > 1:
        d_k = dilation_kernel_size if dilation_kernel_size % 2 == 1 else dilation_kernel_size + 1
        d_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_k, d_k))
        mask = cv2.dilate(mask, d_elem, iterations=1)

    # 4. Morphological Closing to seal interior holes
    if closing_kernel_size > 1:
        c_k = closing_kernel_size if closing_kernel_size % 2 == 1 else closing_kernel_size + 1
        c_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c_k, c_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, c_elem, iterations=1)

    # 5. Fill interior holes in contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_component_area:
            cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return filled_mask


def create_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    color: Tuple[int, int, int] = (255, 30, 30),
    draw_contours: bool = True,
) -> np.ndarray:
    """
    Overlay binary mask onto the RGB image with semi-transparent tint and crisp outline.
    """
    overlay = image_rgb.copy()
    h, w = image_rgb.shape[:2]

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    binary_bool = mask > 127
    if not np.any(binary_bool):
        return overlay

    color_layer = np.zeros_like(image_rgb)
    color_layer[:, :] = color

    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, color_layer, alpha, 0)
    overlay[binary_bool] = blended[binary_bool]

    if draw_contours:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)

    return overlay


def mask_to_bounding_boxes(mask: np.ndarray, min_area: int = 100) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes (x, y, w, h) for all connected suspicious regions in mask.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            boxes.append(cv2.boundingRect(cnt))
    return boxes


def fuse_detection_masks(
    copy_move_mask: np.ndarray,
    ela_mask: Optional[np.ndarray] = None,
    mode: str = "copy_move_primary",
) -> np.ndarray:
    """
    Combine copy-move spatial mask with ELA anomaly mask.
    """
    if ela_mask is None or mode == "copy_move_primary":
        return copy_move_mask.copy()

    if ela_mask.shape != copy_move_mask.shape:
        ela_mask = cv2.resize(
            ela_mask,
            (copy_move_mask.shape[1], copy_move_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if mode == "union":
        return cv2.bitwise_or(copy_move_mask, ela_mask)
    elif mode == "intersection":
        return cv2.bitwise_and(copy_move_mask, ela_mask)
    else:
        return copy_move_mask.copy()
