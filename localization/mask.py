"""
Suspicious region mask generation from sparse matched keypoints and morphological operations.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


def generate_suspicious_mask(
    image_shape: Tuple[int, int],
    matched_points_src: List[Tuple[float, float]],
    matched_points_dst: List[Tuple[float, float]],
    point_radius: int = 18,
    dilation_kernel_size: int = 25,
    closing_kernel_size: int = 21,
    min_component_area: int = 250,
    use_convex_hull_clusters: bool = True,
) -> np.ndarray:
    """
    Convert sparse matched keypoint pairs into a dense binary mask of suspected forged regions.

    Args:
        image_shape: (H, W) or (H, W, C)
        matched_points_src: list of (x, y) coords for source points
        matched_points_dst: list of (x, y) coords for destination points
        point_radius: initial radius drawn around each matched keypoint
        dilation_kernel_size: size of structuring element for dilation
        closing_kernel_size: size of structuring element for closing
        min_component_area: minimum area (in pixels) to keep a detected region
        use_convex_hull_clusters: whether to fill the convex hull of localized point clusters

    Returns:
        binary mask of shape (H, W), dtype uint8 with values 0 (authentic) and 255 (suspicious)
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    all_points = list(matched_points_src) + list(matched_points_dst)
    if not all_points:
        return mask

    # 1. Draw solid circles around every detected keypoint
    for pt in all_points:
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(mask, (cx, cy), point_radius, 255, -1)

    # 2. Local spatial clustering: compute convex hull only for tightly localized sub-clusters
    if use_convex_hull_clusters and len(all_points) >= 4:
        from sklearn.cluster import DBSCAN
        pts_np = np.array(all_points, dtype=np.float32)
        # Cluster points within localized proximity (e.g. 45 px)
        db = DBSCAN(eps=45.0, min_samples=3).fit(pts_np)
        for label in set(db.labels_):
            if label == -1:
                continue
            cluster_pts = pts_np[db.labels_ == label]
            if len(cluster_pts) >= 4:
                hull = cv2.convexHull(cluster_pts.astype(np.int32).reshape(-1, 1, 2))
                cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)

    # 3. Morphological Dilation with an elliptical structuring element
    if dilation_kernel_size > 1:
        d_k = dilation_kernel_size if dilation_kernel_size % 2 == 1 else dilation_kernel_size + 1
        d_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_k, d_k))
        mask = cv2.dilate(mask, d_elem, iterations=1)

    # 4. Morphological Closing to fill gaps between keypoints in the same object
    if closing_kernel_size > 1:
        c_k = closing_kernel_size if closing_kernel_size % 2 == 1 else closing_kernel_size + 1
        c_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c_k, c_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, c_elem, iterations=2)

    # 5. Fill interior holes in contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(mask)
    for cnt in contours:
        cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # 6. Connected component analysis to remove tiny isolated noise artifacts
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled_mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            clean_mask[labels == i] = 255

    return clean_mask


def create_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
    color: Tuple[int, int, int] = (255, 30, 30),
    draw_contours: bool = True,
) -> np.ndarray:
    """
    Overlay binary mask onto the RGB image with semi-transparent tint and crisp outline.

    Args:
        image_rgb: base image (H, W, 3) uint8
        mask: binary mask (H, W) uint8 (0 or 255)
        alpha: opacity of highlight tint (0.0 to 1.0)
        color: RGB tuple for highlighting (default bright red)
        draw_contours: whether to draw a solid outline around detected regions

    Returns:
        RGB image with overlay applied
    """
    overlay = image_rgb.copy()
    h, w = image_rgb.shape[:2]

    # Ensure mask matches image dimensions
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    binary_bool = mask > 127
    if not np.any(binary_bool):
        return overlay

    # Create solid color layer
    color_layer = np.zeros_like(image_rgb)
    color_layer[:, :] = color

    # Blend highlighted pixels
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, color_layer, alpha, 0)
    overlay[binary_bool] = blended[binary_bool]

    # Draw solid border around detected contours
    if draw_contours:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)

    return overlay


def mask_to_bounding_boxes(mask: np.ndarray, min_area: int = 150) -> List[Tuple[int, int, int, int]]:
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

    Modes:
    - 'copy_move_primary': Use copy-move mask primarily; if copy-move is detected, it is returned.
    - 'union': Logical OR of copy-move and ELA.
    - 'intersection': Logical AND where both methods agree.
    """
    if ela_mask is None or mode == "copy_move_primary":
        return copy_move_mask.copy()

    # Align dimensions if necessary
    if ela_mask.shape != copy_move_mask.shape:
        ela_mask = cv2.resize(ela_mask, (copy_move_mask.shape[1], copy_move_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    if mode == "union":
        return cv2.bitwise_or(copy_move_mask, ela_mask)
    elif mode == "intersection":
        return cv2.bitwise_and(copy_move_mask, ela_mask)
    else:
        return copy_move_mask.copy()
