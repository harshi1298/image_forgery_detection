"""
ORB (Oriented FAST and Rotated BRIEF) keypoint detection and descriptor computation.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np


def detect_keypoints_and_descriptors(
    gray_image: np.ndarray,
    n_features: int = 3000,
    scale_factor: float = 1.2,
    n_levels: int = 8,
    edge_threshold: int = 31,
    patch_size: int = 31,
    fast_threshold: int = 20,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Detect ORB keypoints and compute 256-bit binary descriptors.

    Args:
        gray_image: 2D uint8 grayscale image
        n_features: maximum number of features to retain
        scale_factor: pyramid decimation ratio
        n_levels: number of pyramid levels
        edge_threshold: threshold size from image borders
        patch_size: size of the patch used by oriented BRIEF
        fast_threshold: FAST detector threshold

    Returns:
        keypoints: list of cv2.KeyPoint objects
        descriptors: (N, 32) uint8 numpy array, or None if no features found
    """
    if gray_image.ndim != 2:
        raise ValueError("ORB detector expects a 2D grayscale image.")

    orb = cv2.ORB_create(
        nfeatures=n_features,
        scaleFactor=scale_factor,
        nlevels=n_levels,
        edgeThreshold=edge_threshold,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=patch_size,
        fastThreshold=fast_threshold,
    )

    keypoints, descriptors = orb.detectAndCompute(gray_image, mask=None)

    if keypoints is None:
        keypoints = []

    return keypoints, descriptors


def keypoints_to_coords(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
    """
    Extract (x, y) coordinates from a list of cv2.KeyPoint into a float32 array of shape (N, 2).
    """
    if not keypoints:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)


def draw_keypoints(
    image_rgb: np.ndarray,
    keypoints: List[cv2.KeyPoint],
    color: Tuple[int, int, int] = (0, 255, 128),
    rich: bool = False,
) -> np.ndarray:
    """
    Render keypoints onto the image.

    Args:
        image_rgb: RGB image
        keypoints: list of KeyPoints
        color: RGB tuple for drawing
        rich: If True, draws circle with size and orientation

    Returns:
        Annotated RGB image
    """
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    # cv2.drawKeypoints expects BGR or gray, so convert RGB -> BGR -> draw -> BGR -> RGB
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # color in BGR format
    bgr_color = (int(color[2]), int(color[1]), int(color[0]))
    annotated_bgr = cv2.drawKeypoints(bgr, keypoints, None, color=bgr_color, flags=flags)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

