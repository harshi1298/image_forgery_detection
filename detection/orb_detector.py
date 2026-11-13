"""
Feature keypoint detection and descriptor computation supporting both SIFT
(scale & rotation invariant) and multi-scale ORB.
"""

from typing import List, Tuple, Optional, Union
import cv2
import numpy as np


def detect_keypoints_and_descriptors(
    gray_image: np.ndarray,
    n_features: int = 3000,
    method: str = "sift",
    scale_factor: float = 1.15,
    n_levels: int = 10,
    edge_threshold: int = 31,
    patch_size: int = 31,
    fast_threshold: int = 20,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Detect feature keypoints and compute descriptors.

    Supports:
        - 'sift': Scale-Invariant Feature Transform (best for scaled/zoomed-in/out and rotated copy-move)
        - 'orb': Multi-scale Oriented FAST and Rotated BRIEF (fast binary features)
        - 'hybrid': SIFT with enhanced multi-scale coverage

    Args:
        gray_image: 2D uint8 grayscale image
        n_features: maximum number of features to retain
        method: 'sift', 'orb', or 'hybrid'
        scale_factor: pyramid decimation ratio for ORB
        n_levels: number of pyramid levels for ORB
        edge_threshold: edge threshold for ORB
        patch_size: size of the patch used by oriented BRIEF
        fast_threshold: FAST detector threshold for ORB

    Returns:
        keypoints: list of cv2.KeyPoint objects
        descriptors: numpy array (float32 for SIFT, uint8 for ORB), or None
    """
    if gray_image.ndim != 2:
        raise ValueError("Feature detector expects a 2D grayscale image.")

    method_clean = str(method).lower().strip()

    if method_clean in ("sift", "hybrid"):
        # SIFT provides rigorous continuous scale-space invariance
        sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=3,
            contrastThreshold=0.03,
            edgeThreshold=10,
            sigma=1.6,
        )
        keypoints, descriptors = sift.detectAndCompute(gray_image, mask=None)
        if keypoints is None:
            keypoints = []
        return keypoints, descriptors

    # Default / fallback to ORB with fine multi-scale pyramid
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
    """
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    bgr_color = (int(color[2]), int(color[1]), int(color[0]))
    annotated_bgr = cv2.drawKeypoints(bgr, keypoints, None, color=bgr_color, flags=flags)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


