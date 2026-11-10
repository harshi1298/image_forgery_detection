"""
Feature matching, Normalized Cross-Correlation (NCC) verification,
and geometric RANSAC filtering for Intra-Image Copy-Move detection.
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np


def compute_patch_ncc(
    gray_image: np.ndarray,
    pt1: Tuple[float, float],
    pt2: Tuple[float, float],
    patch_size: int = 17,
) -> float:
    """
    Compute Normalized Cross-Correlation (NCC / Pearson correlation) between two image patches.
    In genuine copy-move manipulation, cloned regions have high cross-correlation (>= 0.50 - 0.95).
    Spurious keypoint matches on unrelated textures have near-zero or low correlation.
    """
    h, w = gray_image.shape[:2]
    r = patch_size // 2
    x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
    x2, y2 = int(round(pt2[0])), int(round(pt2[1]))

    if x1 - r < 0 or x1 + r >= w or y1 - r < 0 or y1 + r >= h:
        return 0.0
    if x2 - r < 0 or x2 + r >= w or y2 - r < 0 or y2 + r >= h:
        return 0.0

    p1 = gray_image[y1 - r : y1 + r + 1, x1 - r : x1 + r + 1].astype(np.float32)
    p2 = gray_image[y2 - r : y2 + r + 1, x2 - r : x2 + r + 1].astype(np.float32)

    s1 = float(np.std(p1))
    s2 = float(np.std(p2))
    if s1 < 2.0 or s2 < 2.0:
        return 0.0

    p1_norm = (p1 - np.mean(p1)) / s1
    p2_norm = (p2 - np.mean(p2)) / s2
    return float(np.mean(p1_norm * p2_norm))


def match_keypoints_intra_image(
    keypoints: List[cv2.KeyPoint],
    descriptors: Optional[np.ndarray],
    gray_image: Optional[np.ndarray] = None,
    ratio_threshold: float = 0.75,
    min_spatial_distance: float = 35.0,
    max_hamming_distance: int = 50,
    min_patch_ncc: float = 0.45,
    k_neighbors: int = 10,
) -> List[Tuple[int, int, float, float, float]]:
    """
    Perform intra-image feature matching with spatial radius constraint,
    Lowe's ratio test, and patch Normalized Cross-Correlation (NCC) verification.

    Returns:
        List of match tuples: (idx1, idx2, hamming_dist, spatial_dist, ncc_score)
    """
    if descriptors is None or len(descriptors) < 4:
        return []

    num_kps = len(keypoints)
    if num_kps < 4:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    k = min(k_neighbors, num_kps)
    raw_knn = bf.knnMatch(descriptors, descriptors, k=k)

    coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    candidate_matches = []
    seen_pairs = set()

    for i, matches in enumerate(raw_knn):
        if not matches:
            continue

        pt_i = coords[i]
        valid_candidates = []
        for m in matches:
            j = m.trainIdx
            if j == i:
                continue
            pt_j = coords[j]
            spatial_dist = float(np.hypot(pt_i[0] - pt_j[0], pt_i[1] - pt_j[1]))
            if spatial_dist >= min_spatial_distance:
                valid_candidates.append((m, spatial_dist))

        if len(valid_candidates) < 2:
            continue

        best_m, best_spatial = valid_candidates[0]
        second_m, _ = valid_candidates[1]

        best_dist = best_m.distance
        second_dist = max(second_m.distance, 1e-4)

        if best_dist > max_hamming_distance:
            continue

        # Lowe's ratio test against the second nearest non-local candidate
        if (best_dist / second_dist) > ratio_threshold:
            continue

        p1_idx, p2_idx = min(i, best_m.trainIdx), max(i, best_m.trainIdx)
        if (p1_idx, p2_idx) in seen_pairs:
            continue
        seen_pairs.add((p1_idx, p2_idx))

        # Patch cross-correlation verification
        ncc_score = 1.0
        if gray_image is not None:
            ncc_score = compute_patch_ncc(gray_image, coords[p1_idx], coords[p2_idx])
            if ncc_score < min_patch_ncc:
                continue

        candidate_matches.append((p1_idx, p2_idx, float(best_dist), best_spatial, float(ncc_score)))

    return candidate_matches


def cluster_displacement_vectors(
    keypoints: List[cv2.KeyPoint],
    matches: List[Tuple],
    bin_size: float = 20.0,
    min_cluster_size: int = 5,
) -> List[List[Tuple]]:
    """
    Cluster putative matches by displacement vector (dx, dy).
    In pure translation copy-move, the displacement vector is consistent.
    """
    if not matches:
        return []

    clusters: Dict[Tuple[int, int], List[Tuple]] = {}
    for m in matches:
        i, j = m[0], m[1]
        p1 = keypoints[i].pt
        p2 = keypoints[j].pt

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # Canonicalize vector direction
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy

        bx = int(round(dx / bin_size))
        by = int(round(dy / bin_size))
        clusters.setdefault((bx, by), []).append(m)

    valid_clusters = [cl for cl in clusters.values() if len(cl) >= min_cluster_size]
    valid_clusters.sort(key=len, reverse=True)
    return valid_clusters


def apply_ransac_filtering(
    keypoints: List[cv2.KeyPoint],
    matches: List[Tuple],
    reproj_threshold: float = 6.0,
    min_inliers: int = 8,
    min_spatial_distance: float = 35.0,
) -> Tuple[List[Tuple], Optional[np.ndarray], np.ndarray]:
    """
    Apply RANSAC geometric verification to remove geometrically inconsistent false matches.
    Validates that the estimated affine transformation is physically realistic.
    """
    if len(matches) < min_inliers:
        return [], None, np.zeros(len(matches), dtype=bool)

    coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    src_pts = np.array([coords[m[0]] for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array([coords[m[1]] for m in matches], dtype=np.float32).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold,
        maxIters=3000,
        confidence=0.99,
    )

    inlier_mask = inliers.ravel().astype(bool) if inliers is not None else np.zeros(len(matches), dtype=bool)
    inlier_count = int(np.sum(inlier_mask))

    # Geometric sanity check on affine matrix
    is_valid_transform = False
    if M is not None and inlier_count >= min_inliers:
        scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        trans_dist = np.hypot(M[0, 2], M[1, 2])
        # Scale must be realistic and translation must exceed minimum spatial distance
        if 0.60 <= scale <= 1.65 and trans_dist >= min_spatial_distance * 0.70:
            is_valid_transform = True

    # Fallback to displacement vector clustering if affine model failed
    if not is_valid_transform or inlier_count < min_inliers:
        clusters = cluster_displacement_vectors(
            keypoints, matches, bin_size=20.0, min_cluster_size=min_inliers
        )
        if clusters:
            best_cluster = clusters[0]
            cluster_set = set(best_cluster)
            inlier_mask = np.array([m in cluster_set for m in matches], dtype=bool)
            inlier_count = int(np.sum(inlier_mask))
            M = None

    inlier_matches = [m for idx, m in enumerate(matches) if inlier_mask[idx]]
    return inlier_matches, M, inlier_mask
