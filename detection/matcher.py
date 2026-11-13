"""
Feature matching, Normalized Cross-Correlation (NCC) verification,
and geometric RANSAC filtering for Intra-Image Copy-Move detection.
Supports both SIFT (scale & rotation invariant) and ORB features.
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np


def compute_patch_ncc(
    gray_image: np.ndarray,
    pt1: Tuple[float, float],
    pt2: Tuple[float, float],
    patch_size: int = 17,
    scale_ratio: float = 1.0,
) -> float:
    """
    Compute Normalized Cross-Correlation (NCC) between two image patches.
    Supports scale-aware sampling when an object was zoomed in or zoomed out.
    """
    h, w = gray_image.shape[:2]
    r1 = patch_size // 2
    
    # Scale patch 2 according to the relative keypoint scale ratio
    s = float(np.clip(scale_ratio, 0.2, 5.0))
    r2 = max(2, int(round(r1 * s)))

    x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
    x2, y2 = int(round(pt2[0])), int(round(pt2[1]))

    if x1 - r1 < 0 or x1 + r1 >= w or y1 - r1 < 0 or y1 + r1 >= h:
        return 0.0
    if x2 - r2 < 0 or x2 + r2 >= w or y2 - r2 < 0 or y2 + r2 >= h:
        return 0.0

    p1 = gray_image[y1 - r1 : y1 + r1 + 1, x1 - r1 : x1 + r1 + 1].astype(np.float32)
    p2 = gray_image[y2 - r2 : y2 + r2 + 1, x2 - r2 : x2 + r2 + 1].astype(np.float32)

    # Resample patch 2 to match patch 1 dimensions if scaled
    if p2.shape[:2] != p1.shape[:2]:
        p2 = cv2.resize(p2, (p1.shape[1], p1.shape[0]), interpolation=cv2.INTER_LINEAR)

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
    ratio_threshold: float = 0.78,
    min_spatial_distance: float = 30.0,
    max_hamming_distance: int = 58,
    min_patch_ncc: float = 0.35,
    k_neighbors: int = 10,
) -> List[Tuple[int, int, float, float, float]]:
    """
    Perform intra-image feature matching with spatial radius constraint,
    Lowe's ratio test, and scale-aware patch Normalized Cross-Correlation (NCC).

    Automatically handles:
      - SIFT descriptors (float32, L2 norm)
      - ORB descriptors (uint8, Hamming norm)

    Returns:
        List of match tuples: (idx1, idx2, distance, spatial_dist, ncc_score)
    """
    if descriptors is None or len(descriptors) < 4:
        return []

    num_kps = len(keypoints)
    if num_kps < 4:
        return []

    # Choose distance metric based on descriptor type
    is_float = descriptors.dtype == np.float32 or descriptors.dtype == np.float64
    norm_type = cv2.NORM_L2 if is_float else cv2.NORM_HAMMING
    max_dist = 280.0 if is_float else float(max_hamming_distance)

    bf = cv2.BFMatcher(norm_type, crossCheck=False)
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

        if best_dist > max_dist:
            continue

        # Lowe's ratio test against the second nearest non-local candidate
        ratio = best_dist / second_dist
        if ratio > ratio_threshold:
            continue

        p1_idx, p2_idx = min(i, best_m.trainIdx), max(i, best_m.trainIdx)
        if (p1_idx, p2_idx) in seen_pairs:
            continue
        seen_pairs.add((p1_idx, p2_idx))

        # Scale-aware patch cross-correlation verification
        ncc_score = 1.0
        if gray_image is not None and min_patch_ncc > 0.0:
            kp1 = keypoints[p1_idx]
            kp2 = keypoints[p2_idx]
            scale_ratio = kp2.size / max(kp1.size, 1e-4) if (kp1.size > 0 and kp2.size > 0) else 1.0
            ncc_score = compute_patch_ncc(gray_image, coords[p1_idx], coords[p2_idx], scale_ratio=scale_ratio)

            # Soft NCC gating: only drop if NCC is low AND descriptor ratio is ambiguous
            if ncc_score < min_patch_ncc and ratio > 0.70:
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
    min_spatial_distance: float = 30.0,
    min_scale: float = 0.20,
    max_scale: float = 5.0,
) -> Tuple[List[Tuple], Optional[np.ndarray], np.ndarray, float]:
    """
    Apply RANSAC geometric verification to remove geometrically inconsistent false matches.
    Evaluates multi-orientation canonical alignment to handle zoom in/out (scaling)
    and arbitrary match ordering between source and destination regions.

    Returns:
        inlier_matches: list of geometrically verified match tuples
        transform_matrix: 2x3 affine matrix (or None)
        inlier_mask: boolean mask of inliers
        estimated_scale: estimated zoom/scale factor (e.g. 0.70x, 1.45x)
    """
    if len(matches) < min_inliers:
        return [], None, np.zeros(len(matches), dtype=bool), 1.0

    coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    best_inliers: List[Tuple] = []
    best_M: Optional[np.ndarray] = None
    best_mask = np.zeros(len(matches), dtype=bool)
    best_scale: float = 1.0

    # Test directional canonical alignments so source -> target vectors are consistent
    orientations = [
        lambda a, b: (a, b) if coords[a][0] <= coords[b][0] else (b, a),
        lambda a, b: (b, a) if coords[a][0] <= coords[b][0] else (a, b),
        lambda a, b: (a, b) if coords[a][1] <= coords[b][1] else (b, a),
        lambda a, b: (b, a) if coords[a][1] <= coords[b][1] else (a, b),
    ]

    for orient_fn in orientations:
        aligned_pairs = [orient_fn(m[0], m[1]) for m in matches]
        src_pts = np.array([coords[a] for a, b in aligned_pairs], dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array([coords[b] for a, b in aligned_pairs], dtype=np.float32).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold,
            maxIters=3000,
            confidence=0.99,
        )

        if M is not None and inliers is not None:
            s_est = float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2))
            t_est = float(np.hypot(M[0, 2], M[1, 2]))
            inlier_count = int(np.sum(inliers))

            # Validate physically realistic scale and translation
            if min_scale <= s_est <= max_scale and t_est >= min_spatial_distance * 0.50:
                if inlier_count > len(best_inliers):
                    mask = inliers.ravel().astype(bool)
                    best_inliers = [matches[idx] for idx, val in enumerate(mask) if val]
                    best_M = M
                    best_mask = mask
                    best_scale = s_est

    # Fallback to displacement vector clustering if affine model did not reach min_inliers
    if len(best_inliers) < min_inliers:
        clusters = cluster_displacement_vectors(
            keypoints, matches, bin_size=20.0, min_cluster_size=min_inliers
        )
        if clusters:
            best_cluster = clusters[0]
            cluster_set = set(best_cluster)
            best_mask = np.array([m in cluster_set for m in matches], dtype=bool)
            best_inliers = [m for idx, m in enumerate(matches) if best_mask[idx]]
            best_M = None
            best_scale = 1.0

    return best_inliers, best_M, best_mask, best_scale

