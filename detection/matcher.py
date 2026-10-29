"""
Feature matching and geometric verification for Intra-Image Copy-Move detection.
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np


def match_keypoints_intra_image(
    keypoints: List[cv2.KeyPoint],
    descriptors: Optional[np.ndarray],
    ratio_threshold: float = 0.78,
    min_spatial_distance: float = 35.0,
    max_hamming_distance: int = 60,
    k_neighbors: int = 10,
) -> List[Tuple[int, int, float, float]]:
    """
    Perform intra-image feature matching to discover identical or transformed regions.

    Since we match descriptors against the same image:
    1. Self-matches (i == j) are ignored.
    2. Neighbors within a small spatial radius (min_spatial_distance) are discarded
       to prevent spurious matches between adjacent texture pixels.
    3. Ratio test is performed between the best non-local candidate and the next candidate.
    4. Symmetrical pairs (i, j) and (j, i) are deduplicated (i < j).

    Returns:
        List of match tuples: (idx1, idx2, hamming_distance, spatial_distance)
    """
    if descriptors is None or len(descriptors) < 4:
        return []

    num_kps = len(keypoints)
    if num_kps < 4:
        return []

    # BFMatcher with NORM_HAMMING for binary ORB descriptors
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

        # Filter out self-match and spatial neighbors that are too close
        valid_candidates = []
        for m in matches:
            j = m.trainIdx
            if j == i:
                continue
            pt_j = coords[j]
            spatial_dist = float(np.hypot(pt_i[0] - pt_j[0], pt_i[1] - pt_j[1]))
            if spatial_dist >= min_spatial_distance:
                valid_candidates.append((m, spatial_dist))

        if not valid_candidates:
            continue

        best_m, best_spatial = valid_candidates[0]
        best_j = best_m.trainIdx
        best_dist = best_m.distance

        if best_dist > max_hamming_distance:
            continue

        # Ratio test against the second non-local neighbor if available
        if len(valid_candidates) >= 2:
            second_m, _ = valid_candidates[1]
            second_dist = max(second_m.distance, 1e-5)
            if (best_dist / second_dist) > ratio_threshold:
                continue

        # Canonical pair (min_idx, max_idx) to avoid duplicates
        p1, p2 = (min(i, best_j), max(i, best_j))
        if (p1, p2) not in seen_pairs:
            seen_pairs.add((p1, p2))
            spatial_dist = float(np.hypot(coords[p1][0] - coords[p2][0], coords[p1][1] - coords[p2][1]))
            candidate_matches.append((p1, p2, float(best_dist), spatial_dist))

    return candidate_matches


def cluster_displacement_vectors(
    keypoints: List[cv2.KeyPoint],
    matches: List[Tuple[int, int, float, float]],
    bin_size: float = 30.0,
    min_cluster_size: int = 4,
) -> List[List[Tuple[int, int, float, float]]]:
    """
    Cluster matches by displacement vector (dx, dy).
    In pure translation copy-move, the displacement vector (dx, dy) is constant
    across all duplicated keypoints.
    """
    if not matches:
        return []

    clusters: Dict[Tuple[int, int], List[Tuple[int, int, float, float]]] = {}

    for m in matches:
        i, j, _, _ = m
        pt1 = keypoints[i].pt
        pt2 = keypoints[j].pt

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]

        # Canonicalize vector direction so (dx, dy) and (-dx, -dy) group together
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy

        bin_x = int(round(dx / bin_size))
        bin_y = int(round(dy / bin_size))
        bin_key = (bin_x, bin_y)

        if bin_key not in clusters:
            clusters[bin_key] = []
        clusters[bin_key].append(m)

    # Return clusters meeting the minimum size threshold, sorted largest first
    valid_clusters = [cl for cl in clusters.values() if len(cl) >= min_cluster_size]
    valid_clusters.sort(key=len, reverse=True)
    return valid_clusters


def apply_ransac_filtering(
    keypoints: List[cv2.KeyPoint],
    matches: List[Tuple[int, int, float, float]],
    reproj_threshold: float = 8.0,
    min_inliers: int = 5,
) -> Tuple[List[Tuple[int, int, float, float]], Optional[np.ndarray], np.ndarray]:
    """
    Apply RANSAC geometric verification to remove geometrically inconsistent false matches.

    Attempts to fit an affine transformation (or homography) between source and destination points.
    Also handles translation/rotation clusters.

    Returns:
        inlier_matches: list of filtered matches
        transform_matrix: 2x3 affine matrix or 3x3 homography (or None)
        inlier_mask: boolean or uint8 mask of inliers
    """
    if len(matches) < min_inliers:
        return [], None, np.zeros(len(matches), dtype=bool)

    coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    src_pts = np.array([coords[m[0]] for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array([coords[m[1]] for m in matches], dtype=np.float32).reshape(-1, 1, 2)

    # First attempt: Affine partial 2D (translation + rotation + uniform scale)
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

    # If affine didn't yield enough inliers, try homography (handles perspective distortion)
    if inlier_count < min_inliers and len(matches) >= 4:
        H, h_inliers = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold,
            maxIters=3000,
            confidence=0.99,
        )
        if h_inliers is not None and np.sum(h_inliers) > inlier_count:
            M = H
            inlier_mask = h_inliers.ravel().astype(bool)
            inlier_count = int(np.sum(inlier_mask))

    # If RANSAC geometric model struggled (e.g., complex multi-patch clone),
    # check displacement vector clustering as a reliable backup
    if inlier_count < min_inliers:
        clusters = cluster_displacement_vectors(keypoints, matches, bin_size=25.0, min_cluster_size=min_inliers)
        if clusters:
            # Union top consistent clusters
            cluster_matches = []
            for cl in clusters:
                cluster_matches.extend(cl)
            cluster_set = set(cluster_matches)
            inlier_mask = np.array([m in cluster_set for m in matches], dtype=bool)
            inlier_count = int(np.sum(inlier_mask))

    inlier_matches = [m for idx, m in enumerate(matches) if inlier_mask[idx]]
    return inlier_matches, M, inlier_mask

