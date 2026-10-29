"""
Detection Module: ORB keypoint extraction, intra-image feature matching, RANSAC, and copy-move analysis.
"""

from .orb_detector import detect_keypoints_and_descriptors, draw_keypoints
from .matcher import match_keypoints_intra_image, apply_ransac_filtering
from .copy_move import detect_copy_move, visualize_copy_move_matches

__all__ = [
    "detect_keypoints_and_descriptors",
    "draw_keypoints",
    "match_keypoints_intra_image",
    "apply_ransac_filtering",
    "detect_copy_move",
    "visualize_copy_move_matches",
]

