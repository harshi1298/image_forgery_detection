"""
Localization Module: Dense Suspicious Region Mask Generation and Visual Overlay.
"""

from .mask import (
    generate_suspicious_mask,
    create_overlay,
    fuse_detection_masks,
    mask_to_bounding_boxes,
)

__all__ = [
    "generate_suspicious_mask",
    "create_overlay",
    "fuse_detection_masks",
    "mask_to_bounding_boxes",
]

