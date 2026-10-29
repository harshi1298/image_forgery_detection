"""
Evaluation Module: IoU, Precision, Recall, F1, and forensic visualization.
"""

from .metrics import (
    calculate_iou,
    calculate_pixel_metrics,
    evaluate_localization,
    make_final_decision,
)
from .visualization import (
    plot_detection_summary,
    create_comparison_figure,
    create_confusion_overlay,
)

__all__ = [
    "calculate_iou",
    "calculate_pixel_metrics",
    "evaluate_localization",
    "make_final_decision",
    "plot_detection_summary",
    "create_comparison_figure",
    "create_confusion_overlay",
]

