"""
Forensic visual reporting and side-by-side comparison plots.
"""

from typing import Optional, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_confusion_overlay(
    image_rgb: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Produce a color-coded confusion overlay showing:
    - Green: True Positives (Prediction ∩ GroundTruth)
    - Red: False Positives (Prediction \\ GroundTruth)
    - Blue/Yellow: False Negatives (GroundTruth \\ Prediction)
    """
    h, w = image_rgb.shape[:2]
    p_bool = (predicted_mask > 127).astype(bool)
    gt_bool = (ground_truth_mask > 127).astype(bool)

    if p_bool.shape != (h, w):
        p_resized = cv2.resize(predicted_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        p_bool = p_resized > 127
    if gt_bool.shape != (h, w):
        gt_resized = cv2.resize(ground_truth_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        gt_bool = gt_resized > 127

    tp = np.logical_and(p_bool, gt_bool)
    fp = np.logical_and(p_bool, np.logical_not(gt_bool))
    fn = np.logical_and(np.logical_not(p_bool), gt_bool)

    overlay = image_rgb.copy().astype(np.float32)

    # Color codes in RGB:
    # TP: Lime Green (46, 204, 113)
    # FP: Coral Red (231, 76, 60)
    # FN: Electric Blue (52, 152, 219)
    color_tp = np.array([46, 204, 113], dtype=np.float32)
    color_fp = np.array([231, 76, 60], dtype=np.float32)
    color_fn = np.array([52, 152, 219], dtype=np.float32)

    if np.any(tp):
        overlay[tp] = (1.0 - alpha) * overlay[tp] + alpha * color_tp
    if np.any(fp):
        overlay[fp] = (1.0 - alpha) * overlay[fp] + alpha * color_fp
    if np.any(fn):
        overlay[fn] = (1.0 - alpha) * overlay[fn] + alpha * color_fn

    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_detection_summary(
    image_rgb: np.ndarray,
    matches_img: np.ndarray,
    ela_img: np.ndarray,
    ela_heatmap: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth_mask: Optional[np.ndarray] = None,
    overlay_img: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, float]] = None,
    decision: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    """
    Generate an all-in-one multi-panel summary figure for academic presentations / reports.
    """
    has_gt = ground_truth_mask is not None

    cols = 3
    rows = 2 if not has_gt else 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 9))
    axes = axes.ravel()

    # Panel 1: Original Image
    axes[0].imshow(image_rgb)
    axes[0].set_title("1. Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: ORB Copy-Move Matches
    axes[1].imshow(matches_img)
    inliers = decision.get("inliers_count", 0) if decision else 0
    axes[1].set_title(f"2. ORB Feature Matches ({inliers} Inliers)", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: ELA Compression Heatmap
    axes[2].imshow(ela_heatmap)
    axes[2].set_title("3. ELA Compression Heatmap", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # Panel 4: Predicted Suspicious Mask
    axes[3].imshow(predicted_mask, cmap="gray")
    axes[3].set_title("4. Predicted Suspicious Mask", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # Panel 5: Localization Overlay
    if overlay_img is not None:
        axes[4].imshow(overlay_img)
        axes[4].set_title("5. Suspicious Region Highlight", fontsize=11, fontweight="bold")
    else:
        axes[4].imshow(image_rgb)
        axes[4].set_title("5. Target", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    # Panel 6: Ground Truth or Decision Card
    if has_gt:
        axes[5].imshow(ground_truth_mask, cmap="gray")
        title_str = "6. Ground Truth Mask"
        if metrics:
            title_str += f"\n(IoU: {metrics['iou']:.2f} | F1: {metrics['f1']:.2f})"
        axes[5].set_title(title_str, fontsize=11, fontweight="bold")
    else:
        axes[5].axis("off")
        verdict = decision.get("verdict", "N/A") if decision else "N/A"
        conf = decision.get("confidence_pct", 0) if decision else 0
        risk = decision.get("risk_level", "N/A") if decision else "N/A"
        text_info = f"FORENSIC VERDICT\n\nResult: {verdict}\nConfidence: {conf}%\nRisk Level: {risk}\n"
        if decision and "reasons" in decision:
            text_info += "\nKey Findings:\n" + "\n".join(f"• {r}" for r in decision["reasons"][:3])
        axes[5].text(
            0.1, 0.5, text_info,
            fontsize=11, verticalalignment="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f0f2f6", edgecolor="#4a90e2", alpha=0.9),
        )

    axes[5].axis("off")
    plt.tight_layout()
    return fig


def create_comparison_figure(
    image_rgb: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    metrics: Dict[str, float],
) -> plt.Figure:
    """
    Generate side-by-side localization comparison: Predicted vs Ground Truth vs Confusion.
    """
    confusion = create_confusion_overlay(image_rgb, predicted_mask, ground_truth_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ground_truth_mask, cmap="gray")
    axes[0].set_title("Ground Truth Mask", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(predicted_mask, cmap="gray")
    axes[1].set_title(
        f"Predicted Mask\nIoU: {metrics['iou']:.3f} | F1: {metrics['f1']:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(confusion)
    axes[2].set_title(
        "Pixel Confusion Overlay\nGreen=TP | Red=FP | Blue=FN",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].axis("off")

    plt.tight_layout()
    return fig

