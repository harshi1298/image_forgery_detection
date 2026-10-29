"""
Forensic evaluation metrics: IoU, Precision, Recall, F1-Score, and multi-signal decision engine.
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np


def calculate_iou(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index) between predicted and ground truth binary masks.

    Formula:
        IoU = |Prediction ∩ GroundTruth| / |Prediction ∪ GroundTruth|

    Edge cases:
        - If both masks are empty (authentic image correctly predicted as authentic): IoU = 1.0
        - If union is empty: IoU = 1.0 (both empty) or 0.0
    """
    p_bool = (predicted_mask > 127).astype(bool)
    gt_bool = (ground_truth_mask > 127).astype(bool)

    if p_bool.shape != gt_bool.shape:
        # Resize predicted mask to match ground truth
        p_resized = cv2.resize(
            predicted_mask.astype(np.uint8),
            (gt_bool.shape[1], gt_bool.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        p_bool = p_resized > 127

    intersection = np.logical_and(p_bool, gt_bool).sum()
    union = np.logical_or(p_bool, gt_bool).sum()

    if union == 0:
        return 1.0 if (p_bool.sum() == 0 and gt_bool.sum() == 0) else 0.0

    return float(intersection / union)


def calculate_pixel_metrics(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Compute pixel-level classification metrics:
    - True Positives (TP): predicted forged, actually forged
    - False Positives (FP): predicted forged, actually authentic
    - True Negatives (TN): predicted authentic, actually authentic
    - False Negatives (FN): predicted authentic, actually forged
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
    - Accuracy: (TP + TN) / Total
    """
    p_bool = (predicted_mask > 127).astype(bool)
    gt_bool = (ground_truth_mask > 127).astype(bool)

    if p_bool.shape != gt_bool.shape:
        p_resized = cv2.resize(
            predicted_mask.astype(np.uint8),
            (gt_bool.shape[1], gt_bool.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        p_bool = p_resized > 127

    tp = int(np.logical_and(p_bool, gt_bool).sum())
    fp = int(np.logical_and(p_bool, np.logical_not(gt_bool)).sum())
    fn = int(np.logical_and(np.logical_not(p_bool), gt_bool).sum())
    tn = int(np.logical_and(np.logical_not(p_bool), np.logical_not(gt_bool)).sum())
    total = tp + fp + fn + tn

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if gt_bool.sum() == 0 and p_bool.sum() == 0 else 0.0)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if gt_bool.sum() == 0 else 0.0)

    if (precision + recall) > 0:
        f1 = float(2.0 * (precision * recall) / (precision + recall))
    else:
        f1 = 0.0

    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else (1.0 if gt_bool.sum() == 0 and p_bool.sum() == 0 else 0.0)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "pixel_accuracy": accuracy,
        "gt_forged_pixels": int(gt_bool.sum()),
        "pred_forged_pixels": int(p_bool.sum()),
    }


def evaluate_localization(
    predicted_mask: np.ndarray,
    ground_truth_mask: Optional[np.ndarray],
) -> Optional[Dict[str, float]]:
    """
    Convenience wrapper returning metrics if ground truth mask is present, or None if absent.
    """
    if ground_truth_mask is None:
        return None
    return calculate_pixel_metrics(predicted_mask, ground_truth_mask)


def make_final_decision(
    copy_move_result: Dict[str, Any],
    ela_result: Dict[str, Any],
    predicted_mask: np.ndarray,
    min_inliers: int = 6,
) -> Dict[str, Any]:
    """
    Synthesize copy-move feature matching and ELA compression evidence into a balanced verdict.

    Returns:
        verdict: "POSSIBLY FORGED" or "LIKELY AUTHENTIC"
        confidence: overall confidence percentage
        risk_level: "High", "Medium", "Low"
        reasons: list of evidence statements explaining the rationale
    """
    inliers = len(copy_move_result.get("inlier_matches", []))
    raw_matches = len(copy_move_result.get("all_matches", []))
    cm_detected = copy_move_result.get("is_detected", False)
    cm_conf = copy_move_result.get("confidence", 0.0)

    ela_score = ela_result.get("suspicious_score", 0.0)
    ela_std = ela_result.get("std_error", 0.0)

    mask_area = int((predicted_mask > 127).sum())
    total_pixels = predicted_mask.shape[0] * predicted_mask.shape[1]
    mask_ratio = mask_area / max(1, total_pixels)

    reasons = []

    # Weighted composite score
    # Copy-move carries 70% weight, ELA carries 30% weight (as recommended: ELA is supporting evidence)
    composite_score = 0.70 * cm_conf + 0.30 * ela_score

    if inliers >= min_inliers:
        reasons.append(f"Identified {inliers} geometrically verified copy-move feature pairs (RANSAC verified).")
    elif raw_matches > 0:
        reasons.append(f"Found {raw_matches} candidate feature matches, but only {inliers} satisfied geometric consistency.")
    else:
        reasons.append("No significant duplicated keypoint clusters detected.")

    if ela_score > 0.65:
        reasons.append(f"High compression error variance (ELA anomaly score: {ela_score:.2f}), suggesting different compression histories.")
    elif ela_score > 0.40:
        reasons.append(f"Moderate ELA variance (score: {ela_score:.2f}).")
    else:
        reasons.append(f"Uniform compression error distribution (ELA score: {ela_score:.2f}).")

    if mask_area > 0:
        reasons.append(f"Suspicious region localized covering {mask_area:,} pixels ({mask_ratio * 100:.2f}% of image).")

    is_forged = cm_detected or (inliers >= min_inliers) or (composite_score >= 0.50 and mask_area > 500)

    if is_forged:
        verdict = "POSSIBLY FORGED"
        if composite_score > 0.75 or inliers >= 15:
            risk_level = "High"
        else:
            risk_level = "Medium"
    else:
        verdict = "LIKELY AUTHENTIC"
        risk_level = "Low"

    return {
        "verdict": verdict,
        "confidence": float(composite_score),
        "confidence_pct": round(composite_score * 100, 1),
        "risk_level": risk_level,
        "reasons": reasons,
        "inliers_count": inliers,
        "raw_matches_count": raw_matches,
        "ela_anomaly_score": round(ela_score, 3),
        "mask_pixel_count": mask_area,
    }

