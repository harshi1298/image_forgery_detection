"""
Forensic evaluation metrics: IoU, Precision, Recall, F1-Score, and calibrated decision engine.
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
        - If both masks are empty (authentic image correctly predicted): IoU = 1.0
        - If one is empty and other is non-empty: IoU = 0.0
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

    p_sum = int(p_bool.sum())
    gt_sum = int(gt_bool.sum())

    if p_sum == 0 and gt_sum == 0:
        return 1.0

    intersection = np.logical_and(p_bool, gt_bool).sum()
    union = np.logical_or(p_bool, gt_bool).sum()

    if union == 0:
        return 1.0
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

    # Clean 1-pixel image boundary artifact in CoMoFoD masks
    if gt_bool.shape[0] > 10 and gt_bool.shape[1] > 10:
        gt_bool_clean = gt_bool.copy()
        gt_bool_clean[:, 0] = False
        gt_bool_clean[:, -1] = False
        gt_bool_clean[0, :] = False
        gt_bool_clean[-1, :] = False
        # If boundary had non-zero pixels that were artifacts, use cleaned
        if gt_bool_clean.sum() > 0:
            gt_bool = gt_bool_clean

    if p_bool.shape != gt_bool.shape:
        p_resized = cv2.resize(
            predicted_mask.astype(np.uint8),
            (gt_bool.shape[1], gt_bool.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        p_bool = p_resized > 127

    p_sum = int(p_bool.sum())
    gt_sum = int(gt_bool.sum())

    tp = int(np.logical_and(p_bool, gt_bool).sum())
    fp = int(np.logical_and(p_bool, np.logical_not(gt_bool)).sum())
    fn = int(np.logical_and(np.logical_not(p_bool), gt_bool).sum())
    tn = int(np.logical_and(np.logical_not(p_bool), np.logical_not(gt_bool)).sum())
    total = tp + fp + fn + tn

    # Edge case: authentic image (ground truth has 0 forged pixels)
    if gt_sum == 0:
        if p_sum == 0:
            return {
                "tp": 0, "fp": 0, "fn": 0, "tn": total,
                "precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0,
                "pixel_accuracy": 1.0,
                "gt_forged_pixels": 0, "pred_forged_pixels": 0,
            }
        else:
            return {
                "tp": 0, "fp": fp, "fn": 0, "tn": tn,
                "precision": 0.0, "recall": 1.0, "f1": 0.0, "iou": 0.0,
                "pixel_accuracy": float(tn / total) if total > 0 else 0.0,
                "gt_forged_pixels": 0, "pred_forged_pixels": p_sum,
            }

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    if (precision + recall) > 0:
        f1 = float(2.0 * (precision * recall) / (precision + recall))
    else:
        f1 = 0.0

    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 0.0
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

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
        "gt_forged_pixels": gt_sum,
        "pred_forged_pixels": p_sum,
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
    min_inliers: int = 10,
) -> Dict[str, Any]:
    """
    Synthesize copy-move feature matching and ELA compression evidence into a balanced, calibrated verdict.

    Returns:
        verdict: "POSSIBLY FORGED" or "LIKELY AUTHENTIC"
        confidence_pct: calibrated confidence percentage in the chosen verdict
        forgery_prob: probability of image being forged in [0.0, 1.0]
        authenticity_prob: probability of image being authentic in [0.0, 1.0]
        risk_level: "High", "Medium", "Low"
        reasons: list of forensic evidence statements
    """
    inliers = len(copy_move_result.get("inlier_matches", []))
    raw_matches = len(copy_move_result.get("all_matches", []))
    cm_detected = copy_move_result.get("is_detected", False)
    cm_conf = copy_move_result.get("confidence", 0.0)
    mean_ncc = copy_move_result.get("mean_ncc", 0.0)

    ela_score = ela_result.get("suspicious_score", 0.0)

    mask_area = int((predicted_mask > 127).sum())
    total_pixels = predicted_mask.shape[0] * predicted_mask.shape[1]
    mask_ratio = mask_area / max(1, total_pixels)

    reasons = []

    # Calibrate probability of forgery
    if cm_detected:
        forgery_prob = float(cm_conf)
        # ELA acts as slight reinforcement if high
        if ela_score > 0.40:
            forgery_prob = min(0.99, forgery_prob + 0.05 * ela_score)
    else:
        # Authentic baseline: low forgery probability
        forgery_prob = float(cm_conf)
        if ela_score > 0.50:
            forgery_prob = min(0.40, forgery_prob + 0.15 * ela_score)

    authenticity_prob = float(1.0 - forgery_prob)
    is_forged = cm_detected and (inliers >= min_inliers)

    if inliers >= min_inliers:
        reasons.append(f"Identified {inliers} verified copy-move feature pairs with high patch correlation (mean NCC: {mean_ncc:.2f}).")
    elif raw_matches > 0:
        reasons.append(f"Found {raw_matches} candidate feature matches, but only {inliers} satisfied geometric consistency (threshold: {min_inliers}).")
    else:
        reasons.append("No duplicated feature clusters detected in the image.")

    if ela_score > 0.50:
        reasons.append(f"Localized compression error inconsistency detected (ELA anomaly score: {ela_score:.2f}).")
    else:
        reasons.append(f"Uniform compression error distribution across the image (ELA score: {ela_score:.2f}).")

    if mask_area > 0 and is_forged:
        reasons.append(f"Suspicious region localized covering {mask_area:,} pixels ({mask_ratio * 100:.2f}% of image).")

    if is_forged:
        verdict = "POSSIBLY FORGED"
        decision_confidence = round(forgery_prob * 100.0, 1)
        risk_level = "High" if (forgery_prob > 0.85 or inliers >= 20) else "Medium"
    else:
        verdict = "LIKELY AUTHENTIC"
        decision_confidence = round(authenticity_prob * 100.0, 1)
        risk_level = "Low"

    return {
        "verdict": verdict,
        "confidence": decision_confidence / 100.0,
        "confidence_pct": decision_confidence,
        "forgery_prob": round(forgery_prob, 3),
        "authenticity_prob": round(authenticity_prob, 3),
        "risk_level": risk_level,
        "reasons": reasons,
        "inliers_count": inliers,
        "raw_matches_count": raw_matches,
        "ela_anomaly_score": round(ela_score, 3),
        "mask_pixel_count": mask_area,
    }
