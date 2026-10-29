"""
Pipeline verification test on CoMoFoD sample 001.
"""

from pathlib import Path
from preprocessing import preprocess_pipeline
from detection import detect_copy_move, visualize_copy_move_matches
from ela import compute_ela, generate_ela_mask, visualize_ela_heatmap
from localization import generate_suspicious_mask, create_overlay
from evaluation import calculate_pixel_metrics, make_final_decision
from dataset_loader import CoMoFoDDataset


def main():
    ds = CoMoFoDDataset("dataset")
    print(f"Total discovered CoMoFoD samples: {len(ds.samples)}")

    sample = ds.get_sample("001_F")
    if not sample:
        print("Sample 001_F not found, trying first sample...")
        sample = ds.samples[0]

    print(f"Testing on sample: {sample['name']}")
    forged_rgb, orig_rgb, gt_mask = ds.load_sample_images(sample)

    # 1. Preprocessing
    prep = preprocess_pipeline(forged_rgb, max_dim=800)
    resized_rgb = prep["resized_rgb"]
    filtered_gray = prep["filtered_gray"]
    print(f"Resized image shape: {resized_rgb.shape}")

    # 2. Copy-Move Detection (ORB + Matcher + RANSAC)
    cm_res = detect_copy_move(
        filtered_gray,
        n_features=3500,
        ratio_threshold=0.80,
        min_spatial_distance=35.0,
        max_hamming_distance=60,
    )
    print(f"Keypoints: {len(cm_res['keypoints'])}")
    print(f"Candidate Matches: {len(cm_res['all_matches'])}")
    print(f"RANSAC Inlier Matches: {len(cm_res['inlier_matches'])}")
    print(f"Copy-Move Detected: {cm_res['is_detected']} (Confidence: {cm_res['confidence']:.2f})")

    # 3. Dense Mask Generation
    pred_mask = generate_suspicious_mask(
        image_shape=resized_rgb.shape[:2],
        matched_points_src=cm_res["matched_points_src"],
        matched_points_dst=cm_res["matched_points_dst"],
        point_radius=20,
        dilation_kernel_size=25,
        closing_kernel_size=21,
    )
    pred_pixels = (pred_mask > 127).sum()
    print(f"Predicted suspicious mask area: {pred_pixels} pixels")

    # 4. ELA
    ela_res = compute_ela(resized_rgb, quality=90, scale=15.0)
    print(f"ELA Anomaly Score: {ela_res['suspicious_score']:.2f} (Std Error: {ela_res['std_error']:.2f})")

    # 5. Final Decision
    decision = make_final_decision(cm_res, ela_res, pred_mask)
    print(f"Verdict: {decision['verdict']} (Confidence: {decision['confidence_pct']}%, Risk: {decision['risk_level']})")

    # 6. Evaluation against Ground Truth
    if gt_mask is not None:
        metrics = calculate_pixel_metrics(pred_mask, gt_mask)
        print("\n--- Ground Truth Evaluation Metrics ---")
        print(f"IoU:       {metrics['iou']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"Accuracy:  {metrics['pixel_accuracy']:.4f}")

    print("\nPipeline test completed successfully!")


if __name__ == "__main__":
    main()
