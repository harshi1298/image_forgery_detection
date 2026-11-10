"""
Streamlit Web Application: Image Forgery Detection and Localization
Single Image Analysis using Copy-Move Detection + ORB + ELA.
"""

from pathlib import Path
import time
import io
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Import internal modules
from preprocessing import preprocess_pipeline, load_image
from detection import detect_copy_move, visualize_copy_move_matches, draw_keypoints
from ela import compute_ela, generate_ela_mask, visualize_ela_heatmap
from localization import generate_suspicious_mask, create_overlay, mask_to_bounding_boxes
from evaluation import (
    calculate_pixel_metrics,
    make_final_decision,
    create_confusion_overlay,
    plot_detection_summary,
)

# Page configuration
st.set_page_config(
    page_title="Image Forgery Detector | ORB + ELA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .verdict-box-forged {
        background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 18px;
        color: #991B1B;
        margin-bottom: 1rem;
    }
    .verdict-box-auth {
        background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
        border: 2px solid #22C55E;
        border-radius: 12px;
        padding: 18px;
        color: #166534;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.image("https://img.icons8.com/fluency/96/search-in-browser.png", width=64)
st.sidebar.title("Forgery Detector")
st.sidebar.caption("ORB + ELA + Localization Pipeline")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Detection Parameters")

with st.sidebar.expander("ORB & Matcher Settings", expanded=True):
    n_features = st.slider("Max ORB Features", 1000, 5000, 3500, step=250)
    ratio_thresh = st.slider("Lowe's Ratio Threshold", 0.60, 0.90, 0.75, step=0.02)
    min_spatial_dist = st.slider("Min Spatial Distance (px)", 15, 80, 35, step=5)
    max_hamming = st.slider("Max Hamming Distance", 30, 80, 50, step=2)
    ransac_thresh = st.slider("RANSAC Reproj Threshold", 2.0, 15.0, 6.0, step=0.5)
    min_inliers = st.slider("Min Inliers for Detection", 3, 25, 10, step=1)

with st.sidebar.expander("Localization & Mask Settings", expanded=False):
    point_radius = st.slider("Keypoint Splat Radius (px)", 4, 25, 6, step=1)
    morph_kernel = st.slider("Morphological Kernel (px)", 3, 25, 7, step=2)
    use_hulls = st.checkbox("Cluster Convex Hulls", value=False)

with st.sidebar.expander("ELA Settings", expanded=False):
    ela_quality = st.slider("JPEG Compression Quality", 70, 98, 90, step=2)
    ela_scale = st.slider("ELA Difference Amplification", 5.0, 30.0, 15.0, step=1.0)
    auto_scale_ela = st.checkbox("Auto-scale ELA to Peak", value=False)


# ----------------- MAIN INTERFACE: SINGLE IMAGE UPLOAD -----------------
st.markdown('<div class="main-header">Image Forgery Detection & Localization</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload an image to inspect for Copy-Move manipulation and localized compression anomalies (ORB + RANSAC + ELA).</div>',
    unsafe_allow_html=True,
)

col_up1, col_up2 = st.columns(2)
with col_up1:
    uploaded_file = st.file_uploader(
        "Upload Suspicious Image (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
        help="Upload the image you want to investigate for forgery.",
    )
with col_up2:
    uploaded_mask = st.file_uploader(
        "Optional: Upload Ground-Truth Binary Mask (PNG, JPG):",
        type=["png", "jpg"],
        help="Optional: Upload corresponding ground-truth mask to evaluate IoU, Precision, Recall, and F1-score.",
    )

image_rgb = None
gt_mask = None
image_stem = "custom_image"

if uploaded_file is not None:
    image_rgb = load_image(uploaded_file.read())
    image_stem = Path(uploaded_file.name).stem

if uploaded_mask is not None:
    mask_raw = load_image(uploaded_mask.read())
    gt_mask = cv2.cvtColor(mask_raw, cv2.COLOR_RGB2GRAY)
    _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

if image_rgb is None:
    st.info("👆 Please upload an image using the file uploader above to begin forensic analysis.")
else:
    # Run Analysis Button
    run_btn = st.button("🚀 Run Comprehensive Forgery Analysis", type="primary", use_container_width=True)

    if run_btn or "last_analyzed" in st.session_state:
        st.session_state["last_analyzed"] = True

        with st.spinner("Analyzing keypoints, patch correlations, and compression residuals..."):
            start_time = time.time()

            # 1. Preprocessing
            prep = preprocess_pipeline(image_rgb, max_dim=800)
            resized_rgb = prep["resized_rgb"]
            filtered_gray = prep["filtered_gray"]
            h_proc, w_proc = resized_rgb.shape[:2]

            # 2. Copy-Move Detection
            cm_result = detect_copy_move(
                gray_image=filtered_gray,
                n_features=n_features,
                ratio_threshold=ratio_thresh,
                min_spatial_distance=min_spatial_dist,
                max_hamming_distance=max_hamming,
                reproj_threshold=ransac_thresh,
                min_inliers_for_forgery=min_inliers,
            )

            # 3. Dense Suspicious Region Mask Generation
            pred_mask = generate_suspicious_mask(
                image_shape=(h_proc, w_proc),
                matched_points_src=cm_result["matched_points_src"],
                matched_points_dst=cm_result["matched_points_dst"],
                is_detected=cm_result["is_detected"],
                point_radius=point_radius,
                dilation_kernel_size=morph_kernel,
                closing_kernel_size=morph_kernel,
                use_convex_hull_clusters=use_hulls,
            )

            # 4. Error Level Analysis
            ela_result = compute_ela(
                image_rgb=resized_rgb,
                quality=ela_quality,
                scale=ela_scale,
                auto_scale=auto_scale_ela,
            )
            ela_heatmap = visualize_ela_heatmap(ela_result)
            ela_binary_mask = generate_ela_mask(ela_result)

            # 5. Visual Overlays
            matches_vis = visualize_copy_move_matches(
                image_rgb=resized_rgb,
                keypoints=cm_result["keypoints"],
                inlier_matches=cm_result["inlier_matches"],
                raw_matches=cm_result["all_matches"],
                show_outliers=False,
            )
            overlay_vis = create_overlay(resized_rgb, pred_mask, alpha=0.45, color=(255, 40, 40))

            # 6. Final Decision & Evaluation
            decision = make_final_decision(
                copy_move_result=cm_result,
                ela_result=ela_result,
                predicted_mask=pred_mask,
                min_inliers=min_inliers,
            )

            metrics = None
            if gt_mask is not None:
                gt_aligned = cv2.resize(gt_mask, (w_proc, h_proc), interpolation=cv2.INTER_NEAREST)
                metrics = calculate_pixel_metrics(pred_mask, gt_aligned)

            elapsed = time.time() - start_time

        # ----------------- VERDICT & METRIC CARDS -----------------
        is_forged = decision["verdict"] == "POSSIBLY FORGED"
        box_cls = "verdict-box-forged" if is_forged else "verdict-box-auth"
        icon = "⚠️" if is_forged else "✅"

        st.markdown(
            f"""
            <div class="{box_cls}">
                <h3 style="margin:0; font-size:1.4rem;">{icon} Decision: {decision['verdict']}</h3>
                <p style="margin:6px 0 0 0; font-size:1.0rem;">
                    <strong>Decision Confidence:</strong> {decision['confidence_pct']}% &nbsp;|&nbsp; 
                    <strong>Forgery Probability:</strong> {decision.get('forgery_prob', 0.0)*100:.1f}% &nbsp;|&nbsp; 
                    <strong>Authenticity:</strong> {decision.get('authenticity_prob', 0.0)*100:.1f}% &nbsp;|&nbsp; 
                    <strong>Risk Level:</strong> {decision['risk_level']} &nbsp;|&nbsp; 
                    <strong>Computation Time:</strong> {elapsed:.2f}s
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metric Columns
        if metrics:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("RANSAC Inliers", f"{len(cm_result['inlier_matches'])}", f"of {len(cm_result['all_matches'])} candidates")
            c2.metric("IoU (Localization)", f"{metrics['iou']:.3f}", "Intersection/Union")
            c3.metric("F1-Score", f"{metrics['f1']:.3f}", "Harmonic Mean")
            c4.metric("Precision", f"{metrics['precision']:.3f}", "TP / (TP+FP)")
            c5.metric("Recall", f"{metrics['recall']:.3f}", "TP / (TP+FN)")
            c6.metric("ELA Score", f"{decision['ela_anomaly_score']:.2f}", f"Std: {ela_result['std_error']:.1f}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ORB Keypoints", f"{len(cm_result['keypoints']):,}")
            c2.metric("Candidate Matches", f"{len(cm_result['all_matches']):,}")
            c3.metric("RANSAC Inlier Pairs", f"{len(cm_result['inlier_matches']):,}")
            c4.metric("ELA Anomaly Score", f"{decision['ela_anomaly_score']:.2f}")

        # ----------------- TABS FOR VISUALIZATION -----------------
        t_main, t_compare, t_orb, t_ela, t_report = st.tabs(
            [
                "🖼️ Primary Visual Results",
                "🎯 Ground Truth & IoU Evaluation",
                "🔍 ORB Matcher Deep Dive",
                "⚡ ELA Compression Analysis",
                "📝 Forensic Report & Export",
            ]
        )

        with t_main:
            st.subheader("Multi-Signal Detection & Suspicious Region Localization")
            r1_c1, r1_c2 = st.columns(2)
            with r1_c1:
                st.image(resized_rgb, caption="1. Input Image", use_container_width=True)
            with r1_c2:
                st.image(
                    matches_vis,
                    caption=f"2. ORB Feature Matches ({len(cm_result['inlier_matches'])} Geometric Inliers, Mean NCC: {cm_result['mean_ncc']:.2f})",
                    use_container_width=True,
                )

            r2_c1, r2_c2 = st.columns(2)
            with r2_c1:
                st.image(ela_heatmap, caption=f"3. ELA Heatmap (Compression Quality: {ela_quality})", use_container_width=True)
            with r2_c2:
                st.image(overlay_vis, caption="4. Suspicious Region Highlight (Localized Mask)", use_container_width=True)

        with t_compare:
            st.subheader("Localization Mask vs Ground-Truth Comparison")
            if gt_mask is not None:
                gt_aligned = cv2.resize(gt_mask, (w_proc, h_proc), interpolation=cv2.INTER_NEAREST)
                confusion_overlay = create_confusion_overlay(resized_rgb, pred_mask, gt_aligned)

                col_gt1, col_gt2, col_gt3 = st.columns(3)
                with col_gt1:
                    st.image(pred_mask, caption=f"Predicted Mask ({int((pred_mask > 127).sum()):,} px)", use_container_width=True)
                with col_gt2:
                    st.image(gt_aligned, caption=f"Ground Truth Mask ({int((gt_aligned > 127).sum()):,} px)", use_container_width=True)
                with col_gt3:
                    st.image(confusion_overlay, caption="Confusion: Green=TP, Red=FP, Blue=FN", use_container_width=True)

                st.markdown("#### Detailed Pixel-Level Evaluation Matrix")
                m_df = pd.DataFrame(
                    [
                        {"Metric": "Intersection over Union (IoU)", "Value": f"{metrics['iou']:.4f}", "Interpretation": "Overlap degree between predicted and actual forged regions"},
                        {"Metric": "F1-Score", "Value": f"{metrics['f1']:.4f}", "Interpretation": "Balanced harmonic mean of precision and recall"},
                        {"Metric": "Precision", "Value": f"{metrics['precision']:.4f}", "Interpretation": "Fraction of predicted forged pixels that are genuinely forged"},
                        {"Metric": "Recall (Sensitivity)", "Value": f"{metrics['recall']:.4f}", "Interpretation": "Fraction of genuinely forged pixels successfully detected"},
                        {"Metric": "Overall Pixel Accuracy", "Value": f"{metrics['pixel_accuracy']:.4f}", "Interpretation": "(TP + TN) / Total Image Pixels"},
                        {"Metric": "True Positive Pixels (TP)", "Value": f"{metrics['tp']:,}", "Interpretation": "Correctly detected manipulated pixels"},
                        {"Metric": "False Positive Pixels (FP)", "Value": f"{metrics['fp']:,}", "Interpretation": "Authentic pixels mistakenly flagged"},
                        {"Metric": "False Negative Pixels (FN)", "Value": f"{metrics['fn']:,}", "Interpretation": "Manipulated pixels missed by detector"},
                    ]
                )
                st.table(m_df)
            else:
                st.info("No ground-truth mask uploaded for this image. To compute IoU, Precision, and F1-score, upload a corresponding binary ground-truth mask.")
                st.image(pred_mask, caption=f"Predicted Suspicious Mask ({int((pred_mask > 127).sum()):,} px)", width=450)

        with t_orb:
            st.subheader("ORB Keypoint Extraction and Inlier Analysis")
            kp_img = draw_keypoints(resized_rgb, cm_result["keypoints"], color=(0, 255, 120), rich=False)

            k_col1, k_col2 = st.columns(2)
            with k_col1:
                st.image(kp_img, caption=f"All Extracted ORB Keypoints ({len(cm_result['keypoints'])})", use_container_width=True)
            with k_col2:
                st.image(matches_vis, caption=f"RANSAC Inlier Matches ({len(cm_result['inlier_matches'])})", use_container_width=True)

            if cm_result["inlier_matches"]:
                st.markdown("#### Sample Matched Inlier Keypoint Coordinates")
                inlier_rows = []
                for idx, m in enumerate(cm_result["inlier_matches"][:15]):
                    p1 = cm_result["keypoints"][m[0]].pt
                    p2 = cm_result["keypoints"][m[1]].pt
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    ncc_val = f"{m[4]:.2f}" if len(m) >= 5 else "N/A"
                    inlier_rows.append({
                        "Pair": f"#{idx+1}",
                        "Source (x, y)": f"({p1[0]:.1f}, {p1[1]:.1f})",
                        "Destination (x, y)": f"({p2[0]:.1f}, {p2[1]:.1f})",
                        "Displacement (dx, dy)": f"({dx:+.1f}, {dy:+.1f})",
                        "Patch NCC": ncc_val,
                        "Hamming Dist": int(m[2]),
                        "Spatial Dist": f"{m[3]:.1f} px",
                    })
                st.dataframe(pd.DataFrame(inlier_rows), use_container_width=True)

        with t_ela:
            st.subheader("Error Level Analysis (ELA) Compression Residuals")
            st.markdown(
                """
                Error Level Analysis highlights compression discrepancies across the image by resaving at a specified JPEG quality
                and calculating the absolute difference. Uniform surfaces exhibit consistent noise, while spliced or copied regions
                from different origins often display abnormal error amplitudes.
                """
            )
            e_col1, e_col2, e_col3 = st.columns(3)
            with e_col1:
                st.image(ela_result["recompressed_rgb"], caption=f"Recompressed JPEG (Q={ela_quality})", use_container_width=True)
            with e_col2:
                st.image(ela_result["ela_image"], caption=f"Amplified Difference (Scale: {ela_result['scale_used']:.1f}x)", use_container_width=True)
            with e_col3:
                st.image(ela_heatmap, caption="JET Color Heatmap", use_container_width=True)

            fig_hist, ax_hist = plt.subplots(figsize=(8, 2.5))
            raw_diff_flat = ela_result["raw_diff"].flatten()
            ax_hist.hist(raw_diff_flat, bins=50, color="#3B82F6", alpha=0.8, edgecolor="black")
            ax_hist.set_title("Distribution of Compression Residual Errors", fontsize=10)
            ax_hist.set_xlabel("Absolute Difference")
            ax_hist.set_ylabel("Pixel Count")
            st.pyplot(fig_hist)

        with t_report:
            st.subheader("Forensic Inspection Report")

            report_text = f"""================================================================================
IMAGE FORGERY INVESTIGATION REPORT
================================================================================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Input Source: {uploaded_file.name if uploaded_file is not None else 'Custom Image'}
Dimensions Analyzed: {w_proc} x {h_proc} px

1. EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
FINAL VERDICT: {decision['verdict']}
DECISION CONFIDENCE: {decision['confidence_pct']}%
FORGERY PROBABILITY: {decision.get('forgery_prob', 0.0)*100:.1f}%
AUTHENTICITY PROBABILITY: {decision.get('authenticity_prob', 0.0)*100:.1f}%
RISK LEVEL: {decision['risk_level']}

FINDINGS:
{chr(10).join('- ' + r for r in decision['reasons'])}

2. TECHNICAL EVIDENCE BREAKDOWN
--------------------------------------------------------------------------------
A. COPY-MOVE FEATURE ANALYSIS:
   - ORB Features Detected: {len(cm_result['keypoints'])}
   - Candidate Matches Found: {len(cm_result['all_matches'])}
   - RANSAC Verified Inliers: {len(cm_result['inlier_matches'])}
   - Mean Inlier Patch NCC: {cm_result['mean_ncc']:.2f}
   - Copy-Move Behavior Detected: {cm_result['is_detected']}

B. ERROR LEVEL ANALYSIS (ELA):
   - JPEG Quality Factor: {ela_quality}
   - Mean Compression Error: {ela_result['mean_error']:.2f}
   - Std Deviation of Error: {ela_result['std_error']:.2f}
   - 95th Percentile Error: {ela_result['p95_error']:.2f}
   - ELA Anomaly Score: {ela_result['suspicious_score']:.2f}

C. SUSPICIOUS REGION LOCALIZATION:
   - Suspicious Pixels Flagged: {decision['mask_pixel_count']:,}
   - Approximate Area Coverage: {(decision['mask_pixel_count'] / (w_proc * h_proc)) * 100:.2f}%
"""
            if metrics:
                report_text += f"""
3. GROUND TRUTH VERIFICATION METRICS
--------------------------------------------------------------------------------
   - Intersection over Union (IoU): {metrics['iou']:.4f}
   - F1-Score: {metrics['f1']:.4f}
   - Precision: {metrics['precision']:.4f}
   - Recall: {metrics['recall']:.4f}
   - Pixel Accuracy: {metrics['pixel_accuracy']:.4f}
   - True Positives (TP): {metrics['tp']:,} px
   - False Positives (FP): {metrics['fp']:,} px
   - False Negatives (FN): {metrics['fn']:,} px
"""
            st.text_area("Forensic Summary Text", report_text, height=320)

            st.download_button(
                label="📥 Download Forensic Report (.txt)",
                data=report_text,
                file_name=f"forgery_report_{image_stem}.txt",
                mime="text/plain",
            )

            summary_fig = plot_detection_summary(
                image_rgb=resized_rgb,
                matches_img=matches_vis,
                ela_img=ela_result["ela_image"],
                ela_heatmap=ela_heatmap,
                predicted_mask=pred_mask,
                ground_truth_mask=cv2.resize(gt_mask, (w_proc, h_proc), interpolation=cv2.INTER_NEAREST) if gt_mask is not None else None,
                overlay_img=overlay_vis,
                metrics=metrics,
                decision=decision,
            )
            buf = io.BytesIO()
            summary_fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                label="🖼️ Download Academic Summary Figure (.png)",
                data=buf,
                file_name=f"forgery_summary_{image_stem}.png",
                mime="image/png",
            )
