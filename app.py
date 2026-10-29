"""
Streamlit Web Application: Image Forgery Detection and Localization
Using Copy-Move Detection + ORB + ELA + CoMoFoD Ground Truth Verification.
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
from dataset_loader import CoMoFoDDataset, generate_synthetic_sample

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


@st.cache_resource
def get_dataset():
    ds = CoMoFoDDataset("dataset")
    if len(ds.samples) == 0:
        # Create synthetic if none exists
        generate_synthetic_sample()
        ds = CoMoFoDDataset("dataset")
    return ds


ds = get_dataset()

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://img.icons8.com/fluency/96/search-in-browser.png", width=64)
st.sidebar.title("Forgery Detector")
st.sidebar.caption("ORB + ELA + Localization Mini-Project")

app_mode = st.sidebar.radio(
    "Navigation Mode",
    [
        "🔬 Single Image Analysis",
        "📊 Batch CoMoFoD Benchmark",
        "📐 System Architecture & Methodology",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Algorithm Parameters")

with st.sidebar.expander("ORB & Matcher Settings", expanded=False):
    n_features = st.slider("Max ORB Features", 1000, 5000, 3500, step=250)
    ratio_thresh = st.slider("Lowe's Ratio Threshold", 0.60, 0.90, 0.78, step=0.02)
    min_spatial_dist = st.slider("Min Spatial Distance (px)", 15, 80, 35, step=5)
    max_hamming = st.slider("Max Hamming Distance", 30, 80, 60, step=2)
    ransac_thresh = st.slider("RANSAC Reproj Threshold", 2.0, 15.0, 8.0, step=0.5)
    min_inliers = st.slider("Min Inliers for Detection", 3, 20, 6, step=1)

with st.sidebar.expander("Localization & Mask Settings", expanded=False):
    point_radius = st.slider("Keypoint Splat Radius (px)", 4, 25, 10, step=1)
    morph_kernel = st.slider("Morphological Kernel (px)", 5, 25, 11, step=2)
    use_hulls = st.checkbox("Cluster Convex Hulls", value=True)

with st.sidebar.expander("ELA Settings", expanded=False):
    ela_quality = st.slider("JPEG Compression Quality", 70, 98, 90, step=2)
    ela_scale = st.slider("ELA Difference Amplification", 5.0, 30.0, 15.0, step=1.0)
    auto_scale_ela = st.checkbox("Auto-scale ELA to Peak", value=False)


# ----------------- MODE 1: SINGLE IMAGE ANALYSIS -----------------
if app_mode == "🔬 Single Image Analysis":
    st.markdown('<div class="main-header">Image Forgery Detection & Suspicious Region Localization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Combines Intra-Image ORB Keypoint Matching, Geometric RANSAC, Error Level Analysis (ELA), and Ground-Truth Evaluation.</div>', unsafe_allow_html=True)

    input_source = st.radio(
        "Select Input Source:",
        ["Select Sample from CoMoFoD Dataset", "Upload Custom Image"],
        horizontal=True,
    )

    image_rgb = None
    gt_mask = None
    sample_info_str = ""

    if input_source == "Select Sample from CoMoFoD Dataset":
        sample_names = ds.get_sample_names()
        if not sample_names:
            st.warning("No samples discovered in dataset directory. Creating a synthetic sample.")
            generate_synthetic_sample()
            ds = get_dataset()
            sample_names = ds.get_sample_names()

        # Provide a friendly dropdown with notable samples first
        notable_defaults = ["006_F", "003_F", "005_F", "001_F", "004_F", "002_F", "010_F"]
        default_idx = 0
        for nd in notable_defaults:
            if nd in sample_names:
                default_idx = sample_names.index(nd)
                break

        selected_name = st.selectbox("Choose a CoMoFoD Sample:", sample_names, index=default_idx)
        sample = ds.get_sample(selected_name)
        if sample:
            forged_img, orig_img, mask_img = ds.load_sample_images(sample)
            image_rgb = forged_img
            gt_mask = mask_img
            sample_info_str = f"Sample ID: **{sample['id']}** | File: `{sample['name']}`"
            if sample.get("original_path"):
                sample_info_str += " | Original Reference Available"
            if sample.get("mask_path"):
                sample_info_str += " | Ground Truth Mask Loaded"

    else:
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            uploaded_file = st.file_uploader("Upload Suspicious Image (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image_rgb = load_image(uploaded_file.read())
        with col_up2:
            uploaded_mask = st.file_uploader("Optional: Upload Ground-Truth Binary Mask (PNG, JPG):", type=["png", "jpg"])
            if uploaded_mask is not None:
                mask_raw = load_image(uploaded_mask.read())
                gt_mask = cv2.cvtColor(mask_raw, cv2.COLOR_RGB2GRAY)
                _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

    if image_rgb is not None:
        if sample_info_str:
            st.info(sample_info_str)

        run_btn = st.button("🚀 Run Comprehensive Forgery Analysis", type="primary", use_container_width=True)

        if run_btn or "last_analyzed" in st.session_state:
            st.session_state["last_analyzed"] = True

            with st.spinner("Analyzing image keypoints, geometric displacement, and compression residuals..."):
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

                # 5. Overlays and Visualizations
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
                    # Align gt_mask size to resized_rgb
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
                        <strong>Confidence:</strong> {decision['confidence_pct']}% &nbsp;|&nbsp; 
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
                    st.image(matches_vis, caption=f"2. ORB Feature Matches ({len(cm_result['inlier_matches'])} Geometric Inliers)", use_container_width=True)

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
                    st.info("No ground-truth mask provided for this image. Upload a ground truth mask or choose a CoMoFoD sample to calculate IoU and F1-score.")
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
                        inlier_rows.append({
                            "Pair": f"#{idx+1}",
                            "Source (x, y)": f"({p1[0]:.1f}, {p1[1]:.1f})",
                            "Destination (x, y)": f"({p2[0]:.1f}, {p2[1]:.1f})",
                            "Displacement (dx, dy)": f"({dx:+.1f}, {dy:+.1f})",
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

                # Histogram of ELA residuals
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
Input Source: {selected_name if input_source == 'Select Sample from CoMoFoD Dataset' else 'Custom Upload'}
Dimensions Analyzed: {w_proc} x {h_proc} px

1. EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
FINAL VERDICT: {decision['verdict']}
CONFIDENCE: {decision['confidence_pct']}%
RISK LEVEL: {decision['risk_level']}

FINDINGS:
{chr(10).join('- ' + r for r in decision['reasons'])}

2. TECHNICAL EVIDENCE BREAKDOWN
--------------------------------------------------------------------------------
A. COPY-MOVE FEATURE ANALYSIS:
   - ORB Features Detected: {len(cm_result['keypoints'])}
   - Candidate Matches Found: {len(cm_result['all_matches'])}
   - RANSAC Verified Inliers: {len(cm_result['inlier_matches'])}
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
3. GROUND TRUTH VERIFICATION METRICS (CoMoFoD)
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
                    file_name=f"forgery_report_{selected_name if input_source == 'Select Sample from CoMoFoD Dataset' else 'custom'}.txt",
                    mime="text/plain",
                )

                # Exportable All-In-One Figure
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
                    file_name=f"forgery_summary_{selected_name if input_source == 'Select Sample from CoMoFoD Dataset' else 'custom'}.png",
                    mime="image/png",
                )


# ----------------- MODE 2: BATCH BENCHMARK -----------------
elif app_mode == "📊 Batch CoMoFoD Benchmark":
    st.markdown('<div class="main-header">Batch CoMoFoD Dataset Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate detection accuracy, IoU, Precision, Recall, and F1 across multiple dataset samples.</div>', unsafe_allow_html=True)

    available_samples = [s for s in ds.samples if s["name"].endswith("_F") and s.get("mask_path")]

    st.write(f"Discovered **{len(available_samples)}** benchmarkable CoMoFoD copy-move samples with ground-truth masks.")

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        num_to_eval = st.slider("Number of samples to evaluate in batch:", 3, min(30, len(available_samples)), 10)
    with col_b2:
        start_batch_btn = st.button("▶️ Run Batch Evaluation", type="primary")

    if start_batch_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()

        results_list = []
        subset = available_samples[:num_to_eval]

        for i, s in enumerate(subset):
            status_text.text(f"Evaluating ({i+1}/{num_to_eval}): {s['name']}...")
            forged_rgb, orig_rgb, gt_mask = ds.load_sample_images(s)
            if forged_rgb is None or gt_mask is None:
                continue

            prep = preprocess_pipeline(forged_rgb, max_dim=800)
            resized_rgb = prep["resized_rgb"]
            filtered_gray = prep["filtered_gray"]
            h_proc, w_proc = resized_rgb.shape[:2]

            cm_res = detect_copy_move(
                gray_image=filtered_gray,
                n_features=n_features,
                ratio_threshold=ratio_thresh,
                min_spatial_distance=min_spatial_dist,
                max_hamming_distance=max_hamming,
                reproj_threshold=ransac_thresh,
                min_inliers_for_forgery=min_inliers,
            )

            pred_mask = generate_suspicious_mask(
                image_shape=(h_proc, w_proc),
                matched_points_src=cm_res["matched_points_src"],
                matched_points_dst=cm_res["matched_points_dst"],
                point_radius=point_radius,
                dilation_kernel_size=morph_kernel,
                closing_kernel_size=morph_kernel,
                use_convex_hull_clusters=use_hulls,
            )

            gt_aligned = cv2.resize(gt_mask, (w_proc, h_proc), interpolation=cv2.INTER_NEAREST)
            m = calculate_pixel_metrics(pred_mask, gt_aligned)

            results_list.append({
                "Sample": s["name"],
                "Keypoints": len(cm_res["keypoints"]),
                "Inliers": len(cm_res["inlier_matches"]),
                "Detected": "Yes" if cm_res["is_detected"] else "No",
                "IoU": round(m["iou"], 3),
                "F1": round(m["f1"], 3),
                "Precision": round(m["precision"], 3),
                "Recall": round(m["recall"], 3),
                "Pixel Acc": round(m["pixel_accuracy"], 3),
            })

            progress_bar.progress((i + 1) / num_to_eval)

        status_text.success(f"Batch evaluation completed on {len(results_list)} samples!")
        df_results = pd.DataFrame(results_list)

        # Summary averages
        avg_iou = df_results["IoU"].mean()
        avg_f1 = df_results["F1"].mean()
        avg_prec = df_results["Precision"].mean()
        avg_rec = df_results["Recall"].mean()
        det_rate = (df_results["Detected"] == "Yes").mean() * 100.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Detection Rate", f"{det_rate:.1f}%")
        m2.metric("Mean IoU", f"{avg_iou:.3f}")
        m3.metric("Mean F1-Score", f"{avg_f1:.3f}")
        m4.metric("Mean Precision", f"{avg_prec:.3f}")
        m5.metric("Mean Recall", f"{avg_rec:.3f}")

        st.markdown("### Per-Sample Benchmark Table")
        st.dataframe(df_results, use_container_width=True)

        # Bar chart comparison
        fig_bar, ax_bar = plt.subplots(figsize=(12, 4))
        x = np.arange(len(df_results))
        width = 0.35
        ax_bar.bar(x - width / 2, df_results["IoU"], width, label="IoU", color="#3B82F6")
        ax_bar.bar(x + width / 2, df_results["F1"], width, label="F1-Score", color="#10B981")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(df_results["Sample"], rotation=45, ha="right")
        ax_bar.set_ylabel("Score")
        ax_bar.set_title("IoU and F1-Score Across CoMoFoD Samples", fontsize=11, fontweight="bold")
        ax_bar.legend()
        ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig_bar)

        csv = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Benchmark Results (.csv)",
            data=csv,
            file_name="comofod_batch_benchmark.csv",
            mime="text/csv",
        )


# ----------------- MODE 3: ARCHITECTURE & METHODOLOGY -----------------
else:
    st.markdown('<div class="main-header">System Architecture & Theoretical Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Presentation & Viva Reference for Image Forgery Detection & Localization Mini-Project</div>', unsafe_allow_html=True)

    st.markdown(
        r"""
        ### 1. Project Overview & Problem Formulation
        Digital image forgery has become pervasive with photo-editing software. This system addresses two core challenges:
        1. **Task A — Detection:** Determine whether an image is authentic or has been manipulated.
        2. **Task B — Localization:** Pinpoint the exact spatial region that underwent duplication or tampering.

        ---

        ### 2. Multi-Signal Architecture
        ```
                                 ┌───────────────┐
                                 │  Input Image  │
                                 └───────┬───────┘
                                         │
                                  Preprocessing
                             (Resize, Gray, Denoise)
                                         │
                          ┌──────────────┴──────────────┐
                          │                             │
                          ▼                             ▼
                     ORB Detection                 ELA Analysis
                          │                             │
                     Keypoints                     JPEG Re-save
                    & Descriptors                 & Error Diff
                          │                             │
                  Intra-Image Matching                  │
                 (KNN + Lowe's Ratio)                   │
                          │                             │
                  RANSAC Verification                   │
               (Affine / Homography Model)              │
                          │                             │
                          ▼                             ▼
                  Copy-Move Evidence            Compression Evidence
                          │                             │
                          └──────────────┬──────────────┘
                                         ▼
                            Suspicious Mask Generation
                            (Dilation, Closing, Hulls)
                                         │
                                         ▼
                                   Final Decision
                          ("POSSIBLY FORGED" / "AUTHENTIC")
                                         │
                            ┌────────────┴────────────┐
                            ▼                         ▼
                       Visual Overlay            IoU / F1 Metrics
                     (Predicted Region)       (Ground-Truth Mask)
        ```

        ---

        ### 3. Key Methodological Modules

        #### A. Oriented FAST and Rotated BRIEF (ORB)
        - FAST algorithm detects corner keypoints at multi-scale pyramid levels.
        - Intensity centroid computes the corner orientation angle $\theta$:
          $$m_{pq} = \sum_{x,y} x^p y^q I(x, y), \quad \theta = \text{atan2}(m_{01}, m_{10})$$
        - Steered BRIEF descriptor generates a rotation-invariant 256-bit binary string.
        - Binary representation enables rapid Hamming distance computation via CPU POPCNT instruction.

        #### B. Intra-Image Feature Matching & Bad Match Filtering
        - Query each descriptor against the entire image's descriptors using `BFMatcher(cv2.NORM_HAMMING)`.
        - **Spatial Distance Filter:** Exclude self-matches and neighbors within a spatial radius ($d_{\text{spatial}} < 35\text{ px}$) to eliminate trivial adjacent texture matches.
        - **Lowe's Ratio Test:** Compare best non-local match $d_1$ to secondary match $d_2$:
          $$\frac{d_1}{d_2} < \tau_{\text{ratio}} \quad (\approx 0.78)$$

        #### C. Geometric RANSAC Verification
        - Random Sample Consensus fits an affine transformation matrix between source and destination points:
          $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a & -b \\ b & a \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$
        - Inliers must satisfy reprojection error: $\|p' - \mathcal{T}(p)\| < \epsilon_{\text{reproj}}$.

        #### D. Suspicious Region Mask Generation
        - Converts sparse inlier points into dense suspicious regions using:
          1. Point radius splatting
          2. Elliptical dilation: $\mathcal{M} \oplus \mathcal{B}$
          3. Morphological closing: $(\mathcal{M} \oplus \mathcal{B}) \ominus \mathcal{B}$
          4. Local DBSCAN cluster convex hulls
          5. Connected component filtering to remove isolated noise specks.

        #### E. Error Level Analysis (ELA)
        - Recompresses the image at known JPEG quality factor (typically 90%):
          $$\Delta_{\text{ELA}}(x, y) = |I(x, y) - I_{\text{recomp}}(x, y)|$$
        - Scaled for visualization: $I_{\text{vis}} = \min(255, \Delta \cdot \alpha)$.
        - High local variance or divergent error patterns signal spliced or modified content.

        #### F. Ground-Truth Quantitative Metrics
        - **Intersection over Union (IoU):**
          $$\text{IoU} = \frac{|P \cap GT|}{|P \cup GT|}$$
        - **Precision:** $\frac{TP}{TP + FP}$
        - **Recall:** $\frac{TP}{TP + FN}$
        - **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

        ---

        ### 4. Team Division of Labor (2-Member Scope)
        | Member | Focus Areas | Responsibilities |
        | :--- | :--- | :--- |
        | **Member 1 (Algorithm Lead)** | Core CV Pipeline | Preprocessing, ORB detection, intra-image matching, RANSAC geometric filtering, morphological mask synthesis. |
        | **Member 2 (Application & Evaluation)** | Forensics & Interface | ELA analysis, Streamlit UI, IoU/F1 calculation, visual confusion overlays, batch benchmarking, forensic reporting. |
        """
    )
