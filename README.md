# Image Forgery Detection and Localization
### Using Copy-Move Detection + ORB + Error Level Analysis (ELA) + CoMoFoD Ground-Truth Evaluation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Dataset: CoMoFoD](https://img.shields.io/badge/Dataset-CoMoFoD-orange.svg)](https://comofod.wordpress.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision and digital forensics application developed for a **2-member, 1-week mini-project**. The system detects image manipulation (specifically Copy-Move forgery) and localizes suspicious regions at the pixel level using **Oriented FAST and Rotated BRIEF (ORB)**, **RANSAC geometric verification**, and **Error Level Analysis (ELA)**, evaluating localization accuracy against **CoMoFoD ground-truth masks** using **Intersection over Union (IoU)**, **Precision**, **Recall**, and **F1-Score**.

---

## 📑 Table of Contents
1. [Core Features](#-core-features)
2. [Problem Statement & Scope](#-problem-statement--scope)
3. [System Architecture](#-system-architecture)
4. [Project Directory Structure](#-project-directory-structure)
5. [Installation & Virtual Environment Setup](#-installation--virtual-environment-setup)
6. [How to Run](#-how-to-run)
7. [Streamlit Web Interface Overview](#-streamlit-web-interface-overview)
8. [Evaluation Metrics & Mathematics](#-evaluation-metrics--mathematics)
9. [Benchmark Results on CoMoFoD Dataset](#-benchmark-results-on-comofod-dataset)
10. [2-Member Team Division of Work](#-2-member-team-division-of-work)
11. [Complete Technical Documentation](#-complete-technical-documentation)

---

## 🌟 Core Features

- **Intra-Image ORB Keypoint Matching:** Multi-scale corner detection with orientation compensation and 256-bit steered binary descriptors (BRIEF).
- **False-Match Suppression:** Eliminates trivial adjacent-texture matches using Euclidean spatial distance thresholding ($d_{\text{spatial}} \ge 35\text{ px}$) and Lowe's ratio test ($r \le 0.78$).
- **RANSAC Geometric Verification:** Fits partial affine (translation, rotation, scale) and homography models to identify genuine cloned patches while rejecting random feature pairings.
- **Dense Suspicious Region Mask Synthesis:** Converts sparse keypoint coordinates into continuous binary masks using circular splatting, elliptical dilation, morphological closing, and local DBSCAN cluster convex hulls.
- **Error Level Analysis (ELA):** Independent compression residual analysis highlighting JPEG quality factor discrepancies and local error variance.
- **Ground-Truth Evaluation:** Quantitative comparison against CoMoFoD dataset ground-truth masks using IoU, Precision, Recall, F1-Score, and a color-coded confusion overlay (Green = TP, Red = FP, Blue = FN).
- **Interactive Streamlit Web Dashboard (`app.py`):**
  - Single Image Analysis mode with pre-loaded CoMoFoD samples or custom uploads.
  - Real-time parameter tuning (feature counts, thresholds, radii).
  - Batch dataset benchmarking with progress tracking, bar charts, and CSV export.
  - One-click export of official Forensic Reports (`.txt`) and Academic Summary Figures (`.png`).

---

## 🎯 Problem Statement & Scope

Digital image tampering has become pervasive with modern photo-editing software. This system solves two distinct forensic tasks:

```text
                             Input Image
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
         Task A: Detection               Task B: Localization
       "Authentic or Forged?"          "Where is the manipulation?"
                  │                               │
                  ▼                               ▼
       Verdict: "POSSIBLY FORGED"        Dense Predicted Mask
           or "LIKELY AUTHENTIC"       vs Ground-Truth (IoU, F1)
```

---

## 📐 System Architecture

```text
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
                     Dense Suspicious Mask
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

## 📁 Project Directory Structure

```text
image-forgery-project/
│
├── app.py                      # Interactive Streamlit Web Dashboard
├── verify_setup.py             # CLI pipeline verification script
├── prepare_dataset.py          # CoMoFoD archive extraction & organizer
├── dataset_loader.py           # Dataset discovery & synthetic generator
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview & quickstart
├── DOCUMENTATION.md            # Exhaustive technical documentation & viva guide
│
├── preprocessing/              # Module 1: Preprocessing
│   ├── __init__.py
│   └── preprocess.py           # Loading, resizing, grayscale, noise filtering
│
├── detection/                  # Module 2: Keypoint Detection & Matching
│   ├── __init__.py
│   ├── orb_detector.py         # ORB keypoint & descriptor extraction
│   ├── matcher.py              # Intra-image KNN matching, ratio test, RANSAC
│   └── copy_move.py            # End-to-end copy-move pipeline & visualizer
│
├── ela/                        # Module 3: Error Level Analysis
│   ├── __init__.py
│   └── ela.py                  # JPEG recompression, difference calculation & heatmaps
│
├── localization/               # Module 4: Region Mask Localization
│   ├── __init__.py
│   └── mask.py                 # Dense mask synthesis (dilation, closing, DBSCAN hulls)
│
├── evaluation/                 # Module 5: Metrics & Decision Engine
│   ├── __init__.py
│   ├── metrics.py              # IoU, Precision, Recall, F1, decision engine
│   └── visualization.py        # Color-coded confusion overlay & academic summary plots
│
└── dataset/                    # CoMoFoD Dataset Samples
    ├── original/               # Authentic reference images (*_O.png)
    ├── forged/                 # Manipulated copy-move images (*_F.png)
    └── masks/                  # Ground-truth binary masks (*_B.png)
```

---

## 💻 Installation & Virtual Environment Setup

### 1. Prerequisites
- **Python 3.10+** (Developed and tested with Python 3.12)
- Virtual Environment tool (`venv`)

### 2. Activate the Virtual Environment
The virtual environment is already set up in the workspace:

```powershell
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt):
.\venv\Scripts\activate.bat

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Extract CoMoFoD Dataset (Optional / First-time setup)
If `comofod_small.rar` is located in the workspace root, extract base samples using:
```bash
python prepare_dataset.py
```
*(Extracts and organizes samples into `dataset/original`, `dataset/forged`, and `dataset/masks`).*

---

## 🚀 How to Run

### Step 1: Run the Terminal Verification Script
Test the complete computer vision pipeline and calculate metrics from the CLI:
```bash
python verify_setup.py
```

### Step 2: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🖥️ Streamlit Web Interface Overview

The Streamlit UI provides three comprehensive modes:

### 1. 🔬 Single Image Analysis
- Choose from pre-loaded CoMoFoD dataset samples (e.g., `006_F`, `003_F`, `005_F`, `001_F`) or upload a custom image with an optional ground-truth mask.
- Dynamic verdict banner (**"POSSIBLY FORGED"** in Red or **"LIKELY AUTHENTIC"** in Green).
- Metric cards displaying RANSAC Inliers, IoU, F1-Score, Precision, Recall, and ELA Anomaly Score.
- **Interactive Visual Tabs:**
  - **Primary Visual Results:** 4-panel view showing Input Image, ORB Matches, ELA Heatmap, and Suspicious Region Overlay.
  - **Ground Truth & IoU Evaluation:** Predicted Mask vs. Ground Truth Mask alongside a color-coded confusion overlay (Green = TP, Red = FP, Blue = FN).
  - **ORB Matcher Deep Dive:** All extracted keypoints, inlier vectors, and coordinates table.
  - **ELA Compression Analysis:** Recompressed JPEG, difference image, and error residual histogram.
  - **Forensic Report & Export:** One-click download of a text report (`.txt`) and publication-ready multi-panel figure (`.png`).

### 2. 📊 Batch CoMoFoD Benchmark
- Run evaluation across multiple CoMoFoD dataset samples simultaneously.
- Real-time progress bar and summary cards (Detection Rate, Mean IoU, Mean F1, Mean Precision, Mean Recall).
- Per-sample benchmark table with interactive sorting and CSV export.

### 3. 📐 System Architecture & Methodology
- In-app technical reference with data flow diagrams, mathematical equations, and viva talking points for your project presentation.

---

## 📊 Evaluation Metrics & Mathematics

### 1. Intersection over Union (IoU / Jaccard Index)
$$\text{IoU} = \frac{|\text{Prediction} \cap \text{GroundTruth}|}{|\text{Prediction} \cup \text{GroundTruth}|} = \frac{TP}{TP + FP + FN}$$

### 2. Precision
$$\text{Precision} = \frac{TP}{TP + FP}$$

### 3. Recall (Sensitivity)
$$\text{Recall} = \frac{TP}{TP + FN}$$

### 4. F1-Score
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

### 5. Error Level Analysis (ELA)
$$\Delta_{\text{ELA}}(x, y) = |I_{\text{original}}(x, y) - I_{\text{recompressed}}(x, y)|$$

---

## 📈 Benchmark Results on CoMoFoD Dataset

Evaluated on official CoMoFoD samples under default settings:

| Sample ID | ORB Keypoints | Candidate Matches | RANSAC Inliers | Final Verdict | IoU | F1-Score | Recall | Precision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`006_F`** | 2,944 | 805 | 470 | **POSSIBLY FORGED** | **0.658** | **0.794** | 0.808 | 0.780 |
| **`003_F`** | 3,341 | 241 | 83 | **POSSIBLY FORGED** | **0.349** | **0.518** | 0.957 | 0.355 |
| **`005_F`** | 2,921 | 714 | 396 | **POSSIBLY FORGED** | **0.354** | **0.523** | 0.832 | 0.381 |
| **`004_F`** | 2,917 | 314 | 143 | **POSSIBLY FORGED** | **0.133** | **0.235** | 1.000 | 0.133 |
| **`001_F`** | 3,417 | 172 | 47 | **POSSIBLY FORGED** | **0.102** | **0.185** | 0.756 | 0.106 |

---

## 👥 2-Member Team Division of Work

| Role | Member Responsibilities | Day-by-Day Focus |
| :--- | :--- | :--- |
| **Member 1: Computer Vision & Algorithm Lead** | • OpenCV preprocessing pipeline<br>• ORB feature detection & descriptor tuning<br>• Intra-image BFMatcher with spatial distance filtering<br>• RANSAC geometric verification & displacement clustering<br>• Dense mask synthesis (morphological closing/dilation/hulls) | Day 1: Dataset setup<br>Day 2: Preprocessing pipeline<br>Day 3: ORB implementation<br>Day 4: Feature matching & RANSAC<br>Day 5: Mask synthesis & optimization |
| **Member 2: Application, Forensics & Evaluation Lead** | • Error Level Analysis (ELA) implementation<br>• Streamlit Web UI development<br>• Metric calculations (IoU, Precision, Recall, F1)<br>• Color-coded confusion visualizer<br>• Batch benchmarking & CSV export<br>• Project report & viva documentation | Day 1: Dataset organization<br>Day 2: ELA pipeline<br>Day 3: Metric evaluation functions<br>Day 4: Streamlit web application<br>Day 5: Visualizations & batch mode<br>Day 6-7: Report, PPT & Viva preparation |

---

## 📚 Complete Technical Documentation

A comprehensive, thesis-grade technical manual is available in **[`DOCUMENTATION.md`](DOCUMENTATION.md)**.

It includes:
- Complete theoretical derivations (FAST test on Bresenham circles, intensity centroids, steered BRIEF, Hamming POPCNT).
- Mathematical formulation of RANSAC affine models and ELA compression residuals.
- Module-by-module API specifications with parameter types and returns.
- Hyperparameter tuning guide with tested ranges and trade-off analysis.
- **15 Viva Voce Questions & Answers** with detailed technical responses for project examinations.
