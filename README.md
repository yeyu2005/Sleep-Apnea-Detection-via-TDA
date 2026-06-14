# Topo-ECG: Topological Analysis for Sleep Apnea Detection

## Short Overview
This repository implements the **Topo‑ECG** pipeline for the PhysioNet Apnea‑ECG database. It provides an end-to-end framework including raw ECG ingestion, noise filtering, R‑peak detection, cubic spline RR interpolation, and feature extraction. 

The core novelty is a dual-stream architecture that fuses **Traditional HRV features** (time/frequency/non-linear domains) with **Topological Data Analysis (TDA)** features (Persistence Images extracted via Takens embedding and Vietoris-Rips filtration). Fused features are evaluated using Random Forest, BiLSTM, and Time Series Transformer models.

---

## 1. Environment Setup

It is highly recommended to use Python 3.10 for compatibility with pre-compiled TDA and ECG processing libraries.

### Create and activate a virtual environment
python -m venv .venv

source .venv/bin/activate  

For Windows: .venv\Scripts\activate

### Install dependencies
pip install -U pip (or python -m pip install -U pip)

Then (python-m) pip install PyYAML giotto-tda neurokit2 wfdb scikit-learn pandas numpy scipy matplotlib ptflops torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu




## 2. Data Preparation
Download the PhysioNet Apnea-ECG Database.

Place all .dat, .hea, and .apn files directly into the data/raw/ directory.
Ensure you download both the training set (a01–c03) and the withheld test set (x01–x35), including the released .apn ground-truth labels for the x records. The latter should be downloaded manually, as they are not included in the ZIP file

Required Folder Structure:

data/

├── raw/         # Place all PhysioNet .dat, .hea, and .apn files here

├── interim/     # Auto-generated: filtered ECG signals and detected R-peaks

└── processed/   # Auto-generated: HRV features, TDA images, and robust variants

## 3. The End-to-End Execution Pipeline
Run these commands in strict sequential order from the root of the repository to train the models.

### Step 1: Raw Signal Preprocessing
Extracts the single-lead ECG, applies bandpass (0.5-40Hz) and notch (50Hz) filters, detects R-peaks via neurokit2, and saves interim beat-to-beat sequences.

python -m scripts.run_all_preprocessing


### Step 2: Feature Extraction & Label Alignment
(The Master Script) Slices the interim signals into 1-minute windows, imputes missing beats via cubic spline, extracts traditional HRV features, aligns them with the ground-truth .apn labels, and enforces the official PhysioNet training split.

python scripts/extract_all_features.py


### Step 3: Topological Parameter Tuning
Calculates the optimal Time Delay ($\tau$) via Average Mutual Information and Embedding Dimension ($d$) via False Nearest Neighbors for the Takens Phase Space reconstruction.

Note: To prevent data leakage, this script strictly samples from the training cohort and ignores the x test records.

python scripts/compute_takens_median.py
(This automatically updates configs/tda_config.yaml).

### Step 4: Topological Feature Extraction (Persistence Images)
Uses giotto-tda to compute persistent homology and caches the flattened 2D Persistence Images to disk. (Ensure the tau and dim arguments match the median values output by Step 3).

python -c "from src.features.topological import cache_persistence_images; cache_persistence_images('data/processed/rr_windows', 'data/processed/tda', tau=3, dim=10, grid_size=32, sigma=0.1)"


### Step 5: Run the Machine Learning Ablation Suite
Executes 5-fold GroupKFold cross-validation (grouped by patient) across the 9 experimental conditions (RF/BiLSTM/TST crossed with HRV/TDA/Fusion) to determine the optimal architecture.

python scripts/run_ablation_suite.py --hrv_dir data/processed/hrv --tda_dir data/processed/tda --labels configs/training_labels.csv --outdir experiments/ablations --seq_len 5

Results, metrics (F1, AUC, Sensitivity), and hardware performance logs are saved to experiments/ablations/ablation_summary.json.

### Step 6: Final State-of-the-Art Test Set Benchmark
Trains the final HRV-Only, TDA-Only, and Fusion models on the training cohort and evaluates strictly on the unseen "x" test cohort.

python scripts/evaluate_test_set.py

## 4. Robustness & Wearable Feasibility
To prove the pipeline's viability for deployment on consumer smartwatches and other devices, we provide scripts to benchmark inference latency and test model robustness against motion artifacts.

### 4.1 Edge Inference Benchmarking
Measures the milliseconds required to process a 1-minute sequence and estimates MACs/FLOPs.

python scripts/benchmark_inference.py --arch tst --seq_len 5 --iters 100

### 4.2 Prove Model Resilience
To prove that Topological features act as a stabilizing anchor against noise, we evaluate the finalized models against severely degraded signals on the unseen test set. This script dynamically zero-imputes destroyed 1-minute segments to preserve the chronological timeline, mimicking a "blank signal" from a smartwatch losing skin contact.

First, generate the 50% signal loss dataset:

python scripts/generate_robustness_variants.py --rr_dir data/processed/test_rr_only --out_dir data/processed/robustness_test --noise 0.0 --downsample 2 --tau 3 --dim 10 --grid_size 32

** Please note that test_rr_only directory contains all the processed rr interval for test set, you should copy the "x" series rr interval from rr_window directory to a new folder data/processed/test_rr_only

Then, evaluate the SOTA models against it:

python scripts/evaluate_robustness_test_set.py --noisy_dir data/processed/robustness_test/downsample_2


## 5. Interpretability
The following results can be used to satisfy clinical explainability requirements:

- Attention Weights: During the evaluation of the fusion sequence models (BiLSTM and TST), the attention layer weights are automatically saved to experiments/ablations/fold_{k}/artifacts/. These demonstrate whether the model relied more on traditional HRV or Topological features for a specific minute.

- Raw Persistence Diagrams: You can pass save_diagrams=True to cache_persistence_images() in Step 4 to save the raw _dgm.npy files for plotting the 0D and 1D topological birth/death cycles.

## Running Tests
Run the unit test suite to verify interpolation logic and mathematical boundaries:

pytest -q
