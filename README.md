# Topo-ECG: Topological Analysis for Sleep Apnea Detection

## Short Overview
This repository implements the **Topo‑ECG** pipeline for the PhysioNet Apnea‑ECG database. It provides an end-to-end framework including raw ECG ingestion, noise filtering, R‑peak detection, cubic spline RR interpolation, and feature extraction. 

The core novelty is a dual-stream architecture that fuses **Traditional HRV features** (time/frequency/non-linear domains) with **Topological Data Analysis (TDA)** features (Persistence Images extracted via Takens embedding and Vietoris-Rips filtration). Fused features are evaluated using Random Forest, BiLSTM, and Time Series Transformer models via strict patient-level GroupKFold cross-validation.

---

## 1. Environment Setup

It is highly recommended to use Python 3.10 for compatibility with pre-compiled TDA and ECG processing libraries.

### Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

### Install dependencies
pip install -U pip
pip install giotto-tda neurokit2 wfdb torch torchvision torchaudio scikit-learn pandas numpy scipy matplotlib


## 2. Data Preparation
Download the PhysioNet Apnea-ECG Database.

Place all .dat, .hea, and .apn files directly into the data/raw/ directory.
file structure: 
data/raw
data/interim
data/processed
(Note: The official test set records x01 through x35 do not have public .apn files provided in the standard zip. They must be evaluated separately or omitted during cross-validation).



## 3. The End-to-End Execution Pipeline
Run these commands in strict sequential order from the root of the repository to train the models.

### Step 1: Raw Signal Preprocessing
Extracts the single-lead ECG, applies bandpass (0.5-40Hz) and notch (50Hz) filters, detects R-peaks via neurokit2, and saves interim beat-to-beat sequences.

python -m scripts.run_all_preprocessing


### Step 2: Feature Extraction & Label Alignment
(The Master Script) Slices the interim signals into 1-minute windows, imputes missing beats via cubic spline, extracts traditional HRV features, aligns them with the ground-truth .apn labels, and enforces the official PhysioNet training split.

python scripts/extract_all_features.py


### Step 3: Topological Parameter Tuning
Calculates the optimal Time Delay ($\tau$) via Average Mutual Information and Embedding Dimension ($d$) via False Nearest Neighbors for the Takens Phase Space reconstruction. Updates configs/tda_config.yaml.

python scripts/compute_takens_median.py


### Step 4: Topological Feature Extraction (Persistence Images)
Uses giotto-tda to compute persistent homology and caches the flattened 2D Persistence Images to disk. (Ensure the tau and dim arguments match the median values output by Step 3).

python -c "from src.features.topological import cache_persistence_images; cache_persistence_images('data/processed/rr_windows', 'data/processed/tda', tau=3, dim=10, grid_size=32, sigma=0.1)"


### Step 5: Run the Machine Learning Ablation Suite
Executes 5-fold GroupKFold cross-validation (grouped by patient) across all 9 experimental conditions (RF/BiLSTM/TST crossed with HRV/TDA/Fusion).

python scripts/run_ablation_suite.py --hrv_dir data/processed/hrv --tda_dir data/processed/tda --labels configs/training_labels.csv --outdir experiments/ablations --seq_len 5

Results, metrics (F1, AUC, Sensitivity), and hardware performance logs are saved to experiments/ablations/ablation_summary.json.


## 4. Robustness & Wearable Feasibility
To test the pipeline's viability for deployment on consumer smartwatches, we provide scripts to generate degraded datasets mirroring motion artifacts and low sampling rates.

### Raw ECG Degradation:
Injects Gaussian noise (mV) directly into the filtered ECG and downsamples the Hz rate before re-running R-peak detection.

python scripts/generate_ecg_robustness.py --interim data/interim --out data/processed/robustness --noise 0.1 0.2 --downsample 50 25


### RR Interval Degradation:
Directly degrades the extracted RR intervals.

python scripts/generate_robustness_variants.py --rr_dir data/processed/rr_windows --out_dir data/processed/robustness --noise 10 20 --downsample 2 3




### Analyze Drop in Performance:
python scripts/analyze_robustness.py --orig_hrv data/processed/hrv --rob_dir data/processed/robustness --out experiments/ablations/robustness_summary.csv



## 5. Interpretability
To satisfy clinical explainability requirements:

Attention Weights: During the evaluation of the fusion sequence models (BiLSTM and TST), the attention layer weights are automatically saved to experiments/ablations/fold_{k}/artifacts/. These demonstrate whether the model relied more on traditional HRV or Topological features for a specific minute.

Raw Persistence Diagrams: You can pass save_diagrams=True to cache_persistence_images() in Step 4 to save the raw _dgm.npy files for plotting the 0D and 1D topological birth/death cycles.

## Running Tests
Run the unit test suite to verify interpolation logic and mathematical boundaries:
