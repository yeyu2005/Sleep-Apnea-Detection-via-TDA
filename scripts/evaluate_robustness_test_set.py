"""Evaluate robustness on the strictly withheld test set for ALL models in one run.

Usage:
  python scripts/evaluate_robustness_test_set.py \
      --noisy_dir data/processed/robustness_test/noise_10.0
"""
import os
import argparse
import numpy as np
import torch
import sys
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.fusion_model import FusionSequenceModel

def load_feature_matrices_robust(hrv_dir, tda_dir, labels_csv):
    """Safely loads features. If signal loss destroyed a window, it zero-imputes it to preserve time."""
    df = pd.read_csv(labels_csv)
    
    # 1. Pre-scan to dynamically find the correct array sizes from a valid file
    expected_h_dim, expected_t_dim = None, None
    hrv_files = {os.path.basename(f): f for f in glob.glob(os.path.join(hrv_dir, "*.npy"))}
    tda_files = {os.path.basename(f): f for f in glob.glob(os.path.join(tda_dir, "*.npy"))}

    for _, row in df.iterrows():
        base = str(row['filename']).replace('.npy', '')
        hm = next((f for k, f in hrv_files.items() if base in k and 'hrv' in k), None)
        tm = next((f for k, f in tda_files.items() if base in k and ('pi' in k or 'tda' in k)), None)
        if hm and tm:
            try:
                h, t = np.load(hm).flatten(), np.load(tm).flatten()
                if expected_h_dim is None and len(h) > 1: expected_h_dim = len(h)
                if expected_t_dim is None and len(t) > 1: expected_t_dim = len(t)
                if expected_h_dim and expected_t_dim: break
            except: continue

    # 2. Load all data, zero-padding the destroyed files
    X_hrv, X_tda, y, groups = [], [], [], []
    missing_count = 0

    for _, row in df.iterrows():
        base = str(row['filename']).replace('.npy', '')
        hrv_match = next((f for k, f in hrv_files.items() if base in k and 'hrv' in k), None)
        tda_match = next((f for k, f in tda_files.items() if base in k and ('pi' in k or 'tda' in k)), None)

        valid = False
        if hrv_match and tda_match:
            try:
                h_arr = np.load(hrv_match).astype(np.float32).flatten()
                t_arr = np.load(tda_match).astype(np.float32).flatten()
                # Check if it survived downsampling math
                if len(h_arr) == expected_h_dim and len(t_arr) == expected_t_dim:
                    X_hrv.append(h_arr)
                    X_tda.append(t_arr)
                    valid = True
            except: pass

        # THE FIX: If the file is corrupted/missing, simulate a blank smartwatch signal
        if not valid:
            missing_count += 1
            X_hrv.append(np.zeros(expected_h_dim, dtype=np.float32))
            X_tda.append(np.zeros(expected_t_dim, dtype=np.float32))

        y.append(int(row['label']))
        groups.append(base.split('_')[0])

    if missing_count > 0:
        print(f"   [!] Zero-imputed {missing_count} corrupted windows to preserve timeline.")

    return np.array(X_hrv), np.array(X_tda), np.array(y), np.array(groups), None

def build_sequences(X_hrv, X_tda, y, groups, seq_len=5):
    """Builds sliding window sequences while respecting patient boundaries."""
    seqs_hrv, seqs_tda, labels, seq_groups = [], [], [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for i in range(len(idx)):
            bh = X_hrv[idx[i:i+seq_len]]
            bt = X_tda[idx[i:i+seq_len]]
            if len(bh) < seq_len:
                pad_len = seq_len - len(bh)
                bh = np.vstack([bh, np.repeat(bh[-1:], pad_len, axis=0)])
                bt = np.vstack([bt, np.repeat(bt[-1:], pad_len, axis=0)])
            seqs_hrv.append(bh)
            seqs_tda.append(bt)
            labels.append(y[idx[min(i + seq_len - 1, len(idx)-1)]])
            seq_groups.append(g)
    return np.stack(seqs_hrv), np.stack(seqs_tda), np.array(labels), np.array(seq_groups)

def calculate_patient_metrics(probs, y_seq, g_seq):
    """Aggregates window predictions to patient level using 10% AHI cutoff."""
    preds = (np.array(probs) >= 0.5).astype(int)
    
    patient_true, patient_pred = [], []
    for p in np.unique(g_seq):
        mask = (g_seq == p)
        true_label = int(np.mean(y_seq[mask]) >= 0.10)
        # THE FIX: Calculate AHI using the binary predictions
        pred_label = int(np.mean(preds[mask]) >= 0.10)
        patient_true.append(true_label)
        patient_pred.append(pred_label)
        
    acc = accuracy_score(patient_true, patient_pred)
    tn, fp, fn, tp = confusion_matrix(patient_true, patient_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return acc, sensitivity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_labels', default='configs/training_labels.csv')
    parser.add_argument('--test_labels', default='configs/test_labels.csv')
    parser.add_argument('--clean_hrv', default='data/processed/hrv')
    parser.add_argument('--tda_dir', default='data/processed/tda')
    parser.add_argument('--noisy_dir', required=True, help="Path to directory containing noisy HRV and TDA features")
    parser.add_argument('--model_dir', default='experiments/final_test', help="Directory containing the 3 .pth files")
    args = parser.parse_args()

    print("--- 1. Loading All Datasets  ---")
    
    # Load Clean Matrices
    X_tr_hrv, X_tr_tda, y_tr, g_tr, _ = load_feature_matrices_robust(args.clean_hrv, args.tda_dir, args.train_labels)
    X_te_hrv_c, X_te_tda_c, y_te, g_te, _ = load_feature_matrices_robust(args.clean_hrv, args.tda_dir, args.test_labels)
    
    # Load NOISY HRV and NOISY TDA
    X_te_hrv_n, X_te_tda_n, y_te_n, g_te_n, _ = load_feature_matrices_robust(args.noisy_dir, args.noisy_dir, args.test_labels)

    seq_len = 5
    print("--- 2. Building Sequences ---")
    X_tr_seq_hrv, X_tr_seq_tda, _, _ = build_sequences(X_tr_hrv, X_tr_tda, y_tr, g_tr, seq_len)
    X_te_seq_hrv_c, X_te_seq_tda_c, y_te_seq, g_te_seq = build_sequences(X_te_hrv_c, X_te_tda_c, y_te, g_te, seq_len)
    
    # FIX: Correctly capture the aligned sequence labels and groups
    X_te_seq_hrv_n, X_te_seq_tda_n, y_te_seq_n, g_te_seq_n = build_sequences(X_te_hrv_n, X_te_tda_n, y_te_n, g_te_n, seq_len)

    print("--- 3. Fitting Scalers ---")
    hrv_dim, tda_dim = X_tr_seq_hrv.shape[2], X_tr_seq_tda.shape[2]
    
    hrv_scaler = StandardScaler().fit(X_tr_seq_hrv.reshape(-1, hrv_dim))
    tda_scaler = MinMaxScaler().fit(X_tr_seq_tda.reshape(-1, tda_dim))

    # Transform Clean Test Set
    X_c_hrv = hrv_scaler.transform(X_te_seq_hrv_c.reshape(-1, hrv_dim)).reshape(X_te_seq_hrv_c.shape)
    X_c_tda = tda_scaler.transform(X_te_seq_tda_c.reshape(-1, tda_dim)).reshape(X_te_seq_tda_c.shape)
    
    # Transform Noisy Test Set 
    X_n_hrv = hrv_scaler.transform(X_te_seq_hrv_n.reshape(-1, hrv_dim)).reshape(X_te_seq_hrv_n.shape)
    X_n_tda = tda_scaler.transform(X_te_seq_tda_n.reshape(-1, tda_dim)).reshape(X_te_seq_tda_n.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {'arch': 'tst', 'nhead': 4, 'nlayers': 2}

    print("\n=======================================================")
    print("TRUE ROBUSTNESS EVALUATION (TEST SET)")
    print("=======================================================")

    modes = [
        ('HRV-Only', 'hrv', 'tst_hrv.pth'),
        ('TDA-Only', 'tda', 'tst_tda.pth'),
        ('Fusion', 'fusion', 'tst_final.pth')
    ]

    for display_name, mode, filename in modes:
        model_path = os.path.join(args.model_dir, filename)
        if not os.path.exists(model_path):
            print(f"\n[!] Missing model for {display_name}: {model_path} not found.")
            continue

        print(f"\nEvaluating: {display_name.upper()}")

        # A. Apply in-memory masking for CLEAN evaluation
        X_eval_c_hrv = np.zeros_like(X_c_hrv) if mode == 'tda' else X_c_hrv
        X_eval_c_tda = np.zeros_like(X_c_tda) if mode == 'hrv' else X_c_tda

        # B. Apply in-memory masking for NOISY evaluation
        X_eval_n_hrv = np.zeros_like(X_n_hrv) if mode == 'tda' else X_n_hrv
        X_eval_n_tda = np.zeros_like(X_n_tda) if mode == 'hrv' else X_n_tda

        # C. Load Model
        model = FusionSequenceModel(hrv_dim, tda_dim, fusion_hidden=64, arch='tst', classifier_kwargs=config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()

        with torch.no_grad():
            # D. Evaluate CLEAN Unseen Data
            logits_clean = model(torch.from_numpy(X_eval_c_hrv).float().to(device), torch.from_numpy(X_eval_c_tda).float().to(device))
            probs_clean = torch.sigmoid(logits_clean).cpu().numpy()
            acc_clean, sens_clean = calculate_patient_metrics(probs_clean.flatten(), y_te_seq, g_te_seq)
            
            # E. Evaluate NOISY Unseen Data
            logits_noisy = model(torch.from_numpy(X_eval_n_hrv).float().to(device), torch.from_numpy(X_eval_n_tda).float().to(device))
            probs_noisy = torch.sigmoid(logits_noisy).cpu().numpy()
            acc_noisy, sens_noisy = calculate_patient_metrics(probs_noisy.flatten(), y_te_seq_n, g_te_seq_n)
            
            print(f"   Clean Data -> Acc: {acc_clean:.3f} | Sens: {sens_clean:.3f}")
            print(f"   Noisy Data -> Acc: {acc_noisy:.3f} | Sens: {sens_noisy:.3f}")
            
            drop = acc_clean - acc_noisy
            if drop > 0:
                print(f"   -> Result: MODEL ACCURACY DROPPED by -{drop:.3f}")
            else:
                print(f"   -> Result: MODEL ANCHORED (Survived safely).")
            
            
if __name__ == '__main__':
    main()