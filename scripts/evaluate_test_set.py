"""Final SOTA Benchmark: Train on all training data, evaluate strictly on the Test Set."""
import os
import json
import numpy as np
import torch
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.train import load_feature_matrices, train_nn, evaluate_preds
from src.models.fusion_model import FusionSequenceModel

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

def main():
    print("Loading datasets...")
    # Load Train Set
    X_tr_hrv, X_tr_tda, y_tr, g_tr, _ = load_feature_matrices('data/processed/hrv', 'data/processed/tda', 'configs/training_labels.csv')
    # Load Test Set
    X_te_hrv, X_te_tda, y_te, g_te, _ = load_feature_matrices('data/processed/hrv', 'data/processed/tda', 'configs/test_labels.csv')

    seq_len = 5
    print("Building sequences...")
    X_tr_seq_hrv, X_tr_seq_tda, y_tr_seq, g_tr_seq = build_sequences(X_tr_hrv, X_tr_tda, y_tr, g_tr, seq_len)
    X_te_seq_hrv, X_te_seq_tda, y_te_seq, g_te_seq = build_sequences(X_te_hrv, X_te_tda, y_te, g_te, seq_len)

    print("Scaling features...")
    hrv_dim, tda_dim = X_tr_seq_hrv.shape[2], X_tr_seq_tda.shape[2]
    
    hrv_scaler = StandardScaler().fit(X_tr_seq_hrv.reshape(-1, hrv_dim))
    tda_scaler = MinMaxScaler().fit(X_tr_seq_tda.reshape(-1, tda_dim))

    X_tr_s_hrv = hrv_scaler.transform(X_tr_seq_hrv.reshape(-1, hrv_dim)).reshape(X_tr_seq_hrv.shape)
    X_tr_s_tda = tda_scaler.transform(X_tr_seq_tda.reshape(-1, tda_dim)).reshape(X_tr_seq_tda.shape)
    
    X_te_s_hrv = hrv_scaler.transform(X_te_seq_hrv.reshape(-1, hrv_dim)).reshape(X_te_seq_hrv.shape)
    X_te_s_tda = tda_scaler.transform(X_te_seq_tda.reshape(-1, tda_dim)).reshape(X_te_seq_tda.shape)

    os.makedirs('experiments/final_test', exist_ok=True)
    config = {'arch': 'tst', 'nhead': 4, 'nlayers': 2, 'epochs': 30, 'batch_size': 64, 'lr': 1e-4}
    
    print("\nCreating internal validation split")
    X_t_hrv, X_v_hrv, X_t_tda, X_v_tda, y_t, y_v = train_test_split(
        X_tr_s_hrv, X_tr_s_tda, y_tr_seq, test_size=0.2, random_state=42
    )

    modes = [
        ('HRV-Only', 'hrv', 'tst_hrv.pth'),
        ('TDA-Only', 'tda', 'tst_tda.pth'),
        ('Fusion', 'fusion', 'tst_final.pth')
    ]

    # --- TRAIN ALL THREE MODELS ---
    for display_name, mode, filename in modes:
        print(f"\n--- Training {display_name} Model ---")
        model_path = os.path.join('experiments/final_test', filename)

        # Apply in-memory masking for the single-modality baselines
        X_train_h = np.zeros_like(X_t_hrv) if mode == 'tda' else X_t_hrv
        X_train_t = np.zeros_like(X_t_tda) if mode == 'hrv' else X_t_tda

        X_val_h = np.zeros_like(X_v_hrv) if mode == 'tda' else X_v_hrv
        X_val_t = np.zeros_like(X_v_tda) if mode == 'hrv' else X_v_tda

        # Train using the internal split
        train_nn((X_train_h, X_train_t), y_t, (X_val_h, X_val_t), y_v, model_path, config)

    print("\n=======================================================")
    print("FINAL TEST SET RESULTS")
    print("=======================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- EVALUATE ALL THREE MODELS ON TEST SET ---
    for display_name, mode, filename in modes:
        print(f"\nEvaluating: {display_name}")
        model_path = os.path.join('experiments/final_test', filename)
        
        X_eval_h = np.zeros_like(X_te_s_hrv) if mode == 'tda' else X_te_s_hrv
        X_eval_t = np.zeros_like(X_te_s_tda) if mode == 'hrv' else X_te_s_tda

        model = FusionSequenceModel(hrv_dim, tda_dim, fusion_hidden=64, arch='tst', classifier_kwargs=config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()

        with torch.no_grad():
            logits = model(torch.from_numpy(X_eval_h).float().to(device), torch.from_numpy(X_eval_t).float().to(device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            y_pred = (probs >= 0.5).astype(int)

        # 1. WINDOW (SEGMENT) LEVEL METRICS
        win_acc = accuracy_score(y_te_seq, y_pred)
        win_f1 = f1_score(y_te_seq, y_pred, zero_division=0)
        win_auc = roc_auc_score(y_te_seq, probs)
        tn_w, fp_w, fn_w, tp_w = confusion_matrix(y_te_seq, y_pred, labels=[0, 1]).ravel()
        win_sens = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0.0

        # 2. PATIENT LEVEL METRICS
        patient_true, patient_pred, patient_probs = [], [], []
        for p in np.unique(g_te_seq):
            mask = (g_te_seq == p)
            # True label: >= 10% of windows are true apnea
            patient_true.append(int(np.mean(y_te_seq[mask]) >= 0.10))
            # Pred label: >= 10% of windows predicted as apnea
            patient_pred.append(int(np.mean(y_pred[mask]) >= 0.10))
            # Patient probability: Average of window probabilities
            patient_probs.append(np.mean(probs[mask]))

        pat_acc = accuracy_score(patient_true, patient_pred)
        pat_f1 = f1_score(patient_true, patient_pred, zero_division=0)
        pat_auc = roc_auc_score(patient_true, patient_probs)
        tn_p, fp_p, fn_p, tp_p = confusion_matrix(patient_true, patient_pred, labels=[0, 1]).ravel()
        pat_sens = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0.0

        # PRINT COMPREHENSIVE STATS
        print(f"   [WINDOW]  Acc: {win_acc:.3f} | Sens: {win_sens:.3f} | F1: {win_f1:.3f} | AUC: {win_auc:.3f}")
        print(f"   [PATIENT] Acc: {pat_acc:.3f} | Sens: {pat_sens:.3f} | F1: {pat_f1:.3f} | AUC: {pat_auc:.3f}")

if __name__ == '__main__':
    main()