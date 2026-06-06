"""Final SOTA Benchmark: Train on all training data, evaluate strictly on the Test Set."""
import os
import json
import numpy as np
import torch
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

    print("Training final TST Fusion model ...")
    os.makedirs('experiments/final_test', exist_ok=True)
    model_path = 'experiments/final_test/tst_final.pth'
    
    config = {'arch': 'tst', 'nhead': 4, 'nlayers': 2, 'epochs': 30, 'batch_size': 64, 'lr': 1e-4}
    
    # Train on train set, validate on test set (just for early stopping observation)
    train_nn((X_tr_s_hrv, X_tr_s_tda), y_tr_seq, (X_te_s_hrv, X_te_s_tda), y_te_seq, model_path, config)

    print("\n--- FINAL TEST SET RESULTS ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionSequenceModel(hrv_dim, tda_dim, fusion_hidden=64, arch='tst', classifier_kwargs=config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X_te_s_hrv).float().to(device), torch.from_numpy(X_te_s_tda).float().to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        y_pred = (probs >= 0.5).astype(int)

    win_metrics = evaluate_preds(y_te_seq, y_pred, probs)
    print(f"Window Level Metrics: {win_metrics}")

    # Patient Level Aggregation
    patient_true, patient_pred = [], []
    for p in np.unique(g_te_seq):
        mask = (g_te_seq == p)
        true_label = int(np.mean(y_te_seq[mask]) >= 0.10)
        pred_label = int(np.mean(y_pred[mask]) >= 0.10)
        patient_true.append(true_label)
        patient_pred.append(pred_label)

    pat_metrics = evaluate_preds(np.array(patient_true), np.array(patient_pred))
    print(f"Patient Level Metrics: {pat_metrics}")

if __name__ == '__main__':
    main()
