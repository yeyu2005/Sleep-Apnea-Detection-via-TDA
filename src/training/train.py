"""Training and cross-validation utilities for Topo-ECG.

Features expected:
- HRV .npy files in `hrv_dir` (per-window), named consistently with `tda_dir` files.
- TDA .npy files in `tda_dir` (flattened Persistence Images).
- `labels_csv` mapping filename -> label and filename -> group (patient_id)

Usage:
    python -m src.training.train --hrv_dir data/processed/hrv --tda_dir data/processed/tda \
        --labels configs/training_labels.csv --outdir experiments/run1 --model rf
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset



def load_feature_matrices(hrv_dir, tda_dir, labels_csv):
    labels_df = pd.read_csv(labels_csv)
    # expect columns: filename,label,group
    assert {'filename','label','group'}.issubset(labels_df.columns), 'labels_csv must contain filename,label,group'

    X_hrv = []
    X_tda = []
    y = []
    groups = []
    fnames = []

    for _, row in labels_df.iterrows():
        raw_fname = str(row['filename'])
        # allow CSV to contain either base names (e.g., a01_win_0001) or full filenames
        base = os.path.splitext(raw_fname)[0]
        # construct expected feature filenames
        hrv_candidates = [
            os.path.join(hrv_dir, f"{base}_hrv.npy"),
            os.path.join(hrv_dir, f"{base}.npy"),
            os.path.join(hrv_dir, raw_fname),
        ]
        tda_candidates = [
            os.path.join(tda_dir, f"{base}_pi.npy"),
            os.path.join(tda_dir, f"{base}_tda.npy"),
            os.path.join(tda_dir, f"{base}.npy"),
            os.path.join(tda_dir, raw_fname),
        ]

        hrv_path = next((p for p in hrv_candidates if os.path.exists(p)), None)
        tda_path = next((p for p in tda_candidates if os.path.exists(p)), None)

        if hrv_path is None or tda_path is None:
            raise FileNotFoundError(f"Missing feature files for base '{base}'. Tried HRV: {hrv_candidates}, TDA: {tda_candidates}")

        X_hrv.append(np.load(hrv_path))
        X_tda.append(np.load(tda_path))
        y.append(int(row['label']))
        groups.append(row['group'])
        fnames.append(base)

    X_hrv = np.vstack(X_hrv)
    X_tda = np.vstack(X_tda)
    
    # 1. Forward-fill HRV to preserve physiological continuity
    # 2. Backward-fill as a fallback (in case the very first minute of the night is noisy)
    X_hrv = pd.DataFrame(X_hrv).ffill().bfill().values
    
    # TDA Persistence Images use strictly 0.0 for empty topological space
    X_tda = np.nan_to_num(X_tda, nan=0.0)
    
    y = np.array(y)
    groups = np.array(groups)
    return X_hrv, X_tda, y, groups, fnames


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_model(arch, input_dim, config):
    arch = (arch or 'mlp').lower()
    if arch == 'mlp':
        return SimpleMLP(input_dim, hidden=config.get('hidden', 128))
    if arch == 'bilstm':
        try:
            from src.models.bilstm import BiLSTMClassifier
            return BiLSTMClassifier(input_dim, hidden_dim=config.get('hidden', 128))
            print("get bilstm")
        except Exception:
            return SimpleMLP(input_dim, hidden=config.get('hidden', 128))
            print("fail, usingn mlp")
    if arch == 'tst':
        try:
            from src.models.tst import TransformerClassifier
            return TransformerClassifier(input_dim, nhead=config.get('nhead', 4), num_layers=config.get('nlayers', 2))
            print("get tst")
        except Exception:
            return SimpleMLP(input_dim, hidden=config.get('hidden', 128))
            print("fail, usingn mlp")
    return SimpleMLP(input_dim, hidden=config.get('hidden', 128))


def train_nn(X_train, y_train, X_val, y_val, outpath, config):
    if torch is None:
        raise RuntimeError('PyTorch is required for NN training')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # support tuple input (HRV_seq, TDA_seq) for fusion sequence models
    is_sequence = False
    use_fusion = False
    if isinstance(X_train, tuple):
        use_fusion = True
        X_train_hrv, X_train_tda = X_train
        X_val_hrv, X_val_tda = X_val
        is_sequence = True
        seq_len, hrv_dim = X_train_hrv.shape[1], X_train_hrv.shape[2]
        tda_dim = X_train_tda.shape[2]
        # build fusion sequence model
        from src.models.fusion_model import FusionSequenceModel
        model = FusionSequenceModel(hrv_dim, tda_dim, fusion_hidden=config.get('fusion_hidden', 64), arch=config.get('arch', 'bilstm'), classifier_kwargs=config).to(device)
    else:
        # detect sequence input (non-fusion)
        is_sequence = X_train.ndim == 3
        if is_sequence:
            seq_len, feat_dim = X_train.shape[1], X_train.shape[2]
            input_dim = feat_dim
        else:
            input_dim = X_train.shape[1]
        model = get_model(config.get('arch', 'mlp'), input_dim, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=config.get('wd', 1e-4))
    # weighted BCE loss: allow pos_weight override from config (computed per-fold)
    if config and 'pos_weight' in config:
        pos_weight = float(config['pos_weight'])
    else:
        pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    if use_fusion:
        train_ds = TensorDataset(torch.from_numpy(X_train_hrv).float(), torch.from_numpy(X_train_tda).float(), torch.from_numpy(y_train).float())
        val_ds = TensorDataset(torch.from_numpy(X_val_hrv).float(), torch.from_numpy(X_val_tda).float(), torch.from_numpy(y_val).float())
    else:
        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 64), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.get('batch_size', 64), shuffle=False)

    best_val_loss = float('inf')
    patience = config.get('patience', 5)
    wait = 0

    for epoch in range(config.get('epochs', 50)):
        model.train()
        for batch in train_loader:
            if use_fusion:
                hrv_b, tda_b, yb = batch
                hrv_b = hrv_b.to(device)
                tda_b = tda_b.to(device)
                yb = yb.to(device)
                logits = model(hrv_b, tda_b)
            else:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                # if sequence input but model expects flat input (e.g., mlp), use last timestep
                if xb.ndim == 3 and xb.shape[1] != 1:
                    try:
                        logits = model(xb)
                    except Exception:
                        logits = model(xb[:, -1, :])
                else:
                    logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_losses = []
        preds = []
        trues = []
        with torch.no_grad():
            for batch in val_loader:
                if use_fusion:
                    hrv_b, tda_b, yb = batch
                    hrv_b = hrv_b.to(device)
                    tda_b = tda_b.to(device)
                    yb = yb.to(device)
                    logits = model(hrv_b, tda_b)
                else:
                    xb, yb = batch
                    xb = xb.to(device)
                    yb = yb.to(device)
                    try:
                        logits = model(xb)
                    except Exception:
                        logits = model(xb[:, -1, :])
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend((probs >= 0.5).astype(int).tolist())
                trues.extend(yb.cpu().numpy().astype(int).tolist())

        mean_val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
        f1 = f1_score(trues, preds)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            wait = 0
            torch.save(model.state_dict(), outpath)
        else:
            wait += 1
            if wait >= patience:
                break

    return outpath


def evaluate_preds(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float('nan')
    except Exception:
        auc = float('nan')
        print("auc nan")
    sens = recall_score(y_true, y_pred)
    return {'accuracy': acc, 'f1': f1, 'auc': auc, 'sensitivity': sens}


def run_group_cv(hrv_dir, tda_dir, labels_csv, outdir, model_type='rf', n_splits=5, config=None):
    X_hrv, X_tda, y, groups, fnames = load_feature_matrices(hrv_dir, tda_dir, labels_csv)
    gkf = GroupKFold(n_splits=n_splits)
    os.makedirs(outdir, exist_ok=True)
    fold_metrics_window = []
    fold_metrics_patient = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_hrv, y, groups)):
        y_train, y_val = y[train_idx], y[val_idx]

        arch = (config or {}).get('arch', 'mlp')
        seq_len = (config or {}).get('seq_len', None)
        is_sequence_model = arch in ('bilstm', 'tst') and seq_len is not None

        if is_sequence_model:
            # Updated to track and return sequence-level patient groups
            def build_sequences_pair(X_hrv_all, X_tda_all, y_all, groups_all, sel_idx, seq_len, stride=1):
                seqs_hrv = []
                seqs_tda = []
                labels = []
                seq_groups = [] # Track patient IDs for the newly formed sequences
                grp_ids = np.unique(groups_all[sel_idx])
                for g in grp_ids:
                    idx_all = np.where(groups_all == g)[0]
                    idx = idx_all[np.isin(idx_all, sel_idx)]
                    idx = np.sort(idx)
                    Xg_hrv = X_hrv_all[idx]
                    Xg_tda = X_tda_all[idx]
                    yg = y_all[idx]
                    for i in range(0, max(1, len(idx)), stride):
                        bh = Xg_hrv[i:i+seq_len]
                        bt = Xg_tda[i:i+seq_len]
                        if bh.shape[0] == 0:
                            continue
                        if bh.shape[0] != seq_len:
                            pad_len = seq_len - bh.shape[0]
                            ph = np.repeat(bh[-1:].copy(), pad_len, axis=0)
                            pt = np.repeat(bt[-1:].copy(), pad_len, axis=0)
                            bh = np.vstack([bh, ph])
                            bt = np.vstack([bt, pt])
                        seqs_hrv.append(bh)
                        seqs_tda.append(bt)
                        
                        label_mode = (config or {}).get('label_mode', 'last')
                        lbl_idx = min(i + seq_len // 2, len(yg)-1) if label_mode == 'center' else min(i + seq_len - 1, len(yg)-1)
                        labels.append(int(yg[lbl_idx]))
                        seq_groups.append(g) # Map sequence to patient ID
                        
                if len(seqs_hrv) == 0:
                    return np.zeros((0, seq_len, X_hrv_all.shape[1])), np.zeros((0, seq_len, X_tda_all.shape[1])), np.array([]), np.array([])
                return np.stack(seqs_hrv), np.stack(seqs_tda), np.array(labels), np.array(seq_groups)

            seq_stride = (config or {}).get('seq_stride', 1)
            X_train_seq_hrv, X_train_seq_tda, y_train_seq, train_seq_groups = build_sequences_pair(X_hrv, X_tda, y, groups, train_idx, seq_len, stride=seq_stride)
            X_val_seq_hrv, X_val_seq_tda, y_val_seq, val_seq_groups = build_sequences_pair(X_hrv, X_tda, y, groups, val_idx, seq_len, stride=seq_stride)

            hrv_dim = X_train_seq_hrv.shape[2]
            tda_dim = X_train_seq_tda.shape[2]
            hrv_flat_train = X_train_seq_hrv.reshape(-1, hrv_dim)
            tda_flat_train = X_train_seq_tda.reshape(-1, tda_dim)
            
            hrv_scaler = StandardScaler()
            tda_scaler = MinMaxScaler(feature_range=(0, 1))
            hrv_scaler.fit(hrv_flat_train)
            tda_scaler.fit(tda_flat_train)

            X_train_s_hrv = hrv_scaler.transform(hrv_flat_train).reshape(X_train_seq_hrv.shape)
            X_train_s_tda = tda_scaler.transform(tda_flat_train).reshape(X_train_seq_tda.shape)

            hrv_flat_val = X_val_seq_hrv.reshape(-1, hrv_dim)
            tda_flat_val = X_val_seq_tda.reshape(-1, tda_dim)
            X_val_s_hrv = hrv_scaler.transform(hrv_flat_val).reshape(X_val_seq_hrv.shape)
            X_val_s_tda = tda_scaler.transform(tda_flat_val).reshape(X_val_seq_tda.shape)
        else:
            X_hrv_train, X_tda_train = X_hrv[train_idx], X_tda[train_idx]
            X_hrv_val, X_tda_val = X_hrv[val_idx], X_tda[val_idx]

            hrv_scaler = StandardScaler()
            tda_scaler = MinMaxScaler(feature_range=(0, 1))
            hrv_scaler.fit(X_hrv_train)
            tda_scaler.fit(X_tda_train)

            X_train_s = np.hstack([hrv_scaler.transform(X_hrv_train), tda_scaler.transform(X_tda_train)])
            X_val_s = np.hstack([hrv_scaler.transform(X_hrv_val), tda_scaler.transform(X_tda_val)])

        fold_out = os.path.join(outdir, f'fold_{fold}')
        os.makedirs(fold_out, exist_ok=True)

        classes = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

        if model_type == 'rf':
            if is_sequence_model:
                X_train_flat = np.concatenate([X_train_s_hrv, X_train_s_tda], axis=2).reshape(X_train_s_hrv.shape[0], -1)
                y_train_use = y_train_seq
                X_val_flat = np.concatenate([X_val_s_hrv, X_val_s_tda], axis=2).reshape(X_val_s_hrv.shape[0], -1)
            else:
                X_train_flat, y_train_use = X_train_s, y_train
                X_val_flat = X_val_s
            
            clf = RandomForestClassifier(class_weight=class_weight_dict, n_estimators=100)
            clf.fit(X_train_flat, y_train_use)
            y_pred = clf.predict(X_val_flat)
            y_prob = clf.predict_proba(X_val_flat)[:,1] if hasattr(clf, "predict_proba") else None
            joblib.dump({'model': clf, 'hrv_scaler': hrv_scaler, 'tda_scaler': tda_scaler}, os.path.join(fold_out, 'rf.joblib'))
        else:
            nn_out = os.path.join(fold_out, 'nn.pth')
            if is_sequence_model:
                n_neg = int(np.sum(y_train_seq == 0))
                n_pos = int(np.sum(y_train_seq == 1))
                pos_weight = float(n_neg / max(1, n_pos))
                c = dict(config or {})
                c['pos_weight'] = pos_weight
                train_nn((X_train_s_hrv, X_train_s_tda), y_train_seq, (X_val_s_hrv, X_val_s_tda), y_val_seq, nn_out, c)
            else:
                n_neg = int(np.sum(y_train == 0))
                n_pos = int(np.sum(y_train == 1))
                pos_weight = float(n_neg / max(1, n_pos))
                c = dict(config or {})
                c['pos_weight'] = pos_weight
                train_nn(X_train_s, y_train, X_val_s, y_val, nn_out, c)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if is_sequence_model:
                from src.models.fusion_model import FusionSequenceModel
                model = FusionSequenceModel(X_train_s_hrv.shape[2], X_train_s_tda.shape[2], fusion_hidden=config.get('fusion_hidden',64), arch=config.get('arch','bilstm'), classifier_kwargs=config)
                model.load_state_dict(torch.load(nn_out, map_location=device))
                model.to(device).eval()
                with torch.no_grad():
                    logits = model(torch.from_numpy(X_val_s_hrv).float().to(device), torch.from_numpy(X_val_s_tda).float().to(device))
                    probs = torch.sigmoid(logits).cpu().numpy()
                    y_pred = (probs >= 0.5).astype(int)
                    y_prob = probs
            else:
                model = get_model(config.get('arch', 'mlp'), X_train_s.shape[1], config)
                model.load_state_dict(torch.load(nn_out, map_location=device))
                model.to(device).eval()
                with torch.no_grad():
                    logits = model(torch.from_numpy(X_val_s).float().to(device))
                    probs = torch.sigmoid(logits).cpu().numpy()
                    y_pred = (probs >= 0.5).astype(int)
                    y_prob = probs

        # Dynamically point target references based on configuration mode
        eval_true = y_val_seq if is_sequence_model else y_val
        eval_groups = val_seq_groups if is_sequence_model else groups[val_idx]

        metrics = evaluate_preds(eval_true, y_pred, y_prob)
        fold_metrics_window.append(metrics)

        # Secure patient-level aggregation against sample length transformations
        patient_true, patient_pred, patient_prob = [], [], []
        for p in np.unique(eval_groups):
            mask = (eval_groups == p)
            true_frac = float(np.mean(eval_true[mask])) if mask.any() else 0.0
            true_label = int(true_frac >= 0.10)
            
            if y_prob is not None:
                win_pred = (y_prob[mask] >= 0.5).astype(int)
                prob_frac = float(np.mean(y_prob[mask])) if mask.any() else 0.0
            else:
                win_pred = np.array(y_pred)[mask]
                prob_frac = float(np.mean(win_pred)) if mask.any() else 0.0
                
            pred_frac = float(np.mean(win_pred)) if mask.any() else 0.0
            pred_label = int(pred_frac >= 0.10)
            
            patient_true.append(true_label)
            patient_pred.append(pred_label)
            patient_prob.append(prob_frac)

        patient_metrics = evaluate_preds(np.array(patient_true), np.array(patient_pred), np.array(patient_prob)) if patient_true else {}
        fold_metrics_patient.append(patient_metrics)
        print(f'Fold {fold} window metrics: {metrics} | patient metrics: {patient_metrics}')

    agg_window = {k: np.mean([m[k] for m in fold_metrics_window if not np.isnan(m[k])]) for k in fold_metrics_window[0]} if fold_metrics_window else {}
    agg_patient = {k: np.mean([m[k] for m in fold_metrics_patient if not np.isnan(m[k])]) for k in fold_metrics_patient[0]} if fold_metrics_patient else {}
    print('Cross-validation mean window metrics:', agg_window)
    print('Cross-validation mean patient metrics:', agg_patient)
    return fold_metrics_window, fold_metrics_patient, {'window': agg_window, 'patient': agg_patient}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hrv_dir', required=True)
    parser.add_argument('--tda_dir', required=True)
    parser.add_argument('--labels', required=True, help='CSV with columns filename,label,group')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--model', choices=['rf','nn'], default='rf')
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--arch', choices=['mlp','bilstm','tst'], default='mlp')
    parser.add_argument('--seq_len', type=int, default=None)
    args = parser.parse_args()

    config = {'epochs':50, 'batch_size':64, 'lr':1e-4, 'wd':1e-4, 'patience':5, 'arch': args.arch, 'seq_len': args.seq_len}
    run_group_cv(args.hrv_dir, args.tda_dir, args.labels, args.outdir, model_type=args.model, n_splits=args.n_splits, config=config)


if __name__ == '__main__':
    main()
