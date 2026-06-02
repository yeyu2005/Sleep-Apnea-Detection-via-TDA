"""PyTorch Dataset for per-window HRV + TDA features."""
import os
import numpy as np
from torch.utils.data import Dataset

class TopoECGDataset(Dataset):
    def __init__(self, hrv_dir, tda_dir, label_map):
        # hrv_dir: folder with per-window .npy files
        self.hrv_files = sorted([os.path.join(hrv_dir,f) for f in os.listdir(hrv_dir) if f.endswith('.npy')])
        self.tda_dir = tda_dir
        self.label_map = label_map  # mapping filename->label

    def __len__(self):
        return len(self.hrv_files)

    def __getitem__(self, idx):
        hrv_path = self.hrv_files[idx]
        fname = os.path.basename(hrv_path)
        hrv = np.load(hrv_path)
        tda_path = os.path.join(self.tda_dir, fname.replace('.npy', '.npy'))
        tda = np.load(tda_path)
        label = self.label_map.get(fname, 0)
        return hrv.astype('float32'), tda.astype('float32'), np.int64(label)
