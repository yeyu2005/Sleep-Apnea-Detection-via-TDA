"""Measure inference time per 1-minute window and optionally estimate FLOPs.

Usage:
  python scripts/benchmark_inference.py --model path/to/nn.pth --arch bilstm --hrv_dim 9 --tda_dim 256 --seq_len 5 --iters 100

If `ptflops` is installed, the script will attempt to estimate FLOPs; otherwise it reports timing only.
"""
import time
import argparse
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None

try:
    from ptflops import get_model_complexity_info
except Exception:
    get_model_complexity_info = None


def build_dummy_model(arch, fused_dim, config):
    if torch is None:
        raise RuntimeError('PyTorch required for benchmarking')
    if arch == 'bilstm':
        from src.models.bilstm import BiLSTMClassifier
        return BiLSTMClassifier(fused_dim, hidden_dim=config.get('hidden',128))
    if arch == 'tst':
        from src.models.tst import TransformerClassifier
        return TransformerClassifier(fused_dim, nhead=config.get('nhead',4), num_layers=config.get('nlayers',2))
    # fallback
    from src.training.train import get_model
    return get_model('mlp', fused_dim, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Optional path to model state dict')
    parser.add_argument('--arch', choices=['bilstm','tst','mlp'], default='bilstm')
    parser.add_argument('--hrv_dim', type=int, default=9)
    parser.add_argument('--tda_dim', type=int, default=256)
    parser.add_argument('--fusion_hidden', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()

    fused_dim = args.fusion_hidden
    config = {'hidden':128, 'nhead':4, 'nlayers':2}
    model = build_dummy_model(args.arch, fused_dim, config)
    # wrap with fusion as in FusionSequenceModel
    from src.models.fusion_model import FusionSequenceModel
    fusion_model = FusionSequenceModel(args.hrv_dim, args.tda_dim, fusion_hidden=args.fusion_hidden, arch=args.arch, classifier_kwargs=config)
    if args.model and os.path.exists(args.model):
        try:
            fusion_model.load_state_dict(torch.load(args.model, map_location='cpu'))
        except Exception:
            pass
    fusion_model.eval()

    # create synthetic batch
    batch_size = 4
    X_hrv = np.random.randn(batch_size, args.seq_len, args.hrv_dim).astype(np.float32)
    X_tda = np.random.randn(batch_size, args.seq_len, args.tda_dim).astype(np.float32)

    import torch
    Xh = torch.from_numpy(X_hrv)
    Xt = torch.from_numpy(X_tda)

    # warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = fusion_model(Xh, Xt)

    # timing
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.iters):
            _ = fusion_model(Xh, Xt)
    elapsed = time.time() - t0
    avg = elapsed / args.iters
    print(f'Average forward time over {args.iters} runs: {avg*1000:.3f} ms')

    # FLOPs estimate if available
    if get_model_complexity_info is not None:
        try:
            # ptflops expects a single-input tensor; we measure classifier part only
            # create a dummy input for transformer/lstm
            with torch.cuda.device('cpu'):
                macs, params = get_model_complexity_info(fusion_model, (args.seq_len, args.hrv_dim), as_strings=True, print_per_layer_stat=False)
            print('Estimated MACs/Params:', macs, params)
        except Exception:
            pass


if __name__ == '__main__':
    main()
