import torch
import torch.nn as nn
from .fusion import AttentionFusion


class FusionSequenceModel(nn.Module):
    """Wrap AttentionFusion and a sequence classifier (BiLSTM or Transformer).

    Inputs:
      - hrv_seq: (batch, seq_len, hrv_dim)
      - tda_seq: (batch, seq_len, tda_dim)
    """
    def __init__(self, hrv_dim, tda_dim, fusion_hidden, arch='bilstm', classifier_kwargs=None):
        super().__init__()
        self.fusion = AttentionFusion(hrv_dim, tda_dim, fusion_hidden)
        fused_dim = fusion_hidden  # AttentionFusion returns 'out' dim = hidden_dim
        self.arch = arch.lower()
        if self.arch == 'bilstm':
            from .bilstm import BiLSTMClassifier
            self.classifier = BiLSTMClassifier(fused_dim, hidden_dim=classifier_kwargs.get('hidden',128))
        elif self.arch == 'tst':
            from .tst import TransformerClassifier
            self.classifier = TransformerClassifier(fused_dim, nhead=classifier_kwargs.get('nhead',4), num_layers=classifier_kwargs.get('nlayers',2))
        else:
            # fallback to simple MLP applied per-timestep and take last
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, fused_dim//2),
                nn.ReLU(),
                nn.Linear(fused_dim//2, 1)
            )

    def forward(self, hrv_seq, tda_seq):
        # apply fusion per timestep
        fused = self.fusion(hrv_seq, tda_seq)  # (b, seq_len, fused_dim)
        # pass to classifier
        logits = self.classifier(fused)
        return logits
