import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim//2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out = self.transformer(x)
        last = out[:, -1, :]
        logits = self.classifier(last).squeeze(-1)
        return logits
