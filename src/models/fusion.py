"""Attention-based feature fusion layer (simple scaled dot-product attention)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, hrv_dim, tda_dim, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hrv_dim, hidden_dim)
        self.k_proj = nn.Linear(tda_dim, hidden_dim)
        self.v_proj = nn.Linear(tda_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim + hrv_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        self._last_weights = None

    def forward(self, hrv, tda):
        # Support both 2D (batch, dim) and 3D (batch, seq_len, dim) inputs
        is_sequence = hrv.dim() == 3
        if not is_sequence:
            q = self.q_proj(hrv).unsqueeze(1)  # (batch, 1, hidden)
            k = self.k_proj(tda).unsqueeze(1)  # (batch, 1, hidden)
            v = self.v_proj(tda).unsqueeze(1)
            attn_scores = torch.bmm(q, k.transpose(1,2)) / self.scale  # (batch,1,1)
            attn_weights = F.softmax(attn_scores, dim=-1)
            self._last_weights = attn_weights.detach().cpu()
            attn_out = torch.bmm(attn_weights, v).squeeze(1)
            fused = torch.cat([hrv, attn_out], dim=-1)
            return self.out(fused)
        else:
            # hrv: (batch, seq_len, hrv_dim)
            b, s, _ = hrv.shape
            q = self.q_proj(hrv)  # (b, s, hidden)
            k = self.k_proj(tda)
            v = self.v_proj(tda)
            # compute attention per timestep (query at each timestep attends to corresponding key)
            # reshape to (b*s, 1, hidden)
            q_r = q.reshape(b*s, 1, -1)
            k_r = k.reshape(b*s, 1, -1)
            v_r = v.reshape(b*s, 1, -1)
            attn_scores = torch.bmm(q_r, k_r.transpose(1,2)) / self.scale  # (b*s,1,1)
            attn_weights = F.softmax(attn_scores, dim=-1)
            self._last_weights = attn_weights.detach().cpu().reshape(b, s, 1, 1)
            attn_out = torch.bmm(attn_weights, v_r).squeeze(1)  # (b*s, hidden)
            attn_out = attn_out.reshape(b, s, -1)
            fused = torch.cat([hrv, attn_out], dim=-1)  # (b, s, hrv_dim+hidden)
            return self.out(fused)

    def last_attention(self):
        return self._last_weights
