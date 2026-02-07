import torch
import torch.nn as nn
from typing import Optional

class TemporalPatchEncoder(nn.Module):
    def __init__(self, patch_len: int = 4, stride: int = 2, d_model: int = 64):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        patches = []
        for i in range(0, T - self.patch_len + 1, self.stride):
            patch = x[:, i : i + self.patch_len, :]
            patch = patch.permute(0, 2, 1)
            proj = self.proj(patch)
            patches.append(proj)
        out = torch.stack(patches, dim=1)
        return out

class CrossAssetAttention(nn.Module):
    """
    Multi-head attention across asset universe to capture instantaneous systemic correlation.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_assets, d_model)
        attn_out, _ = self.mha(x, x, x)
        x = self.norm(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)
