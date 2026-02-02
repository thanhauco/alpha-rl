import torch
import torch.nn as nn
from typing import Optional

class TemporalPatchEncoder(nn.Module):
    """
    PatchTST-inspired temporal patch encoder.
    Slices multivariate time series into fixed-length patches to capture local trends.
    """
    def __init__(self, patch_len: int = 4, stride: int = 2, d_model: int = 64):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D_features)
        B, T, F = x.shape
        patches = []
        for i in range(0, T - self.patch_len + 1, self.stride):
            patch = x[:, i : i + self.patch_len, :] # (B, P, F)
            patch = patch.permute(0, 2, 1) # (B, F, P)
            proj = self.proj(patch) # (B, F, d_model)
            patches.append(proj)
        
        out = torch.stack(patches, dim=1) # (B, N_patches, F, d_model)
        return out
