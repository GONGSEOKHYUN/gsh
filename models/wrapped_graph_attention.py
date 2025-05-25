import torch
import torch.nn as nn
from types import SimpleNamespace
from models.graph_attention import GraphSelfAttention

class WrappedGraphSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, seq_len, window_size, stride_size, dropout=0.1, device='cuda', normalize_before=True):
        super().__init__()
        
        # Build a namespace to mimic the original opt structure
        opt = SimpleNamespace(
            d_model=d_model,
            n_head=n_head,
            d_k=d_model // n_head,
            seq_len=seq_len,
            window_size=window_size,
            stride_size=stride_size,
            dropout=dropout,
            device=device,
            normalize_before=normalize_before
        )

        self.attn = GraphSelfAttention(opt)

    # def forward(self, x):
    #     return self.attn(x)
    def forward(self, x):
        # x: [B, L, D] → reshape to [B, L, H, D//H]
        B, L, D = x.shape
        H = self.attn.n_head
        assert D % H == 0, f"d_model ({D}) must be divisible by n_head ({H})"
        x = x.view(B, L, H, D // H)
    
        out = self.attn(x)  # [B, L, H, D//H]
        out = out.view(B, L, D)  # back to [B, L, D]
        return out
