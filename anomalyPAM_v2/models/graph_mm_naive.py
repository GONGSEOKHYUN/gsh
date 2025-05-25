import torch
import torch.nn.functional as F

# Naive fallback for graph_mm_tvm using simple scaled dot-product attention

# def graph_mm_naive(q, k, q_k_mask=None, k_q_mask=None, is_t1_diagonaled=False, inf_value=-1e9):
#     """
#     Naive version of graph_mm for fallback.
#     Arguments:
#     - q: (B, L, H, D)
#     - k: (B, L, H, D)
#     - Returns attention weights: (B, L, H, L)
#     """
#     # assert q.dim() == 4, f"q.dim = {q.dim()}, shape = {q.shape}"
#     # assert k.dim() == 4, f"k.dim = {k.dim()}, shape = {k.shape}"
    
#     B, L, H, D_k = q.shape
#     # Reshape for matmul
#     q_ = q.permute(0, 2, 1, 3).reshape(B * H, L, D_k)
#     k_ = k.permute(0, 2, 1, 3).reshape(B * H, L, D_k)

#     # Compute raw attention scores
#     scores = torch.bmm(q_, k_.transpose(1, 2))  # (B*H, L, L)
#     scores = scores / (D_k ** 0.5)  # scale
#     attn = F.softmax(scores, dim=-1)  # (B*H, L, L)

#     # Reshape back to (B, L, H, L)
#     attn = attn.reshape(B, H, L, L).permute(0, 2, 1, 3)
#     return attn

def graph_mm_naive(q, k, q_k_mask=None, k_q_mask=None, is_t1_diagonaled=False, inf_value=-1e9):
    B, L = q.shape[0], q.shape[1]

    if is_t1_diagonaled:
        # Case: attn @ value
        if k.dim() == 3:
            # k: [B, L, D] → [B, L, H, D_k]
            D = k.shape[2]
            H = q.shape[2]  # from attn shape [B, L, H, L]
            assert D % H == 0, f"D ({D}) not divisible by H ({H})"
            D_k = D // H
            k = k.view(B, L, H, D_k)
        else:
            D_k = k.shape[-1]
            H = k.shape[2]

        q_ = q.permute(0, 2, 1, 3).contiguous().reshape(B * H, L, L)
        k_ = k.permute(0, 2, 1, 3).contiguous().reshape(B * H, L, D_k)
        out = torch.bmm(q_, k_)  # (B*H, L, D_k)
        out = out.view(B, H, L, D_k).permute(0, 2, 1, 3)
        return out

    else:
        D_k = q.shape[-1]
        H = q.shape[2]

        q_ = q.permute(0, 2, 1, 3).contiguous().reshape(B * H, L, D_k)
        k_ = k.permute(0, 2, 1, 3).contiguous().reshape(B * H, L, D_k)
        scores = torch.bmm(q_, k_.transpose(1, 2)) / (D_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.view(B, H, L, L).permute(0, 2, 1, 3)
        return attn