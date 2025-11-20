"""Implementation of a FlashAttention2 kernel."""

import math
import torch

from torch import autograd, Tensor
from jaxtyping import Float
from einops import einsum


def torch_flash_attn(
    Q: Float[Tensor, "... n_q d_k"],
    K: Float[Tensor, "... n_k d_k"],
    V: Float[Tensor, "... n_k d_v"],
    is_causal: bool,
    Q_TILE_SIZE,
    K_TILE_SIZE,
) -> Float[Tensor, "... n_q d_v"]:
    d_k, d_v, l_dims = Q.shape[-1], V.shape[-1], Q.shape[:-1]
    out_dims = l_dims + (d_v,)
    O = torch.empty(out_dims, device=Q.device, dtype=Q.dtype)
    L = torch.empty(l_dims, device=Q.device, dtype=Q.dtype)

    T_q = Q.shape[-2] // Q_TILE_SIZE
    T_k = K.shape[-2] // K_TILE_SIZE
    for i in range(T_q):
        Q_i = Q[..., i * Q_TILE_SIZE : (i + 1) * Q_TILE_SIZE, :]
        M_i = torch.full(
            Q_i.shape[:-1], -float("inf"), device=Q_i.device, dtype=Q_i.dtype
        )
        L_i = torch.zeros_like(M_i)
        O_i = torch.zeros(
            *Q_i.shape[:-1], d_v, device=Q_i.device, dtype=Q_i.dtype
        )
        q_idx = i * Q_TILE_SIZE + torch.arange(0, Q_TILE_SIZE, device=Q.device)
        for j in range(T_k):
            k_idx = j * K_TILE_SIZE + torch.arange(
                0, K_TILE_SIZE, device=K.device
            )
            K_j = K[..., j * K_TILE_SIZE : (j + 1) * K_TILE_SIZE, :]
            V_j = V[..., j * K_TILE_SIZE : (j + 1) * K_TILE_SIZE, :]
            S_ij = einsum(
                Q_i, K_j, "... b_q d_k, ... b_k d_k -> ... b_q b_k"
            ) / math.sqrt(d_k)
            if is_causal:
                causal_mask = k_idx[None, :] > q_idx[:, None]
                S_ij = torch.where(causal_mask, -float("inf"), S_ij)
            prev_M_i = M_i
            M_i = torch.maximum(M_i, S_ij.max(dim=-1).values)
            P_ij = torch.exp(S_ij - M_i.unsqueeze(-1))
            alpha = torch.exp(prev_M_i - M_i)
            L_i = alpha * L_i + P_ij.sum(dim=-1)
            O_i = alpha.unsqueeze(-1) * O_i + einsum(
                P_ij, V_j, "... b_q b_k, ... b_k d_v -> ... b_q d_v"
            )
        O[..., i * Q_TILE_SIZE : (i + 1) * Q_TILE_SIZE, :] = (
            O_i / L_i.unsqueeze(-1)
        )
        L[..., i * Q_TILE_SIZE : (i + 1) * Q_TILE_SIZE] = M_i + torch.log(L_i)
    return O, L


class TorchFlashAttentionFunc(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "n_q d"],
        K: Float[Tensor, "n_k d"],
        V: Float[Tensor, "n_k d"],
        is_causal: bool = False,
    ):
        # TODO: Tune the tile sizes.
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.is_causal = is_causal
        O, L = torch_flash_attn(
            Q, K, V, is_causal, ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx):
        # TODO: Implement.
        raise NotImplementedError("TODO: Implement FlashAttentionFunc.backward")
