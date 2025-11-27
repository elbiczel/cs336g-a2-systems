"""Implementation of a FlashAttention2 kernel."""

import math
import torch
import triton
import triton.language as tl

from torch import autograd, Tensor
from jaxtyping import Float

from cs336_systems import flash_attention


configs = [
    triton.Config(
        {"Q_TILE_SIZE": 16, "K_TILE_SIZE": 64}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"Q_TILE_SIZE": 32, "K_TILE_SIZE": 64}, num_warps=4, num_stages=2
    ),
    triton.Config(
        {"Q_TILE_SIZE": 32, "K_TILE_SIZE": 128}, num_warps=8, num_stages=3
    ),
    triton.Config(
        {"Q_TILE_SIZE": 64, "K_TILE_SIZE": 64}, num_warps=8, num_stages=3
    ),
]


@triton.autotune(configs=configs, key=["is_causal", "N_QUERIES", "N_KEYS", "D"])
@triton.jit
def flash_attn_fwd(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,  # Data ptrs
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,  # Strides
    N_QUERIES,
    N_KEYS,
    scale: tl.constexpr,
    D: tl.constexpr,
    is_causal: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    if query_tile_index * Q_TILE_SIZE >= N_QUERIES:
        return

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    q_valid = q_idx < N_QUERIES
    M_i = tl.where(q_valid, -float("inf"), 0.0)
    L_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    q_max = tl.minimum(q_idx[-1], N_QUERIES - 1)
    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for k in range(T_k):
        k_idx = k * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        if is_causal and k_idx[0] > q_max:
            break
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S_ij = tl.dot(Q_i, tl.trans(K_j), out_dtype=tl.float32)
        S_ij = S_ij * scale
        k_valid = k_idx < N_KEYS
        S_ij = tl.where(k_valid[None, :], S_ij, -float("inf"))
        S_ij = tl.where(q_valid[:, None], S_ij, -float("inf"))
        if is_causal:
            causal_mask = k_idx[None, :] > q_idx[:, None]
            S_ij = tl.where(causal_mask, -float("inf"), S_ij)

        prev_M_i = M_i
        S_max = tl.max(S_ij, axis=1)
        row_has_valid = tl.isfinite(S_max)
        M_i = tl.where(row_has_valid, tl.maximum(M_i, S_max), M_i)
        logits = tl.where(
            row_has_valid[:, None], S_ij - M_i[:, None], -float("inf")
        )
        P_ij = tl.exp(logits)
        alpha = tl.where(row_has_valid, tl.exp(prev_M_i - M_i), 1.0)
        P_sum = tl.sum(P_ij, axis=1)
        L_i = tl.where(row_has_valid, alpha * L_i + P_sum, L_i)
        O_i = tl.where(
            row_has_valid[:, None],
            alpha[:, None] * O_i + tl.dot(P_ij, V_j, out_dtype=tl.float32),
            O_i,
        )
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    denom = tl.where(L_i[:, None] > 0, L_i[:, None], 1.0)
    O_i = O_i / denom
    L_i = tl.where(L_i > 0, M_i + tl.log(L_i), -float("inf"))
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


class FlashAttentionFunc(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "... n_q d"],
        K: Float[Tensor, "... n_k d"],
        V: Float[Tensor, "... n_k d"],
        is_causal: bool = False,
    ) -> Float[Tensor, "... n_q d"]:
        # Assert shapes
        *l_dims, n_q, D = Q.shape
        n_k = K.shape[-2]
        assert K.shape[-1] == D and V.shape[-1] == D
        assert K.shape[:-2] == tuple(l_dims) and V.shape[:-2] == tuple(l_dims)
        assert n_k == V.shape[-2]
        out_dims = (*l_dims, n_q, D) if l_dims else (n_q, D)
        # Allocate outputs.
        O = torch.empty(out_dims, device=Q.device, dtype=Q.dtype)
        L = torch.empty(
            (*l_dims, n_q) if l_dims else (n_q,),
            device=Q.device,
            dtype=torch.float32,
        )
        # View as 3d data for triton.
        B = int(torch.tensor(l_dims).prod().item()) if l_dims else 1
        Q_3d = Q.reshape(B, n_q, D)
        K_3d = K.reshape(B, n_k, D)
        V_3d = V.reshape(B, n_k, D)
        O_3d = O.reshape(B, n_q, D)
        L_2d = L.reshape(B, n_q)

        ctx.is_causal = is_causal
        scale = 1 / math.sqrt(D)
        grid = lambda meta: (triton.cdiv(n_q, meta["Q_TILE_SIZE"]), B)  # noqa: E731
        flash_attn_fwd[grid](
            Q_3d,
            K_3d,
            V_3d,
            O_3d,
            L_2d,  # Data
            Q_3d.stride(0),
            Q_3d.stride(1),
            Q_3d.stride(2),
            K_3d.stride(0),
            K_3d.stride(1),
            K_3d.stride(2),
            V_3d.stride(0),
            V_3d.stride(1),
            V_3d.stride(2),
            O_3d.stride(0),
            O_3d.stride(1),
            O_3d.stride(2),
            L_2d.stride(0),
            L_2d.stride(1),  # Stides
            N_QUERIES=n_q,
            N_KEYS=n_k,
            scale=scale,
            D=D,
            is_causal=is_causal,
        )
        ctx.save_for_backward(Q, K, V, L)
        ctx.B = B
        ctx.l_dims = tuple(l_dims)
        ctx.n_q = n_q
        ctx.n_k = n_k
        ctx.D = D
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        return flash_attention.TorchFlashAttentionFunc.backward(ctx, dO)
