"""Pallas flash-attention forward/backward kernels."""

from __future__ import annotations

import functools

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from flash_gpt import runtime as rt


def get_attn_mask(q_idx, k_idx, Q_BLOCK, K_BLOCK, Q_SIZE, K_SIZE, BH, T, D):
    del BH, T, D
    x = q_idx * Q_BLOCK + jnp.arange(Q_SIZE, dtype=jnp.int32)[:, None]
    y = k_idx * K_BLOCK + jnp.arange(K_SIZE, dtype=jnp.int32)[None, :]
    basic_mask = x >= y
    return basic_mask[None, ...]


def forward_kernel(
    q_ref, k_T_ref, v_ref, o_ref, m_ref, l_ref,
    Q_BLOCK, K_BLOCK, BH, T, D, get_attn_mask_fn,
):
    del BH
    for q_idx in range((T + Q_BLOCK - 1) // Q_BLOCK):
        q = q_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, :]

        mi = jnp.full((q.shape[0], min(Q_BLOCK, q.shape[1]), 1), rt.NEG_INF, dtype=jnp.float32)
        li = jnp.zeros((q.shape[0], min(Q_BLOCK, q.shape[1]), 1), dtype=jnp.float32)
        o = jnp.zeros((q.shape[0], min(Q_BLOCK, q.shape[1]), D), dtype=jnp.float32)

        for k_idx in range((T + K_BLOCK - 1) // K_BLOCK):
            k_T = k_T_ref[..., k_idx * K_BLOCK : (k_idx + 1) * K_BLOCK]
            v = v_ref[..., k_idx * K_BLOCK : (k_idx + 1) * K_BLOCK, :]

            scores = (q @ k_T) / jnp.sqrt(D)
            attn_mask = get_attn_mask_fn(
                q_idx,
                k_idx,
                Q_BLOCK,
                K_BLOCK,
                min(Q_BLOCK, q.shape[1]),
                min(K_BLOCK, k_T.shape[2]),
                q.shape[0],
                T,
                D,
            )
            scores += jnp.where(attn_mask, 0.0, rt.NEG_INF)

            mij = jnp.max(scores, axis=-1, keepdims=True)
            pij = jnp.exp(scores - mij)
            lij = jnp.sum(pij, axis=-1, keepdims=True)

            mi_new = jnp.maximum(mi, mij)
            alpha = jnp.exp(mi - mi_new)
            beta = jnp.exp(mij - mi_new)
            li_new = alpha * li + beta * lij

            pij_scale = beta / li_new
            pij = pij * pij_scale
            o_scale = (li / li_new) * alpha
            o *= o_scale
            o += pij @ v

            li = li_new
            mi = mi_new

        o_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, :] = o
        m_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, :] = mi
        l_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, :] = li


def backward_kernel_q(
    q_ref, k_ref, k_T_ref, v_T_ref, o_ref, do_ref, m_ref, l_ref, dq_ref, pij_ref, dscores_ref,
    Q_BLOCK, K_BLOCK, BH, T, D, get_attn_mask_fn,
):
    del BH
    for q_idx in range((T + Q_BLOCK - 1) // Q_BLOCK):
        q = q_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, :]
        o = o_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, :]
        do = do_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, :]
        mi = m_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, :]
        li = l_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, :]

        dq = jnp.zeros(q.shape, dtype=jnp.float32)
        for k_idx in range((T + K_BLOCK - 1) // K_BLOCK):
            k = k_ref[..., k_idx * K_BLOCK : (k_idx + 1) * K_BLOCK, :]
            k_T = k_T_ref[..., k_idx * K_BLOCK : (k_idx + 1) * K_BLOCK]
            v_T = v_T_ref[..., k_idx * K_BLOCK : (k_idx + 1) * K_BLOCK]

            scores = (q @ k_T) / jnp.sqrt(D)
            attn_mask = get_attn_mask_fn(
                q_idx,
                k_idx,
                Q_BLOCK,
                K_BLOCK,
                min(Q_BLOCK, q.shape[1]),
                min(K_BLOCK, k.shape[1]),
                q.shape[0],
                T,
                D,
            )
            scores += jnp.where(attn_mask, 0.0, rt.NEG_INF)

            pij = jnp.exp(scores - mi) / li
            dpij = do @ v_T
            ddi = jnp.sum(do * o, axis=-1, keepdims=True)
            dscores = (pij * (dpij - ddi)) / jnp.sqrt(D)

            dq += dscores @ k

            pij_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, k_idx * K_BLOCK : (k_idx + 1) * K_BLOCK] = pij
            dscores_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, k_idx * K_BLOCK : (k_idx + 1) * K_BLOCK] = dscores

        dq_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, :] = dq


def backward_kernel_kv(
    q_ref, do_ref, pij_T_ref, dscores_T_ref, dk_ref, dv_ref,
    Q_BLOCK, K_BLOCK, BH, T, D, get_attn_mask_fn,
):
    del BH, D, get_attn_mask_fn
    for k_idx in range((T + K_BLOCK - 1) // K_BLOCK):
        dk = jnp.zeros((q_ref.shape[0], min(K_BLOCK, T - k_idx * K_BLOCK), q_ref.shape[-1]), dtype=jnp.float32)
        dv = jnp.zeros((q_ref.shape[0], min(K_BLOCK, T - k_idx * K_BLOCK), q_ref.shape[-1]), dtype=jnp.float32)

        for q_idx in range((T + Q_BLOCK - 1) // Q_BLOCK):
            q = q_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, :]
            do = do_ref[..., q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK, :]
            pij_T = pij_T_ref[..., k_idx * K_BLOCK : (k_idx + 1) * K_BLOCK, q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK]
            dscores_T = dscores_T_ref[..., k_idx * K_BLOCK : (k_idx + 1) * K_BLOCK, q_idx * Q_BLOCK : (q_idx + 1) * Q_BLOCK]

            dk += dscores_T @ q
            dv += pij_T @ do

        dk_ref[..., k_idx * K_BLOCK : (k_idx + 1) * K_BLOCK, :] = dk
        dv_ref[..., k_idx * K_BLOCK : (k_idx + 1) * K_BLOCK, :] = dv


def flash_forward(q, k, v):
    BH, T, D = q.shape
    Q_BLOCK = rt.config.model.q_chunk_size
    K_BLOCK = rt.config.model.k_chunk_size
    BH_local = BH // rt.vmap_devices.shape[0]

    k_T = jnp.transpose(k, (0, 2, 1))
    v_T = jnp.transpose(v, (0, 2, 1))

    pforward = pl.pallas_call(
        functools.partial(
            forward_kernel,
            Q_BLOCK=Q_BLOCK,
            K_BLOCK=K_BLOCK,
            BH=BH_local,
            T=T,
            D=D,
            get_attn_mask_fn=get_attn_mask,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((BH_local, T, D), jnp.float32),
            jax.ShapeDtypeStruct((BH_local, T, 1), jnp.float32),
            jax.ShapeDtypeStruct((BH_local, T, 1), jnp.float32),
        ],
        grid=(1,),
        in_specs=[
            pl.BlockSpec(block_shape=(BH_local, T, D), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, D, T), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, D), index_map=lambda i: (0, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec(block_shape=(BH_local, T, D), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, 1), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, 1), index_map=lambda i: (0, 0, 0)),
        ],
        interpret=True,
    )

    o, m, l = jax.jit(
        jax.shard_map(
            pforward,
            mesh=rt.VMAP_MESH,
            in_specs=(P("x", None, None), P("x", None, None), P("x", None, None)),
            out_specs=(P("x", None, None), P("x", None, None), P("x", None, None)),
            check_vma=False,
        )
    )(q, k_T, v)

    return o, (q, k, k_T, v_T, o, m, l)


def flash_backward(res, g):
    q, k, k_T, v_T, o, m, l = res
    do = g
    BH, T, D = q.shape

    Q_BLOCK = rt.config.model.q_chunk_size
    K_BLOCK = rt.config.model.k_chunk_size
    BH_local = BH // rt.vmap_devices.shape[0]

    pbackward_q = pl.pallas_call(
        functools.partial(
            backward_kernel_q,
            Q_BLOCK=Q_BLOCK,
            K_BLOCK=K_BLOCK,
            BH=BH_local,
            T=T,
            D=D,
            get_attn_mask_fn=get_attn_mask,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((BH_local, T, D), jnp.float32),
            jax.ShapeDtypeStruct((BH_local, T, T), jnp.float32),
            jax.ShapeDtypeStruct((BH_local, T, T), jnp.float32),
        ],
        grid=(1,),
        in_specs=[
            pl.BlockSpec(block_shape=(BH_local, T, D), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, D), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, D, T), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, D, T), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, D), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, D), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, 1), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, 1), index_map=lambda i: (0, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec(block_shape=(BH_local, T, D), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, T), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, T), index_map=lambda i: (0, 0, 0)),
        ],
        interpret=True,
    )

    pbackward_kv = pl.pallas_call(
        functools.partial(
            backward_kernel_kv,
            Q_BLOCK=Q_BLOCK,
            K_BLOCK=K_BLOCK,
            BH=BH_local,
            T=T,
            D=D,
            get_attn_mask_fn=get_attn_mask,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((BH_local, T, D), dtype=jnp.float32),
            jax.ShapeDtypeStruct((BH_local, T, D), dtype=jnp.float32),
        ],
        grid=(1,),
        in_specs=[
            pl.BlockSpec(block_shape=(BH_local, T, D), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, D), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, T), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, T), index_map=lambda i: (0, 0, 0)),
        ],
        out_specs=[
            pl.BlockSpec(block_shape=(BH_local, T, D), index_map=lambda i: (0, 0, 0)),
            pl.BlockSpec(block_shape=(BH_local, T, D), index_map=lambda i: (0, 0, 0)),
        ],
        interpret=True,
    )

    dq, pij, dscores = jax.jit(
        jax.shard_map(
            pbackward_q,
            mesh=rt.VMAP_MESH,
            in_specs=(P("x", None, None),) * 8,
            out_specs=(P("x", None, None), P("x", None, None), P("x", None, None)),
            check_vma=False,
        )
    )(q, k, k_T, v_T, o, do, m, l)

    pij_T = jnp.transpose(pij, (0, 2, 1))
    dscores_T = jnp.transpose(dscores, (0, 2, 1))

    dk, dv = jax.jit(
        jax.shard_map(
            pbackward_kv,
            mesh=rt.VMAP_MESH,
            in_specs=(P("x", None, None),) * 4,
            out_specs=(P("x", None, None), P("x", None, None)),
            check_vma=False,
        )
    )(q, do, pij_T, dscores_T)
    return dq, dk, dv
