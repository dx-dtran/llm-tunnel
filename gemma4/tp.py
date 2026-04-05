# Gemma 4 31B - tensor parallelism

import os
from typing import List, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import _functional_collectives as funcol

from model import Attention, FeedForward, Transformer


def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_local():
    return _get_rank() == 0


def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def maybe_init_dist() -> Optional[int]:
    try:
        rank = _get_rank()
        world_size = _get_world_size()
        if world_size < 2:
            return None
    except KeyError:
        return None

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank


rank = _get_rank()
world_size = _get_world_size()


def shard(x, dim):
    assert x.size(dim=dim) % world_size == 0
    return torch.tensor_split(x, world_size, dim=dim)[rank]


def _shard_splits(tensor, dim, weight_splits):
    """Shard a tensor that is composed of multiple concatenated parts."""
    parts = tensor.split(weight_splits, dim=dim)
    sharded = [shard(p, dim) for p in parts]
    return torch.cat(sharded, dim=dim)


def _apply_tp_linear(
    linear: nn.Linear, style: str, weight_splits: List[int] = []
) -> None:
    dim_lookup = {
        "colwise": (0, "out_features"),
        "rowwise": (1, "in_features"),
    }
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    assert getattr(linear, size_attr) % world_size == 0

    if weight_splits:
        sharded_weight = _shard_splits(linear.weight, shard_dim, weight_splits)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = _shard_splits(linear.scales, 0, weight_splits)
    else:
        sharded_weight = shard(linear.weight, shard_dim)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard(linear.scales, 0)

    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)


def _apply_tp_attn(attn: Attention) -> None:
    assert hasattr(attn, "wqkv")
    assert hasattr(attn, "wo")

    if attn.is_sliding:
        weight_splits = [attn.q_dim, attn.kv_dim, attn.kv_dim]
    else:
        weight_splits = [attn.q_dim, attn.kv_dim]

    _apply_tp_linear(attn.wqkv, "colwise", weight_splits)
    _apply_tp_linear(attn.wo, "rowwise")

    # Update per-module dimensions
    attn.n_head = attn.n_head // world_size
    attn.q_dim = attn.q_dim // world_size
    attn.n_kv_heads = attn.n_kv_heads // world_size
    attn.kv_dim = attn.kv_dim // world_size

    attn.register_forward_hook(
        lambda _module, _input, output: funcol.all_reduce(
            output, "sum", list(range(world_size))
        )
    )


def _apply_tp_ffn(ffn: FeedForward) -> None:
    _apply_tp_linear(ffn.w1, "colwise")
    _apply_tp_linear(ffn.w3, "colwise")
    _apply_tp_linear(ffn.w2, "rowwise")

    ffn.register_forward_hook(
        lambda _module, _input, output: funcol.all_reduce(
            output, "sum", list(range(world_size))
        )
    )


def _apply_tp_Transformer(model: Transformer) -> None:
    # Update config for cache setup
    model.config.n_head = model.config.n_head // world_size
    model.config.sliding_n_kv_heads = model.config.sliding_n_kv_heads // world_size
    model.config.global_n_kv_heads = model.config.global_n_kv_heads // world_size


def apply_tp(model: Transformer) -> None:
    _apply_tp_Transformer(model)
    for block in model.layers:
        _apply_tp_attn(block.attention)
        _apply_tp_ffn(block.feed_forward)
