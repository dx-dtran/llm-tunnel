# Gemma 4 31B - gpt-fast implementation
# Hybrid sliding/global attention, GeGLU, tied embeddings, logit softcapping

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


# 5 sliding + 1 global, repeated
LAYER_PATTERN = ["sliding"] * 5 + ["global"]


@dataclass
class ModelArgs:
    block_size: int = 262144
    vocab_size: int = 262144
    n_layer: int = 60
    n_head: int = 32
    dim: int = 5376
    intermediate_size: int = 21504
    norm_eps: float = 1e-6
    # Sliding attention
    sliding_head_dim: int = 256
    sliding_n_kv_heads: int = 16
    sliding_window: int = 1024
    sliding_rope_base: float = 10000.0
    # Global attention
    global_head_dim: int = 512
    global_n_kv_heads: int = 4
    global_rope_base: float = 1000000.0
    global_partial_rotary_factor: float = 0.25
    # Output
    final_logit_softcapping: float = 30.0

    def __post_init__(self):
        self.layer_types: List[str] = (
            LAYER_PATTERN * (self.n_layer // len(LAYER_PATTERN))
        )[: self.n_layer]
        self.global_rotary_dim: int = int(
            self.global_head_dim * self.global_partial_rotary_factor
        )

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        config = [
            c for c in transformer_configs if c.lower() in str(name).lower()
        ]
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "gemma-4-31B-it": dict(),
    "gemma-4-31B": dict(),
}


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(config, i) for i in range(config.n_layer)
        )
        self.norm = GemmaRMSNorm(config.dim, eps=config.norm_eps)

        # Tied embeddings: no separate output linear
        self.embed_scale = config.dim**0.5
        self.final_logit_softcapping = config.final_logit_softcapping

        self.sliding_freqs_cis: Optional[Tensor] = None
        self.global_freqs_cis: Optional[Tensor] = None
        self.causal_mask: Optional[Tensor] = None
        self.sliding_causal_mask: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                b.attention.n_kv_heads,
                b.attention.head_dim,
            )

        self.sliding_freqs_cis = precompute_freqs_cis(
            max_seq_length,
            self.config.sliding_head_dim,
            self.config.sliding_rope_base,
        )
        self.global_freqs_cis = precompute_freqs_cis(
            max_seq_length,
            self.config.global_rotary_dim,
            self.config.global_rope_base,
        )

        self.causal_mask = torch.tril(
            torch.ones(max_seq_length, max_seq_length, dtype=torch.bool)
        )
        positions = torch.arange(max_seq_length)
        self.sliding_causal_mask = (
            positions.unsqueeze(0) >= positions.unsqueeze(1)
        ) & (
            positions.unsqueeze(0) - positions.unsqueeze(1)
            < self.config.sliding_window
        )

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.sliding_freqs_cis is not None, "Caches must be initialized first"

        sliding_freqs = self.sliding_freqs_cis[input_pos]
        global_freqs = self.global_freqs_cis[input_pos]
        sliding_mask = self.sliding_causal_mask[None, None, input_pos]
        causal_mask = self.causal_mask[None, None, input_pos]

        x = self.tok_embeddings(idx) * self.embed_scale

        for layer in self.layers:
            if layer.is_sliding:
                x = layer(x, input_pos, sliding_freqs, sliding_mask)
            else:
                x = layer(x, input_pos, global_freqs, causal_mask)

        x = self.norm(x)
        logits = F.linear(x, self.tok_embeddings.weight)

        cap = self.final_logit_softcapping
        logits = cap * torch.tanh(logits / cap)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int) -> None:
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding"
        self.attention = Attention(config, layer_idx)
        self.feed_forward = FeedForward(config)
        # Gemma sandwich norms: pre + post for both attention and FFN
        self.attention_norm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.post_attention_norm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.post_ffn_norm = GemmaRMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor
    ) -> Tensor:
        r = x
        x = self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        x = self.post_attention_norm(x)
        h = r + x

        r = h
        h = self.feed_forward(self.ffn_norm(h))
        h = self.post_ffn_norm(h)
        out = r + h
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding"

        if self.is_sliding:
            self.head_dim = config.sliding_head_dim
            self.n_kv_heads = config.sliding_n_kv_heads
            self.rotary_dim = config.sliding_head_dim
            kv_dim = self.n_kv_heads * self.head_dim
            total_dim = config.n_head * self.head_dim + 2 * kv_dim
        else:
            self.head_dim = config.global_head_dim
            self.n_kv_heads = config.global_n_kv_heads
            self.rotary_dim = config.global_rotary_dim
            kv_dim = self.n_kv_heads * self.head_dim
            total_dim = config.n_head * self.head_dim + kv_dim  # no V (K=V)

        self.n_head = config.n_head
        self.q_dim = config.n_head * self.head_dim
        self.kv_dim = kv_dim
        self.dim = config.dim

        self.wqkv = nn.Linear(config.dim, total_dim, bias=False)
        self.wo = nn.Linear(self.q_dim, config.dim, bias=False)
        self.kv_cache = None

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            if prefix + "wv.weight" in state_dict:
                wv = state_dict.pop(prefix + "wv.weight")
                state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])
            else:
                state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        proj = self.wqkv(x)

        if self.is_sliding:
            q, k, v = proj.split(
                [self.q_dim, self.kv_dim, self.kv_dim], dim=-1
            )
        else:
            q, kv = proj.split([self.q_dim, self.kv_dim], dim=-1)
            k = kv
            v = kv  # V = K projection output (before RoPE)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if self.is_sliding:
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)
        else:
            q = apply_partial_rotary_emb(q, freqs_cis, self.rotary_dim)
            k = apply_partial_rotary_emb(k, freqs_cis, self.rotary_dim)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_kv_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.q_dim)
        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # GeGLU: gelu_tanh gate * up, then down
        return self.w2(F.gelu(self.w1(x), approximate="tanh") * self.w3(x))


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm: weight is an offset added to 1.0."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * (1.0 + self.weight)


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: float = 10000.0
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0]
            - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0]
            + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def apply_partial_rotary_emb(
    x: Tensor, freqs_cis: Tensor, rotary_dim: int
) -> Tensor:
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x_rot = apply_rotary_emb(x_rot, freqs_cis)
    return torch.cat([x_rot, x_pass], dim=-1)
