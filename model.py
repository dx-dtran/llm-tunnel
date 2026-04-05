"""Pure PyTorch Gemma 4 text model — gpt-fast style.

Supports BF16 inference with optional torchao int4 weight-only quantization.
Text-only (vision tower excluded). Designed for google/gemma-4-27b-it and variants.
"""

import os
import json
import glob
import math
from dataclasses import dataclass, field
from typing import Optional, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Gemma4Config:
    vocab_size: int = 262144
    hidden_size: int = 5376
    intermediate_size: int = 21504
    num_hidden_layers: int = 60
    num_attention_heads: int = 32
    num_key_value_heads: int = 16          # sliding attention
    num_global_key_value_heads: int = 4    # full attention
    head_dim: int = 256                    # sliding
    global_head_dim: int = 512             # full
    rms_norm_eps: float = 1e-6
    rope_theta_sliding: float = 10000.0
    rope_theta_full: float = 1000000.0
    partial_rotary_factor: float = 0.25    # full attention only
    sliding_window: int = 1024
    max_position_embeddings: int = 262144
    final_logit_softcapping: float = 30.0
    attention_k_eq_v: bool = True
    layer_types: list = field(
        default_factory=lambda: (["sliding_attention"] * 5 + ["full_attention"]) * 10
    )

    @staticmethod
    def from_hf(config_path: str) -> "Gemma4Config":
        with open(config_path) as f:
            raw = json.load(f)
        tc = raw.get("text_config", raw)
        rp = tc.get("rope_parameters", {})
        return Gemma4Config(
            vocab_size=tc["vocab_size"],
            hidden_size=tc["hidden_size"],
            intermediate_size=tc["intermediate_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            num_global_key_value_heads=tc.get("num_global_key_value_heads") or tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            global_head_dim=tc.get("global_head_dim", 512),
            rms_norm_eps=tc["rms_norm_eps"],
            rope_theta_sliding=rp.get("sliding_attention", {}).get("rope_theta", 10000.0),
            rope_theta_full=rp.get("full_attention", {}).get("rope_theta", 1000000.0),
            partial_rotary_factor=rp.get("full_attention", {}).get("partial_rotary_factor", 0.25),
            sliding_window=tc.get("sliding_window", 1024),
            max_position_embeddings=tc.get("max_position_embeddings", 262144),
            final_logit_softcapping=tc.get("final_logit_softcapping", 30.0),
            attention_k_eq_v=tc.get("attention_k_eq_v", True),
            layer_types=tc.get(
                "layer_types",
                (["sliding_attention"] * 5 + ["full_attention"]) * 10,
            ),
        )


# ═══════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Gemma 4 RMSNorm — weight initialized to ones, no +1 offset."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(x)


# ─── Rotary position embeddings ────────────────────────────────────────

def _precompute_rope(dim: int, max_len: int, theta: float, device: torch.device):
    """Precompute cos/sin for RoPE.

    Returns cos, sin each of shape [max_len, dim] (full dim, not dim//2).
    Uses the rotate_half convention: pairs are (x[i], x[i + dim//2]).
    """
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # [max_len, half]
    # Double up for rotate_half convention: [cos, cos] so element-wise multiply works
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)  # [max_len, dim]
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)  # [max_len, dim]
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs: split into halves, negate second, swap."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                pos: torch.Tensor, rotary_dim: Optional[int] = None):
    """Apply rotary embeddings using rotate_half convention.

    x:   [batch, n_heads, seq_len, head_dim]
    cos: [max_len, rope_dim]
    sin: [max_len, rope_dim]
    pos: [seq_len] — absolute position indices
    rotary_dim: if set, only rotate the first rotary_dim dimensions (partial rotary)
    """
    if rotary_dim is not None and rotary_dim < x.shape[-1]:
        x_rot, x_pass = x.split([rotary_dim, x.shape[-1] - rotary_dim], dim=-1)
    else:
        x_rot, x_pass = x, None

    # Gather cos/sin at positions: [seq_len, rope_dim] → [1, 1, seq_len, rope_dim]
    c = cos[pos].unsqueeze(0).unsqueeze(0)
    s = sin[pos].unsqueeze(0).unsqueeze(0)

    x_rot = x_rot * c + _rotate_half(x_rot) * s

    if x_pass is not None:
        return torch.cat([x_rot, x_pass], dim=-1)
    return x_rot


# ─── Attention ──────────────────────────────────────────────────────────

class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.is_full = not self.is_sliding

        self.n_heads = config.num_attention_heads
        self.head_dim = config.global_head_dim if self.is_full else config.head_dim
        self.sliding_window = config.sliding_window

        # K=V only applies to full (non-sliding) attention when attention_k_eq_v is set
        self.use_kv_shared = config.attention_k_eq_v and self.is_full
        self.n_kv_heads = (config.num_global_key_value_heads if self.use_kv_shared
                           else config.num_key_value_heads)
        self.kv_groups = self.n_heads // self.n_kv_heads

        # Partial rotary for full attention
        self.rotary_dim = (int(config.partial_rotary_factor * self.head_dim)
                           if self.is_full else self.head_dim)

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = None if self.use_kv_shared else nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        # V gets its own norm (no learnable scale), applied instead of RoPE
        self.v_norm = RMSNorm(self.head_dim, config.rms_norm_eps, with_scale=False)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                pos: torch.Tensor, kv_cache: "KVCache", layer_idx: int,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape

        # Project Q, K
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim)

        # V: either from v_proj (sliding) or shared with K (full attention, k_eq_v)
        # IMPORTANT: assign v BEFORE applying k_norm/RoPE to k
        if self.v_proj is not None:
            v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim)
        else:
            v = k  # pre-norm, pre-RoPE key states

        # Q: norm → RoPE
        q = self.q_norm(q)
        q = q.transpose(1, 2)  # [B, n_heads, L, head_dim]
        q = _apply_rope(q, rope_cos, rope_sin, pos,
                        self.rotary_dim if self.is_full else None)

        # K: norm → RoPE
        k = self.k_norm(k)
        k = k.transpose(1, 2)
        k = _apply_rope(k, rope_cos, rope_sin, pos,
                        self.rotary_dim if self.is_full else None)

        # V: norm only (no RoPE)
        v = self.v_norm(v)
        v = v.transpose(1, 2)

        # Update KV cache and retrieve full cached K, V
        k, v = kv_cache.update(layer_idx, k, v, pos)

        # Expand KV heads for GQA
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


# ─── MLP (GeGLU) ───────────────────────────────────────────────────────

class Gemma4MLP(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


# ─── Decoder layer ─────────────────────────────────────────────────────

class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4MLP(config)

        # Sandwich norms
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Per-layer scalar (buffer, not parameter — initialized to 1.0)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                pos: torch.Tensor, kv_cache: "KVCache", layer_idx: int,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention block (pre-norm → attn → post-norm → residual)
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, rope_cos, rope_sin, pos, kv_cache, layer_idx, mask)
        x = self.post_attention_layernorm(x)
        x = residual + x

        # MLP block (pre-norm → mlp → post-norm → residual)
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        # Per-layer scaling (applied to full hidden state)
        x = x * self.layer_scalar
        return x


# ═══════════════════════════════════════════════════════════════════════
# Full model
# ═══════════════════════════════════════════════════════════════════════

class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.embed_scale = config.hidden_size ** 0.5

        # RoPE tables — populated by setup_rope() after .to(device)
        self._rope_sliding_cos: torch.Tensor
        self._rope_sliding_sin: torch.Tensor
        self._rope_full_cos: torch.Tensor
        self._rope_full_sin: torch.Tensor

    def setup_rope(self, max_len: int, device: torch.device):
        """Precompute RoPE cos/sin tables. Call after moving model to device."""
        # Sliding attention: full rotation on head_dim, theta=10k
        sc, ss = _precompute_rope(self.config.head_dim, max_len,
                                  self.config.rope_theta_sliding, device)
        self._rope_sliding_cos = sc
        self._rope_sliding_sin = ss

        # Full attention: partial rotation on rotary_dim, theta=1M
        rotary_dim = int(self.config.partial_rotary_factor * self.config.global_head_dim)
        fc, fs = _precompute_rope(rotary_dim, max_len,
                                  self.config.rope_theta_full, device)
        self._rope_full_cos = fc
        self._rope_full_sin = fs

    def forward(self, tokens: torch.Tensor, pos: torch.Tensor,
                kv_cache: "KVCache",
                sliding_mask: Optional[torch.Tensor] = None,
                full_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        tokens: [batch, seq_len] — token IDs
        pos:    [seq_len] — absolute position indices
        """
        x = self.embed_tokens(tokens) * self.embed_scale

        for i, layer in enumerate(self.layers):
            is_full = layer.self_attn.is_full
            cos = self._rope_full_cos if is_full else self._rope_sliding_cos
            sin = self._rope_full_sin if is_full else self._rope_sliding_sin
            mask = full_mask if is_full else sliding_mask
            x = layer(x, cos, sin, pos, kv_cache, i, mask)

        x = self.norm(x)

        # Tied output projection
        logits = F.linear(x, self.embed_tokens.weight)

        # Logit softcapping
        cap = self.config.final_logit_softcapping
        logits = torch.tanh(logits / cap) * cap

        return logits


# ═══════════════════════════════════════════════════════════════════════
# KV Cache
# ═══════════════════════════════════════════════════════════════════════

class KVCache:
    """Static KV cache for autoregressive generation.

    - Full attention layers: cache grows up to max_seq_len.
    - Sliding attention layers: ring buffer of size sliding_window.
    """

    def __init__(self, config: Gemma4Config, max_seq_len: int,
                 batch_size: int = 1, device: torch.device = None,
                 dtype: torch.dtype = torch.bfloat16):
        self.config = config
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.k_caches = []
        self.v_caches = []

        for i in range(config.num_hidden_layers):
            is_full = config.layer_types[i] == "full_attention"
            hd = config.global_head_dim if is_full else config.head_dim
            nkv = (config.num_global_key_value_heads if is_full and config.attention_k_eq_v
                   else config.num_key_value_heads)
            cache_len = max_seq_len if is_full else config.sliding_window
            shape = (batch_size, nkv, cache_len, hd)
            self.k_caches.append(torch.zeros(shape, device=device, dtype=dtype))
            self.v_caches.append(torch.zeros(shape, device=device, dtype=dtype))

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
               pos: torch.Tensor):
        """Write new K, V into cache and return the full cached K, V for attention.

        k, v: [batch, n_kv_heads, new_len, head_dim]
        pos:  [new_len] — absolute position indices
        """
        is_full = self.config.layer_types[layer_idx] == "full_attention"

        if is_full:
            self.k_caches[layer_idx][:, :, pos] = k
            self.v_caches[layer_idx][:, :, pos] = v
            end = pos[-1].item() + 1
            return (self.k_caches[layer_idx][:, :, :end],
                    self.v_caches[layer_idx][:, :, :end])
        else:
            # Ring buffer
            window = self.config.sliding_window
            idx = pos % window
            self.k_caches[layer_idx][:, :, idx] = k
            self.v_caches[layer_idx][:, :, idx] = v
            end = min(pos[-1].item() + 1, window)
            return (self.k_caches[layer_idx][:, :, :end],
                    self.v_caches[layer_idx][:, :, :end])

    def reset(self):
        for c in self.k_caches + self.v_caches:
            c.zero_()


# ═══════════════════════════════════════════════════════════════════════
# Masks
# ═══════════════════════════════════════════════════════════════════════

def _make_causal_mask(seq_len: int, device: torch.device):
    """Standard causal mask for prefill. Returns None for seq_len=1 (decode)."""
    if seq_len <= 1:
        return None
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def _make_sliding_causal_mask(seq_len: int, window: int, device: torch.device):
    """Causal + sliding window mask for prefill. Returns None for seq_len=1."""
    if seq_len <= 1:
        return None
    row = torch.arange(seq_len, device=device).unsqueeze(1)
    col = torch.arange(seq_len, device=device).unsqueeze(0)
    return (col <= row) & (row - col < window)


# ═══════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════

def _sample(logits: torch.Tensor, temperature: float = 1.0,
            top_p: Optional[float] = None) -> torch.Tensor:
    """Sample a single token from logits [batch, vocab]."""
    if temperature == 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        mask = cumulative - probs > top_p
        sorted_logits[mask] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def generate(model: Gemma4Model, prompt_tokens: list[int], max_new_tokens: int,
             temperature: float = 1.0, top_p: Optional[float] = None,
             stop_tokens: Optional[set[int]] = None,
             ) -> Generator[int, None, None]:
    """Yield generated token IDs one at a time."""
    device = next(model.parameters()).device
    stop_tokens = stop_tokens or {1, 106}  # Gemma EOS tokens

    tokens = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    seq_len = tokens.shape[1]

    max_seq = seq_len + max_new_tokens
    kv_cache = KVCache(model.config, max_seq, batch_size=1, device=device)

    # ── Prefill ──
    pos = torch.arange(seq_len, device=device)
    sliding_mask = _make_sliding_causal_mask(seq_len, model.config.sliding_window, device)
    full_mask = _make_causal_mask(seq_len, device)

    logits = model(tokens, pos, kv_cache, sliding_mask, full_mask)
    next_token = _sample(logits[:, -1], temperature, top_p)

    if next_token.item() in stop_tokens:
        return
    yield next_token.item()

    # ── Decode loop ──
    cur_pos = seq_len
    while cur_pos < max_seq - 1:
        tokens = next_token.unsqueeze(0).unsqueeze(0)  # [1, 1]
        pos = torch.tensor([cur_pos], device=device)

        logits = model(tokens, pos, kv_cache, None, None)
        next_token = _sample(logits[:, -1], temperature, top_p)

        cur_pos += 1

        if next_token.item() in stop_tokens:
            return
        yield next_token.item()


# ═══════════════════════════════════════════════════════════════════════
# Int4 quantization (native PyTorch — no external deps)
# ═══════════════════════════════════════════════════════════════════════

def _quantize_int4(weight: torch.Tensor, groupsize: int = 128, inner_k_tiles: int = 8):
    """Group-wise asymmetric int4 quantization using PyTorch's native packed format.

    Matches gpt-fast's quantization scheme exactly.
    Returns (weight_int4pack, scales_and_zeros) ready for _weight_int4pack_mm.
      weight_int4pack:  packed int32 tensor (hardware format)
      scales_and_zeros: [K // groupsize, N, 2] bfloat16
                        [..., 0] = scale, [..., 1] = zero (= min_val + scale * 8)
    """
    N, K = weight.shape
    assert K % groupsize == 0, f"K={K} not divisible by groupsize={groupsize}"
    assert K % (inner_k_tiles * 16) == 0, f"K={K} not divisible by {inner_k_tiles * 16}"
    assert N % 8 == 0, f"N={N} not divisible by 8"

    w = weight.float()
    w_grouped = w.reshape(-1, groupsize)   # [N * num_groups, groupsize]

    min_val = w_grouped.amin(dim=1, keepdim=True)
    max_val = w_grouped.amax(dim=1, keepdim=True)

    # scale maps [min_val, max_val] → [0, 15]
    scales = (max_val - min_val).clamp(min=1e-6) / 15.0

    # zeros: float value that maps to quantized midpoint 8
    # dequant formula used by kernel: w ≈ (q - 8) * scale + zeros
    # → zeros = min_val + scale * 8
    zeros = min_val + scales * 8.0

    # Quantize: q = clamp(round((w - min_val) / scale), 0, 15)
    w_int4 = (
        w_grouped.sub(min_val)
        .div(scales)
        .round()
        .clamp_(0, 15)
        .to(torch.uint8)
        .reshape(N, K)
    )

    # _convert_weight_to_int4pack must run on CUDA
    if w_int4.is_cuda:
        weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(w_int4, inner_k_tiles)
    else:
        weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
            w_int4.cuda(), inner_k_tiles
        ).cpu()

    # Pack into [K // groupsize, N, 2] bfloat16 as expected by _weight_int4pack_mm
    scales = scales.reshape(N, -1)   # [N, K // groupsize]
    zeros = zeros.reshape(N, -1)
    scales_and_zeros = (
        torch.cat([scales.unsqueeze(2), zeros.unsqueeze(2)], dim=2)  # [N, K//groupsize, 2]
        .transpose(0, 1)                                               # [K//groupsize, N, 2]
        .contiguous()
        .to(torch.bfloat16)
    )

    return weight_int4pack, scales_and_zeros


class Int4Linear(nn.Module):
    """nn.Linear replacement using native PyTorch int4 weight-only quantization."""

    def __init__(self, in_features: int, out_features: int,
                 groupsize: int = 128, inner_k_tiles: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        # Register buffers with correct shapes so load_state_dict(assign=True) works
        # (also works with meta device — buffers are meta tensors in that context)
        packed_shape = (out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2)
        n_groups = in_features // groupsize
        self.register_buffer("weight", torch.empty(packed_shape, dtype=torch.int32))
        self.register_buffer("scales_and_zeros", torch.empty((n_groups, out_features, 2), dtype=torch.bfloat16))

    @classmethod
    def from_linear(cls, linear: nn.Linear, groupsize: int = 128,
                    inner_k_tiles: int = 8) -> "Int4Linear":
        layer = cls(linear.in_features, linear.out_features, groupsize, inner_k_tiles)
        w_pack, s_z = _quantize_int4(linear.weight.data, groupsize, inner_k_tiles)
        layer.weight = w_pack
        layer.scales_and_zeros = s_z
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        x = x.reshape(-1, self.in_features).to(torch.bfloat16)
        out = torch.ops.aten._weight_int4pack_mm(
            x, self.weight, self.groupsize, self.scales_and_zeros
        )
        return out.reshape(*orig[:-1], self.out_features)


def quantize_model_int4(model: nn.Module, groupsize: int = 128) -> None:
    """Replace all compatible nn.Linear layers in-place with Int4Linear."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            N, K = child.weight.shape
            if K % groupsize == 0 and K % 128 == 0 and N % 8 == 0:
                setattr(model, name, Int4Linear.from_linear(child, groupsize))
        else:
            quantize_model_int4(child, groupsize)


def _setup_int4_structure(model: nn.Module, groupsize: int = 128) -> None:
    """Replace nn.Linear with empty Int4Linear shells (for loading cached quantized weights)."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            N, K = child.out_features, child.in_features
            if K % groupsize == 0 and K % 128 == 0 and N % 8 == 0:
                setattr(model, name, Int4Linear(K, N, groupsize))
        else:
            _setup_int4_structure(child, groupsize)


# ═══════════════════════════════════════════════════════════════════════
# Weight loading
# ═══════════════════════════════════════════════════════════════════════

_HF_PREFIX = "model.language_model."


def _set_param(module: nn.Module, key: str, tensor: torch.Tensor):
    """Navigate dotted key path and set the parameter/buffer in-place."""
    parts = key.split(".")
    mod = module
    for p in parts[:-1]:
        mod = getattr(mod, p)
    name = parts[-1]

    target = getattr(mod, name, None)
    if isinstance(target, nn.Parameter):
        target.data.copy_(tensor)
    elif isinstance(target, torch.Tensor):
        # Registered buffer
        target.copy_(tensor)
    else:
        # Raw attribute
        setattr(mod, name, tensor)


def load_weights(model: Gemma4Model, model_path: str):
    """Load BF16 safetensors weights into the model."""
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")

    loaded = set()
    skipped = set()
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                # Skip vision tower
                if "vision_tower" in key or "embed_vision" in key:
                    continue

                # Map HF key to our model key
                if key.startswith(_HF_PREFIX):
                    local_key = key[len(_HF_PREFIX):]
                else:
                    local_key = key

                # Skip v_proj for full-attention layers (k_eq_v — they share K as V).
                # Sliding layers DO have v_proj.
                if "v_proj" in local_key:
                    # Check if this layer's attention has v_proj
                    parts = local_key.split(".")
                    if len(parts) >= 2 and parts[0] == "layers":
                        layer_idx = int(parts[1])
                        if model.layers[layer_idx].self_attn.v_proj is None:
                            continue  # K=V layer, skip

                tensor = sf.get_tensor(key)
                try:
                    _set_param(model, local_key, tensor)
                    loaded.add(local_key)
                except (AttributeError, KeyError) as e:
                    skipped.add(local_key)

    print(f"  Loaded {len(loaded)} tensors from {len(files)} files")
    if skipped:
        print(f"  Skipped {len(skipped)}: {list(skipped)[:5]}...")


def load_model(model_id: str, quantize: bool = True, max_seq_len: int = 8192,
               device: str = "cuda") -> tuple["Gemma4Model", object]:
    """Download, load, optionally quantize, and return (model, tokenizer)."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    print(f"Downloading {model_id}...")
    model_path = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model"],
    )
    print(f"  Saved to {model_path}")

    # Load config
    config_file = os.path.join(model_path, "config.json")
    config = Gemma4Config.from_hf(config_file)
    print(f"  {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
          f"heads={config.num_attention_heads}")

    cache_path = os.path.join(model_path, "quantized_int4_g128.pt") if quantize else None

    if quantize and os.path.exists(cache_path):
        # Fast path: meta build + load cached quantized weights directly to GPU
        print("Loading cached quantized model...")
        with torch.device("meta"):
            model = Gemma4Model(config)
        _setup_int4_structure(model)
        state_dict = torch.load(cache_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, assign=True)
        model = model.to(dtype=torch.bfloat16)
    else:
        # Slow path: build on CPU, load bf16 weights, quantize
        print("Building model on CPU...")
        model = Gemma4Model(config)

        print("Loading weights...")
        load_weights(model, model_path)

        if quantize:
            print("Quantizing (int4 weight-only)...")
            quantize_model_int4(model)
            torch.cuda.empty_cache()
            print("  Done")
            print("Saving quantized cache for next run...")
            torch.save(model.state_dict(), cache_path)

        print(f"Moving to {device}...")
        model = model.to(device=device, dtype=torch.bfloat16)

    model.eval()
    model.setup_rope(max_seq_len, torch.device(device))

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Ready.")
    return model, tokenizer
