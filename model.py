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
            num_global_key_value_heads=tc.get("num_global_key_value_heads", 4),
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
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))  # Gemma inits to 0, adds 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight + 1.0, self.eps)


# ─── Rotary position embeddings ────────────────────────────────────────

def _precompute_rope(dim: int, max_len: int, theta: float, device: torch.device):
    """Returns cos, sin tensors of shape [max_len, dim//2]."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # [max_len, dim//2]
    return angles.cos(), angles.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                pos: torch.Tensor, rotary_dim: Optional[int] = None):
    """Apply rotary embeddings.

    x:   [batch, n_heads, seq_len, head_dim]
    cos: [max_len, rotary_dim//2]
    sin: [max_len, rotary_dim//2]
    pos: [seq_len] — absolute position indices
    """
    if rotary_dim is not None and rotary_dim < x.shape[-1]:
        x_rot, x_pass = x.split([rotary_dim, x.shape[-1] - rotary_dim], dim=-1)
    else:
        x_rot, x_pass = x, None

    # Gather cos/sin for the given positions: [seq_len, dim//2]
    c = cos[pos]  # [seq_len, dim//2]
    s = sin[pos]
    # Broadcast to [1, 1, seq_len, dim//2]
    c = c.unsqueeze(0).unsqueeze(0)
    s = s.unsqueeze(0).unsqueeze(0)

    # Pair-wise rotation: treat last dim as [..., dim//2, 2]
    x_rot = x_rot.unflatten(-1, (-1, 2))
    x0, x1 = x_rot.unbind(-1)
    x_rot = torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2)

    if x_pass is not None:
        return torch.cat([x_rot, x_pass], dim=-1)
    return x_rot


# ─── Attention ──────────────────────────────────────────────────────────

class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        is_full = self.layer_type == "full_attention"

        self.n_heads = config.num_attention_heads
        self.head_dim = config.global_head_dim if is_full else config.head_dim
        self.n_kv_heads = config.num_global_key_value_heads if is_full else config.num_key_value_heads
        self.kv_groups = self.n_heads // self.n_kv_heads
        self.is_full = is_full
        self.sliding_window = config.sliding_window

        # Partial rotary for full attention
        if is_full:
            self.rotary_dim = int(config.partial_rotary_factor * self.head_dim)
        else:
            self.rotary_dim = self.head_dim  # full rotation

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        # k_eq_v: no separate v_proj — we reuse K as V
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                pos: torch.Tensor, kv_cache: "KVCache", layer_idx: int,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = k  # k_eq_v

        # QK normalization (before RoPE)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        rotary_dim = self.rotary_dim if self.is_full else None
        q = _apply_rope(q, rope_cos, rope_sin, pos, rotary_dim)
        k = _apply_rope(k, rope_cos, rope_sin, pos, rotary_dim)

        # Update KV cache and get full K, V for attention
        k, v = kv_cache.update(layer_idx, k, v, pos)

        # Expand KV heads for GQA
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # Attention
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

        self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                pos: torch.Tensor, kv_cache: "KVCache", layer_idx: int,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention block
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, rope_cos, rope_sin, pos, kv_cache, layer_idx, mask)
        x = self.post_attention_layernorm(x)
        x = residual + x

        # MLP block
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        # Per-layer scaling
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

        # Precomputed RoPE tables (registered as buffers, will move with .to())
        # Sliding: full rotation, theta=10k
        self._rope_sliding_cos: torch.Tensor
        self._rope_sliding_sin: torch.Tensor
        # Full: partial rotation, theta=1M
        self._rope_full_cos: torch.Tensor
        self._rope_full_sin: torch.Tensor

    def setup_rope(self, max_len: int, device: torch.device):
        """Precompute RoPE cos/sin tables. Call after moving model to device."""
        sc, ss = _precompute_rope(self.config.head_dim, max_len,
                                  self.config.rope_theta_sliding, device)
        self._rope_sliding_cos = sc
        self._rope_sliding_sin = ss

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
            nkv = config.num_global_key_value_heads if is_full else config.num_key_value_heads
            cache_len = max_seq_len if is_full else config.sliding_window
            shape = (batch_size, nkv, cache_len, hd)
            self.k_caches.append(torch.zeros(shape, device=device, dtype=dtype))
            self.v_caches.append(torch.zeros(shape, device=device, dtype=dtype))

        # Track how many positions have been filled (for full attention layers)
        self.seq_len = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
               pos: torch.Tensor):
        """Write new K, V into cache and return the full cached K, V for attention.

        k, v: [batch, n_kv_heads, new_len, head_dim]
        pos:  [new_len] — absolute position indices
        """
        is_full = self.config.layer_types[layer_idx] == "full_attention"

        if is_full:
            # Write at the absolute positions
            self.k_caches[layer_idx][:, :, pos] = k
            self.v_caches[layer_idx][:, :, pos] = v
            # Return everything up to current position
            end = pos[-1].item() + 1
            return (self.k_caches[layer_idx][:, :, :end],
                    self.v_caches[layer_idx][:, :, :end])
        else:
            # Ring buffer: write at pos % window_size
            window = self.config.sliding_window
            idx = pos % window
            self.k_caches[layer_idx][:, :, idx] = k
            self.v_caches[layer_idx][:, :, idx] = v
            # Return all valid entries
            end = min(pos[-1].item() + 1, window)
            return (self.k_caches[layer_idx][:, :, :end],
                    self.v_caches[layer_idx][:, :, :end])

    def reset(self):
        for c in self.k_caches + self.v_caches:
            c.zero_()
        self.seq_len = 0


# ═══════════════════════════════════════════════════════════════════════
# Masks
# ═══════════════════════════════════════════════════════════════════════

def _make_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype):
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
    mask = (col <= row) & (row - col < window)
    return mask


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
        # Remove tokens with cumulative probability above top_p
        mask = cumulative - probs > top_p
        sorted_logits[mask] = float("-inf")
        # Scatter back
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
    full_mask = _make_causal_mask(seq_len, device, tokens.dtype)

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
# Weight loading
# ═══════════════════════════════════════════════════════════════════════

_HF_PREFIX = "model.language_model."


def _set_param(model: nn.Module, key: str, tensor: torch.Tensor):
    """Navigate dotted key path and set the parameter/buffer in-place."""
    parts = key.split(".")
    mod = model
    for p in parts[:-1]:
        mod = getattr(mod, p)
    name = parts[-1]
    if isinstance(getattr(mod, name, None), nn.Parameter):
        getattr(mod, name).data.copy_(tensor)
    else:
        # Buffer or raw attribute (e.g. layer_scalar stored as param)
        setattr(mod, name, nn.Parameter(tensor, requires_grad=False))


def load_weights(model: Gemma4Model, model_path: str):
    """Load BF16 safetensors weights into the model."""
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")

    loaded = set()
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                # Skip vision tower
                if "vision_tower" in key or "embed_vision" in key:
                    continue
                # Map HF key to our model key
                if key.startswith(_HF_PREFIX):
                    local_key = key[len(_HF_PREFIX):]
                elif key == "model.embed_vision.embedding_projection.weight":
                    continue
                else:
                    local_key = key

                # Skip v_proj (k_eq_v: K serves as V)
                if "v_proj" in local_key:
                    continue

                tensor = sf.get_tensor(key)
                try:
                    _set_param(model, local_key, tensor)
                    loaded.add(local_key)
                except (AttributeError, KeyError):
                    pass  # skip unexpected keys

    print(f"  Loaded {len(loaded)} tensors from {len(files)} files")


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

    # Build model on CPU
    print("Building model on CPU...")
    model = Gemma4Model(config)

    # Load weights
    print("Loading weights...")
    load_weights(model, model_path)

    # Quantize (int4 weight-only via torchao)
    if quantize:
        print("Quantizing (int4 weight-only)...")
        from torchao.quantization import quantize_, int4_weight_only
        quantize_(model, int4_weight_only(group_size=128))
        print("  Done")

    # Move to device
    print(f"Moving to {device}...")
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    # Setup RoPE tables
    model.setup_rope(max_seq_len, torch.device(device))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Ready.")
    return model, tokenizer
