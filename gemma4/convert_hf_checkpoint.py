# Gemma 4 31B - convert HuggingFace checkpoint to gpt-fast format

import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file as load_safetensors_file

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import ModelArgs, LAYER_PATTERN


def _get_layer_type(layer_idx: int, n_layer: int) -> str:
    layer_types = (LAYER_PATTERN * (n_layer // len(LAYER_PATTERN)))[:n_layer]
    return layer_types[layer_idx]


def permute(w, n_head, head_dim):
    """Convert Q/K weights from HF rotate_half to gpt-fast interleaved RoPE convention."""
    dim = w.shape[1]
    return (
        w.view(n_head, 2, head_dim // 2, dim)
        .transpose(1, 2)
        .reshape(n_head * head_dim, dim)
    )


def permute_partial(w, n_head, head_dim, rotary_dim):
    """Permute only the first rotary_dim elements per head (for partial RoPE)."""
    dim = w.shape[1]
    w = w.view(n_head, head_dim, dim)
    w_rot = w[:, :rotary_dim, :]
    w_pass = w[:, rotary_dim:, :]
    w_rot = (
        w_rot.reshape(n_head, 2, rotary_dim // 2, dim)
        .transpose(1, 2)
        .reshape(n_head, rotary_dim, dim)
    )
    w = torch.cat([w_rot, w_pass], dim=1)
    return w.reshape(n_head * head_dim, dim)


@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("checkpoints/google/gemma-4-31B-it"),
    model_name: Optional[str] = None,
    save: bool = True,
) -> dict:
    if model_name is None:
        model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    print(f"Model config {config.__dict__}")

    # Load the json file containing weight mapping
    model_map_json_safetensors = checkpoint_dir / "model.safetensors.index.json"
    model_map_json = None

    try:
        assert model_map_json_safetensors.is_file()
        model_map_json = model_map_json_safetensors
        print(f"Found safetensors index at {model_map_json_safetensors}")
    except AssertionError:
        print(f"{model_map_json_safetensors} not found")

    if model_map_json is None:
        # Try single safetensors file
        single_file = checkpoint_dir / "model.safetensors"
        if single_file.is_file():
            print(f"Found single safetensors file at {single_file}")
            merged_result = load_safetensors_file(str(single_file), device="cpu")
        else:
            raise Exception("No model weights found!")
    else:
        with open(model_map_json) as json_map:
            bin_index = json.load(json_map)

        bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

        merged_result = {}
        for file in sorted(bin_files):
            print(f"Loading {file}...")
            state_dict = load_safetensors_file(str(file), device="cpu")
            merged_result.update(state_dict)

    # Detect prefix: everything before "embed_tokens.weight" in the language model
    # e.g. "model.embed_tokens.weight" -> prefix = "model."
    # e.g. "model.language_model.embed_tokens.weight" -> prefix = "model.language_model."
    prefix = ""
    for key in merged_result:
        if key.endswith("embed_tokens.weight") and "vision" not in key:
            prefix = key[: -len("embed_tokens.weight")]
            break
    print(f"Using prefix: '{prefix}'")

    # Weight name mapping (HF -> gpt-fast)
    weight_map = {
        f"{prefix}embed_tokens.weight": "tok_embeddings.weight",
        f"{prefix}norm.weight": "norm.weight",
        # Per-layer mappings handled separately
    }

    # Per-layer mapping templates
    layer_map = {
        "self_attn.q_proj.weight": "attention.wq.weight",
        "self_attn.k_proj.weight": "attention.wk.weight",
        "self_attn.v_proj.weight": "attention.wv.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight": "ffn_norm.weight",
        "post_feedforward_layernorm.weight": "post_ffn_norm.weight",
        "self_attn.q_norm.weight": "attention.q_norm.weight",
        "self_attn.k_norm.weight": "attention.k_norm.weight",
        "layer_scalar": "layer_scalar",
    }

    final_result = {}

    for key, value in merged_result.items():
        # Skip non-language-model weights (vision encoder, multi-modal projector)
        if "vision_tower" in key or "multi_modal_projector" in key:
            continue
        # Skip rotary embedding buffers
        if "rotary_emb" in key:
            continue

        # Handle top-level weights
        if key in weight_map:
            final_result[weight_map[key]] = value
            continue

        # Handle per-layer weights
        layer_match = re.search(
            rf"^{re.escape(prefix)}layers\.(\d+)\.(.*)", key
        )
        if layer_match:
            layer_idx = int(layer_match.group(1))
            suffix = layer_match.group(2)

            if suffix not in layer_map:
                print(f"Skipping unknown weight: {key}")
                continue

            # q_norm, k_norm, layer_scalar only exist on global layers in the model
            layer_type = _get_layer_type(layer_idx, config.n_layer)
            global_only = {"self_attn.q_norm.weight", "self_attn.k_norm.weight", "layer_scalar"}
            if layer_type == "sliding" and suffix in global_only:
                continue

            new_suffix = layer_map[suffix]
            new_key = f"layers.{layer_idx}.{new_suffix}"
            final_result[new_key] = value
        else:
            print(f"Skipping unmapped weight: {key}")

    # Now permute Q/K weights and combine into wqkv per layer
    for layer_idx in range(config.n_layer):
        layer_type = _get_layer_type(layer_idx, config.n_layer)
        pre = f"layers.{layer_idx}.attention."

        wq_key = pre + "wq.weight"
        wk_key = pre + "wk.weight"
        wv_key = pre + "wv.weight"

        if wq_key not in final_result:
            print(f"Warning: missing Q weight for layer {layer_idx}")
            continue

        q = final_result.pop(wq_key)
        k = final_result.pop(wk_key)

        if layer_type == "sliding":
            # Full rotary: permute all head elements
            q = permute(q, config.n_head, config.sliding_head_dim)
            k = permute(k, config.sliding_n_kv_heads, config.sliding_head_dim)
            v = final_result.pop(wv_key)
            final_result[pre + "wqkv.weight"] = torch.cat([q, k, v])
        else:
            # Partial rotary: permute only first rotary_dim elements
            q = permute_partial(
                q, config.n_head, config.global_head_dim, config.global_rotary_dim
            )
            k = permute_partial(
                k, config.global_n_kv_heads, config.global_head_dim, config.global_rotary_dim
            )
            # Global layers: no V weight (K=V)
            if wv_key in final_result:
                # Some checkpoints might still have v_proj for global layers
                final_result.pop(wv_key)
            final_result[pre + "wqkv.weight"] = torch.cat([q, k])

    print(f"Converted {len(final_result)} weight tensors")

    if save:
        print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
        torch.save(final_result, checkpoint_dir / "model.pth")
        print("Done!")

    return final_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Gemma 4 checkpoint."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/google/gemma-4-31B-it"),
    )
    parser.add_argument("--model_name", type=str, default=None)

    args = parser.parse_args()
    convert_hf_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
    )
