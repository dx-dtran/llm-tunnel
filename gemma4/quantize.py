# Gemma 4 31B - gpt-fast quantization (int8 + int4)

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Transformer, find_multiple

##### Quantization Primitives ######


def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    eps = torch.finfo(torch.float32).eps

    min_val, max_val = torch.aminmax(x, dim=1)
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales, zero_points


def get_group_qparams(w, n_bit=4, groupsize=128):
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(
        torch.bfloat16
    ).reshape(w.shape[0], -1)


def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )


def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int32 = (
        to_quant.sub(min_val)
        .div(scales)
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int32)
        .reshape_as(w)
    )
    return w_int32


def group_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_group_qparams(w, n_bit, groupsize)
    w_int32 = group_quantize_tensor_from_qparams(w, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int32, scales_and_zeros


##### Weight-only int8 per-channel quantized code ######


def replace_linear_weight_only_int8_per_channel(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                WeightOnlyInt8Linear(child.in_features, child.out_features),
            )
        else:
            replace_linear_weight_only_int8_per_channel(child)


class WeightOnlyInt8QuantHandler:
    def __init__(self, mod):
        self.mod = mod

    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                int8_weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(), -128, 127, torch.int8
                )
                cur_state_dict[f"{fqn}.weight"] = int8_weight
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype)
        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_weight_only_int8_per_channel(self.mod)
        return self.mod


class WeightOnlyInt8Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
        device=None, dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.empty((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales


##### Weight-only int4 per-channel groupwise quantized code ######


def prepare_int4_weight_and_scales_and_zeros(weight_bf16, groupsize, inner_k_tiles):
    weight_int32, scales_and_zeros = group_quantize_tensor(
        weight_bf16, n_bit=4, groupsize=groupsize
    )
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
        weight_int32.to(torch.uint8), inner_k_tiles
    )
    return weight_int4pack, scales_and_zeros


def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(
        x, weight_int4pack, groupsize, scales_and_zeros
    )
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c


def _check_linear_int4_k(k, groupsize=1, inner_k_tiles=1):
    return k % groupsize == 0 and k % (inner_k_tiles * 16) == 0


def replace_linear_int4(module, groupsize, inner_k_tiles, padding):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if _check_linear_int4_k(child.in_features, groupsize, inner_k_tiles):
                setattr(
                    module,
                    name,
                    WeightOnlyInt4Linear(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        groupsize=groupsize,
                        inner_k_tiles=inner_k_tiles,
                        padding=False,
                    ),
                )
            elif padding:
                setattr(
                    module,
                    name,
                    WeightOnlyInt4Linear(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        groupsize=groupsize,
                        inner_k_tiles=inner_k_tiles,
                        padding=True,
                    ),
                )
        else:
            replace_linear_int4(child, groupsize, inner_k_tiles, padding)


class WeightOnlyInt4QuantHandler:
    def __init__(self, mod, groupsize=128, inner_k_tiles=8, padding=True):
        self.mod = mod
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding = padding
        assert groupsize in [32, 64, 128, 256]
        assert inner_k_tiles in [2, 4, 8]

    @torch.no_grad()
    def create_quantized_state_dict(self, use_cuda=True):
        device = "cuda" if use_cuda else "cpu"

        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                assert not mod.bias
                out_features = mod.out_features
                in_features = mod.in_features
                assert out_features % 8 == 0, "require out_features % 8 == 0"
                print(f"linear: {fqn}, in={in_features}, out={out_features}")

                weight = mod.weight.data
                if not _check_linear_int4_k(
                    in_features, self.groupsize, self.inner_k_tiles
                ):
                    if self.padding:
                        print(
                            f"warning: {fqn} is padded to satisfy in_features % 1024 == 0"
                        )
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        print(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                weight_int4pack, scales_and_zeros = (
                    prepare_int4_weight_and_scales_and_zeros(
                        weight.to(torch.bfloat16).to(device=device),
                        self.groupsize,
                        self.inner_k_tiles,
                    )
                )
                cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to("cpu")
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to("cpu")

        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_int4(
            self.mod, self.groupsize, self.inner_k_tiles, self.padding
        )
        return self.mod


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype=None,
        groupsize: int = 128,
        inner_k_tiles: int = 8,
        padding: bool = True,
    ) -> None:
        super().__init__()
        self.padding = padding
        if padding:
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert (
            in_features % (inner_k_tiles * 16) == 0
        ), "require in_features % (innerKTiles * 16) == 0"
        self.register_buffer(
            "weight",
            torch.empty(
                (
                    out_features // 8,
                    in_features // (inner_k_tiles * 16),
                    32,
                    inner_k_tiles // 2,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales_and_zeros",
            torch.empty(
                (in_features // groupsize, out_features, 2), dtype=torch.bfloat16
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.bfloat16)
        if self.padding:
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(
            input,
            self.weight,
            self.scales_and_zeros,
            self.out_features,
            self.groupsize,
        )


def quantize(
    checkpoint_path: Path = Path("checkpoints/google/gemma-4-31B-it/model.pth"),
    mode: str = "int8",
    groupsize: int = 128,
    label: str = "",
    model_name: Optional[str] = None,
) -> None:
    assert checkpoint_path.is_file(), checkpoint_path

    device = "cpu"
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()

    with torch.device("meta"):
        model = Transformer.from_name(model_name or checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(dtype=precision, device=device)

    if mode == "int8":
        print(
            "Quantizing model weights for int8 weight-only symmetric per-channel quantization"
        )
        quant_handler = WeightOnlyInt8QuantHandler(model)
        quantized_state_dict = quant_handler.create_quantized_state_dict()

        dir_name = checkpoint_path.parent
        base_name = checkpoint_path.name
        new_base_name = base_name.replace(".pth", f"{label}int8.pth")

    elif mode == "int4":
        print(
            "Quantizing model weights for int4 weight-only affine per-channel groupwise quantization"
        )
        quant_handler = WeightOnlyInt4QuantHandler(model, groupsize)
        quantized_state_dict = quant_handler.create_quantized_state_dict()

        dir_name = checkpoint_path.parent
        base_name = checkpoint_path.name
        new_base_name = base_name.replace(".pth", f"{label}int4.g{groupsize}.pth")

    else:
        raise ValueError(
            f"Invalid quantization mode {mode} needs to be one of [int8, int4]"
        )

    quantize_path = dir_name / new_base_name
    # Delete the source checkpoint to free disk space before saving
    print(f"Removing {checkpoint_path} to free disk space ...")
    checkpoint_path.unlink(missing_ok=True)
    print(f"Writing quantized weights to {quantize_path}")
    quantize_path.unlink(missing_ok=True)
    torch.save(quantized_state_dict, quantize_path)
    print(f"Quantization complete took {time.time() - t0:.02f} seconds")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize Gemma 4 model.")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/google/gemma-4-31B-it/model.pth"),
        help="Path to the model checkpoint to be quantized.",
    )
    parser.add_argument(
        "--mode",
        "-q",
        type=str,
        default="int8",
        choices=["int8", "int4"],
        help="type of quantization to perform",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        help="Group size for int4 quantization.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="_",
        help="label to add to output filename",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (e.g. gemma-4-31B-it). Required if checkpoint_path parent dir doesn't contain the model name.",
    )

    args = parser.parse_args()
    quantize(args.checkpoint_path, args.mode, args.groupsize, args.label, args.model_name)
