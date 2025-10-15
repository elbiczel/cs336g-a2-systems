import argparse
import torch
import numpy as np
import random
import itertools
import traceback

from typing import Any
from contextlib import nullcontext
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

import pandas as pd

from cs336_basics import model as model_lib
from cs336_systems import bench, utils
from utils import nvtx


def parse_params():
    parser = argparse.ArgumentParser(description="Benchmark configuration")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cpu, cuda, mps).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--context_lengths",
        type=int,
        default=[256],
        nargs="+",
        help="Context lengths to benchmark.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=[32],
        nargs="+",
        help="Head dimensions to benchmark.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="If should compile the model.",
    )
    parser.add_argument(
        "--profile_memory",
        action="store_true",
        help="If memeory profiling is enabled",
    )
    parser.add_argument(
        "--profile_memory_path",
        type=str,
        help="Path to file for mem profile. Only effective if --profile_memory is set.",
    )
    parser.add_argument(
        "--autocast_dtype",
        type=str,
        default="float32",
        help="dtype to use for autocasting the model. If float32 nothing is done.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup steps to perform before measurement.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of steps for the measurement.",
    )

    return parser.parse_args()


def get_data(
    device: str,
    batch_size: int,
    context_length: int,
    d_model: int,
    generator: torch.Generator | None = None,
) -> Float[Tensor, "batch seq d_model"]:
    return torch.rand(
        size=(batch_size, context_length, d_model),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )


class DummyPositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]
    ) -> Float[Tensor, " ... seq d"]:
        return x


def create_model(
    device: str,
    d_model: int,
    compile: bool,
) -> nn.Module:
    model = model_lib.CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=1,
        positional_encoder=DummyPositionalEmbedding(),
    )
    model = model.to(device)
    if compile:
        if device == "mps":
            model = torch.compile(model, backend="aot_eager")
        else:
            # Try mode="max-autotune" for CUDA.
            model = torch.compile(model)
    model = model.train()
    return model


def _benchmark(cfg, d_model, context_length) -> dict[str, Any]:
    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    xb = get_data(cfg.device, cfg.batch_size, context_length, d_model, gen)

    model = create_model(cfg.device, d_model, cfg.compile)
    dtype = utils.get_dtype(cfg.autocast_dtype)
    if cfg.device != "cuda" or dtype == torch.float32:
        cast_ctx = nullcontext()
    else:
        cast_ctx = torch.autocast(device_type="cuda", dtype=dtype)

    # Get memory usage.
    with nvtx.range("mem use"):
        with cast_ctx:
            utils.synchronize(cfg.device)
            allocated = torch.cuda.memory_allocated() / 1024**2  # in MB
            loss = model(xb).sum()
            utils.synchronize(cfg.device)
            fwd_allocated = torch.cuda.memory_allocated() / 1024**2  # in MB
            loss.backward()
            utils.synchronize(cfg.device)
            back_allocated = torch.cuda.memory_allocated() / 1024**2  # in MB
            utils.synchronize(cfg.device)

    # Benchmark
    def fwd():
        with nvtx.range("fwd"):
            with cast_ctx:
                _ = model(xb)
                utils.synchronize(cfg.device)

    def fwd_back():
        with nvtx.range("fwd_back"):
            with cast_ctx:
                with nvtx.range("fwd"):
                    yb = model(xb)
                with nvtx.range("loss"):
                    loss = yb.sum()
                with nvtx.range("back"):
                    loss.backward()
            utils.synchronize(cfg.device)

    profile_memory_path = (
        cfg.profile_memory_path
        if cfg.profile_memory and cfg.device == "cuda"
        else None
    )
    fwd_avg, fwd_std = bench.benchmark(
        fwd, cfg.warmup, cfg.steps, profile_memory_path
    )
    fwd_back_avg, fwd_back_std = bench.benchmark(
        fwd_back, cfg.warmup, cfg.steps, profile_memory_path
    )
    return {
        "context_length": context_length,
        "d_model": d_model,
        "fwd avg (sec)": fwd_avg,
        "fwd std (sec)": fwd_std,
        "back avg (sec)": fwd_back_avg - fwd_avg,
        "back std (sec)": np.sqrt(fwd_back_std**2 + fwd_std**2),
        "fws & back avg (sec)": fwd_back_avg,
        "fws & back std (sec)": fwd_back_std,
        "model mem": allocated,
        "model+fwd mem": fwd_allocated,
        "model+fwd+back mem": back_allocated,
        "fwd mem": fwd_allocated - allocated,
        "back mem": back_allocated - fwd_allocated,
        "status": "ok",
        "dtype": cfg.autocast_dtype,
        "device": cfg.device,
        "batch_size": cfg.batch_size,
        "compile": cfg.compile,
        "profile_memory": cfg.profile_memory,
    }


def benchmark(cfg, d_model, context_length) -> dict[str, Any]:
    try:
        print("Benchmarking: ", d_model, context_length)
        with nvtx.range(f"Attn({d_model}) ctx({context_length})"):
            return _benchmark(cfg, d_model, context_length)
    except Exception as e:
        traceback.print_exc()
        return {
            "context_length": context_length,
            "d_model": d_model,
            "fwd avg (sec)": np.nan,
            "fwd std (sec)": np.nan,
            "back avg (sec)": np.nan,
            "back std (sec)": np.nan,
            "fws & back avg (sec)": np.nan,
            "fws & back std (sec)": np.nan,
            "status": f"{type(e).__name__}: {e}",
            "dtype": cfg.autocast_dtype,
            "device": cfg.device,
            "batch_size": cfg.batch_size,
            "compile": cfg.compile,
            "profile_memory": cfg.profile_memory,
        }


def main():
    cfg = parse_params()
    device = cfg.device
    assert device in ["cuda", "mps", "cpu"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Tried running on CUDA, but it's not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "Tried running on Apple Silicon, but it's not available"
        )
    if device == "mps" and not torch.backends.mps.is_built():
        raise ValueError("Tried running on Apple Silicon, but it's not built")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if device == "cuda":
        model_lib.scaled_dot_product_attention = (
            model_lib.annotated_scaled_dot_product_attention
        )
        if cfg.autocast_dtype == "float32":
            torch.set_float32_matmul_precision("high")

    rows = [
        benchmark(cfg, *full_cfg)
        for full_cfg in itertools.product(cfg.d_model, cfg.context_lengths)
    ]
    df = pd.DataFrame(rows)
    print(df.to_markdown())


if __name__ == "__main__":
    main()
