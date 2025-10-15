import argparse
import torch
import numpy as np
import random
import itertools
import traceback

from typing import Any
from contextlib import nullcontext

import pandas as pd

from cs336_basics import model as model_lib, nn_utils, optimizer
from cs336_systems import bench, settings, utils
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
        "--models",
        type=str,
        default=[],
        nargs="+",
        help="Model names to profile. If empty profiles all.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="If should compile the model.",
    )
    parser.add_argument(
        "--run_opt",
        action="store_true",
        help="If profiling should include optimizer step.",
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
        default=5,
        help="Warmup steps to perform before measurement.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of steps for the measurement.",
    )

    return parser.parse_args()


def get_data(
    device: str,
    context_length: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    vocab_size = 10_000
    seq = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    return seq, targets


def create_model(
    device: str,
    context_length: int,
    config: settings.TransformerConfig,
    compile: bool,
) -> model_lib.BasicsTransformerLM:
    model = model_lib.BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
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


def _benchmark(cfg, transformer_cfg, context_length) -> dict[str, Any]:
    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    xb, yb = get_data(cfg.device, context_length, cfg.batch_size, gen)

    model = create_model(
        cfg.device, context_length, transformer_cfg, cfg.compile
    )
    opt = optimizer.AdamW(model.parameters()) if cfg.run_opt else None
    dtype = utils.get_dtype(cfg.autocast_dtype)

    if cfg.device != "cuda" or dtype == torch.float32:
        cast_ctx = nullcontext()
    else:
        cast_ctx = torch.autocast(device_type="cuda", dtype=dtype)

    # Get memory usage.
    with nvtx.range("mem use"):
        utils.synchronize(cfg.device)
        allocated = torch.cuda.memory_allocated() / 1024**2  # in MB
        loss = model(xb).sum()
        utils.synchronize(cfg.device)
        fwd_allocated = torch.cuda.memory_allocated() / 1024**2  # in MB
        if opt is not None:
            opt.zero_grad()
        loss.backward()
        utils.synchronize(cfg.device)
        back_allocated = torch.cuda.memory_allocated() / 1024**2  # in MB
        if opt is not None:
            opt.step()
            utils.synchronize(cfg.device)
            opt_allocated = torch.cuda.memory_allocated() / 1024**2  # in MB
            utils.synchronize(cfg.device)
        else:
            opt_allocated = np.nan
            utils.synchronize(cfg.device)

    def fwd():
        with nvtx.range("fwd"):
            with cast_ctx:
                _ = model(xb)
                utils.synchronize(cfg.device)

    def fwd_back():
        with nvtx.range("fwd_back"):
            with cast_ctx:
                with nvtx.range("fwd"):
                    logits = model(xb)
                with nvtx.range("loss"):
                    loss = nn_utils.cross_entropy(logits, yb)
                if opt is not None:
                    opt.zero_grad()
                with nvtx.range("back"):
                    loss.backward()
            if opt is not None:
                with nvtx.range("opt_step"):
                    opt.step()
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
        "model": transformer_cfg.name,
        "context_length": context_length,
        "fwd avg (sec)": fwd_avg,
        "fwd std (sec)": fwd_std,
        "back avg (sec)": fwd_back_avg - fwd_avg,
        "back std (sec)": np.sqrt(fwd_back_std**2 + fwd_std**2),
        "fws & back avg (sec)": fwd_back_avg,
        "fws & back std (sec)": fwd_back_std,
        "status": "ok",
        "dtype": cfg.autocast_dtype,
        "device": cfg.device,
        "batch_size": cfg.batch_size,
        "compile": cfg.compile,
        "profile_memory": cfg.profile_memory,
        **(
            {
                "model mem": allocated,
                "model+fwd mem": fwd_allocated,
                "model+fwd+back mem": back_allocated,
                "model+fwd+back_opt mem": opt_allocated,
                "fwd mem": fwd_allocated - allocated,
                "back mem": back_allocated - fwd_allocated,
                "opt mem": opt_allocated - back_allocated,
            }
            if cfg.profile_memory
            else {}
        ),
    }


def benchmark(cfg, transformer_cfg, context_length) -> dict[str, Any]:
    try:
        print("Benchmarking: ", transformer_cfg.name, context_length)
        with nvtx.range(f"Model({transformer_cfg.name}) ctx({context_length})"):
            return _benchmark(cfg, transformer_cfg, context_length)
    except Exception as e:
        traceback.print_exc()
        return {
            "model": transformer_cfg.name,
            "context_length": context_length,
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

    model_cfgs = [
        settings.small(),
        settings.med(),
        settings.large(),
        settings.xl(),
        settings.two_seven_b(),
    ]
    if cfg.models:
        model_cfgs = [m_cfg for m_cfg in model_cfgs if m_cfg.name in cfg.models]

    rows = [
        benchmark(cfg, *full_cfg)
        for full_cfg in itertools.product(model_cfgs, cfg.context_lengths)
    ]
    df = pd.DataFrame(rows)
    print(df.to_markdown())


if __name__ == "__main__":
    main()
