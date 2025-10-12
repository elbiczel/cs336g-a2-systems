import argparse
import torch
import numpy as np
import random
import itertools
import traceback

from typing import Any

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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model (default: True, disable with --no-compile)",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(compile=True)
    parser.add_argument(
        "--run_opt",
        action="store_true",
        help="If should include optimizer step.",
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
    device: str, context_length: int, config: settings.TransformerConfig
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
    if device == "cpu":
        model = torch.compile(model)
    elif device == "mps":
        model = torch.compile(model, backend="aot_eager")
    else:
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model, mode="max-autotune")
    model = model.train()
    return model


def _benchmark(cfg, transformer_cfg, context_length) -> dict[str, Any]:
    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    xb, yb = get_data(cfg.device, context_length, cfg.batch_size, gen)

    model = create_model(cfg.device, context_length, transformer_cfg)
    opt = optimizer.AdamW(model.parameters())

    def fwd():
        with nvtx.range("fwd"):
            _ = model(xb)
            utils.synchronize(cfg.device)

    def fwd_back():
        with nvtx.range("fwd_back"):
            with nvtx.range("fwd"):
                logits = model(xb)
            with nvtx.range("loss"):
                loss = nn_utils.cross_entropy(logits, yb)
            if cfg.run_opt:
                opt.zero_grad()
            with nvtx.range("back"):
                loss.backward()
            if cfg.run_opt:
                with nvtx.range("opt_step"):
                    opt.step()
            utils.synchronize(cfg.device)

    fwd_avg, fwd_std = bench.benchmark(fwd, cfg.warmup, cfg.steps)
    fwd_back_avg, fwd_back_std = bench.benchmark(
        fwd_back, cfg.warmup, cfg.steps
    )
    return {
        "device": cfg.device,
        "batch_size": cfg.batch_size,
        "compile": cfg.compile,
        "model": transformer_cfg.name,
        "context_length": context_length,
        "fwd avg (sec)": fwd_avg,
        "fwd std (sec)": fwd_std,
        "back avg (sec)": fwd_back_avg - fwd_avg,
        "back std (sec)": np.sqrt(fwd_back_std**2 + fwd_std**2),
        "fws & back avg (sec)": fwd_back_avg,
        "fws & back std (sec)": fwd_back_std,
        "status": "ok",
    }


def benchmark(cfg, transformer_cfg, context_length) -> dict[str, Any]:
    try:
        print("Benchmarking: ", transformer_cfg.name, context_length)
        with nvtx.range(f"Model({transformer_cfg.name}) ctx({context_length})"):
            return _benchmark(cfg, transformer_cfg, context_length)
    except Exception as e:
        traceback.print_exc()
        return {
            "device": cfg.device,
            "batch_size": cfg.batch_size,
            "compile": cfg.compile,
            "model": transformer_cfg.name,
            "context_length": context_length,
            "fwd avg (sec)": np.nan,
            "fwd std (sec)": np.nan,
            "back avg (sec)": np.nan,
            "back std (sec)": np.nan,
            "fws & back avg (sec)": np.nan,
            "fws & back std (sec)": np.nan,
            "status": f"{type(e).__name__}: {e}",
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

    model_cfgs = [
        settings.small(),
        settings.med(),
        settings.large(),
        #        settings.xl(),
        #        settings.two_seven_b(),
    ]

    rows = [
        benchmark(cfg, *full_cfg)
        for full_cfg in itertools.product(model_cfgs, cfg.context_lengths)
    ]
    df = pd.DataFrame(rows)
    print(df.to_markdown())


if __name__ == "__main__":
    main()
