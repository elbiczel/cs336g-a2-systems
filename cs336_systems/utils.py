"""Common utilities."""

import argparse
import dataclasses
import json
import logging
import uuid

from contextlib import contextmanager

import torch


def _make_nvtx():
    try:
        import torch
        import torch.cuda.nvtx as real_nvtx

        if torch.version.cuda is None or not torch.cuda.is_available():
            raise RuntimeError("NVTX disabled: no CUDA build or no CUDA device")

        return real_nvtx  # OK to use
    except Exception:
        # Dummy no-op shim with the same API surface you need
        class _DummyNVTX:
            @contextmanager
            def range(self, msg: str = ""):
                yield

            def mark(self, msg: str = ""):
                pass

        return _DummyNVTX()


nvtx = _make_nvtx()


def get_run_name(prefix: str | None) -> str | None:
    """Given an optional prefix, returns a name with a unique suffix"""
    if prefix:
        return f"{prefix}-{uuid.uuid4().hex[:6]}"
    return None


def save_argparse(args: argparse.Namespace, out_path: str) -> None:
    """Serializes the argparse.Namespace to a JSON file.

    Args:
        args: The parsed command-line arguments.
        out_path: The path to save the JSON file.
    """
    config_dict = vars(args)
    with open(out_path, "w") as f:
        json.dump(config_dict, f)


def save_dataclass(obj: object, out_path: str) -> None:
    """Writes the dictionary representation of `dataclass` to disk."""
    if not dataclasses.is_dataclass(obj):
        raise ValueError(f"Object of type {type(obj)} is not a dataclass.")
    obj_dict = dataclasses.asdict(obj)
    with open(out_path, "w") as f:
        json.dump(obj_dict, f)


def synchronize(device: str) -> None:
    if device == "cuda":
        logging.info("Synchronizing CUDA backend...")
        torch.cuda.synchronize()
    elif device == "mps":
        logging.info("Synchronizing MPS backend...")
        torch.mps.synchronize()
    elif device == "cpu":
        logging.info("Execution on CPU, no synchornization required.")
    else:
        raise ValueError(f"Unknown device type: {device.type}.")


def get_dtype(dtype_str: str) -> torch.dtype:
    """Returns a torch dtype for a corresponding string (e.g. 'float16')."""
    return getattr(torch, dtype_str.lower())
