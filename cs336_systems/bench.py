import timeit
import statistics

import torch

from cs336_systems.utils import nvtx


def benchmark(
    fn: callable,
    warmup: int,
    steps: int,
    profile_memory_path: str | None,
) -> tuple[float, float]:
    with nvtx.range("warmup"):
        for _ in range(warmup):
            fn()

    times = []
    if profile_memory_path:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    with nvtx.range("benchmark"):
        for _ in range(steps):
            start_sec = timeit.default_timer()
            fn()
            end_sec = timeit.default_timer()
            times.append(end_sec - start_sec)
    if profile_memory_path:
        torch.cuda.memory._dump_snapshot(profile_memory_path)
        torch.cuda.memory._record_memory_history(enabled=None)
    
    avg = statistics.mean(times)
    std = statistics.stdev(times)

    return avg, std
