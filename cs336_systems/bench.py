import timeit
import statistics


def benchmark(
    fn: callable,
    warmup: int,
    steps: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(steps):
        start_sec = timeit.default_timer()
        fn()
        end_sec = timeit.default_timer()
        times.append(end_sec - start_sec)
    avg = statistics.mean(times)
    std = statistics.stdev(times)

    return avg, std
