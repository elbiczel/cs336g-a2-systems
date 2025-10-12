uv run nsys profile -o data/result --trace=cuda,nvtx,osrt  python cs336_systems/run_benchmark.py --context_lengths 128 256 512 1024  $@
