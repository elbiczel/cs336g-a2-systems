# Assignment report

## End to End Benchmarks (MBAir)
|    | device   |   batch_size | compile   | model   |   context_length |   fwd avg (sec) |   fwd std (sec) |   back avg (sec) |   back std (sec) |   fws & back avg (sec) |   fws & back std (sec) | status   |
|---:|:---------|-------------:|:----------|:--------|-----------------:|----------------:|----------------:|-----------------:|-----------------:|-----------------------:|-----------------------:|:---------|
|  0 | cpu      |            4 | False     | S       |              256 |        0.292967 |       0.0101263 |         0.690617 |        0.0515558 |               0.983585 |              0.0505515 | ok       |
|  1 | cpu      |            4 | False     | M       |              256 |        1.07968  |       0.0161518 |         2.07311  |        0.122161  |               3.15279  |              0.121089  | ok       |
|  2 | cpu      |            4 | False     | L       |              256 |        2.517    |       0.0938035 |         5.54676  |        0.212342  |               8.06376  |              0.1905    | ok       |
|  0 | cpu      |            4 | True      | S       |              256 |        0.295624 |      0.00836923 |         0.671863 |        0.0613006 |               0.967486 |              0.0607266 | ok       |
|  1 | cpu      |            4 | True      | M       |              256 |        1.06595  |      0.0177598  |         2.06168  |        0.0460032 |               3.12763  |              0.0424368 | ok       |
|  2 | cpu      |            4 | True      | L       |              256 |        2.46093  |      0.0541857  |         5.35662  |        0.159464  |               7.81755  |              0.149976  | ok       |
|  0 | mps      |            4 | False     | S       |              256 |        0.152136 |     0.000323014 |         0.350633 |       0.00910346 |               0.502768 |             0.00909772 | ok       |
|  1 | mps      |            4 | False     | M       |              256 |        0.497226 |     0.0113818   |         1.05638  |       0.0234968  |               1.55361  |             0.0205562  | ok       |
|  2 | mps      |            4 | False     | L       |              256 |        1.08069  |     0.0256076   |         2.58213  |       0.252054   |               3.66281  |             0.25075    | ok       |
|  0 | mps      |            4 | True      | S       |              256 |        0.154377 |      0.00253189 |         0.351856 |        0.0111889 |               0.506234 |              0.0108987 | ok       |
|  1 | mps      |            4 | True      | M       |              256 |        0.493877 |      0.00840203 |         1.05395  |        0.0220565 |               1.54783  |              0.0203935 | ok       |
|  2 | mps      |            4 | True      | L       |              256 |        1.09464  |      0.0196739  |         3.15047  |        0.487463  |               4.24511  |              0.487066  | ok       |

## End to End Benchmarks (RTX PRO 6000 WK)

|    | model   |   context_length |   fwd avg (sec) |   fwd std (sec) |   back avg (sec) |   back std (sec) |   fws & back avg (sec) |   fws & back std (sec) | status   | dtype   | device   |   batch_size | compile   | profile_memory   |
|---:|:--------|-----------------:|----------------:|----------------:|-----------------:|-----------------:|-----------------------:|-----------------------:|:---------|:--------|:---------|-------------:|:----------|:-----------------|
|  0 | S       |              256 |       0.0101203 |     4.39494e-05 |        0.0141196 |      0.000145315 |              0.0242399 |            0.000138509 | ok       | float32 | cuda     |            4 | False     | False            |
|  1 | M       |              256 |       0.0202569 |     9.18798e-05 |        0.0340681 |      0.00021485  |              0.0543251 |            0.000194213 | ok       | float32 | cuda     |            4 | False     | False            |
|  2 | L       |              256 |       0.0299634 |     6.69663e-05 |        0.0640944 |      0.000134777 |              0.0940578 |            0.000116963 | ok       | float32 | cuda     |            4 | False     | False            |
|  3 | XL      |              256 |       0.0497967 |     4.44305e-05 |        0.111272  |      7.55653e-05 |              0.161069  |            6.11232e-05 | ok       | float32 | cuda     |            4 | False     | False            |
|  4 | 2p7     |              256 |       0.0603869 |     0.000155709 |        0.160571  |      0.000173022 |              0.220958  |            7.54412e-05 | ok       | float32 | cuda     |            4 | False     | False            |

### No Warm-up

|    | model   |   context_length |   fwd avg (sec) |   fwd std (sec) |   back avg (sec) |   back std (sec) |   fws & back avg (sec) |   fws & back std (sec) | status   | dtype   | device   |   batch_size | compile   | profile_memory   |
|---:|:--------|-----------------:|----------------:|----------------:|-----------------:|-----------------:|-----------------------:|-----------------------:|:---------|:--------|:---------|-------------:|:----------|:-----------------|
|  0 | S       |              256 |       0.0353005 |      0.0790442  |      -0.00274821 |       0.0821705  |              0.0325523 |            0.0224501   | ok       | float32 | cuda     |            4 | False     | False            |
|  1 | M       |              256 |       0.034033  |      0.0423463  |       0.0219129  |       0.0424403  |              0.0559458 |            0.00282343  | ok       | float32 | cuda     |            4 | False     | False            |
|  2 | L       |              256 |       0.0352378 |      0.0151886  |       0.0588802  |       0.0151966  |              0.094118  |            0.000493203 | ok       | float32 | cuda     |            4 | False     | False            |
|  3 | XL      |              256 |       0.052992  |      0.00997852 |       0.110647   |       0.0100072  |              0.163639  |            0.000756646 | ok       | float32 | cuda     |            4 | False     | False            |
|  4 | 2p7     |              256 |       0.062647  |      0.00871239 |       0.158404   |       0.00873731 |              0.221051  |            0.000659412 | ok       | float32 | cuda     |            4 | False     | False            |

### 1 step warm up

|    | model   |   context_length |   fwd avg (sec) |   fwd std (sec) |   back avg (sec) |   back std (sec) |   fws & back avg (sec) |   fws & back std (sec) | status   | dtype   | device   |   batch_size | compile   | profile_memory   |
|---:|:--------|-----------------:|----------------:|----------------:|-----------------:|-----------------:|-----------------------:|-----------------------:|:---------|:--------|:---------|-------------:|:----------|:-----------------|
|  0 | S       |              256 |       0.0103984 |     0.000120235 |        0.0148159 |      0.00131654  |              0.0252143 |            0.00131104  | ok       | float32 | cuda     |            4 | False     | False            |
|  1 | M       |              256 |       0.0305459 |     0.0310978   |        0.0548498 |      0.0825875   |              0.0853957 |            0.0765089   | ok       | float32 | cuda     |            4 | False     | False            |
|  2 | L       |              256 |       0.0305613 |     0.000196254 |        0.0641871 |      0.000258344 |              0.0947484 |            0.000168006 | ok       | float32 | cuda     |            4 | False     | False            |
|  3 | XL      |              256 |       0.0499383 |     8.81006e-05 |        0.114248  |      0.000240845 |              0.164186  |            0.000224153 | ok       | float32 | cuda     |            4 | False     | False            |
|  4 | 2p7     |              256 |       0.0600362 |     0.00103386  |        0.161622  |      0.00105505  |              0.221659  |            0.000210359 | ok       | float32 | cuda     |            4 | False     | False            |

### Mixed Precision (bfloat16)

|    | model   |   context_length |   fwd avg (sec) |   fwd std (sec) |   back avg (sec) |   back std (sec) |   fws & back avg (sec) |   fws & back std (sec) | status   | dtype    | device   |   batch_size | compile   | profile_memory   |
|---:|:--------|-----------------:|----------------:|----------------:|-----------------:|-----------------:|-----------------------:|-----------------------:|:---------|:---------|:---------|-------------:|:----------|:-----------------|
|  0 | S       |              256 |       0.0115468 |     0.00025475  |        0.016251  |      0.00029472  |              0.0277978 |            0.000148198 | ok       | bfloat16 | cuda     |            4 | False     | False            |
|  1 | M       |              256 |       0.0242751 |     0.000599148 |        0.0316248 |      0.000637135 |              0.0558999 |            0.000216708 | ok       | bfloat16 | cuda     |            4 | False     | False            |
|  2 | L       |              256 |       0.0333411 |     6.47849e-05 |        0.0668117 |      9.27532e-05 |              0.100153  |            6.63783e-05 | ok       | bfloat16 | cuda     |            4 | False     | False            |
|  3 | XL      |              256 |       0.0547619 |     6.14746e-05 |        0.112735  |      0.000142991 |              0.167497  |            0.000129102 | ok       | bfloat16 | cuda     |            4 | False     | False            |
|  4 | 2p7     |              256 |       0.0485177 |     4.27052e-05 |        0.115897  |      7.1541e-05  |              0.164415  |            5.73967e-05 | ok       | bfloat16 | cuda     |            4 | False     | False            |

### Compiled

|    | model   |   context_length |   fwd avg (sec) |   fwd std (sec) |   back avg (sec) |   back std (sec) |   fws & back avg (sec) |   fws & back std (sec) | status   | dtype   | device   |   batch_size | compile   | profile_memory   |
|---:|:--------|-----------------:|----------------:|----------------:|-----------------:|-----------------:|-----------------------:|-----------------------:|:---------|:--------|:---------|-------------:|:----------|:-----------------|
|  0 | S       |              256 |      0.00508969 |     2.39174e-05 |       0.00899177 |      2.61624e-05 |              0.0140815 |            1.06034e-05 | ok       | float32 | cuda     |            4 | True      | False            |
|  1 | M       |              256 |      0.0356223  |     0.0792748   |       0.00021299 |      0.079275    |              0.0358352 |            0.00017528  | ok       | float32 | cuda     |            4 | True      | False            |
|  2 | L       |              256 |      0.0179603  |     0.000290359 |       0.0503096  |      0.000323146 |              0.0682699 |            0.000141828 | ok       | float32 | cuda     |            4 | True      | False            |
|  3 | XL      |              256 |      0.0385235  |     0.000203871 |       0.0943493  |      0.000235015 |              0.132873  |            0.000116913 | ok       | float32 | cuda     |            4 | True      | False            |
|  4 | 2p7     |              256 |      0.0516872  |     0.000636176 |       0.141835   |      0.000672963 |              0.193522  |            0.000219452 | ok       | float32 | cuda     |            4 | True      | False            |

### Mixed precision (bfloat16) & compiled

|    | model   |   context_length |   fwd avg (sec) |   fwd std (sec) |   back avg (sec) |   back std (sec) |   fws & back avg (sec) |   fws & back std (sec) | status   | dtype    | device   |   batch_size | compile   | profile_memory   |
|---:|:--------|-----------------:|----------------:|----------------:|-----------------:|-----------------:|-----------------------:|-----------------------:|:---------|:---------|:---------|-------------:|:----------|:-----------------|
|  0 | S       |              256 |      0.00615928 |     4.13104e-05 |       0.00826505 |      0.000401961 |              0.0144243 |            0.000399833 | ok       | bfloat16 | cuda     |            4 | True      | False            |
|  1 | M       |              256 |      0.0132246  |     2.07604e-05 |       0.019979   |      0.000427588 |              0.0332035 |            0.000427084 | ok       | bfloat16 | cuda     |            4 | True      | False            |
|  2 | L       |              256 |      0.0194133  |     7.00106e-05 |       0.0518976  |      0.00020119  |              0.0713109 |            0.000188615 | ok       | bfloat16 | cuda     |            4 | True      | False            |
|  3 | XL      |              256 |      0.0421143  |     7.45255e-05 |       0.0954155  |      0.000139824 |              0.13753   |            0.000118307 | ok       | bfloat16 | cuda     |            4 | True      | False            |
|  4 | 2p7     |              256 |      0.0370287  |     0.000182895 |       0.102367   |      0.000217848 |              0.139396  |            0.000118352 | ok       | bfloat16 | cuda     |            4 | True      | False            |

## Profiling

`uv run nsys profile -o data/result --trace=cuda,nvtx,osrt  python cs336_systems/run_benchmark.py --context_lengths 128 256 512 1024`

### Optimizer Step

`uv run nsys profile -o data/result_step --trace=cuda,nvtx,osrt  python cs336_systems/run_benchmark.py --context_lengths 128 256 512 1024 --run_opt`

### Mixed precision (bfloat16) & compiled

`uv run nsys profile -o data/result_comp_bflt16 --trace=cuda,nvtx,osrt  python cs336_systems/run_benchmark.py --context_lengths 128 256 512 1024 --run_opt --compile --autocast_dtype bfloat16`

### Memory (without AdamW)

`uv run nsys profile -o data/result_mem --trace=cuda,nvtx,osrt  python cs336_systems/run_benchmark.py --context_lengths 128 256 512 --models 2p7 --profile_memory --profile_memory_path data/result_mem_snapshot.pickle`

### Memory

`uv run nsys profile -o data/result_mem_step --trace=cuda,nvtx,osrt  python cs336_systems/run_benchmark.py --context_lengths 128 256 512 --run_opt --models 2p7 --profile_memory --profile_memory_path data/result_mem_step_snapshot.pickle`

### Memory with Mixed precision (bfloat16)

`uv run nsys profile -o data/result_mem_mixed_step --trace=cuda,nvtx,osrt  python cs336_systems/run_benchmark.py --context_lengths 128 256 512 --run_opt --autocast_dtype bfloat16 --models 2p7 --profile_memory --profile_memory_path data/result_mem_mixed_step_snapshot.pickle`

## Mixed precision

### mixed_precision_accumulation

float32: tensor(10.0001)

float16: tensor(9.9531, dtype=torch.float16)

float32 adding float16: tensor(10.0021)

float32 adding float16 manually upcast: tensor(10.0021)


This is due to the precision of float16 that cannot represent 0.01 correctly, even when upcast before addition.

## Self Attention (Apple M4)

### Benchmark

`uv run cs336_systems/run_attn_benchmark.py  --device=mps --context_lengths 16 32 --d_model 256 1024`

|    |   context_length |   d_model |   fwd avg (sec) |   fwd std (sec) |   back avg (sec) |   back std (sec) |   fws & back avg (sec) |   fws & back std (sec) |   model mem |   model+fwd mem |   model+fwd+back mem |   fwd mem |   back mem | status   | dtype   | device   |   batch_size | compile   | profile_memory   |
|---:|-----------------:|----------:|----------------:|----------------:|-----------------:|-----------------:|-----------------------:|-----------------------:|------------:|----------------:|---------------------:|----------:|-----------:|:---------|:--------|:---------|-------------:|:----------|:-----------------|
|  0 |               16 |       256 |     0.000964747 |     0.000140073 |      0.000628642 |      0.000244892 |             0.00159339 |            0.000200877 |           0 |               0 |                    0 |         0 |          0 | ok       | float32 | mps      |            8 | False     | False            |
|  1 |               32 |       256 |     0.000999366 |     0.000207542 |      0.00126654  |      0.000234368 |             0.00226591 |            0.00010888  |           0 |               0 |                    0 |         0 |          0 | ok       | float32 | mps      |            8 | False     | False            |
|  2 |               16 |      1024 |     0.00278449  |     0.000719188 |      0.00711855  |      0.000768673 |             0.00990304 |            0.000271344 |           0 |               0 |                    0 |         0 |          0 | ok       | float32 | mps      |            8 | False     | False            |
|  3 |               32 |      1024 |     0.00197874  |     0.00104569  |      0.00137835  |      0.00104589  |             0.00335709 |            1.9964e-05  |           0 |               0 |                    0 |         0 |          0 | ok       | float32 | mps      |            8 | False     | False            |

### Compiled Benchmark

`uv run cs336_systems/run_attn_benchmark.py  --device=mps --context_lengths 16 32 --d_model 256 1024 --compile`

|    |   context_length |   d_model |   fwd avg (sec) |   fwd std (sec) |   back avg (sec) |   back std (sec) |   fws & back avg (sec) |   fws & back std (sec) |   model mem |   model+fwd mem |   model+fwd+back mem |   fwd mem |   back mem | status   | dtype   | device   |   batch_size | compile   | profile_memory   |
|---:|-----------------:|----------:|----------------:|----------------:|-----------------:|-----------------:|-----------------------:|-----------------------:|------------:|----------------:|---------------------:|----------:|-----------:|:---------|:--------|:---------|-------------:|:----------|:-----------------|
|  0 |               16 |       256 |     0.000878016 |     0.000211566 |       0.00113324 |      0.000303261 |             0.00201126 |            0.000217272 |           0 |               0 |                    0 |         0 |          0 | ok       | float32 | mps      |            8 | True      | False            |
|  1 |               32 |       256 |     0.000910615 |     0.000155417 |       0.00137933 |      0.000228465 |             0.00228994 |            0.000167456 |           0 |               0 |                    0 |         0 |          0 | ok       | float32 | mps      |            8 | True      | False            |
|  2 |               16 |      1024 |     0.00134004  |     0.000286955 |       0.00747273 |      0.000728483 |             0.00881277 |            0.000669585 |           0 |               0 |                    0 |         0 |          0 | ok       | float32 | mps      |            8 | True      | False            |
|  3 |               32 |      1024 |     0.00255449  |     0.00138711  |       0.0110278  |      0.00154581  |             0.0135822  |            0.000682247 |           0 |               0 |                    0 |         0 |          0 | ok       | float32 | mps      |            8 | True      | False            |

## Self Attention (RTX PRO 6000 WK)

### Benchmark

`uv run cs336_systems/run_attn_benchmark.py --context_lengths 16 32 64 128 --d_model 256 1024 4096 8192 16384`

### Compiled Benchmark

`uv run cs336_systems/run_attn_benchmark.py --context_lengths 16 32 64 128 --d_model 256 1024 4096 8192 16384 --compile`
