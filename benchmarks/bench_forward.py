"""测试 func.forward 速度与维度的关系"""

import time

import torch

from tnn_zh import SeparableDimNetwork

RANK = 10
N_1D = 100
DIMS = [5, 10, 20, 50, 100, 200, 500]
N_RUNS = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def bench_dim(dim: int) -> float:
    func = SeparableDimNetwork(dim=dim, rank=RANK).to(DEVICE, DTYPE)
    x = torch.rand(N_1D, dim, device=DEVICE, dtype=DTYPE)

    # warmup
    for _ in range(10):
        _ = func(x)
    sync()

    # benchmark
    start = time.perf_counter()
    for _ in range(N_RUNS):
        _ = func(x)
    sync()
    elapsed = time.perf_counter() - start

    return elapsed / N_RUNS * 1000  # ms


if __name__ == "__main__":
    print(f"device: {DEVICE}, rank={RANK}, n_1d={N_1D}, runs={N_RUNS}")
    print("-" * 30)
    for dim in DIMS:
        t = bench_dim(dim)
        print(f"dim={dim:3d}: {t:.3f} ms")
