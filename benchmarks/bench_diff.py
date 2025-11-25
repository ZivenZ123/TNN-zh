"""测试 grad/grad2/laplace 后 func.forward 的速度"""

import time

import torch

from tnn_zh import TNN, SeparableDimNetwork

DIM = 10
RANK = 10
N_1D = 100
N_RUNS = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def bench_func_forward(func, x, name: str) -> float:
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
    print(
        f"device: {DEVICE}, dim={DIM}, rank={RANK}, n_1d={N_1D}, runs={N_RUNS}"
    )
    print("-" * 40)

    func = SeparableDimNetwork(dim=DIM, rank=RANK).to(DEVICE, DTYPE)
    tnn = TNN(dim=DIM, rank=RANK, func=func).to(DEVICE, DTYPE)
    x = torch.rand(N_1D, DIM, device=DEVICE, dtype=DTYPE)

    # 原始 tnn.func
    t = bench_func_forward(tnn.func, x, "tnn.func")
    print(f"tnn.func.forward:           {t:.3f} ms")

    # grad 后的 func
    tnn_grad = tnn.grad(0)
    t = bench_func_forward(tnn_grad.func, x, "grad.func")
    print(f"tnn.grad(0).func.forward:   {t:.3f} ms")

    # grad2 后的 func
    tnn_grad2 = tnn.grad2(0, 0)
    t = bench_func_forward(tnn_grad2.func, x, "grad2.func")
    print(f"tnn.grad2(0,0).func.forward:{t:.3f} ms")

    # laplace 后的 func
    tnn_lap = tnn.laplace()
    t = bench_func_forward(tnn_lap.func, x, "laplace.func")
    print(f"tnn.laplace().func.forward: {t:.3f} ms")
