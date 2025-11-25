"""测试 加/乘/slice 后 func.forward 的速度"""

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


def bench_func_forward(func, x) -> float:
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


def make_tnn():
    func = SeparableDimNetwork(dim=DIM, rank=RANK).to(DEVICE, DTYPE)
    return TNN(dim=DIM, rank=RANK, func=func).to(DEVICE, DTYPE)


if __name__ == "__main__":
    print(
        f"device: {DEVICE}, dim={DIM}, rank={RANK}, n_1d={N_1D}, runs={N_RUNS}"
    )
    print("-" * 50)

    x = torch.rand(N_1D, DIM, device=DEVICE, dtype=DTYPE)
    tnn1 = make_tnn()

    # 基准
    t = bench_func_forward(tnn1.func, x)
    print(f"tnn.func.forward:              {t:.3f} ms")

    print("-" * 50)

    # 标量乘法
    tnn_scalar = 2.0 * tnn1
    t = bench_func_forward(tnn_scalar.func, x)
    print(f"(2.0*tnn).func.forward:        {t:.3f} ms")

    # 取负
    tnn_neg = -tnn1
    t = bench_func_forward(tnn_neg.func, x)
    print(f"(-tnn).func.forward:           {t:.3f} ms")

    print("-" * 50)

    # 2个相加
    tnn2 = make_tnn()
    tnn_add2 = tnn1 + tnn2
    t = bench_func_forward(tnn_add2.func, x)
    print(f"(tnn1+tnn2).func.forward:      {t:.3f} ms")

    # 3个相加
    tnn3 = make_tnn()
    tnn_add3 = tnn1 + tnn2 + tnn3
    t = bench_func_forward(tnn_add3.func, x)
    print(f"(tnn1+tnn2+tnn3).func.forward: {t:.3f} ms")

    # 5个相加
    tnn4, tnn5 = make_tnn(), make_tnn()
    tnn_add5 = tnn1 + tnn2 + tnn3 + tnn4 + tnn5
    t = bench_func_forward(tnn_add5.func, x)
    print(f"(5个tnn相加).func.forward:     {t:.3f} ms")

    print("-" * 50)

    # 2个相乘
    tnn_mul2 = tnn1 * tnn2
    t = bench_func_forward(tnn_mul2.func, x)
    print(f"(tnn1*tnn2).func.forward:      {t:.3f} ms")

    # 3个相乘
    tnn_mul3 = tnn1 * tnn2 * tnn3
    t = bench_func_forward(tnn_mul3.func, x)
    print(f"(tnn1*tnn2*tnn3).func.forward: {t:.3f} ms")

    print("-" * 50)

    # slice 1个维度
    tnn_slice1 = tnn1.slice({0: 0.5})
    x_slice1 = torch.rand(N_1D, DIM - 1, device=DEVICE, dtype=DTYPE)
    t = bench_func_forward(tnn_slice1.func, x_slice1)
    print(f"slice(1dim).func.forward:      {t:.3f} ms")

    # slice 3个维度
    tnn_slice3 = tnn1.slice({0: 0.5, 1: 0.5, 2: 0.5})
    x_slice3 = torch.rand(N_1D, DIM - 3, device=DEVICE, dtype=DTYPE)
    t = bench_func_forward(tnn_slice3.func, x_slice3)
    print(f"slice(3dim).func.forward:      {t:.3f} ms")
