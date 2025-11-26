"""测试 func.forward / l2_norm / backward 速度对比"""

import time

import torch

from tnn_zh import (
    TNN,
    SeparableDimNetwork,
    generate_quad_points,
    l2_norm,
)

DIM = 10
RANK = 10
N_1D = 100
N_RUNS = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    print(
        f"device: {DEVICE}, dim={DIM}, rank={RANK}, n_1d={N_1D}, runs={N_RUNS}"
    )
    print("-" * 40)

    func = SeparableDimNetwork(dim=DIM, rank=RANK).to(DEVICE, DTYPE)
    tnn = TNN(dim=DIM, rank=RANK, func=func).to(DEVICE, DTYPE)

    # 生成积分点
    bounds = [(0.0, 1.0) for _ in range(DIM)]
    pts, w = generate_quad_points(bounds, device=DEVICE, dtype=DTYPE)

    # warmup
    for _ in range(10):
        _ = func(pts)
        loss = l2_norm(tnn, pts, w)
        loss.backward()
    sync()

    # 1. func.forward
    start = time.perf_counter()
    for _ in range(N_RUNS):
        _ = func(pts)
    sync()
    t_forward = (time.perf_counter() - start) / N_RUNS * 1000

    # 2. l2_norm
    start = time.perf_counter()
    for _ in range(N_RUNS):
        _ = l2_norm(tnn, pts, w)
    sync()
    t_int_l2 = (time.perf_counter() - start) / N_RUNS * 1000

    # 3. l2_norm + backward
    start = time.perf_counter()
    for _ in range(N_RUNS):
        loss = l2_norm(tnn, pts, w)
        loss.backward()
    sync()
    t_backward = (time.perf_counter() - start) / N_RUNS * 1000

    print(f"[单个tnn] func.forward:         {t_forward:.3f} ms")
    print(f"[单个tnn] l2_norm:           {t_int_l2:.3f} ms")
    print(f"[单个tnn] l2_norm+backward:  {t_backward:.3f} ms")

    print("-" * 40)

    # 两个 tnn 相加
    func2 = SeparableDimNetwork(dim=DIM, rank=RANK).to(DEVICE, DTYPE)
    tnn2 = TNN(dim=DIM, rank=RANK, func=func2).to(DEVICE, DTYPE)
    tnn_add = tnn + tnn2

    # warmup
    for _ in range(10):
        loss = l2_norm(tnn_add, pts, w)
        loss.backward()
    sync()

    start = time.perf_counter()
    for _ in range(N_RUNS):
        _ = tnn_add.func(pts)
    sync()
    t_forward_add = (time.perf_counter() - start) / N_RUNS * 1000

    start = time.perf_counter()
    for _ in range(N_RUNS):
        _ = l2_norm(tnn_add, pts, w)
    sync()
    t_int_l2_add = (time.perf_counter() - start) / N_RUNS * 1000

    start = time.perf_counter()
    for _ in range(N_RUNS):
        loss = l2_norm(tnn_add, pts, w)
        loss.backward()
    sync()
    t_backward_add = (time.perf_counter() - start) / N_RUNS * 1000

    print(f"[tnn+tnn] func.forward:         {t_forward_add:.3f} ms")
    print(f"[tnn+tnn] l2_norm:           {t_int_l2_add:.3f} ms")
    print(f"[tnn+tnn] l2_norm+backward:  {t_backward_add:.3f} ms")

    print("-" * 40)

    # 三个 tnn 相加
    func3 = SeparableDimNetwork(dim=DIM, rank=RANK).to(DEVICE, DTYPE)
    tnn3 = TNN(dim=DIM, rank=RANK, func=func3).to(DEVICE, DTYPE)
    tnn_add3 = tnn + tnn2 + tnn3

    # warmup
    for _ in range(10):
        loss = l2_norm(tnn_add3, pts, w)
        loss.backward()
    sync()

    start = time.perf_counter()
    for _ in range(N_RUNS):
        _ = tnn_add3.func(pts)
    sync()
    t_forward_add3 = (time.perf_counter() - start) / N_RUNS * 1000

    start = time.perf_counter()
    for _ in range(N_RUNS):
        _ = l2_norm(tnn_add3, pts, w)
    sync()
    t_int_l2_add3 = (time.perf_counter() - start) / N_RUNS * 1000

    start = time.perf_counter()
    for _ in range(N_RUNS):
        loss = l2_norm(tnn_add3, pts, w)
        loss.backward()
    sync()
    t_backward_add3 = (time.perf_counter() - start) / N_RUNS * 1000

    print(f"[tnn+tnn+tnn] func.forward:         {t_forward_add3:.3f} ms")
    print(f"[tnn+tnn+tnn] l2_norm:           {t_int_l2_add3:.3f} ms")
    print(f"[tnn+tnn+tnn] l2_norm+backward:  {t_backward_add3:.3f} ms")
