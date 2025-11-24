"""
使用TNN求解高维Laplace方程
方程: -Δu = f 在 [0,1]^d 上
真解: u(x) = prod(sin(πx_i))
源项: f(x) = d * π^2 * prod(sin(πx_i))
"""

import math

import torch
import torch.nn as nn
from tnn_zh import (
    TNN,
    SeparableDimNetworkGELU,
    generate_quad_points,
    int_tnn_L2,
)

# 配置
DIM = 5
RANK = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
PI = math.pi


class SourceTermNet(nn.Module):
    """辅助网络, 将f(x)表示为秩1的TNN分量"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x形状: (n_1d, dim) 或 (dim,)
        # 返回形状: (n_1d, rank=1, dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 每个维度计算sin(pi * x)
        # val: (n_1d, dim)
        val = torch.sin(PI * x)

        # 添加rank维度 -> (n_1d, 1, dim)
        return val.unsqueeze(1)


class LaplacePDELoss(nn.Module):
    def __init__(self, tnn_model: TNN, domain_bounds):
        super().__init__()
        self.tnn = tnn_model

        # 生成积分点
        self.quad_points, self.quad_weights = generate_quad_points(
            domain_bounds,
            n_quad_points=16,
            sub_intervals=10,
            device=DEVICE,
            dtype=DTYPE,
        )

        # 将f(x)构造为TNN
        source_func = SourceTermNet(DIM)
        self.f_tnn = (DIM * PI**2) * TNN(dim=DIM, rank=1, func=source_func).to(
            DEVICE, dtype=DTYPE
        )

    def forward(self):
        residual = -self.tnn.laplace() - self.f_tnn

        return int_tnn_L2(
            residual, self.quad_points, self.quad_weights
        )  # L2损失


def solve():
    print(f"求解{DIM}维Laplace方程...")

    # 1. 模型构建
    # 边界条件: u=0 在[0,1]^d的所有边界上
    # 这与sin(pi*x)匹配, 它在0和1处为0.
    boundary: list[tuple[float | None, float | None]] = [
        (0.0, 1.0) for _ in range(DIM)
    ]

    # 满足强制边界条件的基础网络
    u_tnn_func = (
        SeparableDimNetworkGELU(dim=DIM, rank=RANK)
        .apply_dirichlet_bd(boundary)
        .to(DEVICE, dtype=DTYPE)
    )

    u_tnn = TNN(dim=DIM, rank=RANK, func=u_tnn_func).to(DEVICE, dtype=DTYPE)

    # 2. 损失函数定义
    bounds = [(0.0, 1.0) for _ in range(DIM)]
    loss_fn = LaplacePDELoss(u_tnn, bounds)

    # 3. 训练
    print("开始训练...")
    u_tnn.fit(
        loss_fn=loss_fn,
        phases=[
            {"type": "adam", "lr": 0.01, "epochs": 2000, "grad_clip": 1.0}
        ],
    )
    print("训练完成. ")

    return u_tnn


def evaluate(u_tnn: TNN):
    print("\n评估误差...")
    n_test = 1000
    # [0,1]^d中的随机点
    test_points = torch.rand((n_test, DIM), device=DEVICE, dtype=DTYPE)

    # 构造真解TNN: u(x) = prod(sin(πx_i))
    # SourceTermNet恰好实现了该逻辑
    u_true_tnn = TNN(dim=DIM, rank=1, func=SourceTermNet(DIM)).to(
        DEVICE, dtype=DTYPE
    )

    with torch.no_grad():
        u_pred = u_tnn(test_points)
        u_true = u_true_tnn(test_points)

        # 相对L2误差
        diff = u_pred - u_true
        l2_err = torch.norm(diff) / torch.norm(u_true)

    print(f"相对L2误差: {l2_err.item():.2e}")


if __name__ == "__main__":
    u_tnn = solve()
    evaluate(u_tnn)
