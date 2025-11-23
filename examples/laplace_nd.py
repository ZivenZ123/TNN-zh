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
    TNNTrainer,
    apply_dirichlet_bd,
    generate_quad_points,
    int_tnn_L2,
)

# 配置
DIM = 5
RANK = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
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
        )

        # 将f(x)构造为TNN
        source_func = SourceTermNet(DIM)
        self.f_tnn = (DIM * PI**2) * TNN(dim=DIM, rank=1, func=source_func).to(
            DEVICE, dtype=DTYPE
        )

    def forward(self):
        residual = -self.tnn.laplace() - self.f_tnn

        # L2损失
        return int_tnn_L2(residual, self.quad_points, self.quad_weights)


def get_true_solution(x):
    """在x处计算真解: (batch_size, dim) -> (batch_size,)"""
    # prod(sin(pi * x_i))
    return torch.prod(torch.sin(PI * x), dim=-1)


def solve():
    print(f"求解{DIM}维Laplace方程...")

    # 1. 模型构建
    # 边界条件: u=0 在[0,1]^d的所有边界上
    # 这与sin(pi*x)匹配, 它在0和1处为0.
    boundary_spec: list[tuple[float | None, float | None]] = [
        (0.0, 1.0) for _ in range(DIM)
    ]

    # 应用边界条件的基础网络
    base_func = SeparableDimNetworkGELU(dim=DIM, rank=RANK).to(
        DEVICE, dtype=DTYPE
    )
    bounded_func = apply_dirichlet_bd(boundary_spec)(base_func)

    model = TNN(dim=DIM, rank=RANK, func=bounded_func).to(DEVICE, dtype=DTYPE)

    # 2. 损失函数定义
    bounds = [(0.0, 1.0) for _ in range(DIM)]
    loss_fn = LaplacePDELoss(model, bounds)

    # 3. 训练
    # 使用简化的训练计划进行演示
    phases = [
        {"type": "adam", "lr": 0.01, "epochs": 1000, "grad_clip": 1.0},
        {"type": "adam", "lr": 0.001, "epochs": 1000, "grad_clip": 0.5},
    ]

    print("开始训练...")
    trainer = TNNTrainer(model, loss_fn, print_interval=200)
    trainer.multi_phase(phases)
    print("训练完成. ")

    return model


def evaluate(model):
    print("\n评估误差...")
    n_test = 1000
    # [0,1]^d中的随机点
    test_points = torch.rand((n_test, DIM), device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        u_pred = model(test_points)
        u_true = get_true_solution(test_points)

        # 相对L2误差
        diff = u_pred - u_true
        l2_err = torch.norm(diff) / torch.norm(u_true)

    print(f"相对L2误差: {l2_err.item():.2e}")


if __name__ == "__main__":
    model = solve()
    evaluate(model)
