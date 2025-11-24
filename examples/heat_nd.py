"""
使用TNN求解N维热传导方程 (Heat Equation)
方程: u_t = nu * Δu
域: [0, 1]^d x [0, 0.5]
边界条件: u=0 在空间边界
初始条件: u(x, 0) = prod(sin(pi * x_i))

采用两阶段训练法(Two-Phase Training):
阶段1: 训练TNN u拟合初始条件 u(x, 0) = prod(sin(pi * x_i))
阶段2: 固定u, 训练修正项v, 使得 u+v 满足PDE
最终解: solution(x) = u(x) + v(x)
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

# 全局配置
SPATIAL_DIM = 5  # 空间维度 d
RANK = 10
NU = 0.1  # 扩散系数
TIME_END = 0.5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
PI = math.pi


class SourceTermNet(nn.Module):
    """辅助网络, 将 prod(sin(π*x_i)) 表示为秩1的TNN分量"""

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


class InitialConditionLoss(nn.Module):
    """初始条件损失: ||u(x,0) - prod(sin(π*x_i))||²"""

    def __init__(self, u_tnn: TNN, spatial_dim: int):
        super().__init__()
        self.u_tnn = u_tnn
        self.spatial_dim = spatial_dim

        # 在 t=0 处的积分点
        bounds = [(0.0, 1.0)] * spatial_dim
        self.quad_points, self.quad_weights = generate_quad_points(
            bounds,
            n_quad_points=16,
            sub_intervals=10,
            device=DEVICE,
            dtype=DTYPE,
        )

        # 目标函数：prod(sin(π*x_i))
        target_func = SourceTermNet(spatial_dim)
        self.target = TNN(dim=spatial_dim, rank=1, func=target_func).to(
            DEVICE, DTYPE
        )

    def forward(self):
        # 提取 t=0 切片
        u_0 = self.u_tnn.slice(fixed_dims={self.spatial_dim: 0.0})

        diff = u_0 - self.target
        return int_tnn_L2(diff, self.quad_points, self.quad_weights)


class HeatPDELoss(nn.Module):
    """PDE残差损失"""

    def __init__(self, v_tnn: TNN, u_tnn: TNN, spatial_dim: int):
        super().__init__()
        self.v_tnn = v_tnn
        self.u_tnn = u_tnn  # 固定的初始条件拟合
        self.spatial_dim = spatial_dim

        # 生成积分点：空间+时间
        bounds = [(0.0, 1.0)] * spatial_dim + [(0.0, TIME_END)]
        self.quad_points, self.quad_weights = generate_quad_points(
            bounds,
            n_quad_points=16,
            sub_intervals=10,
            device=DEVICE,
            dtype=DTYPE,
        )

    def forward(self):
        # 组合解 u = u_tnn + v_tnn
        u = self.u_tnn + self.v_tnn

        # 计算时间导数和空间拉普拉斯
        u_t = u.grad(self.spatial_dim)
        laplace_all = u.laplace()
        laplace_spatial = laplace_all - u.grad2(
            self.spatial_dim, self.spatial_dim
        )

        # PDE残差
        residual = u_t - NU * laplace_spatial

        return int_tnn_L2(residual, self.quad_points, self.quad_weights)


def solve():
    print(f"求解 {SPATIAL_DIM}维 空间 + 1维 时间 热传导方程 (两阶段法)...")
    print(f"总维度: {SPATIAL_DIM + 1}")

    # ===== 第一阶段：拟合初始条件 =====
    print("\n阶段1: 拟合初始条件 u(x, 0) = prod(sin(π*x_i))...")

    # 构造 u_tnn，满足空间边界条件和 t=0 处无约束
    boundary_u = [(0.0, 1.0)] * SPATIAL_DIM + [(None, None)]
    u_func = (
        SeparableDimNetworkGELU(dim=SPATIAL_DIM + 1, rank=RANK)
        .apply_dirichlet_bd(boundary_u)
        .to(DEVICE, DTYPE)
    )
    u_tnn = TNN(dim=SPATIAL_DIM + 1, rank=RANK, func=u_func).to(DEVICE, DTYPE)

    # 训练 u 拟合初始条件
    ic_loss = InitialConditionLoss(u_tnn, SPATIAL_DIM)
    phases_u = [
        {"type": "adam", "lr": 0.01, "epochs": 1000},
        {"type": "adam", "lr": 0.001, "epochs": 500},
    ]
    u_tnn.fit(ic_loss, phases_u)

    # ===== 第二阶段：求解PDE =====
    print("\n阶段2: 求解PDE，训练修正项 v...")

    # 固定 u_tnn
    for p in u_tnn.parameters():
        p.requires_grad = False

    # 构造 v_tnn，满足空间边界条件和 t=0 处为0
    boundary_v = [(0.0, 1.0)] * SPATIAL_DIM + [(0.0, None)]
    v_func = (
        SeparableDimNetworkGELU(dim=SPATIAL_DIM + 1, rank=RANK)
        .apply_dirichlet_bd(boundary_v)
        .to(DEVICE, DTYPE)
    )
    v_tnn = TNN(dim=SPATIAL_DIM + 1, rank=RANK, func=v_func).to(DEVICE, DTYPE)

    # 训练 v 使得 u+v 满足PDE
    pde_loss = HeatPDELoss(v_tnn, u_tnn, SPATIAL_DIM)
    phases_v = [
        {"type": "adam", "lr": 0.01, "epochs": 1000},
        {"type": "adam", "lr": 0.001, "epochs": 1000},
    ]
    v_tnn.fit(pde_loss, phases_v)

    print("训练完成.")

    # 返回组合解
    def solution_tnn(x):
        return u_tnn(x) + v_tnn(x)

    return solution_tnn


def evaluate(solution_fn):
    print("\n评估误差...")
    n_test = 2000
    total_dim = SPATIAL_DIM + 1

    # 随机采样测试点
    # 空间 [0, 1], 时间 [0, TIME_END]
    test_points = torch.rand((n_test, total_dim), device=DEVICE, dtype=DTYPE)
    test_points[:, -1] = test_points[:, -1] * TIME_END

    # 计算真解
    # u_true = exp(-nu * d * pi^2 * t) * prod(sin(pi * x_i))

    spatial_x = test_points[:, :SPATIAL_DIM]
    t = test_points[:, SPATIAL_DIM]

    # prod(sin(pi * x_i))
    spatial_term = torch.prod(torch.sin(PI * spatial_x), dim=1)

    # exp term
    decay = -NU * SPATIAL_DIM * (PI**2)
    time_term = torch.exp(decay * t)

    u_true = spatial_term * time_term

    with torch.no_grad():
        u_pred = solution_fn(test_points)

        diff = u_pred - u_true
        l2_err = torch.norm(diff) / torch.norm(u_true)

    print(f"相对L2误差: {l2_err.item():.2e}")


if __name__ == "__main__":
    sol = solve()
    evaluate(sol)
