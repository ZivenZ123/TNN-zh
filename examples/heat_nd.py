"""
使用TNN求解N维热传导方程
方程: u_t = nu * Δu
域: [0, 1]^d x [0, 0.5]
边界条件: u=0 在空间边界
初始条件: u(x, 0) = prod(sin(pi * x_i))

采用两阶段训练法:
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
    l2_norm,
)

# 配置
SPATIAL_DIM = 5
RANK = 10
NU = 0.1  # 扩散系数
TIME_END = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
PI = math.pi


class SourceFunc(nn.Module):
    """辅助网络, 将 prod(sin(π*x_i)) 表示为秩1的TNN分量"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x形状: (n_1d, dim) 或 (dim,)
        # 返回形状: (n_1d, rank=1, dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        val = torch.sin(PI * x)
        return val.unsqueeze(1)


class InitialConditionLoss(nn.Module):
    """初始条件损失: ||u(x,0) - prod(sin(π*x_i))||²"""

    def __init__(self, u_tnn: TNN):
        super().__init__()
        self.u_tnn = u_tnn

        # 在 t=0 处的积分点
        domain_bounds = [(0.0, 1.0)] * SPATIAL_DIM
        self.pts, self.w = generate_quad_points(
            domain_bounds,
            device=DEVICE,
            dtype=DTYPE,
        )

        # 目标函数: prod(sin(π*x_i))
        target_func = SourceFunc(SPATIAL_DIM)
        self.target: TNN = TNN(
            dim=SPATIAL_DIM, rank=1, func=target_func, theta=False
        ).to(DEVICE, DTYPE)

    def forward(self):
        # 提取 t=0 切片
        u_0: TNN = self.u_tnn.slice(fixed_dims={-1: 0.0})
        residual: TNN = u_0 - self.target
        return l2_norm(residual, self.pts, self.w)


class HeatPDELoss(nn.Module):
    """PDE残差损失"""

    def __init__(self, v_tnn: TNN, u_tnn: TNN):
        super().__init__()
        self.v_tnn = v_tnn
        self.u_tnn = u_tnn  # 固定的初始条件拟合

        # 生成积分点: 空间+时间
        domain_bounds = [(0.0, 1.0)] * SPATIAL_DIM + [(0.0, TIME_END)]
        self.pts, self.w = generate_quad_points(
            domain_bounds,
            device=DEVICE,
            dtype=DTYPE,
        )

    def forward(self):
        u: TNN = self.u_tnn + self.v_tnn

        # 计算时间导数和空间拉普拉斯
        u_t: TNN = u.grad(grad_dim=-1)
        laplace_spatial: TNN = u.laplace() - u.grad2(dim1=-1, dim2=-1)

        residual: TNN = u_t - NU * laplace_spatial
        return l2_norm(residual, self.pts, self.w)


def solve() -> TNN:
    print(f"求解 {SPATIAL_DIM}维 空间 + 1维 时间 热传导方程 (两阶段法)...")

    # >>>>>> 第一阶段: 拟合初始条件 <<<<<<
    print("阶段1: 拟合初始条件 u(x, 0) = prod(sin(π*x_i))...")

    # 构造 u_tnn, 满足空间边界条件和 t=0 处无约束
    boundary_u = [(0.0, 1.0)] * SPATIAL_DIM + [(None, None)]
    u_func = (
        SeparableDimNetworkGELU(dim=SPATIAL_DIM + 1, rank=RANK)
        .apply_dirichlet_bd(boundary_u)
        .to(DEVICE, DTYPE)
    )
    u_tnn: TNN = TNN(dim=SPATIAL_DIM + 1, rank=RANK, func=u_func).to(
        DEVICE, DTYPE
    )

    # 训练 u 拟合初始条件
    loss_fn = InitialConditionLoss(u_tnn)
    u_tnn.fit(
        loss_fn,
        phases=[
            {"type": "adam", "lr": 0.005, "epochs": 10000},
        ],
    )

    # >>>>>> 第二阶段: 求解PDE <<<<<<
    print("阶段2: 求解PDE, 训练修正项 v...")
    # 固定 u_tnn 的参数不进行学习
    for p in u_tnn.parameters():
        p.requires_grad = False

    # 构造 v_tnn, 满足空间边界条件和 t=0 处为0
    boundary_v = [(0.0, 1.0)] * SPATIAL_DIM + [(0.0, None)]
    v_func = (
        SeparableDimNetworkGELU(dim=SPATIAL_DIM + 1, rank=RANK)
        .apply_dirichlet_bd(boundary_v)
        .to(DEVICE, DTYPE)
    )
    v_tnn: TNN = TNN(dim=SPATIAL_DIM + 1, rank=RANK, func=v_func).to(
        DEVICE, DTYPE
    )

    # 训练 v 使得 u+v 满足PDE
    pde_loss = HeatPDELoss(v_tnn, u_tnn)
    v_tnn.fit(
        pde_loss,
        phases=[
            {"type": "adam", "lr": 0.01, "epochs": 1000},
            {"type": "lbfgs", "lr": 1.0, "epochs": 50},
        ],
    )

    return u_tnn + v_tnn


def evaluate(solution_tnn: TNN):
    print("\n评估误差...")
    n_test = 1_000_000

    # 随机采样测试点
    # 空间 [0, 1], 时间 [0, TIME_END]
    test_points = torch.rand(
        (n_test, SPATIAL_DIM + 1), device=DEVICE, dtype=DTYPE
    )
    test_points[:, -1] = test_points[:, -1] * TIME_END

    # 计算真解
    u_true = torch.prod(
        torch.sin(PI * test_points[:, :SPATIAL_DIM]), dim=1
    ) * torch.exp(-NU * SPATIAL_DIM * (PI**2) * test_points[:, SPATIAL_DIM])

    with torch.no_grad():
        u_pred = solution_tnn(test_points)
        diff = u_pred - u_true
        l2_err = torch.norm(diff) / torch.norm(u_true)

    print(f"相对L2误差: {l2_err.item():.2e}")


if __name__ == "__main__":
    solution_tnn: TNN = solve()
    evaluate(solution_tnn)
