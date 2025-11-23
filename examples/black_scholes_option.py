"""
Black-Scholes期权定价模型求解器 (简化版)

基于张量神经网络(TNN)求解Black-Scholes偏微分方程.

Black-Scholes方程:
∂C/∂t + (1/2)σ²S²(∂²C/∂S²) + rS(∂C/∂S) - rC = 0

边界条件 (欧式看涨期权):
- C(S, T, σ) = max(S - K, 0) (到期时的收益)
- C(0, t, σ) = 0 (标的价格为0时期权价值为0)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tnn_zh import (
    TNN,
    SeparableDimNetworkGELU,
    generate_quad_points,
    int_tnn_L2,
    wrap_1d_func_as_tnn,
)

# 设备和精度配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# Black-Scholes模型参数
K = 0.5  # 执行价格
r = 0.02  # 无风险利率
T = 1.0  # 到期时间
S_max = 2.0  # 最大标的价格
sigma_min = 0.1  # 最小波动率
sigma_max = 0.5  # 最大波动率

# 域边界
DOMAIN_BOUNDS = [
    (0.0, S_max),
    (0.0, T),
    (sigma_min, sigma_max),
]


class Step1BoundaryLoss(nn.Module):
    """第一步边界损失: ||u(S,T,σ) - g(S)||²"""

    def __init__(self, u_tnn: TNN, domain_bounds):
        super().__init__()
        self.u_tnn = u_tnn
        self.domain_bounds = domain_bounds

        # 预先生成积分点
        boundary_domain1 = [domain_bounds[0], domain_bounds[2]]
        self.quad_points1, self.quad_weights1 = generate_quad_points(
            boundary_domain1, n_quad_points=16, sub_intervals=20, device=DEVICE
        )

        boundary_domain2 = [(K - 0.05, K + 0.05), domain_bounds[2]]
        self.quad_points2, self.quad_weights2 = generate_quad_points(
            boundary_domain2, n_quad_points=16, sub_intervals=20, device=DEVICE
        )

        # 目标边界函数 g(S) = max(S-K, 0)
        self.g_2d = wrap_1d_func_as_tnn(dim=2, target_dim=0)(
            lambda S: torch.maximum(S - K, torch.zeros_like(S))
        ).to(DEVICE)

    def forward(self):
        # 提取t=T处的切片
        u_at_T = self.u_tnn.slice(fixed_dims={1: T})

        # 误差
        error = u_at_T - self.g_2d

        # L2范数(全域 + 执行价附近加权)
        loss1 = int_tnn_L2(error, self.quad_points1, self.quad_weights1)
        loss2 = int_tnn_L2(error, self.quad_points2, self.quad_weights2)

        return loss1 + 100 * loss2


class Step2PDELoss(nn.Module):
    """第二步PDE损失: ||L(C)||², C = u + v"""

    def __init__(self, v_tnn, u_tnn, domain_bounds):
        super().__init__()
        self.v_tnn = v_tnn
        self.u_tnn = u_tnn

        # 预先生成积分点
        self.quad_points, self.quad_weights = generate_quad_points(
            domain_bounds, n_quad_points=16, sub_intervals=20, device=DEVICE
        )

        # PDE系数函数
        self.sigma_squared = wrap_1d_func_as_tnn(dim=3, target_dim=2)(
            lambda sigma: 0.5 * sigma**2
        ).to(DEVICE)

        self.S_squared = wrap_1d_func_as_tnn(dim=3, target_dim=0)(
            lambda S: S**2
        ).to(DEVICE)

        self.rS = wrap_1d_func_as_tnn(dim=3, target_dim=0)(lambda S: r * S).to(
            DEVICE
        )

    def forward(self):
        # 完整解 C = u + v
        C = self.u_tnn + self.v_tnn

        # Black-Scholes算子: L(C) = ∂C/∂t + (1/2)σ²S²(∂²C/∂S²) + rS(∂C/∂S) - rC
        residual = (
            C.grad(1)
            + self.sigma_squared * self.S_squared * C.grad2(0, 0)
            + self.rS * C.grad(0)
            - r * C
        )

        return int_tnn_L2(residual, self.quad_points, self.quad_weights)


def create_tnn(rank, boundary_spec):
    """创建带边界条件的TNN"""
    bounded_func = (
        SeparableDimNetworkGELU(dim=3, rank=rank)
        .apply_dirichlet_bd(boundary_spec)
        .to(DEVICE, dtype=DTYPE)
    )
    return TNN(dim=3, rank=rank, func=bounded_func).to(DEVICE, dtype=DTYPE)


def train_step(tnn: TNN, loss_module, phases, step_name):
    """通用训练函数"""
    print(f"\n>>> {step_name} <<<")
    losses = tnn.fit(loss_module, phases)
    print(f"{step_name}完成! 最终损失: {losses[-1]:.8f}")
    return tnn


def solve():
    """两步法求解Black-Scholes方程"""
    rank = 20

    # 第一步: 学习边界函数 u
    # 边界条件: S=0时u=0
    u_tnn = create_tnn(rank, [(0.0, None), (None, None), (None, None)])
    u_loss = Step1BoundaryLoss(u_tnn, DOMAIN_BOUNDS)

    u_tnn = train_step(
        u_tnn,
        u_loss,
        [
            {"type": "adam", "lr": 0.01, "epochs": 1000, "grad_clip": 1.0},
            {"type": "adam", "lr": 0.001, "epochs": 1000, "grad_clip": 5.0},
            {"type": "adam", "lr": 0.0001, "epochs": 8000, "grad_clip": 3.0},
        ],
        "第一步: 学习边界函数 u(S,t,σ)",
    )

    # 第二步: 求解PDE
    # 边界条件: S=0时v=0, t=T时v=0
    for param in u_tnn.parameters():
        param.requires_grad = False

    v_tnn = create_tnn(rank, [(0.0, None), (None, T), (None, None)])
    v_loss = Step2PDELoss(v_tnn, u_tnn, DOMAIN_BOUNDS)

    v_tnn = train_step(
        v_tnn,
        v_loss,
        [
            {"type": "adam", "lr": 0.01, "epochs": 2000, "grad_clip": 1.0},
            {"type": "adam", "lr": 0.001, "epochs": 3000, "grad_clip": 0.5},
            {"type": "adam", "lr": 0.0001, "epochs": 10000, "grad_clip": 0.3},
        ],
        "第二步: 求解PDE, 学习 v(S,t,σ)",
    )

    # 完整解 C = u + v
    def C_solution(test_points):
        with torch.no_grad():
            return u_tnn(test_points) + v_tnn(test_points)

    print("\n>>> 两步法求解完成! <<<")
    return C_solution


def visualize(C_solution, sigma_fixed=0.3):
    """绘制TNN解的3D曲面图"""
    print(f"\n>>> 生成三维曲面图 (σ={sigma_fixed}) <<<")

    # 创建网格
    n_points = 80
    S_range = np.linspace(0.01, S_max * 0.475, n_points)
    t_range = np.linspace(0.01, T * 0.95, n_points)
    S_grid, t_grid = np.meshgrid(S_range, t_range)

    # 准备输入
    test_points = torch.stack(
        [
            torch.tensor(S_grid.flatten(), dtype=DTYPE),
            torch.tensor(t_grid.flatten(), dtype=DTYPE),
            torch.full((n_points * n_points,), sigma_fixed, dtype=DTYPE),
        ],
        dim=1,
    ).to(DEVICE)

    # TNN预测
    with torch.no_grad():
        tnn_values = (
            C_solution(test_points).cpu().numpy().reshape(n_points, n_points)
        )

    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        t_grid,
        S_grid,
        tnn_values,
        cmap="viridis",
        alpha=0.9,
        linewidth=0,
        antialiased=True,
        rcount=70,
        ccount=70,
    )

    # 添加底部投影
    ax.contourf(
        t_grid,
        S_grid,
        tnn_values,
        zdir="z",
        offset=0,
        cmap="viridis",
        alpha=0.3,
    )

    ax.set_xlabel("Time t", fontsize=12, fontweight="bold")
    ax.set_ylabel("Stock Price S", fontsize=12, fontweight="bold")
    ax.set_zlabel("Option Value C", fontsize=12, fontweight="bold")
    ax.set_title(
        "TNN Solution: Black-Scholes Option Pricing",
        fontsize=14,
        fontweight="bold",
    )
    ax.view_init(elev=20, azim=125)

    cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
    cbar.set_label("Value", fontsize=11, fontweight="bold")

    plt.tight_layout()

    # 保存
    filename = f"bs_tnn_solution_sigma_{sigma_fixed}.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"图像已保存至: {filename}")
    print(f"TNN解范围: [{np.min(tnn_values):.4f}, {np.max(tnn_values):.4f}]")


def main():
    """主函数"""
    C_solution = solve()
    visualize(C_solution, sigma_fixed=0.3)
    return C_solution


if __name__ == "__main__":
    result = main()
