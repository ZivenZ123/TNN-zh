"""
Black-Scholes期权定价模型求解器 (简化版)

基于张量神经网络(TNN)求解Black-Scholes偏微分方程.

方程: ∂C/∂t + (1/2)σ²S²(∂²C/∂S²) + rS(∂C/∂S) - rC = 0
边界条件 (欧式看涨): C(S, T) = max(S - K, 0), C(0, t) = 0
"""

import numpy as np
import torch
import torch.nn as nn
from tnn_zh import (
    TNN,
    SeparableDimNetworkGELU,
    generate_quad_points,
    int_tnn_L2,
    wrap_1d_func_as_tnn,
)

# 配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# BS参数
K = 0.5  # 执行价格
r = 0.02  # 无风险利率
T = 1.0  # 到期时间
S_max = 2.0  # 最大标的价格
sigma_min = 0.1
sigma_max = 0.5

# 域边界: (S, t, sigma)
BOUNDS = [
    (0.0, S_max),
    (0.0, T),
    (sigma_min, sigma_max),
]


class BCLoss(nn.Module):
    """边界条件损失: ||u(S,T,σ) - max(S-K, 0)||²"""

    def __init__(self, u_tnn: TNN):
        super().__init__()
        self.u_tnn = u_tnn

        # 积分点: S在[0, S_max], sigma在[sigma_min, sigma_max]
        bounds = [BOUNDS[0], BOUNDS[2]]
        self.pts1, self.w1 = generate_quad_points(
            bounds,
            n_quad_points=20,
            sub_intervals=10,
            device=DEVICE,
            dtype=DTYPE,
        )

        # 在K附近加密采样
        bounds_k = [(K - 0.1, K + 0.1), BOUNDS[2]]
        self.pts2, self.w2 = generate_quad_points(
            bounds_k,
            n_quad_points=20,
            sub_intervals=10,
            device=DEVICE,
            dtype=DTYPE,
        )

        # 目标函数 g(S) = max(S-K, 0)
        # 注意: 输入是(S, sigma), 目标函数只与S有关
        self.g = wrap_1d_func_as_tnn(dim=2, target_dim=0)(
            lambda S: torch.relu(S - K)
        ).to(DEVICE)

    def forward(self):
        # 提取t=T切片 (第1维固定为T)
        u_T = self.u_tnn.slice(fixed_dims={1: T})

        diff = u_T - self.g
        return int_tnn_L2(diff, self.pts1, self.w1) + 10.0 * int_tnn_L2(
            diff, self.pts2, self.w2
        )


class PDELoss(nn.Module):
    """PDE残差损失"""

    def __init__(self, v_tnn, u_tnn):
        super().__init__()
        self.v_tnn = v_tnn  # 待求部分
        self.u_tnn = u_tnn  # 已知边界部分 (参数固定)

        self.pts, self.w = generate_quad_points(
            BOUNDS,
            n_quad_points=16,
            sub_intervals=10,
            device=DEVICE,
            dtype=DTYPE,
        )

        # 系数函数
        # dim=3: (S, t, sigma)
        self.sigma2 = wrap_1d_func_as_tnn(3, 2)(lambda s: 0.5 * s**2).to(
            DEVICE
        )
        self.S2 = wrap_1d_func_as_tnn(3, 0)(lambda S: S**2).to(DEVICE)
        self.rS = wrap_1d_func_as_tnn(3, 0)(lambda S: r * S).to(DEVICE)

    def forward(self):
        C = self.u_tnn + self.v_tnn

        # L(C) = C_t + 0.5*sigma^2*S^2*C_SS + r*S*C_S - r*C
        res = (
            C.grad(1)
            + self.sigma2 * self.S2 * C.grad2(0, 0)
            + self.rS * C.grad(0)
            - r * C
        )
        return int_tnn_L2(res, self.pts, self.w)


def bs_exact(S, t, sigma, K, r, T):
    """Black-Scholes 精确解"""
    tau = T - t
    # 避免除零
    tau = torch.clamp(tau, min=1e-8)
    sigma = torch.clamp(sigma, min=1e-8)

    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (
        sigma * torch.sqrt(tau)
    )
    d2 = d1 - sigma * torch.sqrt(tau)

    # 标准正态分布CDF
    N_d1 = 0.5 * (1 + torch.erf(d1 / np.sqrt(2)))
    N_d2 = 0.5 * (1 + torch.erf(d2 / np.sqrt(2)))

    C = S * N_d1 - K * torch.exp(-r * tau) * N_d2
    return torch.where(tau < 1e-7, torch.relu(S - K), C)


def solve():
    print("求解Black-Scholes方程 (两阶段法)...")
    rank = 20

    # 1. 学习边界函数 u (满足终值条件)
    print("Phase 1: 拟合终值条件...")
    # 强制 S=0 时 u=0
    u_func = (
        SeparableDimNetworkGELU(3, rank)
        .apply_dirichlet_bd([(0.0, None), (None, None), (None, None)])
        .to(DEVICE, DTYPE)
    )
    u_tnn = TNN(3, rank, u_func).to(DEVICE, DTYPE)

    loss_fn = BCLoss(u_tnn)
    phases = [{"type": "adam", "lr": 0.01, "epochs": 1000}]
    u_tnn.fit(loss_fn, phases)

    # 2. 学习修正项 v (使得 u+v 满足PDE)
    print("Phase 2: 求解PDE...")
    # 固定 u
    for p in u_tnn.parameters():
        p.requires_grad = False

    # v 在 S=0 和 t=T 处为0
    v_func = (
        SeparableDimNetworkGELU(3, rank)
        .apply_dirichlet_bd([(0.0, None), (None, T), (None, None)])
        .to(DEVICE, DTYPE)
    )
    v_tnn = TNN(3, rank, v_func).to(DEVICE, DTYPE)

    loss_fn = PDELoss(v_tnn, u_tnn)
    phases = [
        {"type": "adam", "lr": 0.005, "epochs": 2000},
        {"type": "adam", "lr": 0.0005, "epochs": 2000},
    ]
    v_tnn.fit(loss_fn, phases)

    # 返回完整解
    def solution(x):
        return u_tnn(x) + v_tnn(x)

    return solution


def evaluate(model):
    print("\n评估准确度...")
    n_test = 5000

    # 随机采样测试点
    pts = torch.rand(n_test, 3, device=DEVICE)
    pts[:, 0] = pts[:, 0] * S_max  # S
    pts[:, 1] = pts[:, 1] * T  # t
    pts[:, 2] = pts[:, 2] * (sigma_max - sigma_min) + sigma_min  # sigma

    with torch.no_grad():
        pred = model(pts)
        exact = bs_exact(pts[:, 0], pts[:, 1], pts[:, 2], K, r, T)

        # 避免数值问题，只在 t < T-epsilon 处评估 PDE 内部
        mask = pts[:, 1] < (T - 0.01)
        rel_l2 = torch.norm((pred - exact)[mask]) / torch.norm(exact[mask])

    print(f"相对L2误差: {rel_l2:.2e}")


if __name__ == "__main__":
    model = solve()
    evaluate(model)
