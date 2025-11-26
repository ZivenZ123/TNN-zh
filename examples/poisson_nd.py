"""
使用TNN求解高维Poisson方程
方程: -Δu = f 在 [0,1]^d 上
真解: u(x) = prod(sin(πx_i))
源项: f(x) = d * π^2 * prod(sin(πx_i))
"""

import math

import torch
import torch.nn as nn

from tnn_zh import (
    TNN,
    SeparableDimNetwork,
    generate_quad_points,
    l2_norm,
)

# 配置
DIM = 5
RANK = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
PI = math.pi


class SourceFunc(nn.Module):
    """辅助网络, 将f(x)表示为秩1的TNN分量"""

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


class PoissonPDELoss(nn.Module):
    def __init__(self, tnn_model: TNN):
        super().__init__()
        self.tnn: TNN = tnn_model

        # 生成积分点
        domain_bounds = [(0.0, 1.0) for _ in range(DIM)]
        self.pts, self.w = generate_quad_points(
            domain_bounds,
            device=DEVICE,
            dtype=DTYPE,
        )

        # 将f(x)构造为tnn
        source_func = SourceFunc(DIM)
        self.f_tnn: TNN = (DIM * PI**2) * TNN(
            dim=DIM, rank=1, func=source_func, theta=False
        ).to(DEVICE, DTYPE)

    def forward(self):
        residual: TNN = -self.tnn.laplace() - self.f_tnn
        return l2_norm(residual, self.pts, self.w)


def solve() -> TNN:
    print(f"求解{DIM}维Poisson方程...")

    # 创建满足强制边界条件的func网络
    boundary_conditions = [(0.0, 1.0) for _ in range(DIM)]
    u_tnn_func = (
        SeparableDimNetwork(dim=DIM, rank=RANK)
        .apply_dirichlet_bd(boundary_conditions)
        .to(DEVICE, DTYPE)
    )

    # 构建tnn模型
    u_tnn: TNN = TNN(dim=DIM, rank=RANK, func=u_tnn_func).to(DEVICE, DTYPE)

    # 实例化loss
    loss_fn = PoissonPDELoss(u_tnn)

    # 训练
    u_tnn.fit(
        loss_fn,
        phases=[
            {
                "type": "adam",
                "lr": 0.0005,
                "epochs": 20000,
                "save": "poisson_adam",
            },
            {"type": "lbfgs", "lr": 1.0, "epochs": 100},
        ],
    )

    # # 加载adam训练结果并开始lbfgs训练
    # u_tnn = TNN.load("poisson_adam", device=DEVICE, dtype=DTYPE)
    # loss_fn = PoissonPDELoss(u_tnn)
    # u_tnn.fit(
    #     loss_fn,
    #     phases=[
    #         {"type": "lbfgs", "lr": 1.0, "epochs": 100},
    #     ],
    # )

    return u_tnn


def evaluate(u_tnn: TNN):
    print("\n评估误差...")
    n_test = 1_000_000
    # [0,1]^d中的随机点
    test_points = torch.rand((n_test, DIM), device=DEVICE, dtype=DTYPE)

    # 构造真解TNN
    u_true_tnn: TNN = TNN(dim=DIM, rank=1, func=SourceFunc(DIM)).to(
        DEVICE, DTYPE
    )

    with torch.no_grad():
        u_pred = u_tnn(test_points)
        u_true = u_true_tnn(test_points)

        # 计算相对L2误差
        diff = u_pred - u_true
        l2_err = torch.norm(diff) / torch.norm(u_true)

    print(f"相对L2误差: {l2_err.item():.2e}")


if __name__ == "__main__":
    u_tnn: TNN = solve()
    evaluate(u_tnn)
