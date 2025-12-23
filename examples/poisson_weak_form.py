"""
使用TNN + Galerkin弱形式求解高维Poisson方程

方程: -Δu = f 在 [0,1]^d 上, 齐次Dirichlet边界条件
真解: u(x) = prod(sin(πx_i))
源项: f(x) = d * π^2 * prod(sin(πx_i))

Galerkin弱形式:
    找 u = Σᵣ θᵣ φᵣ, 使得对所有测试函数 v = φⱼ:
    ∫ ∇u · ∇v dx = ∫ fv dx
    即: Sθ = b

训练策略:
1. 每步前向传播时, θ通过解析求解 θ* = S⁻¹b 得到
2. 用求解得到的 u = Σᵣ θᵣ* φᵣ 计算 PDE 残差 ||-Δu - f||²
3. 网络参数通过最小化残差来优化

关键点: θ不是学习参数, 而是网络参数的函数!
梯度通过 torch.linalg.solve 反向传播到网络参数.
"""

import math

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from tnn_zh import (
    TNN,
    SeparableDimNetworkSin,
    ThetaModule,
    assemble_stiffness_matrix,
    generate_quad_points,
)

# 配置
DIM = 5
RANK = 20
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
PI = math.pi


class SourceFunc(nn.Module):
    """源项 f(x) = d * π² * prod(sin(πx_i)) 的秩1表示"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        val = torch.sin(PI * x)
        return val.unsqueeze(1)


def assemble_load_vector(
    u_tnn: TNN,
    f_tnn: TNN,
    quad_points: torch.Tensor,
    quad_weights: torch.Tensor,
) -> torch.Tensor:
    """
    组装载荷向量 b_i = ∫ f φᵢ dx
    """
    phi_vals = u_tnn.func(quad_points)  # (n_1d, rank_u, dim)
    f_vals = f_tnn.func(quad_points)  # (n_1d, rank_f, dim)

    weighted_phi = torch.einsum("nd,nid->ind", quad_weights, phi_vals)
    integral_per_dim = torch.einsum("ind,njd->ijd", weighted_phi, f_vals)
    prod_over_dim = integral_per_dim.prod(dim=-1)  # (rank_u, rank_f)
    b = torch.einsum("ij,j->i", prod_over_dim, f_tnn.theta)

    return b


class GalerkinLoss(nn.Module):
    """
    Galerkin弱形式 + PDE残差 Loss

    每次前向传播:
    1. 组装刚度矩阵 S 和载荷向量 b
    2. 解析求解 θ* = S⁻¹b
    3. 计算弱形式残差 ||Sθ* - b||² (应该接近0)
    4. 但我们优化的是让基函数更好地表达解

    注意: 由于 θ* = S⁻¹b 是最优的, Sθ* - b = 0.
    所以我们需要另一个 loss: 比如让 u 接近真解, 或者最小化能量泛函.
    这里我们用 Galerkin 投影残差: ||r||² 其中 r = -Δu - f 投影到基函数空间.
    """

    def __init__(
        self, u_func: nn.Module, f_tnn: TNN, quad_points, quad_weights
    ):
        super().__init__()
        self.u_func = u_func
        self.f_tnn = f_tnn
        self.quad_points = quad_points
        self.quad_weights = quad_weights
        self.rank = u_func.original_func.rank

    def forward(self):
        # 创建临时TNN用于组装矩阵 (theta=1)
        u_tnn_temp = TNN(
            dim=DIM, rank=self.rank, func=self.u_func, theta=False
        )

        # 组装刚度矩阵 S
        S = assemble_stiffness_matrix(
            u_tnn_temp, self.quad_points, self.quad_weights
        )

        # 组装载荷向量 b
        b = assemble_load_vector(
            u_tnn_temp, self.f_tnn, self.quad_points, self.quad_weights
        )

        # 解析求解 θ* = S⁻¹b
        # 添加正则化以提高数值稳定性
        reg = 1e-6 * torch.eye(self.rank, device=S.device, dtype=S.dtype)
        theta_star = torch.linalg.solve(S + reg, b)

        # 计算能量泛函: J(θ) = (1/2)θᵀSθ - bᵀθ
        # 在最优点: J* = -(1/2)bᵀθ*
        # 我们希望 J* 尽可能小 (负得更多), 即基函数能更好地表达解
        # 但直接最小化 J* 会导致 b->0 的平凡解
        #
        # 正确的做法: 用 L2 误差作为 loss
        # 直接计算 u = Σ θ*_r φ_r 在采样点的值, 然后与真解比较
        #
        # 但这需要知道真解, 不太通用. 这里我们用一个替代方案:
        # 最大化 |b|² / |S| 或者类似的度量, 鼓励基函数能"捕获"更多源项

        # 简单方案: 用能量的绝对值作为 loss (越负越好, 所以最小化 J*)
        # 但要防止 b->0, 加一个正则项鼓励 |b| 不为 0
        energy = -0.5 * torch.dot(b, theta_star)

        # 正则化: 鼓励 b 不为 0
        b_norm = torch.norm(b)
        reg_loss = 1.0 / (b_norm + 1e-8)

        # 总 loss = energy + λ * reg_loss
        # 注意: energy 是负的, 我们想让它更负; reg_loss 阻止 b->0
        loss = energy + 0.1 * reg_loss

        return loss, theta_star


def solve():
    print(f"使用 Galerkin弱形式 + TNN 求解 {DIM} 维 Poisson 方程...")
    print(f"基函数数量 (rank): {RANK}")

    # 1. 创建满足边界条件的基函数网络
    boundary_conditions = [(0.0, 1.0) for _ in range(DIM)]
    u_func = (
        SeparableDimNetworkSin(dim=DIM, rank=RANK)
        .apply_dirichlet_bd(boundary_conditions)
        .to(DEVICE, DTYPE)
    )

    # 2. 生成积分点
    domain_bounds = [(0.0, 1.0) for _ in range(DIM)]
    quad_points, quad_weights = generate_quad_points(
        domain_bounds,
        n_quad_points=16,
        sub_intervals=10,
        device=DEVICE,
        dtype=DTYPE,
    )

    # 3. 构造源项TNN
    f_tnn = (DIM * PI**2) * TNN(
        dim=DIM, rank=1, func=SourceFunc(DIM), theta=False
    ).to(DEVICE, DTYPE)

    # 4. 创建loss函数
    loss_fn = GalerkinLoss(u_func, f_tnn, quad_points, quad_weights)

    # 5. 优化器 - 只优化网络参数
    optimizer = optim.Adam(u_func.parameters(), lr=1e-3)

    # 6. 训练
    print("\n开始训练...")
    epochs = 10000
    final_theta = None
    final_loss = None

    with tqdm(range(epochs), desc="训练") as pbar:
        for step in pbar:
            optimizer.zero_grad()

            loss, theta_star = loss_fn()
            loss.backward()

            optimizer.step()

            final_theta = theta_star.detach().clone()
            final_loss = loss.item()

            if step % 100 == 0 or step == epochs - 1:
                pbar.set_postfix(loss=f"{final_loss:.2e}")

    print(f"\n训练完成, 最终能量: {final_loss:.2e}")
    print(f"最优theta: {final_theta}")

    # 7. 构建最终的TNN
    theta_module = ThetaModule(
        rank=RANK, learnable=False, initial_values=final_theta
    )
    u_tnn_solved = TNN(dim=DIM, rank=RANK, func=u_func, theta=theta_module).to(
        DEVICE, DTYPE
    )

    return u_tnn_solved


def evaluate(u_tnn: TNN):
    print("\n评估误差...")
    n_test = 100_000

    test_points = torch.rand((n_test, DIM), device=DEVICE, dtype=DTYPE)

    # 真解: u_true(x) = prod(sin(πx_i))
    u_true_tnn = TNN(dim=DIM, rank=1, func=SourceFunc(DIM), theta=False).to(
        DEVICE, DTYPE
    )

    with torch.no_grad():
        u_pred = u_tnn(test_points)
        u_true = u_true_tnn(test_points)

        diff = u_pred - u_true
        l2_err = torch.norm(diff) / torch.norm(u_true)
        max_err = torch.max(torch.abs(diff))

    print(f"相对L2误差: {l2_err.item():.2e}")
    print(f"最大绝对误差: {max_err.item():.2e}")


if __name__ == "__main__":
    u_tnn = solve()
    evaluate(u_tnn)
