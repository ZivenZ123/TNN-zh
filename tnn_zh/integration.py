"""
TNN积分模块

提供针对TNN的数值积分功能.

主要功能:
- GaussLegendre: 高斯-勒让德积分类
- generate_quad_points: 生成积分点和权重(张量积形式)
- int_tnn: 计算单个TNN的积分
- int_tnn_product: 计算两个TNN乘积的积分(内存优化)
- l2_norm: 计算TNN的L2范数
- assemble_mass_matrix: 组装质量矩阵 M_{ij} = ∫ φᵢ φⱼ dx
- assemble_stiffness_matrix: 组装刚度矩阵 S_{ij} = ∫ ∇φᵢ · ∇φⱼ dx

张量积网格优势:
- 内存: O(n_1d × dim) 而非 O(n_1d^dim × dim)
- 适用于高维问题 (dim=3,4,5...)
- 实际代表 n_1d^dim 个积分点

使用方式:
# 步骤1: 生成积分点和权重(只需生成一次)
# 注意: 每个维度可以有不同的边界
quad_points, quad_weights = generate_quad_points(
    domain_bounds=[(0, 1), (0, 2)],  # 每个维度可以有不同的边界
    n_quad_points=16,
    sub_intervals=2,
    device=device
)
# quad_points: (n_1d, dim) 形状的张量 - 张量积表示
# quad_weights: (n_1d, dim) 形状的张量 - 每个维度有独立的权重

# 步骤2: 在积分点上进行积分计算
result1 = int_tnn(tnn, quad_points, quad_weights)
result2 = int_tnn_product(tnn1, tnn2, quad_points, quad_weights)
result3 = l2_norm(tnn, quad_points, quad_weights)

# 弱形式求解PDE - 组装矩阵
# 将TNN的rank个秩1函数视为基函数, 用Galerkin方法求解
M = assemble_mass_matrix(tnn, quad_points, quad_weights)  # (rank, rank)
S = assemble_stiffness_matrix(tnn, quad_points, quad_weights)  # (rank, rank)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .core import TNN

DTYPE = torch.float32


class GaussLegendre:
    """
    高斯-勒让德积分器类
    """

    def _standard_gauss_points(
        self, n_points: int, dtype: torch.dtype = DTYPE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取标准区间[-1,1]上的高斯-勒让德积分点和权重

        Args:
            n_points: 积分点数量
            dtype: 数据类型,默认为torch.float32

        Returns:
            (积分点, 权重)
        """
        points, weights = np.polynomial.legendre.leggauss(n_points)
        return (
            torch.tensor(points, dtype=dtype),
            torch.tensor(weights, dtype=dtype),
        )

    def gauss_points(
        self, n_points: int, a: float, b: float, dtype: torch.dtype = DTYPE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定区间[a,b]上的高斯-勒让德积分点和权重

        Args:
            n_points: 积分点数量
            a: 积分区间左端点
            b: 积分区间右端点
            dtype: 数据类型,默认为torch.float32

        Returns:
            (变换后的积分点, 变换后的权重)
        """
        points, weights = self._standard_gauss_points(n_points, dtype=dtype)
        scale = (b - a) * 0.5
        offset = (a + b) * 0.5
        transformed_points = points * scale + offset
        transformed_weights = weights * scale
        return transformed_points, transformed_weights

    def gauss_points_with_subdivision(
        self,
        n_points: int,
        a: float,
        b: float,
        sub_intervals: int = 1,
        dtype: torch.dtype = DTYPE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定区间[a,b]上的高斯-勒让德积分点和权重,支持区间细分

        Args:
            n_points: 积分点数量
            a: 积分区间左端点
            b: 积分区间右端点
            sub_intervals: 子区间数量,默认为1(不细分)
            dtype: 数据类型,默认为torch.float32

        Returns:
            (合并后的积分点, 合并后的权重)
        """
        if sub_intervals == 1:
            return self.gauss_points(n_points, a, b, dtype=dtype)

        standard_points, standard_weights = self._standard_gauss_points(
            n_points, dtype=dtype
        )

        sub_interval_length = (b - a) / sub_intervals
        i_indices = torch.arange(sub_intervals)
        sub_a_list = a + i_indices * sub_interval_length
        sub_b_list = a + (i_indices + 1) * sub_interval_length

        sub_scales = (sub_b_list - sub_a_list) * 0.5
        sub_offsets = (sub_a_list + sub_b_list) * 0.5

        all_points = standard_points.unsqueeze(0) * sub_scales.unsqueeze(
            1
        ) + sub_offsets.unsqueeze(1)
        all_weights = standard_weights.unsqueeze(0) * sub_scales.unsqueeze(1)

        merged_points = all_points.flatten()
        merged_weights = all_weights.flatten()

        return merged_points, merged_weights


def generate_quad_points(
    domain_bounds: list[tuple[float, float]],
    n_quad_points: int = 16,
    sub_intervals: int = 10,
    device: torch.device | None = None,
    dtype: torch.dtype = DTYPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    为各维度生成高斯积分点和权重(张量积形式)

    每个维度可以有不同的边界区间, 积分点和权重都独立生成.

    Args:
        domain_bounds: 积分域边界,格式为[(a₁, b₁), (a₂, b₂), ...]
                      domain_bounds[i] 对应 quad_points[:, i] 的边界
        n_quad_points: 积分点数(所有维度共用)
        sub_intervals: 子区间数量(所有维度共用)
        device: 目标设备,默认为None(CPU)
        dtype: 数据类型,默认为torch.float32

    Returns:
        (quad_points, quad_weights):
        - quad_points: 形状(n_1d, dim)的积分点张量,每列对应一个维度的采样点
        - quad_weights: 形状(n_1d, dim)的积分权重张量,每列对应一个维度的权重

    注意: 张量积网格实际代表 n_1d^dim 个点,但只需 O(n_1d*dim) 内存
    """
    gl = GaussLegendre()
    dim = len(domain_bounds)

    # 1. 生成标准高斯点和权重 (只需一次)
    standard_points, standard_weights = gl._standard_gauss_points(
        n_quad_points, dtype=dtype
    )
    # standard_points, standard_weights: (n_points,)

    # 2. 把边界转成张量
    bounds = torch.tensor(domain_bounds, dtype=dtype)  # (dim, 2)
    a_vec = bounds[:, 0]  # (dim,)
    b_vec = bounds[:, 1]  # (dim,)

    # 3. 计算子区间 (向量化)
    sub_interval_length = (b_vec - a_vec) / sub_intervals  # (dim,)
    i_indices = torch.arange(sub_intervals, dtype=dtype)  # (sub_intervals,)

    # sub_a: (sub_intervals, dim)
    sub_a = a_vec.unsqueeze(0) + i_indices.unsqueeze(
        1
    ) * sub_interval_length.unsqueeze(0)
    sub_b = sub_a + sub_interval_length.unsqueeze(0)

    sub_scales = (sub_b - sub_a) * 0.5  # (sub_intervals, dim)
    sub_offsets = (sub_a + sub_b) * 0.5  # (sub_intervals, dim)

    # 4. 计算所有点和权重
    # standard_points: (n_points,) -> (n_points, 1, 1)
    # sub_scales: (sub_intervals, dim) -> (1, sub_intervals, dim)
    all_points = standard_points.view(-1, 1, 1) * sub_scales.unsqueeze(
        0
    ) + sub_offsets.unsqueeze(0)  # (n_points, sub_intervals, dim)

    all_weights = standard_weights.view(-1, 1, 1) * sub_scales.unsqueeze(0)
    # (n_points, sub_intervals, dim)

    # 5. reshape 为 (n_1d, dim), 其中 n_1d = n_points * sub_intervals
    # 注意: 循环版本的顺序是先遍历子区间内的点, 再遍历子区间
    # 所以需要 permute(1, 0, 2) 把 sub_intervals 放到前面
    n_1d = n_quad_points * sub_intervals
    quad_points = all_points.permute(1, 0, 2).reshape(n_1d, dim)
    quad_weights = all_weights.permute(1, 0, 2).reshape(n_1d, dim)

    if device is not None:
        quad_points = quad_points.to(device)
        quad_weights = quad_weights.to(device)

    return quad_points, quad_weights


def int_tnn(
    tnn: TNN,
    quad_points: torch.Tensor,
    quad_weights: torch.Tensor,
) -> torch.Tensor:
    """
    计算单个TNN的积分(张量积网格)

    利用TNN结构和张量积网格进行高效积分计算:
    对于TNN: u(x) = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} func_d(x_d)[r]
    积分: ∫ u(x) dx = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} (Σ_i ω_d[i] func_d(x_d^i)[r])

    Args:
        tnn: 待积分的TNN实例
        quad_points: 积分点张量,形状(n_1d, dim) - 张量积表示
        quad_weights: 积分权重张量,形状(n_1d, dim) - 每个维度有独立的权重

    Returns:
        TNN在指定域上的积分值
    """
    # 获取函数输出: (n_1d, dim) → (n_1d, rank, dim)
    output = tnn.func(quad_points)

    # 使用 einsum 一步完成加权求和和维度乘积
    # 'nd,nrd->rd': 每个维度用自己的权重加权求和 → (rank, dim)
    # 然后对 dim 维度求积 → (rank,)
    weighted = torch.einsum("nd,nrd->rd", quad_weights, output)
    prod_result = weighted.prod(dim=-1)  # (rank,)

    # 与theta加权求和
    result = torch.einsum("r,r->", tnn.theta, prod_result)

    return result


def int_tnn_product(
    tnn1: TNN,
    tnn2: TNN,
    quad_points: torch.Tensor,
    quad_weights: torch.Tensor,
) -> torch.Tensor:
    """
    计算两个TNN乘积的积分(张量积网格,内存优化)

    高效计算两个TNN乘积的积分:
    ∫ (tnn1 * tnn2) dx = Σ_{r1,r2} (θ1_{r1}*θ2_{r2}) Π_d [Σ_i ω_d[i] f1_d^{r1}(x_d^i) f2_d^{r2}(x_d^i)]

    通过避免显式构造 (n_1d, rank1, rank2, dim) 的大中间变量来优化内存.

    复杂度: O(n_1d × rank1 × rank2 × dim)
    内存: O(max(rank1×n_1d×dim, rank1×rank2×dim))

    Args:
        tnn1: 第一个TNN实例
        tnn2: 第二个TNN实例
        quad_points: 积分点张量,形状(n_1d, dim) - 张量积表示
        quad_weights: 积分权重张量,形状(n_1d, dim) - 每个维度有独立的权重

    Returns:
        两个TNN乘积的积分值
    """
    if tnn1.dim != tnn2.dim:
        raise ValueError(f"两个TNN的维度必须相同,但得到{tnn1.dim}和{tnn2.dim}")

    # 1. 获取函数输出
    output1 = tnn1.func(quad_points)  # (n_1d, rank1, dim)
    output2 = tnn2.func(quad_points)  # (n_1d, rank2, dim)

    # 2. 用权重加权 output1 并转置为 (rank1, n_1d, dim)
    weighted_output1 = torch.einsum("nd,nrd->rnd", quad_weights, output1)
    # (rank1, n_1d, dim)

    # 3. 与 output2 做缩并 (einsum 避免大中间变量)
    integral_per_dim = torch.einsum("rnd,nsd->rsd", weighted_output1, output2)
    # (rank1, n_1d, dim) × (n_1d, rank2, dim) → (rank1, rank2, dim)

    # 4. 对dim维度求积
    prod_over_dim = integral_per_dim.prod(dim=-1)  # (rank1, rank2)

    # 5. 与theta外积求和
    result = torch.einsum("r,s,rs->", tnn1.theta, tnn2.theta, prod_over_dim)

    return result


def l2_norm(
    tnn: TNN,
    quad_points: torch.Tensor,
    quad_weights: torch.Tensor,
) -> torch.Tensor:
    """
    计算TNN的L2范数(张量积网格)

    计算 ||tnn||_L2 = sqrt(∫ (tnn)² dx)

    Args:
        tnn: 待计算L2范数的TNN实例
        quad_points: 积分点张量,形状(n_1d, dim) - 张量积表示
        quad_weights: 积分权重张量,形状(n_1d, dim) - 每个维度有独立的权重

    Returns:
        TNN的L2范数
    """
    return int_tnn_product(tnn, tnn, quad_points, quad_weights).sqrt()


def assemble_mass_matrix(
    tnn: TNN,
    quad_points: torch.Tensor,
    quad_weights: torch.Tensor,
) -> torch.Tensor:
    """
    组装质量矩阵 (Mass Matrix)

    对于TNN的基函数 φᵣ(x) = Π_d f_d^{(r)}(x_d), 计算:
    M_{ij} = ∫ φᵢ(x) φⱼ(x) dx = Π_d [∫ f_d^{(i)}(x_d) f_d^{(j)}(x_d) dx_d]

    利用变量分离结构, 定义各维度的积分因子:
    A_{ij}^{(d)} = ∫ f_d^{(i)}(x_d) f_d^{(j)}(x_d) dx_d

    则质量矩阵为: M_{ij} = Π_d A_{ij}^{(d)}

    Args:
        tnn: TNN实例, 其 func 提供基函数
        quad_points: 积分点张量, 形状(n_1d, dim)
        quad_weights: 积分权重张量, 形状(n_1d, dim)

    Returns:
        质量矩阵, 形状(rank, rank)
    """
    # 获取函数值: (n_1d, rank, dim)
    vals = tnn.func(quad_points)

    # 计算各维度的积分因子 A_{ij}^{(d)}
    # A[i,j,d] = Σ_n weights[n,d] * vals[n,i,d] * vals[n,j,d]
    A = torch.einsum("nd,nid,njd->ijd", quad_weights, vals, vals)
    # A: (rank, rank, dim)

    # 质量矩阵: M_{ij} = Π_d A_{ij}^{(d)}
    M = A.prod(dim=-1)  # (rank, rank)

    return M


def assemble_stiffness_matrix(
    tnn: TNN,
    quad_points: torch.Tensor,
    quad_weights: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    组装刚度矩阵 (Stiffness Matrix)

    对于TNN的基函数 φᵣ(x) = Π_d f_d^{(r)}(x_d), 计算:
    S_{ij} = ∫ ∇φᵢ · ∇φⱼ dx = Σ_k ∫ (∂φᵢ/∂x_k)(∂φⱼ/∂x_k) dx

    利用变量分离结构:
    ∂φᵣ/∂x_k = (∂f_k^{(r)}/∂x_k) · Π_{d≠k} f_d^{(r)}(x_d)

    定义:
    A_{ij}^{(d)} = ∫ f_d^{(i)} f_d^{(j)} dx_d  (质量因子)
    B_{ij}^{(d)} = ∫ (∂f_d^{(i)}/∂x_d)(∂f_d^{(j)}/∂x_d) dx_d  (刚度因子)

    则刚度矩阵为:
    S_{ij} = Σ_k B_{ij}^{(k)} · Π_{d≠k} A_{ij}^{(d)}
           = Σ_k B_{ij}^{(k)} · M_{ij} / A_{ij}^{(k)}

    其中 M_{ij} = Π_d A_{ij}^{(d)} 是质量矩阵.

    Args:
        tnn: TNN实例, 其 func 需要支持 forward_all_grad2 方法
        quad_points: 积分点张量, 形状(n_1d, dim)
        quad_weights: 积分权重张量, 形状(n_1d, dim)
        eps: 防止除零的小量, 默认1e-12

    Returns:
        刚度矩阵, 形状(rank, rank)
    """
    # 获取函数值和一阶导数 (忽略二阶导数)
    # vals, grads: (n_1d, rank, dim)
    vals, grads, _ = tnn.func.forward_all_grad2(quad_points)

    # 计算质量因子 A_{ij}^{(d)} 和刚度因子 B_{ij}^{(d)}
    # A[i,j,d] = Σ_n weights[n,d] * vals[n,i,d] * vals[n,j,d]
    # B[i,j,d] = Σ_n weights[n,d] * grads[n,i,d] * grads[n,j,d]
    A = torch.einsum("nd,nid,njd->ijd", quad_weights, vals, vals)
    B = torch.einsum("nd,nid,njd->ijd", quad_weights, grads, grads)
    # A, B: (rank, rank, dim)

    # 质量矩阵: M_{ij} = Π_d A_{ij}^{(d)}
    M = A.prod(dim=-1)  # (rank, rank)

    # 刚度矩阵: S_{ij} = Σ_k B_{ij}^{(k)} · M_{ij} / A_{ij}^{(k)}
    # 使用 M / A 技巧避免循环, 加 eps 防止除零
    S = (B * M.unsqueeze(-1) / (A + eps)).sum(dim=-1)  # (rank, rank)

    return S
