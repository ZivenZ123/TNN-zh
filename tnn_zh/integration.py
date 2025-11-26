"""
TNN积分模块

提供针对TNN的数值积分功能.

主要功能:
- GaussLegendre: 高斯-勒让德积分类
- generate_quad_points: 生成积分点和权重(张量积形式)
- int_tnn: 计算单个TNN的积分
- int_tnn_product: 计算两个TNN乘积的积分(内存优化)
- l2_norm: 计算TNN的L2范数

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
# quad_weights: (n_1d,) 形状的张量 - 所有维度共用

# 步骤2: 在积分点上进行积分计算
result1 = int_tnn(tnn, quad_points, quad_weights)
result2 = int_tnn_product(tnn1, tnn2, quad_points, quad_weights)
result3 = l2_norm(tnn, quad_points, quad_weights)
"""

import numpy as np
import torch

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

    每个维度可以有不同的边界区间,但使用相同的采样点数量和权重结构.

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
        - quad_weights: 形状(n_1d,)的积分权重张量,所有维度共用相同的权重结构

    注意: 张量积网格实际代表 n_1d^dim 个点,但只需 O(n_1d*dim) 内存
    """
    gl = GaussLegendre()
    dim = len(domain_bounds)

    # 为每个维度独立生成积分点
    points_list = []
    for d in range(dim):
        a, b = domain_bounds[d]
        points, weights = gl.gauss_points_with_subdivision(
            n_quad_points, a, b, sub_intervals, dtype=dtype
        )
        if device is not None:
            points = points.to(device)
        points_list.append(points)

    # 堆叠为 (n_1d, dim)
    quad_points = torch.stack(points_list, dim=1)

    # 权重在第一个维度生成(所有维度相同的权重结构)
    # 注意: 权重与具体的[a,b]区间无关,只与区间长度有关
    # 这里使用标准区间[0,1]生成权重结构
    _, quad_weights = gl.gauss_points_with_subdivision(
        n_quad_points, 0, 1, sub_intervals, dtype=dtype
    )
    if device is not None:
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
    积分: ∫ u(x) dx = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} (Σ_i ω[i] func_d(x_d^i)[r])

    Args:
        tnn: 待积分的TNN实例
        quad_points: 积分点张量,形状(n_1d, dim) - 张量积表示
        quad_weights: 积分权重张量,形状(n_1d,) - 所有维度共用

    Returns:
        TNN在指定域上的积分值
    """
    # 获取函数输出: (n_1d, dim) → (n_1d, rank, dim)
    output = tnn.func(quad_points)

    # 使用 einsum 一步完成加权求和和维度乘积
    # 'n,nrd->rd': 对 n_1d 维度加权求和 → (rank, dim)
    # 然后对 dim 维度求积 → (rank,)
    weighted = torch.einsum("n,nrd->rd", quad_weights, output)
    prod_result = weighted.prod(dim=-1)  # (rank,)

    # 与theta加权求和 (einsum 更统一,但 dot 对一维向量点积更直观)
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
    ∫ (tnn1 * tnn2) dx = Σ_{r1,r2} (θ1_{r1}*θ2_{r2}) Π_d [Σ_i ω[i] f1_d^{r1}(x_d^i) f2_d^{r2}(x_d^i)]

    通过避免显式构造 (n_1d, rank1, rank2, dim) 的大中间变量来优化内存.

    复杂度: O(n_1d × rank1 × rank2 × dim)
    内存: O(max(rank1×n_1d×dim, rank1×rank2×dim))

    Args:
        tnn1: 第一个TNN实例
        tnn2: 第二个TNN实例
        quad_points: 积分点张量,形状(n_1d, dim) - 张量积表示
        quad_weights: 积分权重张量,形状(n_1d,) - 所有维度共用

    Returns:
        两个TNN乘积的积分值
    """
    if tnn1.dim != tnn2.dim:
        raise ValueError(f"两个TNN的维度必须相同,但得到{tnn1.dim}和{tnn2.dim}")

    # 1. 获取函数输出
    output1 = tnn1.func(quad_points)  # (n_1d, rank1, dim)
    output2 = tnn2.func(quad_points)  # (n_1d, rank2, dim)

    # 2. 用权重加权 output1 并转置为 (rank1, n_1d, dim)
    weighted_output1 = torch.einsum("n,nrd->rnd", quad_weights, output1)
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
        quad_weights: 积分权重张量,形状(n_1d,) - 所有维度共用

    Returns:
        TNN的L2范数
    """
    return int_tnn_product(tnn, tnn, quad_points, quad_weights).sqrt()
