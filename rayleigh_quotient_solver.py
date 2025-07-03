"""
Rayleigh商求解器 - 用于特征值问题的求解

基于张量神经网络(TNN)的Rayleigh商计算和拉普拉斯特征值问题求解
"""

import math

import torch
import torch.optim as optim

from tnn_reconstructed import TensorNeuralNetwork, TNNIntegrator


def gradient_squared_integral(
    tnn: TensorNeuralNetwork,
    domain_bounds: list[tuple[float, float]],
    integrator: TNNIntegrator,
):
    """
    计算梯度的L²范数积分: ∫|∇u|² dx

    这个函数计算TNN函数u(x)的梯度平方在给定域上的积分.

    数学原理:
    梯度的平方范数定义为: |∇u|² = Σᵢ (∂u/∂xᵢ)²
    因此积分可以分解为: ∫|∇u|² dx = Σᵢ ∫(∂u/∂xᵢ)² dx

    算法步骤:
    1. 对每个空间维度i, 创建梯度分量的TNN表示: ∂u/∂xᵢ
    2. 使用tnn_int2方法计算每个梯度分量的平方积分: ∫(∂u/∂xᵢ)² dx
    3. 将所有维度的贡献求和得到最终结果

    Args:
        tnn: 待计算梯度积分的TensorNeuralNetwork实例
        domain_bounds: 积分域的边界, 格式为[(a₁,b₁), (a₂,b₂), ..., (aₙ,bₙ)]
        integrator: TNNIntegrator实例, 用于执行张量积分

    Returns:
        torch.Tensor: 标量张量, 表示∫|∇u|²dx的数值结果, 支持自动微分

    Note:
        - 返回值始终为非负数
        - 计算涉及自动微分, 可能比u²积分稍慢
        - 对于特征值问题, 这个积分与最小特征值直接相关
    """
    total_integral = torch.tensor(0.0, requires_grad=True)

    for i in range(tnn.dim):
        # 创建第i个梯度分量的TNN
        grad_i_tnn = tnn.grad(i)

        # 计算 (∂u/∂xᵢ)² 的积分, 使用tnn_int2方法
        grad_i_squared_integral = integrator.tnn_int2(
            grad_i_tnn, grad_i_tnn, domain_bounds
        )

        total_integral = total_integral + grad_i_squared_integral

    return total_integral


def rayleigh_quotient(
    tnn: TensorNeuralNetwork,
    domain_bounds: list[tuple[float, float]],
    integrator: TNNIntegrator,
    potential_func=None,
):
    """
    计算TNN函数的Rayleigh商

    这个函数是特征值求解的核心, 通过计算Rayleigh商来评估当前TNN函数的"特征值".
    Rayleigh商提供了特征值的上界估计, 通过最小化这个商可以逼近真实的最小特征值.

    数学公式:
    R[u] = (∫|∇u|² dx + ∫V(x)|u|² dx) / (∫|u|² dx)

    计算步骤:
    1. 动能项: 使用gradient_squared_integral计算∫|∇u|² dx
    2. 势能项: 计算∫V(x)|u|² dx (当前版本暂未实现)
    3. 归一化项: 使用tnn_int2计算∫|u|² dx
    4. 组合得到Rayleigh商

    Args:
        tnn: TensorNeuralNetwork实例, 待计算Rayleigh商的函数
        domain_bounds: 积分域的边界, 格式为[(a₁,b₁), (a₂,b₂), ..., (aₙ,bₙ)]
        integrator: TNNIntegrator实例, 用于执行张量积分
        potential_func: 势函数V(x), 如果为None则忽略势能项

    Returns:
        torch.Tensor: Rayleigh商的数值, 支持自动微分用于梯度优化

    Note:
        - 返回值越小, 对应的特征值估计越准确
        - 分母添加小的正则化项1e-8防止除零
        - 当前版本的势能项设为0, 适用于纯拉普拉斯特征值问题
    """

    # 计算动能项: ∫|∇u|² dx
    kinetic_term = gradient_squared_integral(tnn, domain_bounds, integrator)

    # 计算势能项: ∫V|u|² dx (暂时忽略势能, 设为0)
    potential_term = 0.0
    if potential_func is not None:
        # 如果有势能, 需要额外处理
        # 这里简化为0, 实际应用中需要扩展
        potential_term = 0.0

    # 计算归一化项: ∫|u|² dx
    normalization_term = integrator.tnn_int2(tnn, tnn, domain_bounds)

    # 计算Rayleigh商
    rayleigh_quotient = (kinetic_term + potential_term) / (
        normalization_term + 1e-8
    )

    return rayleigh_quotient


def laplacian_eigenvalue_problem():
    """
    求解高维拉普拉斯特征值问题

    本函数演示如何使用张量神经网络求解高维拉普拉斯特征值问题.

    问题描述:
        求解偏微分方程: -Δu = λu 在域 Ω = [0,1]^d 上
        边界条件: u|∂Ω = 0 (齐次Dirichlet边界条件)
        目标: 寻找最小特征值 λ 和对应的特征函数 u

    理论解:
        对于d维单位超立方体上的拉普拉斯算子, 最小特征值为:
        λ_min = d * π²

    为什么使用Rayleigh商方法:
        Rayleigh商是求解特征值问题的经典变分方法, 具有以下重要性质:

        1. 变分原理: 对于自伴算子-Δ, Rayleigh商 R[u] = (u, -Δu)/(u, u)
           提供了最小特征值的上界估计. 根据变分原理:
           λ_min = min_{u≠0} R[u] = min_{u≠0} ∫|∇u|²dx / ∫|u|²dx

        2. 优化友好: Rayleigh商是关于函数u的连续泛函, 可以通过梯度下降等
           优化算法进行最小化, 非常适合神经网络的训练框架

        3. 边界条件自然满足: 通过构造满足齐次Dirichlet边界条件的TNN,
           Rayleigh商的最小化自动在正确的函数空间中进行

        4. 数值稳定性: 相比直接求解特征值方程, Rayleigh商方法避免了
           微分算子的直接离散化, 减少了数值误差的累积

        5. 高维适应性: 传统有限元方法在高维问题中面临维数灾难,
           而基于TNN的Rayleigh商方法通过张量分解有效缓解了这一问题

    计算方法:
        1. 构造满足边界条件的张量神经网络
        2. 使用Rayleigh商方法逼近特征值: R[u] = ∫|∇u|²dx / ∫|u|²dx
        3. 通过三阶段优化策略最小化Rayleigh商以逼近最小特征值

    Returns:
        tuple: (tnn, losses) - 训练后的TNN和损失历史

    Note:
        - 使用正弦激活函数以更好地逼近特征函数的振荡性质
        - 采用层归一化和系数缩放提高训练稳定性
        - 优化过程分为Adam快速下降, LBFGS精细优化和最终微调三个阶段
    """
    dim = 5  # 维度
    rank = 15  # 张量秩

    print(f">>> {dim} 维拉普拉斯特征值问题 <<<")
    print(f"张量秩: {rank}")
    print(f"子网络总数: {rank * dim}")

    # 定义域边界
    domain_bounds = [(0.0, 1.0) for _ in range(dim)]

    # 创建重构的TNN
    tnn = TensorNeuralNetwork(
        dim=dim,
        rank=rank,
        domain_bounds=domain_bounds,
    )

    print(f"TNN参数总数: {sum(p.numel() for p in tnn.parameters())}")

    # 创建积分器
    integrator = TNNIntegrator(n_quad_points=8)

    theoretical_eigenvalue = dim * math.pi**2
    print(f"理论最小特征值: {theoretical_eigenvalue:.6f}")

    # ================== 三阶段优化 ==================

    # 优化过程
    print("\n开始优化...")

    losses = []

    # 阶段1: Adam快速下降 (5个epoch)
    print(">>> 阶段1: Adam 快速下降 <<<")
    optimizer1 = optim.Adam(tnn.parameters(), lr=0.001)

    for epoch in range(5):
        optimizer1.zero_grad()
        loss = rayleigh_quotient(tnn, domain_bounds, integrator)
        loss.backward()
        optimizer1.step()

        losses.append(loss.item())
        relative_error = (
            abs(loss.item() - theoretical_eigenvalue)
            / theoretical_eigenvalue
            * 100
        )
        print(
            f"Epoch {epoch}, 特征值: {loss.item():.6f}, 相对误差: {relative_error:.3f}%"
        )

    # 阶段2: Adam精细调优 (10个epoch)
    print("\n>>> 阶段2: Adam 精细调优 <<<")
    optimizer2 = optim.Adam(tnn.parameters(), lr=0.0001)  # 更小的学习率

    for epoch in range(10):
        optimizer2.zero_grad()
        loss = rayleigh_quotient(tnn, domain_bounds, integrator)
        loss.backward()
        optimizer2.step()

        losses.append(loss.item())
        relative_error = (
            abs(loss.item() - theoretical_eigenvalue)
            / theoretical_eigenvalue
            * 100
        )
        print(
            f"Epoch {5 + epoch}, 特征值: {loss.item():.6f}, 相对误差: {relative_error:.3f}%"
        )

    # 阶段3: LBFGS精确求解 (5个epoch)
    print("\n>>> 阶段3: LBFGS 精确求解 <<<")
    optimizer3 = optim.LBFGS(tnn.parameters(), lr=1.0)

    for epoch in range(5):

        def closure():
            optimizer3.zero_grad()
            loss = rayleigh_quotient(tnn, domain_bounds, integrator)
            loss.backward()
            return loss.item()

        loss = optimizer3.step(closure)
        losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)

        current_loss = losses[-1]
        relative_error = (
            abs(current_loss - theoretical_eigenvalue)
            / theoretical_eigenvalue
            * 100
        )
        print(
            f"Epoch {15 + epoch}, 特征值: {current_loss:.8f}, 相对误差: {relative_error:.4f}%"
        )

    # ================== 结果总结 ==================
    final_eigenvalue = losses[-1]
    final_error = (
        abs(final_eigenvalue - theoretical_eigenvalue)
        / theoretical_eigenvalue
        * 100
    )

    print(f"\n{'=' * 50}")
    print("优化完成!")
    print(f"理论特征值: {theoretical_eigenvalue:.8f}")
    print(f"最终特征值: {final_eigenvalue:.8f}")
    print(f"最终相对误差: {final_error:.4f}%")
    print(f"总训练轮数: {len(losses)}")

    # 分析收敛过程
    print("\n收敛分析:")
    print(f"阶段1 (Adam快速): {losses[0]:.6f} -> {losses[4]:.6f}")
    print(f"阶段2 (Adam精细): {losses[4]:.6f} -> {losses[14]:.6f}")
    print(f"阶段3 (LBFGS): {losses[14]:.6f} -> {losses[-1]:.8f}")

    return tnn, losses


# 运行示例
if __name__ == "__main__":
    tnn_result, losses_result = laplacian_eigenvalue_problem()
