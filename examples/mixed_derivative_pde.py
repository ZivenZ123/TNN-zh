"""
混合导数特征值问题求解器

基于张量神经网络(TNN)求解带混合导数的二元函数特征值问题
"""

import math

import torch
import torch.optim as optim

from tnn import TensorNeuralNetwork, TNNIntegrator


def mixed_derivative_eigenvalue_problem():
    """
    求解带混合导数的二元函数特征值问题

    问题描述:
        计算损失函数: ||-Du(x,y) + V₁(x)u(x,y) + V₂(y)u(x,y) - λu(x,y)||²₂

        其中:
        - D = ∂²/∂x² + 2∂²/∂x∂y + ∂²/∂y² (混合导数算子)
        - V₁(x) = sin(2πx)
        - V₂(y) = sin(√2πy)
        - λ 是特征值参数

    使用张量神经网络逼近解函数u(x,y), 并通过最小化损失函数来求解

    Returns:
        tuple: (tnn, losses) - 训练后的TNN和损失历史
    """

    # ruff: noqa: N802, N806
    print(">>> 二元函数混合导数特征值问题 <<<")

    # 问题设置
    dim = 2  # 二维问题
    rank = 5  # 张量秩
    lambda_val = 20.0  # 特征值参数, 可调整

    # 定义域边界 [0,1] × [0,1]
    domain_bounds = [(0.0, 1.0), (0.0, 1.0)]

    # 创建TNN
    tnn = TensorNeuralNetwork(
        dim=dim,
        rank=rank,
        domain_bounds=domain_bounds,
    )

    print(f"TNN参数总数: {sum(p.numel() for p in tnn.parameters())}")
    print(f"张量秩: {rank}")
    print(f"特征值λ: {lambda_val}")

    # 定义势函数
    def V1_func(x):
        """V₁(x) = sin(2πx)"""
        return torch.sin(2 * math.pi * x)

    def V2_func(y):
        """V₂(y) = sin(√2πy)"""
        return torch.sin(torch.sqrt(torch.tensor(2 * math.pi)) * y)

    # 创建积分器
    integrator = TNNIntegrator(n_quad_points=16)

    def compute_loss_function():
        """
        计算损失函数: ||-Du + V₁u + V₂u - λu||²₂

        步骤:
        1. 计算D*u = ∂²u/∂x² + 2∂²u/∂x∂y + ∂²u/∂y²
        2. 计算V₁(x)*u(x,y)和V₂(y)*u(x,y)
        3. 组合得到: operator_u = -D*u + V₁*u + V₂*u - λ*u
        4. 计算L²范数: ∫∫ |operator_u|² dx dy
        """

        # 计算各阶导数的TNN表示
        # u_x = tnn.grad(0)  # ∂u/∂x
        # u_y = tnn.grad(1)  # ∂u/∂y
        u_xx = tnn.grad(0).grad(0)  # ∂²u/∂x²
        u_yy = tnn.grad(1).grad(1)  # ∂²u/∂y²
        u_xy = tnn.grad(0).grad(1)  # ∂²u/∂x∂y

        # 构造微分算子 Du = ∂²u/∂x² + 2∂²u/∂x∂y + ∂²u/∂y²
        Du = u_xx + (2.0 * u_xy) + u_yy

        # 计算势能项: V₁(x)*u(x,y) 和 V₂(y)*u(x,y)
        V1_u = tnn.multiply_1d_function(
            V1_func, target_dim=0
        )  # V₁(x) * u(x,y)
        V2_u = tnn.multiply_1d_function(
            V2_func, target_dim=1
        )  # V₂(y) * u(x,y)

        # 计算 λu
        lambda_u = lambda_val * tnn

        # 构造算子: operator_u = -Du + V₁u + V₂u - λu
        operator_u = (-Du) + V1_u + V2_u + (-lambda_u)

        # 计算L²范数: ∫∫ |operator_u|² dx dy
        l2_norm_squared = integrator.tnn_int2(
            operator_u, operator_u, domain_bounds
        )

        return l2_norm_squared

    # 优化过程
    print("\n开始优化...")

    # 三阶段优化策略
    losses = []

    # 阶段1: Adam快速下降
    print(">>> 阶段1: Adam 快速下降 <<<")
    optimizer1 = optim.Adam(tnn.parameters(), lr=0.001)

    for epoch in range(20):
        optimizer1.zero_grad()
        loss = compute_loss_function()
        loss.backward()
        optimizer1.step()

        losses.append(loss.item())
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.8f}")

    # 阶段2: Adam精细调优
    print("\n>>> 阶段2: Adam 精细调优 <<<")
    optimizer2 = optim.Adam(tnn.parameters(), lr=0.0001)

    for epoch in range(20):
        optimizer2.zero_grad()
        loss = compute_loss_function()
        loss.backward()
        optimizer2.step()

        losses.append(loss.item())
        if epoch % 5 == 0:
            print(f"Epoch {20 + epoch}: Loss = {loss.item():.8f}")

    # 阶段3: LBFGS精确求解
    print("\n>>> 阶段3: LBFGS 精确求解 <<<")
    optimizer3 = optim.LBFGS(tnn.parameters(), lr=1.0)

    for epoch in range(10):

        def closure():
            optimizer3.zero_grad()
            loss = compute_loss_function()
            loss.backward()
            return loss.item()

        loss = optimizer3.step(closure)
        losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)

        if epoch % 2 == 0:
            print(f"Epoch {40 + epoch}: Loss = {losses[-1]:.10f}")

    print("\n优化完成!")
    print(f"最终损失: {losses[-1]:.10f}")

    # 分析收敛过程
    print("\n收敛分析:")
    print(f"阶段1 (Adam快速): {losses[0]:.8f} -> {losses[19]:.8f}")
    print(f"阶段2 (Adam精细): {losses[19]:.8f} -> {losses[39]:.8f}")
    print(f"阶段3 (LBFGS): {losses[39]:.8f} -> {losses[-1]:.10f}")

    # 测试函数在几个点的值
    test_points = torch.tensor(
        [[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]], requires_grad=True
    )

    print("\n函数值测试:")
    with torch.no_grad():
        values = tnn(test_points)
        for point, value in zip(test_points, values, strict=False):
            print(f"u({point[0]:.2f}, {point[1]:.2f}) = {value.item():.6f}")

    # 计算各项分量在测试点的值
    print("\n各项分量分析:")
    # 注意: 这里不能使用torch.no_grad(), 因为需要计算梯度
    # 选择一个测试点进行详细分析
    test_point = torch.tensor([[0.5, 0.5]], requires_grad=True)

    # 计算u值
    u_val = tnn(test_point)

    # 计算导数项
    u_x = tnn.grad(0)
    u_y = tnn.grad(1)
    u_xx = u_x.grad(0)
    u_yy = u_y.grad(1)
    u_xy = u_x.grad(1)

    Du_val = (u_xx + 2.0 * u_xy + u_yy)(test_point)

    # 计算势能项
    V1_u_val = tnn.multiply_1d_function(V1_func, 0)(test_point)
    V2_u_val = tnn.multiply_1d_function(V2_func, 1)(test_point)

    print("在点(0.5, 0.5)处:")
    print(f"  u = {u_val.item():.6f}")
    print(f"  Du = {Du_val.item():.6f}")
    print(f"  V₁u = {V1_u_val.item():.6f}")
    print(f"  V₂u = {V2_u_val.item():.6f}")
    print(f"  λu = {lambda_val * u_val.item():.6f}")

    return tnn, losses


# 运行示例
if __name__ == "__main__":
    tnn_mixed, losses_mixed = mixed_derivative_eigenvalue_problem()
