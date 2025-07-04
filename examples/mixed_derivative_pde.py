"""
混合导数特征值问题求解器

基于张量神经网络(TNN)求解带混合导数的二元函数特征值问题

GPU使用说明:
1. 代码会自动检测并使用GPU (如果可用)
2. 全局变量DEVICE自动设置为"cuda"或"cpu"
3. 所有张量和模型会自动移动到正确的设备
4. 如需强制使用CPU, 可以在代码中修改DEVICE变量
"""

import math

import torch

from tnn import (
    DEVICE,
    TensorNeuralNetwork,
    TNNIntegrator,
    TNNTrainer,
)


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
    print(f"张量秩: {rank}")
    print(f"特征值λ: {lambda_val}")
    print(f"计算设备: {DEVICE}")

    # 定义域边界 [0,1] × [0,1]
    domain_bounds = [(0.0, 1.0), (0.0, 1.0)]

    # 创建TNN
    tnn = TensorNeuralNetwork(
        dim=dim,
        rank=rank,
        domain_bounds=domain_bounds,
    ).to(DEVICE)

    # 创建积分器
    integrator = TNNIntegrator(n_quad_points=16)

    # 定义势函数
    def V1_func(x):
        """V₁(x) = sin(2πx)"""
        return torch.sin(2 * math.pi * x)

    def V2_func(y):
        """V₂(y) = sin(√2πy)"""
        # 确保常数和输入张量在同一设备上
        sqrt_2pi = torch.sqrt(torch.tensor(2 * math.pi, device=y.device))
        return torch.sin(sqrt_2pi * y)

    def loss_fn():
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

    # 创建训练器
    trainer = TNNTrainer(tnn, loss_fn)

    # 配置训练阶段
    training_phases = [
        {
            "type": "adam",
            "lr": 0.001,
            "epochs": 10,
            "name": "Adam 快速下降",
        },
        {
            "type": "adam",
            "lr": 0.0001,
            "epochs": 10,
            "name": "Adam 精细调优",
        },
        {
            "type": "lbfgs",
            "lr": 1.0,
            "epochs": 1,
            "name": "LBFGS 精确求解",
        },
    ]

    # 执行训练
    losses, training_time = trainer.multi_phase(training_phases)

    print(f"\n最终损失: {losses[-1]:.10f}")
    print(f"总训练时间: {training_time:.2f} 秒")

    return tnn, losses


# 运行示例
if __name__ == "__main__":
    tnn_result, losses_result = mixed_derivative_eigenvalue_problem()
