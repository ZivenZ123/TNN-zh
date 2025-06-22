# %%
"""
五维张量神经网络(TNN)求解椭圆偏微分方程的实现

本模块使用张量神经网络方法求解五维椭圆偏微分方程:
-Δu = f(x), x ∈ [-1,1]^5
其中 f(x) = Σ_{k=1}^5 sin(2πx_k) * Π_{i≠k}^5 sin(πx_i)

主要功能:
1. 构建五维张量神经网络模型
2. 定义损失函数和求积规则
3. 使用Adam和LBFGS优化器进行训练
4. 计算L2和H1范数下的误差估计
"""

import os
import time

import numpy as np
import quadrature
import torch
import torch.optim as optim
from integration import (
    error0_estimate,
    error1_estimate,
    int2_tnn,
    int2_tnn_amend_1d,
)
from tnn import Tnn, TnnSin

pi = 3.14159265358979323846

# %%
# ********** 选择数据类型和设备 **********
# 设置张量的数据类型为双精度浮点数, 确保计算精度
dtype = torch.double
# dtype = torch.float  # 可选择单精度浮点数以提高计算速度

# 设置计算设备为CPU
device = "cpu"
# device = "cuda"  # 可选择GPU以加速计算


# ********** 设置求积规则 **********
# 定义计算域: [a,b]^dim, 这里是五维超立方体[-1,1]^5
a = -1  # 计算域左边界
b = 1  # 计算域右边界
dim = 5  # 问题维度

# 设置求积规则参数:
quad = 16  # 单个区间上的求积点数量
n_subintervals = 200  # 每个维度上的分区数量

# 生成一维复合求积点和权重
# 这些点将用于数值积分计算
x, w = quadrature.composite_quadrature_1d(
    quad, (a, b), n_subintervals, device=device, dtype=dtype
)
n_quadrature_points = len(x)  # 总的求积点数量
print(f"求积点总数: {n_quadrature_points}")

# ********** 创建神经网络模型 **********
# 设置网络参数
p = 50  # 输出层神经元数量, 对应基函数的数量
size = [1, 100, 100, 100, p]  # 网络结构: 输入层->3个隐藏层->输出层
activation = TnnSin  # 使用正弦激活函数


def bd(x):
    """
    强制边界条件函数

    使得神经网络在边界上自动满足齐次Dirichlet边界条件
    u(±1, x2, x3, x4, x5) = 0

    Args:
        x: 输入坐标点

    Returns:
        边界条件函数值 (x-a)(b-x) = (x+1)(1-x)
    """
    return (x - a) * (b - x)


def grad_bd(x):
    """
    边界条件函数的一阶导数

    Args:
        x: 输入坐标点

    Returns:
        边界条件函数的导数值
    """
    return -2 * x + a + b


def grad_grad_bd(x):
    """
    边界条件函数的二阶导数

    Args:
        x: 输入坐标点

    Returns:
        边界条件函数的二阶导数值 (常数-2)
    """
    return -2 * torch.ones_like(x)


# 创建张量神经网络模型
model = (
    Tnn(
        dim,  # 输入维度
        size,  # 网络结构
        activation,  # 激活函数
        bd=bd,  # 边界条件函数
        grad_bd=grad_bd,  # 边界条件函数的一阶导数
        grad_grad_bd=grad_grad_bd,  # 边界条件函数的二阶导数
        scaling=False,  # 不使用输入缩放
    )
    .to(dtype)  # 设置数据类型
    .to(device)  # 设置计算设备
)
print(f"神经网络模型结构:\n{model}")

# %%
# 构造右端项函数 F(x_1, x_2, ..., x_d) = Σ_{k=1}^d sin(2πx_k) * Π_{i≠k}^d sin(πx_i)
# 这是一个d维函数, 每一项都是某个坐标的2π正弦乘以其他坐标的π正弦的乘积
sin_pi_x = torch.sin(pi * x)  # sin(πx_i) for all dimensions
sin_2pi_x = torch.sin(2 * pi * x)  # sin(2πx_i) for all dimensions

# 初始化F张量, 形状为 (dim, dim, n_quadrature_points)
F = torch.zeros((dim, dim, n_quadrature_points), dtype=dtype, device=device)

# 对每个k (第一个维度索引), 构造第k项: sin(2πx_k) * Π_{i≠k} sin(πx_i)
for k in range(dim):
    # 从所有维度的sin(πx_i)开始
    F[k] = sin_pi_x.unsqueeze(0).expand(dim, -1)  # 复制到所有维度
    # 将第k个维度替换为sin(2πx_k)
    F[k, k, :] = sin_2pi_x

# 系数向量, 用于组合不同的函数项
alpha_F = torch.ones(dim, dtype=dtype, device=device)

# 构造右端项函数的梯度
# ∇F 的每个分量对应相应的导数
grad_F = torch.ones(
    (dim, dim, n_quadrature_points), device=device, dtype=dtype
)
grad_F = pi * torch.cos(pi * x) * grad_F  # 基础梯度: π cos(πx_i)
for i in range(dim):
    grad_F[i, i, :] = (
        2 * pi * torch.cos(2 * pi * x)
    )  # 第i项中第i个坐标的导数: 2π cos(2πx_i)


# %%
# ********** 定义损失函数 **********
def criterion(model, w, x):
    """
    计算变分形式的损失函数

    基于弱形式求解椭圆方程: ∫∇u·∇v dx = ∫f·v dx

    损失函数表达式: L = ||Δu - f||²_L2 = ∫(Δu - f)² dx
    其中:
    - Δu 是神经网络解的拉普拉斯算子
    - f 是右端项函数
    - 展开后为: L = ∫(Δu)² dx + ∫f² dx - 2∫(Δu)·f dx

    Args:
        model: 神经网络模型
        w: 求积权重
        x: 求积点坐标

    Returns:
        损失函数值
    """
    # 计算神经网络输出及其一阶、二阶导数
    phi, grad_phi, grad_grad_phi = model(w, x, need_grad=2)

    # 基函数的系数向量
    alpha = torch.ones(p, device=device, dtype=dtype)

    # 计算刚度矩阵 A: ∫∇φ_i·∇φ_j dx
    part1 = int2_tnn_amend_1d(
        w, w, alpha, phi, alpha, phi, grad_phi, grad_phi, if_sum=False
    )

    # 计算载荷向量 B: ∫f·φ_i dx
    part2 = int2_tnn(w, alpha, phi, alpha_F, F, if_sum=False)
    part2 = torch.sum(part2, dim=-1)

    # 求解线性方程组 AC = B, 得到展开系数
    A = part1
    B = (dim + 3) * np.pi**2 * part2
    C = torch.linalg.solve(A, B)

    # 构造拉普拉斯算子 Δu 的离散表示
    phi_expand = phi.expand(dim, -1, -1, -1).clone()
    phi_expand[torch.arange(dim), torch.arange(dim), :, :] = grad_grad_phi
    grad_grad_phi_new = phi_expand.transpose(0, 1).flatten(1, 2)
    C_new = C.repeat(dim)

    # 计算损失函数的三个部分
    # part1: ∫(Δu)^2 dx
    part1 = int2_tnn(w, C_new, grad_grad_phi_new, C_new, grad_grad_phi_new)

    # part2: ∫f^2 dx
    part2 = int2_tnn(w, alpha_F, F, alpha_F, F)

    # part3: ∫(Δu)·f dx
    part3 = int2_tnn(w, C_new, grad_grad_phi_new, alpha_F, F)

    # 总损失: ||Δu - f||^2 = ||Δu||^2 + ||f||^2 - 2⟨Δu, f⟩
    loss = (
        part1
        + (dim + 3) ** 2 * np.pi**4 * part2
        + 2 * (dim + 3) * np.pi**2 * part3
    )

    return loss


# %%
# ********** 解的评估函数 **********
def compute_errors(model, w, x):
    """
    计算数值解的误差估计

    在训练过程中和训练结束后计算数值解与精确解之间的L2和H1范数误差.
    这个函数用于监控训练进度和评估最终解的精度.

    Args:
        model: 训练好的神经网络模型
        w: 求积权重
        x: 求积点坐标
    """
    # 计算神经网络输出及其梯度
    phi, grad_phi = model(w, x, need_grad=1)
    alpha = torch.ones(p, device=device, dtype=dtype)

    # 重新计算展开系数 (与损失函数中相同的过程)
    part1 = int2_tnn_amend_1d(
        w, w, alpha, phi, alpha, phi, grad_phi, grad_phi, if_sum=False
    )

    part2 = int2_tnn(w, alpha, phi, alpha_F, F, if_sum=False)
    part2 = torch.sum(part2, dim=-1)

    A = part1
    B = (dim + 3) * np.pi**2 * part2
    C = torch.linalg.solve(A, B)

    # 计算L2范数误差 (误差0)
    error0 = (
        error0_estimate(w, alpha_F, F, C, phi, projection=False)
        / torch.sqrt(int2_tnn(w, alpha_F, F, alpha_F, F))
        / ((dim + 3) * pi**2)
    )

    # 计算H1范数误差 (误差1)
    error1 = (
        error1_estimate(
            w, alpha_F, F, C, phi, grad_F, grad_phi, projection=False
        )
        / torch.sqrt(int2_tnn(w, alpha_F, F, alpha_F, F))
        / ((dim + 3) * pi**2)
    )

    print("{:<9}{:<25}".format("L2误差 = ", error0.item()))
    print("{:<9}{:<25}".format("H1误差 = ", error1.item()))
    return


# %%
# ********** Adam优化器训练过程 **********
print("=" * 50)
print("开始Adam优化器训练")
print("=" * 50)

# 训练参数设置
lr = 0.003  # 学习率
epochs = 50000  # 训练轮数
print_every = 100  # 打印间隔
save = False  # 是否保存模型

# 创建Adam优化器
optimizer = optim.Adam(model.parameters(), lr=lr)

# 记录训练开始时间
starttime = time.time()

# 开始训练循环
for e in range(epochs):
    # 计算当前损失
    loss = criterion(model, w, x)

    # 打印初始信息
    if e == 0:
        print("*" * 40)
        print("{:<9}{:<25}".format("轮次 = ", e))
        print("{:<9}{:<25}".format("损失 = ", loss.item()))
        # 计算并打印误差
        compute_errors(model, w, x)
        # 可选: 保存初始模型
        if save:
            if not os.path.exists("model"):
                os.mkdir("model")
            torch.save(model, f"model/model{e}.pkl")

    # 执行一步优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数

    # 定期打印训练进度
    if (e + 1) % print_every == 0:
        print("*" * 40)
        print("{:<9}{:<25}".format("轮次 = ", e + 1))
        print("{:<9}{:<25}".format("损失 = ", loss.item()))
        # 计算并打印当前误差
        compute_errors(model, w, x)
        # 可选: 保存当前模型
        if save:
            torch.save(model, f"model/model{e + 1}.pkl")

print("*" * 40)
print("Adam训练完成!")

# 计算并打印Adam训练耗时
endtime = time.time()
print(f"Adam训练耗时: {endtime - starttime:.2f}秒")

print("*" * 20, "LBFGS优化器", "*" * 20)


# %%
# ********** LBFGS优化器训练过程 **********
# LBFGS是一种拟牛顿法, 通常在神经网络训练的后期使用以获得更精确的解

# LBFGS训练参数
lr = 1  # LBFGS的学习率通常设为1
epochs = 10000  # LBFGS训练轮数
print_every = 100  # 打印间隔
save = True  # 保存LBFGS训练的模型

# 创建LBFGS优化器
optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)

# LBFGS训练循环
for e in range(epochs):

    def closure():
        """
        LBFGS优化器需要的闭包函数

        在每次迭代中, LBFGS可能需要多次计算损失函数和梯度
        因此需要将损失计算封装在闭包中

        Returns:
            当前的损失函数值
        """
        loss = criterion(model, w, x)
        optimizer.zero_grad()
        loss.backward()
        return loss.item()

    # 执行LBFGS优化步骤
    loss = optimizer.step(closure)

    # 打印初始信息
    if e == 0:
        print("*" * 40)
        print("{:<9}{:<25}".format("轮次 = ", e))
        print("{:<9}{:<25}".format("损失 = ", loss))
        # 计算并打印误差
        compute_errors(model, w, x)

    # 定期打印训练进度
    if (e + 1) % print_every == 0:
        print("*" * 40)
        print("{:<9}{:<25}".format("轮次 = ", e + 1))
        print("{:<9}{:<25}".format("损失 = ", loss))
        # 计算并打印当前误差
        compute_errors(model, w, x)

print("*" * 40)
print("LBFGS训练完成!")
print("=" * 50)
print("所有训练过程结束")
print("=" * 50)
