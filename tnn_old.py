# %%
"""
张量神经网络 (TNN) 重构实现

基于论文 "Tensor Neural Network and Its Numerical Integration"
以及手写推导的数学公式

核心数学表达式:
tnn(x₁, x₂, ..., x_dim) = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} subtnn_d^{(r)}(x_d)

关键改进:
1. 子网络从 R → R^{rank} 改为 R → R (一维到一维)
2. 需要 rank × dim 个子网络
3. 在每层激活前添加归一化
4. 优化积分方法: tnn_int2 使用 _create_product_tnn + tnn_int1
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# %%
class SubTensorNeuralNetwork(nn.Module):
    """
    SubTensorNeuralNetwork - TNN的子网络 (一维到一维)

    每个子网络处理一个维度的一个输入, 输出一个标量.
    这是TNN架构的基础组件, 负责处理单个维度的输入并输出标量值.

    数学表示: subtnn_d^{(r)}: R → R
    subtnn_d^{(r)}(x_d) ∈ R

    特点:
    - 在每层激活前进行归一化, 提高训练稳定性
    - 支持多种激活函数
    - 可选择是否使用层归一化
    - 内置边界条件处理, 支持齐次Dirichlet边界条件

    Args:
        input_dim: 输入维度, 必须为1
        output_dim: 输出维度, 必须为1
        hidden_dims: 隐藏层维度的元组, 例如(50, 50)表示两个隐藏层各50个神经元
        activation: 激活函数类型, 支持'sin', 'relu', 'tanh'
        use_layer_norm: 是否使用层归一化, 默认为True
        boundary_condition: 边界条件设置, 格式为(a, b)表示在区间[a,b]上应用齐次Dirichlet边界条件
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_dims: tuple[int, ...] = (50, 50),
        activation: str = "sin",
        use_layer_norm: bool = True,
        boundary_condition: tuple[float, float] | None = None,
    ):
        super().__init__()

        # 边界条件设置
        self.boundary_condition = boundary_condition

        # 构建网络层
        layers = []
        layer_norms = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layer_norms.append(nn.LayerNorm(hidden_dim))
            else:
                layer_norms.append(nn.Identity())
            prev_dim = hidden_dim

        # 输出层 - 不使用归一化, 保持输出值的原始大小用于张量积运算
        layers.append(nn.Linear(prev_dim, output_dim))

        self.layers = nn.ModuleList(layers)
        self.layer_norms = nn.ModuleList(layer_norms)
        self.use_layer_norm = use_layer_norm

        # 激活函数
        if activation == "sin":
            self.activation = torch.sin
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = torch.relu

    def forward(self, x):
        """
        前向传播计算

        对输入的一维数据进行前向传播, 通过多层全连接网络输出标量值.
        隐藏层使用归一化和激活函数, 输出层只使用激活函数(不归一化).
        如果设置了边界条件, 会自动应用齐次Dirichlet边界条件.

        Args:
            x: 输入张量, shape为(batch_size, 1)

        Returns:
            输出张量, shape为(batch_size, 1), 已应用边界条件
        """
        # 神经网络的前向传播
        output = x

        # 处理隐藏层 (有归一化)
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            layer_norm = self.layer_norms[i]
            output = layer(output)
            # 在激活前进行归一化
            output = layer_norm(output)
            output = self.activation(output)

        # 处理输出层 (无归一化)
        output = self.layers[-1](output)
        output = self.activation(output)

        # 应用边界条件
        if self.boundary_condition is not None:
            a, b = self.boundary_condition
            # 齐次Dirichlet边界条件: (x - a)(b - x) * network_output
            boundary_factor = (x - a) * (b - x)
            # 归一化boundary_factor到[0, 2]范围
            # 最大值在区间中点处: ((b-a)/2)^2
            max_boundary_value = ((b - a) / 2) ** 2
            boundary_factor = 2 * boundary_factor / max_boundary_value
            output = boundary_factor * output

        return output

    def grad(self) -> "SubTensorNeuralNetwork":
        """
        创建当前子网络的梯度版本

        这个方法创建一个新的子网络, 其输出是当前子网络输出对输入的梯度.
        新的子网络保持相同的边界条件和基本配置.

        数学原理:
        对于子网络 f(x), 创建一个新的子网络 g(x) = df/dx

        实现原理:
        使用PyTorch的自动微分功能torch.autograd.grad来计算梯度.
        创建一个特殊的子网络类, 其forward方法自动计算并返回dy/dx.

        Returns:
            SubTensorNeuralNetwork: 新的梯度子网络实例

        Example:
            >>> subnet = SubTensorNeuralNetwork()
            >>> grad_subnet = subnet.grad()
            >>> x = torch.tensor([[0.5]], requires_grad=True)
            >>> y = subnet(x)        # 原函数值
            >>> dy_dx = grad_subnet(x)  # 梯度值

        Note:
            - 返回全新的子网络对象, 不修改原始子网络
            - 保存了对原始子网络的引用以便计算梯度
            - 新的forward方法支持梯度反向传播
            - 支持链式求导: subnet.grad().grad() 计算二阶导数
        """

        class DerivativeSubTensorNeuralNetwork(SubTensorNeuralNetwork):
            """导数子网络的特殊实现"""

            def __init__(self, original_subnet: "SubTensorNeuralNetwork"):
                # 继承原始子网络的基本配置
                super().__init__(
                    input_dim=1,
                    output_dim=1,
                    hidden_dims=(1,),  # 最小配置, 因为我们会重写forward
                    activation="sin",
                    use_layer_norm=False,
                    boundary_condition=original_subnet.boundary_condition,
                )

                # 保存原始子网络的引用
                self.original_subnet = original_subnet

            def forward(self, x):
                """计算原始子网络输出对输入的梯度"""
                x.requires_grad_(True)
                y = self.original_subnet(x)
                # 计算梯度
                grad_y = torch.autograd.grad(
                    y,
                    x,
                    grad_outputs=torch.ones_like(y),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                return grad_y

        # 返回新的梯度子网络
        return DerivativeSubTensorNeuralNetwork(self)

    def multiply_by_function(self, func) -> "SubTensorNeuralNetwork":
        """
        创建一元函数与子网络乘积的新子网络

        这个方法创建一个新的子网络, 其输出是原子网络输出与给定一元函数的乘积.
        新子网络保持与原子网络相同的边界条件和基本配置.

        数学表示: h(x) = f(x) * g(x)
        其中f是给定的一元函数, g是原子网络的输出函数

        Args:
            func: 一元函数, 接受torch.Tensor输入并返回torch.Tensor

        Returns:
            SubTensorNeuralNetwork: 新的乘积子网络

        Note:
            - 返回的是一个全新的子网络对象
            - 计算过程保持自动微分的兼容性
            - 保持原子网络的边界条件设置
        """

        class FunctionProductSubTensorNeuralNetwork(SubTensorNeuralNetwork):
            """函数乘积子网络的特殊实现"""

            def __init__(
                self, original_subnet: "SubTensorNeuralNetwork", func
            ):
                # 继承原始子网络的基本配置
                super().__init__(
                    input_dim=1,
                    output_dim=1,
                    hidden_dims=(1,),  # 最小配置, 因为我们会重写forward
                    activation="sin",
                    use_layer_norm=False,
                    boundary_condition=original_subnet.boundary_condition,
                )

                # 保存原始子网络和函数的引用
                self.original_subnet = original_subnet
                self.func = func

            def forward(self, x):
                """计算f(x) * subnet(x)"""
                subnet_output = self.original_subnet(x)
                func_output = self.func(x)
                return func_output * subnet_output

        # 返回新的函数乘积子网络
        return FunctionProductSubTensorNeuralNetwork(self, func)


class TensorNeuralNetwork(nn.Module):
    """
    TensorNeuralNetwork (TNN)

    张量神经网络, 基于张量分解的神经网络架构. 通过将高维函数表示为多个低维函数的张量积形式,
    实现高效的函数逼近和计算.

    数学表达式:
    tnn(x) = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} subtnn_d^{(r)}(x_d)

    其中:
    - rank: 张量分解的秩, 控制模型的表达能力
    - dim: 输入维度
    - θᵣ: 第r个分量的可学习系数
    - subtnn_d^{(r)}: 第r个秩分量中第d维的子网络 (R → R 映射)

    结构:
    1. 包含 rank × dim 个子网络, 每个都是 R → R 的映射
    2. θᵣ是可学习的系数, 用于平衡不同秩分量的贡献
    3. 支持齐次Dirichlet边界条件
    4. 支持多种激活函数和层归一化

    Args:
        dim: 输入维度
        rank: 张量分解的秩, 默认为10
        hidden_dims: 子网络的隐藏层维度, 默认为(50, 50)
        activation: 激活函数类型, 支持"sin", "relu", "tanh", 默认为"sin"
        domain_bounds: 定义域边界, 用于边界条件处理, 默认为None
            domain_bounds 数据结构:
            - 类型: list[tuple[float, float]] | None
            - 格式: [(a₁, b₁), (a₂, b₂), ..., (aₙ, bₙ)]
            - 含义: 第i个元组(aᵢ, bᵢ)表示第i维的定义域区间[aᵢ, bᵢ]
            - 示例:
                * 1维: [(0.0, 1.0)]  # x ∈ [0,1]
                * 2维: [(0.0, 1.0), (-1.0, 1.0)]  # x₁ ∈ [0,1], x₂ ∈ [-1,1]
                * 3维: [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # 单位立方体
                * 高维: [(0.0, 1.0) for _ in range(dim)]  # 简洁写法
        use_coefficients: 是否使用可学习的θᵣ系数, 默认为True
        use_layer_norm: 是否在子网络中使用层归一化, 默认为True
    """

    def __init__(
        self,
        dim: int,
        rank: int = 10,
        hidden_dims: tuple[int, ...] = (50, 50),
        activation: str = "sin",
        domain_bounds: list[tuple[float, float]] | None = None,
        use_coefficients: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.rank = rank
        self.domain_bounds = domain_bounds
        self.use_coefficients = use_coefficients

        # 创建 rank × dim 个子网络
        # subnetworks[r][d] 对应 subtnn_d^{(r)}
        self.subnetworks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SubTensorNeuralNetwork(
                            input_dim=1,
                            output_dim=1,
                            hidden_dims=hidden_dims,
                            activation=activation,
                            use_layer_norm=use_layer_norm,
                            boundary_condition=domain_bounds[d]
                            if domain_bounds
                            else None,
                        )
                        for d in range(dim)
                    ]
                )
                for _ in range(rank)
            ]
        )

        # 可学习的系数θᵣ
        if use_coefficients:
            self.theta = nn.Parameter(torch.ones(rank))
        else:
            self.register_buffer("theta", torch.ones(rank))

    def forward(self, x):
        """
        前向传播计算

        实现TNN的核心计算逻辑:
        tnn(x) = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} subtnn_d^{(r)}(x_d)

        对于每个秩分量r:
        1. 计算所有维度上子网络输出的乘积
        2. 乘以对应的系数θᵣ
        3. 累加到最终结果中

        Args:
            x: 输入张量, shape为(batch_size, dim)

        Returns:
            输出张量, shape为(batch_size, 1)
        """
        batch_size = x.shape[0]
        device = x.device

        # 计算TNN输出
        result = torch.zeros(batch_size, 1, device=device)

        # 对每个rank分量r
        for r in range(self.rank):
            # 计算 Π_{d=1}^{dim} subtnn_d^{(r)}(x_d)
            product = torch.ones(batch_size, 1, device=device)

            for d in range(self.dim):
                # 提取第d维输入
                x_d = x[:, d : d + 1]  # shape: (batch_size, 1)

                # 获取第r个rank的第d个子网络
                subnet = self.subnet(r, d)

                # 累乘到product中
                product *= subnet(x_d)

            # 乘以系数θᵣ并累加到结果
            result += self.theta[r] * product

        return result

    def subnet(self, rank_idx: int, dim_idx: int) -> SubTensorNeuralNetwork:
        """
        获取指定的子网络 subtnn_d^{(r)}

        根据秩索引和维度索引获取对应的子网络. 这个方法用于访问和分析
        特定的子网络组件.

        Args:
            rank_idx: 秩索引, 范围为[0, rank-1]
            dim_idx: 维度索引, 范围为[0, dim-1]

        Returns:
            SubTensorNeuralNetwork: 指定的子网络
        """
        return self.subnetworks[rank_idx][dim_idx]  # type: ignore

    def grad(self, grad_dim: int) -> "TensorNeuralNetwork":
        """
        创建TNN在指定维度上的梯度的TNN表示

        这个方法构造一个新的TNN, 用于表示原TNN函数在某个特定维度上的偏导数.
        新的TNN保持相同的张量积结构, 但其中一个维度的子网络被替换为导数版本.

        数学原理:
        对于原TNN: u(x) = Σᵣ θᵣ Πᵈ subtnn_d^{(r)}(x_d)
        其在第i维的偏导数为:
        ∂u/∂x_i = Σᵣ θᵣ (∂subtnn_i^{(r)}/∂x_i) Πᵈ≠ᵢ subtnn_d^{(r)}(x_d)

        实现策略:
        1. 创建一个新的TNN, 具有相同的维度和秩结构
        2. 复制原TNN的所有系数θᵣ
        3. 对于梯度维度: 创建导数子网络(使用自动微分)
        4. 对于其他维度: 直接复制原子网络的参数

        Args:
            grad_dim: 计算梯度的维度索引, 范围为[0, self.dim-1]

        Returns:
            TensorNeuralNetwork: 新的TNN实例, 表示∂u/∂x_{grad_dim}

        Raises:
            AssertionError: 当grad_dim超出有效范围时

        Example:
            >>> tnn = TensorNeuralNetwork(dim=3, rank=5)
            >>> grad_x1 = tnn.grad(0)  # 对第1维求偏导
            >>> grad_x2 = tnn.grad(1)  # 对第2维求偏导

        Note:
            - 返回的TNN与原TNN具有相同的输入输出维度
            - 梯度计算通过自动微分实现, 保持计算图的连续性
            - 新TNN在非梯度维度的参数与原TNN共享, 梯度维度使用导数子网络
            - 支持链式求导: tnn.grad(0).grad(1) 计算二阶偏导数
        """
        assert 0 <= grad_dim < self.dim, (
            f"grad_dim {grad_dim} 超出有效范围 [0, {self.dim - 1}]"
        )

        # 创建梯度TNN, rank保持不变
        # 注意: 这里的hidden_dims等参数不会实际使用, 因为我们会直接替换子网络
        grad_tnn = TensorNeuralNetwork(
            dim=self.dim,
            rank=self.rank,
            hidden_dims=(10,),  # 使用最小配置, 避免不必要的内存分配
            activation="sin",
            domain_bounds=self.domain_bounds,
            use_coefficients=True,
            use_layer_norm=True,
        )

        # 构造梯度TNN的子网络和系数
        grad_subnetworks = []  # 存储所有rank分量的子网络
        grad_coefficients = []  # 存储所有rank分量的系数

        for r in range(self.rank):
            # 复制系数
            grad_coefficients.append(self.theta[r])

            # 为这个rank分量创建所有维度的子网络
            rank_subnets = []
            for d in range(self.dim):
                if d == grad_dim:
                    # 对于梯度维度, 创建梯度子网络
                    derivative_subnet = self.subnet(r, d).grad()
                    rank_subnets.append(derivative_subnet)
                else:
                    # 对于其他维度, 直接引用原子网络
                    rank_subnets.append(self.subnet(r, d))

            grad_subnetworks.append(rank_subnets)

        # 直接替换子网络结构和系数
        with torch.no_grad():
            # 设置系数
            for idx, coeff in enumerate(grad_coefficients):
                grad_tnn.theta[idx] = coeff

            # 替换子网络结构
            grad_tnn.subnetworks = nn.ModuleList(
                [
                    nn.ModuleList(rank_subnets)
                    for rank_subnets in grad_subnetworks
                ]
            )

        return grad_tnn

    def __add__(self, other: "TensorNeuralNetwork") -> "TensorNeuralNetwork":
        """
        重载加法操作符, 实现两个TNN的直接相加

        这个方法允许使用 tnn1 + tnn2 的语法来创建两个TNN的和.
        利用张量分解的线性性质, 两个TNN的和仍然保持张量积的结构.

        数学原理:
        对于两个TNN:
        u1(x) = Σ_{r=1}^{rank1} α_r Π_{d=1}^{dim} f_d^{(r)}(x_d)
        u2(x) = Σ_{s=1}^{rank2} β_s Π_{d=1}^{dim} g_d^{(s)}(x_d)

        它们的和为:
        u1(x) + u2(x) = Σ_{r=1}^{rank1} α_r Π_{d=1}^{dim} f_d^{(r)}(x_d) + Σ_{s=1}^{rank2} β_s Π_{d=1}^{dim} g_d^{(s)}(x_d)

        这可以重写为一个新的TNN:
        u_sum(x) = Σ_{k=1}^{rank1+rank2} γ_k Π_{d=1}^{dim} h_d^{(k)}(x_d)

        其中前rank1个分量来自第一个TNN, 后rank2个分量来自第二个TNN.

        实现策略:
        1. 创建一个新的TNN骨架结构
        2. 直接引用原始TNN的子网络, 避免参数复制
        3. 设置正确的系数和网络结构

        Args:
            other: 另一个TensorNeuralNetwork实例

        Returns:
            TensorNeuralNetwork: 新的TNN实例, 表示两个TNN的和

        Raises:
            AssertionError: 当两个TNN的维度不匹配时
            TypeError: 当other不是TensorNeuralNetwork实例时

        Example:
            >>> tnn1 = TensorNeuralNetwork(dim=2, rank=3)
            >>> tnn2 = TensorNeuralNetwork(dim=2, rank=2)
            >>> sum_tnn = tnn1 + tnn2  # 直接使用+操作符
            >>> print(sum_tnn.rank)  # 输出: 5 (3+2)

            # 支持链式加法
            >>> result = tnn1 + tnn2 + tnn3

        Note:
            - 新TNN的维度与输入TNN相同
            - 新TNN的rank等于两个输入TNN的rank之和
            - 直接引用原始子网络, 避免参数不匹配问题
            - 支持自动微分和梯度优化
            - 计算复杂度线性增长, 保持高效性
        """
        if not isinstance(other, TensorNeuralNetwork):
            raise TypeError(
                f"不支持TensorNeuralNetwork与{type(other)}的加法操作"
            )

        assert self.dim == other.dim, "两个TNN的维度必须相同"

        new_rank = self.rank + other.rank

        # 首先收集所有子网络和系数
        sum_subnetworks = []  # 存储所有rank分量的子网络
        sum_coefficients = []  # 存储所有rank分量的系数

        # 前rank1个分量: 来自self
        for r in range(self.rank):
            sum_coefficients.append(self.theta[r])
            rank_subnets = []
            for d in range(self.dim):
                rank_subnets.append(self.subnet(r, d))
            sum_subnetworks.append(rank_subnets)

        # 后rank2个分量: 来自other
        for s in range(other.rank):
            sum_coefficients.append(other.theta[s])
            rank_subnets = []
            for d in range(other.dim):
                rank_subnets.append(other.subnet(s, d))
            sum_subnetworks.append(rank_subnets)

        # 创建新的TNN
        sum_tnn = TensorNeuralNetwork(
            dim=self.dim,
            rank=new_rank,
            hidden_dims=(40, 40),  # 这个参数不会被使用
            activation="sin",
            domain_bounds=self.domain_bounds,
            use_coefficients=True,
            use_layer_norm=True,
        )

        # 直接替换子网络结构和系数
        with torch.no_grad():
            # 设置系数
            for idx, coeff in enumerate(sum_coefficients):
                sum_tnn.theta[idx] = coeff

            # 替换子网络结构
            sum_tnn.subnetworks = nn.ModuleList(
                [
                    nn.ModuleList(rank_subnets)
                    for rank_subnets in sum_subnetworks
                ]
            )

        return sum_tnn

    def __mul__(self, scalar) -> "TensorNeuralNetwork":
        """
        重载乘法操作符, 实现TNN与标量的乘法: tnn * c

        这个方法允许使用 tnn * c 的语法来创建TNN与常数的乘积.
        利用张量分解的线性性质, TNN乘以常数仍然保持张量积的结构.

        数学原理:
        对于TNN: u(x) = Σ_{r=1}^{rank} θᵣ Πᵈ subtnn_d^{(r)}(x_d)
        其与常数c的乘积为: c·u(x) = Σ_{r=1}^{rank} (c·θᵣ) Πᵈ subtnn_d^{(r)}(x_d)

        实现策略:
        1. 创建一个新的TNN骨架结构
        2. 直接引用原始TNN的子网络, 避免参数复制
        3. 将所有系数θᵣ乘以标量c

        Args:
            scalar: 标量常数, 支持int, float, torch.Tensor(标量)

        Returns:
            TensorNeuralNetwork: 新的TNN实例, 表示c * tnn

        Raises:
            TypeError: 当scalar不是支持的类型时

        Example:
            >>> tnn = TensorNeuralNetwork(dim=2, rank=3)
            >>> scaled_tnn = tnn * 2.5  # 使用*操作符
            >>> scaled_tnn2 = tnn * torch.tensor(3.0)

        Note:
            - 新TNN的维度和秩与原TNN相同
            - 直接引用原始子网络, 避免参数不匹配问题
            - 支持自动微分和梯度优化
            - 计算复杂度不变, 保持高效性
        """
        # 类型检查和转换
        if isinstance(scalar, int | float):
            scalar = torch.tensor(float(scalar))
        elif isinstance(scalar, torch.Tensor):
            if scalar.numel() != 1:
                raise TypeError("标量乘法只支持单元素张量")
            scalar = scalar.item()
            scalar = torch.tensor(float(scalar))
        else:
            raise TypeError(
                f"不支持TensorNeuralNetwork与{type(scalar)}的乘法操作"
            )

        # 创建新的TNN
        scaled_tnn = TensorNeuralNetwork(
            dim=self.dim,
            rank=self.rank,
            hidden_dims=(40, 40),  # 这个参数不会被使用
            activation="sin",
            domain_bounds=self.domain_bounds,
            use_coefficients=True,
            use_layer_norm=True,
        )

        # 收集缩放后的系数和原始子网络
        scaled_coefficients = []
        scaled_subnetworks = []

        for r in range(self.rank):
            # 缩放系数
            scaled_coefficients.append(self.theta[r] * scalar)

            # 直接引用原子网络
            rank_subnets = []
            for d in range(self.dim):
                rank_subnets.append(self.subnet(r, d))
            scaled_subnetworks.append(rank_subnets)

        # 直接替换子网络结构和系数
        with torch.no_grad():
            # 设置缩放后的系数
            for idx, coeff in enumerate(scaled_coefficients):
                scaled_tnn.theta[idx] = coeff

            # 替换子网络结构
            scaled_tnn.subnetworks = nn.ModuleList(
                [
                    nn.ModuleList(rank_subnets)
                    for rank_subnets in scaled_subnetworks
                ]
            )

        return scaled_tnn

    def __rmul__(self, scalar) -> "TensorNeuralNetwork":
        """
        重载右乘法操作符, 实现标量与TNN的乘法: c * tnn

        这个方法允许使用 c * tnn 的语法来创建常数与TNN的乘积.
        由于乘法的交换律, 直接调用 __mul__ 方法即可.

        Args:
            scalar: 标量常数, 支持int, float, torch.Tensor(标量)

        Returns:
            TensorNeuralNetwork: 新的TNN实例, 表示c * tnn

        Example:
            >>> tnn = TensorNeuralNetwork(dim=2, rank=3)
            >>> scaled_tnn = 2.5 * tnn  # 使用右乘法
            >>> scaled_tnn2 = torch.tensor(3.0) * tnn

        Note:
            - 功能与 __mul__ 完全相同
            - 支持链式操作: 2 * (3 * tnn) = 6 * tnn
        """
        return self.__mul__(scalar)

    def __neg__(self) -> "TensorNeuralNetwork":
        """
        重载一元负号操作符, 实现TNN的取负: -tnn

        这个方法允许使用 -tnn 的语法来创建TNN的负值.
        实际上通过将TNN乘以-1来实现: (-1) * tnn

        Returns:
            TensorNeuralNetwork: 新的TNN实例, 表示-tnn

        Example:
            >>> tnn = TensorNeuralNetwork(dim=2, rank=3)
            >>> neg_tnn = -tnn  # 使用一元负号
            >>> # 等价于: neg_tnn = -1.0 * tnn

        Note:
            - 功能与 (-1.0) * tnn 完全相同
            - 支持链式操作: -(2 * tnn) = -2 * tnn
        """
        return (-1.0) * self

    def __sub__(self, other) -> "TensorNeuralNetwork":
        """
        重载减法操作符, 实现TNN的减法: tnn1 - tnn2

        这个方法允许使用 tnn1 - tnn2 的语法来创建两个TNN的差.
        实际上通过将第二个TNN取负然后相加来实现: tnn1 + (-tnn2)

        Args:
            other: 另一个TensorNeuralNetwork实例

        Returns:
            TensorNeuralNetwork: 新的TNN实例, 表示tnn1 - tnn2

        Example:
            >>> tnn1 = TensorNeuralNetwork(dim=2, rank=3)
            >>> tnn2 = TensorNeuralNetwork(dim=2, rank=2)
            >>> diff_tnn = tnn1 - tnn2

        Note:
            - 支持TNN与TNN的减法
            - 通过 tnn1 + (-tnn2) 实现, 使用了__neg__方法
        """
        if isinstance(other, TensorNeuralNetwork):
            # TNN - TNN: 使用加法和一元负号
            return self + (-other)
        else:
            raise TypeError(
                f"不支持TensorNeuralNetwork与{type(other)}的减法操作"
            )

    def multiply_1d_function(
        self, func, target_dim: int
    ) -> "TensorNeuralNetwork":
        """
        将一维函数乘以TNN, 结果仍为TNN

        这个方法实现一维函数f(x_i)与TNN的乘法运算. 由于TNN具有张量积结构,
        一维函数可以直接乘进对应维度的所有子网络中, 保持张量积的形式.

        数学原理:
        对于TNN: u(x) = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} subtnn_d^{(r)}(x_d)
        一维函数f(x_i)与TNN的乘积为:
        f(x_i) * u(x) = Σ_{r=1}^{rank} θᵣ * (f(x_i) * subtnn_i^{(r)}(x_i)) * Π_{d≠i} subtnn_d^{(r)}(x_d)

        实现策略:
        1. 创建一个新的TNN, 具有相同的维度和秩结构
        2. 复制原TNN的所有系数θᵣ
        3. 对于目标维度: 创建函数乘积子网络
        4. 对于其他维度: 直接引用原子网络

        Args:
            func: 一维函数, 接受torch.Tensor输入并返回torch.Tensor
                  函数签名应为 func(x: torch.Tensor) -> torch.Tensor
                  其中x的shape为(batch_size, 1), 必须是一维到一维的映射
            target_dim: 目标维度索引, 范围为[0, self.dim-1]
                       表示函数f作用在第target_dim个维度上

        Returns:
            TensorNeuralNetwork: 新的TNN实例, 表示f(x_{target_dim}) * tnn(x)

        Raises:
            AssertionError: 当target_dim超出有效范围时
            TypeError: 当func不是可调用对象时
            ValueError: 当func不是一维函数时 (输入或输出维度不为1)

        Example:
            >>> import torch
            >>> import math
            >>> tnn = TensorNeuralNetwork(dim=2, rank=3)
            >>>
            >>> # 定义一维函数: f(x) = sin(2πx)
            >>> def sin_func(x):
            ...     return torch.sin(2 * math.pi * x)
            >>>
            >>> # f(x₁) * tnn(x₁, x₂)
            >>> result_tnn = tnn.multiply_1d_function(sin_func, target_dim=0)
            >>>
            >>> # f(x₂) * tnn(x₁, x₂)
            >>> result_tnn2 = tnn.multiply_1d_function(sin_func, target_dim=1)

        Note:
            - 返回的TNN与原TNN具有相同的输入输出维度和秩
            - 只有target_dim维度的子网络被修改, 其他维度的子网络直接引用
            - 支持复合操作: tnn.multiply_1d_function(f, 0).multiply_1d_function(g, 1)
            - 一维函数应当支持批量计算和自动微分
            - 新TNN保持原有的边界条件设置
            - 严格要求输入函数必须是一维函数 (R → R)
        """
        assert 0 <= target_dim < self.dim, (
            f"target_dim {target_dim} 超出有效范围 [0, {self.dim - 1}]"
        )

        if not callable(func):
            raise TypeError("func 必须是可调用对象")

        # 验证函数是否为一维函数
        self._validate_1d_function(func)

        # 创建新的TNN
        result_tnn = TensorNeuralNetwork(
            dim=self.dim,
            rank=self.rank,
            hidden_dims=(10,),  # 使用最小配置, 避免不必要的内存分配
            activation="sin",
            domain_bounds=self.domain_bounds,
            use_coefficients=True,
            use_layer_norm=True,
        )

        # 构造新TNN的子网络和系数
        result_subnetworks = []  # 存储所有rank分量的子网络
        result_coefficients = []  # 存储所有rank分量的系数

        for r in range(self.rank):
            # 复制系数
            result_coefficients.append(self.theta[r])

            # 为这个rank分量创建所有维度的子网络
            rank_subnets = []
            for d in range(self.dim):
                if d == target_dim:
                    # 对于目标维度, 创建函数乘积子网络
                    function_product_subnet = self.subnet(
                        r, d
                    ).multiply_by_function(func)
                    rank_subnets.append(function_product_subnet)
                else:
                    # 对于其他维度, 直接引用原子网络
                    rank_subnets.append(self.subnet(r, d))

            result_subnetworks.append(rank_subnets)

        # 直接替换子网络结构和系数
        with torch.no_grad():
            # 设置系数
            for idx, coeff in enumerate(result_coefficients):
                result_tnn.theta[idx] = coeff

            # 替换子网络结构
            result_tnn.subnetworks = nn.ModuleList(
                [
                    nn.ModuleList(rank_subnets)
                    for rank_subnets in result_subnetworks
                ]
            )

        return result_tnn

    def _validate_1d_function(self, func):
        """
        验证函数是否为一维函数 (R → R)

        通过测试函数的输入输出维度来验证其是否为一维函数.
        一维函数必须满足: 输入shape为(batch_size, 1), 输出shape为(batch_size, 1)

        Args:
            func: 待验证的函数

        Raises:
            ValueError: 当func不是一维函数时
        """
        try:
            # 创建测试输入: shape为(2, 1)的张量
            test_input = torch.tensor([[0.5], [0.7]], requires_grad=True)

            # 调用函数
            test_output = func(test_input)

            # 检查输出是否为张量
            if not isinstance(test_output, torch.Tensor):
                raise ValueError(
                    f"函数输出必须是torch.Tensor类型, 但得到了{type(test_output)}"
                )

            # 检查输出维度
            if test_output.shape != (2, 1):
                raise ValueError(
                    f"函数必须是一维函数 (R → R). "
                    f"期望输出shape为(2, 1), 但得到了{test_output.shape}. "
                    f"输入shape为{test_input.shape}"
                )

            # 检查是否支持自动微分
            if test_input.requires_grad and not test_output.requires_grad:
                raise ValueError(
                    "函数必须支持自动微分. "
                    "请确保函数内部使用torch操作而非numpy等不支持梯度的操作"
                )

        except Exception as e:
            if isinstance(e, ValueError):
                # 重新抛出我们自定义的ValueError
                raise e
            else:
                # 其他异常转换为ValueError
                raise ValueError(
                    f"函数验证失败: {str(e)}. "
                    f"请确保函数是一维函数 (R → R) 且支持torch张量计算"
                ) from e


# %%
class TNNIntegrator(nn.Module):
    """
    TNN的张量积分器 - 集成高斯积分功能

    这个类专门用于计算张量神经网络(TNN)的数值积分. 通过利用TNN的张量积分解结构,
    将高维积分问题转化为多个一维积分的乘积, 从而避免维数灾难并提供高精度的数值积分.

    数学原理:
    对于TNN函数: u(x) = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} subtnn_d^{(r)}(x_d)

    其在域Ω = [a₁,b₁] × [a₂,b₂] × ... × [aₙ,bₙ]上的积分可以分解为:
    ∫_Ω u(x) dx = ∫_{a₁}^{b₁} ∫_{a₂}^{b₂} ... ∫_{aₙ}^{bₙ} u(x₁,x₂,...,xₙ) dx₁dx₂...dxₙ
                 = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} (∫_{aₐ}^{bₐ} subtnn_d^{(r)}(x_d) dx_d)

    核心优势:
    1. 维数分离: 将d维积分分解为d个独立的1维积分
    2. 高精度: 使用高斯-勒让德积分方法, 对多项式函数具有最优精度
    3. 高效计算: 避免了传统高维数值积分的指数复杂度
    4. 自动微分兼容: 支持PyTorch的梯度计算

    积分方法:
    - 使用高斯-勒让德积分公式进行1维数值积分
    - 支持任意区间[a,b]的积分变换
    - 可配置积分点数以平衡精度和计算效率

    主要功能:
    1. tnn_int1: 计算单个TNN的积分
    2. tnn_int2: 计算两个TNN乘积的积分
    3. 支持复杂的函数积分计算

    Args:
        n_quad_points: 高斯-勒让德积分的节点数, 默认为16
            - 更多节点提供更高精度, 但计算成本更高
            - 对于光滑函数, 16个节点通常足够
            - 对于振荡函数或不光滑函数, 可能需要更多节点

    Attributes:
        n_quad_points: 积分节点数
        points: 标准区间[-1,1]上的高斯-勒让德积分点
        weights: 对应的积分权重

    Note:
        - 继承自nn.Module以支持GPU计算和自动微分
        - 积分精度主要取决于子网络的光滑性和积分点数
        - 对于特征值问题和PDE求解, 通常提供足够的精度
    """

    def __init__(self, n_quad_points: int = 16):
        super().__init__()
        self.n_quad_points = n_quad_points
        # 获取高斯-勒让德积分点和权重
        self.points, self.weights = self._get_gauss_legendre_points(
            n_quad_points
        )

    def _get_gauss_legendre_points(self, n):
        """获取标准区间[-1,1]上的高斯-勒让德积分点和权重"""
        points, weights = np.polynomial.legendre.leggauss(n)
        return torch.tensor(points, dtype=torch.float32), torch.tensor(
            weights, dtype=torch.float32
        )

    def _transform_to_interval(self, a: float, b: float):
        """将积分点从[-1,1]变换到[a,b]"""
        # x = (b-a)/2 * t + (b+a)/2, dx = (b-a)/2 * dt
        transformed_points = (b - a) / 2 * self.points + (b + a) / 2
        transformed_weights = (b - a) / 2 * self.weights
        return transformed_points, transformed_weights

    def tnn_int1(
        self,
        tnn: TensorNeuralNetwork,
        domain_bounds: list[tuple[float, float]],
    ):
        """
        单个TNN的积分计算

        利用TNN结构进行高效积分计算:
        对于TNN: u(x) = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} subtnn_d^{(r)}(x_d)

        积分: ∫ u(x) dx = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} (∫ subtnn_d^{(r)}(x_d) dx_d)
        """
        # 为每个rank和每个维度计算1维积分
        integral_matrix_1d = torch.zeros(tnn.rank, tnn.dim)

        for r in range(tnn.rank):
            for d in range(tnn.dim):
                a_d, b_d = domain_bounds[d]
                pts, wts = self._transform_to_interval(a_d, b_d)

                # 将积分点扩展为batch
                x_d = pts.unsqueeze(1)  # shape: (n_points, 1)

                # 获取第r个rank的第d个子网络
                subnet = tnn.subnet(r, d)

                # 计算1维积分
                integral_matrix_1d[r, d] = torch.sum(
                    wts * subnet(x_d).squeeze()
                )

        # 计算总积分: 对每个rank, 计算所有维度积分的乘积, 然后求和
        total_integral = torch.tensor(0.0, requires_grad=True)
        for r in range(tnn.rank):
            dimensional_integral_product = torch.prod(integral_matrix_1d[r, :])
            total_integral = (
                total_integral + tnn.theta[r] * dimensional_integral_product
            )

        return total_integral

    def tnn_int2(
        self,
        tnn1: TensorNeuralNetwork,
        tnn2: TensorNeuralNetwork,
        domain_bounds: list[tuple[float, float]],
    ):
        """
        两个TNN乘积的积分计算

        1. 使用 _product_tnn 将两个TNN的乘积转换为一个新的TNN
        2. 使用 tnn_int1 计算新TNN的积分
        """
        # 创建乘积TNN
        product_tnn = self._product_tnn(tnn1, tnn2)

        # 使用tnn_int1计算乘积TNN的积分
        return self.tnn_int1(product_tnn, domain_bounds)

    def _product_tnn(
        self, tnn1: TensorNeuralNetwork, tnn2: TensorNeuralNetwork
    ) -> TensorNeuralNetwork:
        """
        创建两个TNN乘积的TNN表示

        u1(x) × u2(x) = (Σᵣ α_r Πᵈ f_d^{(r)}(x_d)) × (Σₛ β_s Πᵈ g_d^{(s)}(x_d))
                      = Σᵣₛ α_r β_s Πᵈ (f_d^{(r)}(x_d) × g_d^{(s)}(x_d))

        新的TNN有 rank1 × rank2 个rank分量
        """
        assert tnn1.dim == tnn2.dim, "两个TNN的维度必须相同"

        new_rank = tnn1.rank * tnn2.rank

        # 首先收集所有乘积子网络
        product_subnetworks = []  # 存储所有rank分量的子网络, 长度为 new_rank
        product_coefficients = []  # 存储所有rank分量的系数, 长度为 new_rank

        for r in range(tnn1.rank):
            for s in range(tnn2.rank):
                # 计算系数
                coeff = tnn1.theta[r] * tnn2.theta[s]
                product_coefficients.append(coeff)

                # 为这个rank分量创建所有维度的乘积子网络
                rank_subnets = []  # 长度为 tnn1.dim, 存储当前rank分量在各维度的乘积子网络
                for d in range(tnn1.dim):
                    subnet = self._product_subnet(
                        subnet1=tnn1.subnet(r, d),
                        subnet2=tnn2.subnet(s, d),
                    )
                    rank_subnets.append(subnet)

                product_subnetworks.append(rank_subnets)

        # 创建新的TNN
        tnn = TensorNeuralNetwork(
            dim=tnn1.dim,
            rank=new_rank,
            hidden_dims=(40, 40),
            activation="sin",
            domain_bounds=tnn1.domain_bounds,
            use_coefficients=True,
            use_layer_norm=True,
        )

        # 替换子网络和系数
        with torch.no_grad():
            # 设置系数
            for idx, coeff in enumerate(product_coefficients):
                tnn.theta[idx] = coeff

            # 替换子网络结构
            tnn.subnetworks = nn.ModuleList(
                [
                    nn.ModuleList(rank_subnets)
                    for rank_subnets in product_subnetworks
                ]
            )

        return tnn

    def _product_subnet(
        self,
        subnet1: SubTensorNeuralNetwork,
        subnet2: SubTensorNeuralNetwork,
    ) -> SubTensorNeuralNetwork:
        """
        创建两个子网络乘积的子网络

        将两个子网络的输出进行逐元素乘法, 生成一个新的乘积子网络.
        这是实现TNN乘积运算的核心组件.

        数学表示: h(x) = f(x) * g(x)
        其中f和g分别是subnet1和subnet2的输出函数

        Args:
            subnet1: 第一个子网络, 提供乘法运算的第一个操作数
            subnet2: 第二个子网络, 提供乘法运算的第二个操作数

        Returns:
            SubTensorNeuralNetwork: 新的乘积子网络, 计算两个输入子网络的乘积

        注意:
            - 返回的是一个全新的子网络对象
            - 计算过程保持自动微分的兼容性
        """

        class ProductSubTensorNeuralNetwork(SubTensorNeuralNetwork):
            """乘积子网络的特殊实现"""

            def __init__(
                self,
                subnet1: SubTensorNeuralNetwork,
                subnet2: SubTensorNeuralNetwork,
            ):
                # 继承第一个子网络的基本配置
                super().__init__(
                    input_dim=1,
                    output_dim=1,
                    hidden_dims=(1,),  # 最小配置, 因为我们会重写forward
                    activation="sin",
                    use_layer_norm=False,
                    boundary_condition=subnet1.boundary_condition,
                )

                # 保存原始子网络的引用
                self.subnet1 = subnet1
                self.subnet2 = subnet2

            # 重写forward
            def forward(self, x):
                """计算两个子网络输出的乘积"""
                y1 = self.subnet1(x)
                y2 = self.subnet2(x)
                return y1 * y2

        # 返回新的乘积子网络
        return ProductSubTensorNeuralNetwork(subnet1, subnet2)


# %%
class TNNRayleighQuotientCalculator:
    """
    基于TNN的Rayleigh商计算器

    这个类专门用于计算TNN函数的Rayleigh商, 主要应用于特征值问题的变分求解.
    通过利用TNN的张量积结构, 可以高效地计算高维函数的Rayleigh商积分.

    数学背景:
    Rayleigh商定义为: R[u] = (∫|∇u|² dx + ∫V(x)|u|² dx) / (∫|u|² dx)

    其中:
    - ∇u是函数u的梯度
    - V(x)是可选的势函数
    - 积分在给定域Ω上进行

    核心功能:
    1. 计算动能项: ∫|∇u|² dx (梯度平方的积分)
    2. 计算势能项: ∫V(x)|u|² dx (势函数与函数平方的积分)
    3. 计算归一化项: ∫|u|² dx (函数平方的积分)
    4. 组合得到完整的Rayleigh商

    算法优势:
    - 利用TNN的张量积分解避免高维积分的维数灾难
    - 支持自动微分, 可用于梯度优化
    - 高精度的数值积分方法
    - 支持任意维度的问题

    Args:
        tnn: TensorNeuralNetwork实例, 待计算Rayleigh商的函数
        potential_func: 势函数V(x), 如果为None则忽略势能项
        n_quad_points: 数值积分的高斯求积点数, 影响积分精度

    Attributes:
        tnn: 存储的TNN实例
        potential_func: 势函数
        integrator: TNNIntegrator实例, 用于执行张量积分
    """

    def __init__(
        self,
        tnn: TensorNeuralNetwork,
        potential_func=None,
        n_quad_points: int = 16,
    ):
        self.tnn = tnn
        self.potential_func = potential_func
        self.integrator = TNNIntegrator(n_quad_points)

    def gradient_squared_integral(
        self,
        tnn: TensorNeuralNetwork,
        domain_bounds: list[tuple[float, float]],
    ):
        """
        计算梯度的L²范数积分: ∫|∇u|² dx

        这个方法计算TNN函数u(x)的梯度平方在给定域上的积分.

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

            # 计算 (∂u/∂xᵢ)² 的积分, 使用新的tnn_int2方法
            grad_i_squared_integral = self.integrator.tnn_int2(
                grad_i_tnn, grad_i_tnn, domain_bounds
            )

            total_integral = total_integral + grad_i_squared_integral

        return total_integral

    def rayleigh_quotient(self, domain_bounds: list[tuple[float, float]]):
        """
        使用张量分解方式计算Rayleigh商

        这个方法是特征值求解的核心, 通过计算Rayleigh商来评估当前TNN函数的"特征值".
        Rayleigh商提供了特征值的上界估计, 通过最小化这个商可以逼近真实的最小特征值.

        数学公式:
        R[u] = (∫|∇u|² dx + ∫V(x)|u|² dx) / (∫|u|² dx)

        计算步骤:
        1. 动能项: 使用gradient_squared_integral计算∫|∇u|² dx
        2. 势能项: 计算∫V(x)|u|² dx (当前版本暂未实现)
        3. 归一化项: 使用tnn_int2计算∫|u|² dx
        4. 组合得到Rayleigh商

        Args:
            domain_bounds: 积分域的边界, 格式为[(a₁,b₁), (a₂,b₂), ..., (aₙ,bₙ)]

        Returns:
            torch.Tensor: Rayleigh商的数值, 支持自动微分用于梯度优化

        Note:
            - 返回值越小, 对应的特征值估计越准确
            - 分母添加小的正则化项1e-8防止除零
            - 当前版本的势能项设为0, 适用于纯拉普拉斯特征值问题
        """

        # 计算动能项: ∫|∇u|² dx
        kinetic_term = self.gradient_squared_integral(self.tnn, domain_bounds)

        # 计算势能项: ∫V|u|² dx (暂时忽略势能, 设为0)
        potential_term = 0.0
        if self.potential_func is not None:
            # 如果有势能, 需要额外处理
            # 这里简化为0, 实际应用中需要扩展
            potential_term = 0.0

        # 计算归一化项: ∫|u|² dx
        normalization_term = self.integrator.tnn_int2(
            self.tnn, self.tnn, domain_bounds
        )

        # 计算Rayleigh商
        rayleigh_quotient = (kinetic_term + potential_term) / (
            normalization_term + 1e-8
        )

        return rayleigh_quotient


# %%
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
        None: 函数执行完毕后打印结果, 不返回值

    Note:
        - 使用正弦激活函数以更好地逼近特征函数的振荡性质
        - 采用层归一化和系数缩放提高训练稳定性
        - 优化过程分为Adam快速下降, LBFGS精细优化和最终微调三个阶段
    """
    dim = 5  # 维度
    rank = 15  # 张量秩

    print(f"=== {dim} 维拉普拉斯特征值问题 ===")
    print(f"张量秩: {rank}")
    print(f"子网络总数: {rank * dim}")

    # 定义域边界
    domain_bounds = [(0.0, 1.0) for _ in range(dim)]

    # 创建重构的TNN
    tnn = TensorNeuralNetwork(
        dim=dim,
        rank=rank,
        hidden_dims=(40, 40),
        activation="sin",
        domain_bounds=domain_bounds,
        use_coefficients=True,
        use_layer_norm=True,
    )

    print(f"TNN参数总数: {sum(p.numel() for p in tnn.parameters())}")

    # 使用TNN Rayleigh商计算器
    solver = TNNRayleighQuotientCalculator(
        tnn, potential_func=None, n_quad_points=8
    )

    theoretical_eigenvalue = dim * math.pi**2
    print(f"理论最小特征值: {theoretical_eigenvalue:.6f}")

    # ================== 三阶段优化 ==================

    losses = []

    # 阶段1: Adam快速下降 (5个epoch)
    print("=== 阶段1: Adam 快速下降 ===")
    optimizer1 = optim.Adam(tnn.parameters(), lr=0.001)

    for epoch in range(5):
        optimizer1.zero_grad()
        loss = solver.rayleigh_quotient(domain_bounds)
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
    print("\n=== 阶段2: Adam 精细调优 ===")
    optimizer2 = optim.Adam(tnn.parameters(), lr=0.0001)  # 更小的学习率

    for epoch in range(10):
        optimizer2.zero_grad()
        loss = solver.rayleigh_quotient(domain_bounds)
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
    print("\n=== 阶段3: LBFGS 精确求解 ===")
    optimizer3 = optim.LBFGS(tnn.parameters(), lr=1.0)

    for epoch in range(5):

        def closure():
            optimizer3.zero_grad()
            loss = solver.rayleigh_quotient(domain_bounds)
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


# %%
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
    """

    # ruff: noqa: N802, N806
    print("=== 二元函数混合导数特征值问题 ===")

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
        hidden_dims=(20, 20),
        activation="sin",
        domain_bounds=domain_bounds,
        use_coefficients=True,
        use_layer_norm=True,
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
    print("=== 阶段1: Adam 快速下降 ===")
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
    print("\n=== 阶段2: Adam 精细调优 ===")
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
    print("\n=== 阶段3: LBFGS 精确求解 ===")
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
