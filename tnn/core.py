"""
张量神经网络 (TNN)

核心数学表达式:
tnn(x₁, x₂, ..., x_dim) = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} subtnn_d^{(r)}(x_d)
"""

import numpy as np
import torch
import torch.nn as nn

# 全局设备设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DefaultSubNet(nn.Module):
    """
    默认的子网络实现 - 用于TensorNeuralNetwork的标准子网络

    这是TensorNeuralNetwork中使用的默认子网络实现, 是一个标准的全连接神经网络.
    每个子网络负责处理单个维度的输入并输出对应的标量值.

    网络结构:
    - 输入维度: 1 (处理单个维度的标量输入)
    - 输出维度: rank (对应TNN中的张量秩)
    - 激活函数: sin激活函数, 适合逼近振荡函数和特征函数
    - 归一化: 层归一化, 提高训练稳定性
    - 最终激活: Tanh激活, 将输出限制在[-1,1]范围内

    这个默认实现为TensorNeuralNetwork提供了开箱即用的子网络结构,
    用户也可以根据具体问题需求自定义其他类型的子网络.
    """

    class SinActivation(nn.Module):
        """Sin激活函数"""

        def forward(self, x):
            return torch.sin(x)

    def __init__(self, rank: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 50),
            nn.LayerNorm(50),
            self.SinActivation(),
            nn.Linear(50, 50),
            nn.LayerNorm(50),
            self.SinActivation(),
            nn.Linear(50, rank),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


class SubTensorNeuralNetwork(nn.Module):
    """
    SubTensorNeuralNetwork - TNN的子网络 (一维到一维)

    每个子网络处理一个维度的一个输入, 输出一个标量.
    这是TNN架构的基础组件, 负责处理单个维度的输入并输出标量值.

    数学表示: subtnn_d^{(r)}: R → R
    subtnn_d^{(r)}(x_d) ∈ R

    特点:
    - 接受外部定义的网络结构作为参数
    - 内置边界条件处理, 支持齐次Dirichlet边界条件
    - 支持梯度计算和函数乘积运算

    Args:
        network: 神经网络模块 (nn.Module), 输入输出都应该是一维的
        boundary_condition: 边界条件设置, 格式为(a, b)表示在区间[a,b]上应用齐次Dirichlet边界条件
    """

    def __init__(
        self,
        network: nn.Module,
        boundary_condition: tuple[float, float] | None = None,
    ):
        super().__init__()

        # 存储传入的网络, 参数也会被自动包含
        self.network = network

        # 边界条件设置
        self.boundary_condition = boundary_condition

    def forward(self, x: torch.Tensor):  # todo: 把所有运算都明确为张量的运算
        """
        前向传播计算

        对输入的一维数据进行前向传播, 通过传入的神经网络输出标量值.
        如果设置了边界条件, 会强制应用齐次Dirichlet边界条件.

        Args:
            x: 输入张量, shape为(batch_size, 1)

        Returns:
            输出张量, shape为(batch_size, 1), 已应用边界条件
        """
        # 通过传入的网络进行前向传播
        output = self.network(x)

        # >>> 强制应用边界条件 <<<
        if self.boundary_condition is not None:
            a, b = self.boundary_condition

            # 检查输入x是否在边界条件范围内
            if torch.any(x < a) or torch.any(x > b):
                raise ValueError(
                    f"输入x必须在边界条件范围[{a}, {b}]内, 但接收到的x范围为[{x.min().item():.6f}, {x.max().item():.6f}]"
                )

            # 齐次Dirichlet边界条件: boundary_factor * output
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
            >>> subnet = SubTensorNeuralNetwork(network)
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

        class DerivativeNetwork(nn.Module):
            """导数网络的特殊实现"""

            def __init__(self, original_subnet: "SubTensorNeuralNetwork"):
                super().__init__()
                # 保存原始子网络的引用, 原始子网络的参数会被自动包含
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

        # 创建导数网络
        derivative_network = DerivativeNetwork(self)

        # 返回新的梯度子网络
        return SubTensorNeuralNetwork(
            network=derivative_network,
            boundary_condition=None,  # 不用强制应用边界条件, 否则不是真正的导数
        )

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

        class FunctionProductNetwork(nn.Module):
            """函数乘积网络的特殊实现"""

            def __init__(
                self, original_subnet: "SubTensorNeuralNetwork", func
            ):
                super().__init__()
                # 保存原始子网络和函数的引用
                self.original_subnet = original_subnet
                self.func = func

            def forward(self, x):
                """计算f(x) * subnet(x)"""
                subnet_output = self.original_subnet(x)
                func_output = self.func(x)
                return func_output * subnet_output

        # 创建函数乘积网络
        function_product_network = FunctionProductNetwork(self, func)

        # 返回新的函数乘积子网络
        return SubTensorNeuralNetwork(
            network=function_product_network,
            boundary_condition=None,  # 不用强制应用边界条件, 否则不是真正的乘积
        )

    def multiply_by_subnet(
        self, other: "SubTensorNeuralNetwork"
    ) -> "SubTensorNeuralNetwork":
        """
        创建两个子网络乘积的新子网络

        将两个子网络的输出进行逐元素乘法, 生成一个新的乘积子网络.
        这是实现TNN乘积运算的核心组件.

        数学表示: h(x) = f(x) * g(x)
        其中f和g分别是当前子网络和other子网络的输出函数

        Args:
            other: 另一个SubTensorNeuralNetwork实例

        Returns:
            SubTensorNeuralNetwork: 新的乘积子网络, 计算两个输入子网络的乘积

        Note:
            - 返回的是一个全新的子网络对象
            - 计算过程保持自动微分的兼容性
            - 保持原子网络的边界条件设置
        """

        class SubnetProductNetwork(nn.Module):
            """子网络乘积网络的特殊实现"""

            def __init__(
                self,
                subnet1: "SubTensorNeuralNetwork",
                subnet2: "SubTensorNeuralNetwork",
            ):
                super().__init__()
                # 保存原始子网络的引用
                self.subnet1 = subnet1
                self.subnet2 = subnet2

            def forward(self, x):
                """计算两个子网络输出的乘积"""
                y1 = self.subnet1(x)
                y2 = self.subnet2(x)
                return y1 * y2

        # 创建子网络乘积网络
        subnet_product_network = SubnetProductNetwork(self, other)

        # 返回新的子网络乘积子网络
        return SubTensorNeuralNetwork(
            network=subnet_product_network,
            boundary_condition=None,  # 不用强制应用边界条件, 否则不是真正的乘积
        )


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
        rank: 张量分解的秩
        subnetworks: 二维列表, 包含rank×dim个SubTensorNeuralNetwork实例
            数据结构: list[list[SubTensorNeuralNetwork]]
            格式: subnetworks[r][d] 对应第r个秩分量的第d维子网络
            形状: [rank, dim] - 外层列表长度为rank, 内层列表长度为dim
            示例:
                * 2×3 TNN: [[subnet₀₀, subnet₀₁, subnet₀₂], [subnet₁₀, subnet₁₁, subnet₁₂]]
                * 每个subnet都是SubTensorNeuralNetwork实例, 处理R→R映射
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
    """

    class ThetaModule(nn.Module):
        """
        TNN系数θ模块

        专门用于管理TNN中的可学习系数θᵣ的模块. 支持可学习和固定两种模式,
        并提供标量乘法等操作.

        Args:
            rank: 张量分解的秩, 即系数的数量
            initial_values: 初始系数值, 默认为None (使用全1初始化)
        """

        def __init__(
            self,
            rank: int,
            initial_values: torch.Tensor | None = None,
        ):
            super().__init__()
            self.rank = rank

            if initial_values is None:
                initial_values = torch.ones(rank).to(DEVICE)
            else:
                # 验证initial_values必须是一阶张量
                if not isinstance(initial_values, torch.Tensor):
                    raise TypeError(
                        f"initial_values必须是torch.Tensor类型, 但得到了{type(initial_values)}"
                    )
                if initial_values.dim() != 1:
                    raise ValueError(
                        f"initial_values必须是一阶张量, 但得到了{initial_values.dim()}阶张量"
                    )
                if initial_values.size(0) != rank:
                    raise ValueError(
                        f"initial_values的长度{initial_values.size(0)}与rank{rank}不匹配"
                    )
                initial_values = initial_values.to(DEVICE)

            # 可学习参数: θᵣ会在训练过程中更新
            self.theta = nn.Parameter(initial_values)

        def forward(self):
            """返回系数张量"""
            return self.theta

    def __init__(
        self,
        dim: int,
        rank: int,
        subnetworks: list[list[SubTensorNeuralNetwork]] | None = None,
        domain_bounds: list[tuple[float, float]] | None = None,
    ):
        super().__init__()

        self.dim = dim
        self.rank = rank
        self.domain_bounds = domain_bounds

        # 如果没有提供子网络, 则创建默认的子网络
        if subnetworks is None:
            # 为每个维度创建一个共享的DefaultSubNet实例
            # 这样同一维度的不同rank会共享参数, 大幅减少参数冗余
            shared_default_subnets = nn.ModuleList(
                [DefaultSubNet(self.rank) for _ in range(self.dim)]
            )

            # 创建一个引用共享DefaultSubNet的子网络类
            class SharedSingleComponentSubNet(nn.Module):
                """
                共享单分量子网络 - 从共享的DefaultSubNet中提取单个分量

                这个类引用共享的DefaultSubNet实例, 从中提取第r个分量,
                实现参数共享以减少内存占用和提高训练效率.
                """

                def __init__(
                    self, shared_subnet: nn.Module, component_index: int
                ):
                    super().__init__()
                    self.shared_subnet = shared_subnet
                    self.component_index = component_index

                def forward(self, x):
                    # 获取共享DefaultSubNet的完整输出, 然后提取第component_index个分量
                    full_output = self.shared_subnet(
                        x
                    )  # shape: (batch_size, rank)
                    return full_output[
                        :, self.component_index : self.component_index + 1
                    ]  # shape: (batch_size, 1)

            subnetworks = [
                [
                    SubTensorNeuralNetwork(
                        network=SharedSingleComponentSubNet(
                            shared_subnet=shared_default_subnets[d],
                            component_index=r,
                        ),
                        boundary_condition=domain_bounds[d]  # 不同维度的边界
                        if domain_bounds
                        else None,
                    )
                    for d in range(self.dim)
                ]
                for r in range(self.rank)
            ]

            # 将共享的子网络保存为模块属性, 确保参数被正确管理
            self.shared_default_subnets = shared_default_subnets

        # 验证子网络结构的一致性
        assert len(subnetworks) == self.rank, (
            f"传入的子网络rank数量{len(subnetworks)}与指定的rank{self.rank}不匹配"
        )
        for r in range(self.rank):
            assert len(subnetworks[r]) == self.dim, (
                f"第{r}个rank的子网络数量{len(subnetworks[r])}与指定的dim{self.dim}不匹配"
            )

        # 直接使用传入的子网络, 同时传入子网络的所有参数
        # subnetworks[r][d] 对应 subtnn_d^{(r)}
        self.subnetworks = nn.ModuleList(
            [nn.ModuleList(subnetworks[r]) for r in range(self.rank)]
        )

        # 使用ThetaModule管理系数θᵣ
        # 设计为module的好处是可以区分要学习的theta参数和TNN中表达式的theta值
        # 因为在对TNN做运算的时候, 需要学习的是原始的theta, 但表达式的theta会发生改变
        self.theta_module = self.ThetaModule(self.rank)

    @property
    def theta(self):
        """获取系数张量"""
        # 也可以直接用 self.theta_module() 来表示TNN中theta的值
        # 但是这样重写一下会更符合数学直观表达, 可以用 self.theta 来表示TNN中theta的值
        return self.theta_module()

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

        # 计算TNN输出
        result = torch.zeros(batch_size, 1).to(x.device)

        # 对每个rank分量r
        for r in range(self.rank):
            # 计算 Π_{d=1}^{dim} subtnn_d^{(r)}(x_d)
            product = torch.ones(batch_size, 1).to(x.device)

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
        4. 对于其他维度: 直接引用原子网络

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

        # 构建梯度TNN的子网络结构
        grad_subnetworks: list[list[SubTensorNeuralNetwork]] = [
            [
                self.subnet(r, d).grad()
                if d == grad_dim
                else self.subnet(r, d)
                for d in range(self.dim)
            ]
            for r in range(self.rank)
        ]

        # 使用构造好的子网络直接创建梯度TNN
        grad_tnn = TensorNeuralNetwork(
            dim=self.dim,
            rank=self.rank,
            subnetworks=grad_subnetworks,
            domain_bounds=self.domain_bounds,
        )

        # 直接共享系数内存
        grad_tnn.theta_module = self.theta_module

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
        1. 验证两个TNN的维度和定义域兼容性
        2. 直接引用原始TNN的子网络, 避免参数复制
        3. 正确拼接系数并处理定义域边界

        Args:
            other: 另一个TensorNeuralNetwork实例

        Returns:
            TensorNeuralNetwork: 新的TNN实例, 表示两个TNN的和

        Raises:
            AssertionError: 当两个TNN的维度不匹配时
            TypeError: 当other不是TensorNeuralNetwork实例时
            ValueError: 当两个TNN的定义域不兼容时

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
            - 需要处理定义域兼容性检查
        """
        if not isinstance(other, TensorNeuralNetwork):
            raise TypeError(
                f"不支持TensorNeuralNetwork与{type(other)}的加法操作"
            )

        assert self.dim == other.dim, "两个TNN的维度必须相同"

        # 直接构建加法TNN的所有子网络结构
        subnetworks: list[list[SubTensorNeuralNetwork]] = [
            # 前self.rank个分量: 来自self
            *[
                [self.subnet(r, d) for d in range(self.dim)]
                for r in range(self.rank)
            ],
            # 后other.rank个分量: 来自other
            *[
                [other.subnet(s, d) for d in range(other.dim)]
                for s in range(other.rank)
            ],
        ]

        # 使用构造好的子网络直接创建加法TNN
        sum_tnn = TensorNeuralNetwork(
            dim=self.dim,
            rank=self.rank + other.rank,
            subnetworks=subnetworks,
            domain_bounds=self.domain_bounds,
        )

        # 创建新的theta模块来处理系数拼接
        class SumThetaModule(nn.Module):
            """加法TNN的系数模块, 动态拼接两个原始TNN的系数"""

            def __init__(
                self,
                theta_module1: "TensorNeuralNetwork.ThetaModule",
                theta_module2: "TensorNeuralNetwork.ThetaModule",
            ):
                super().__init__()
                self.theta_module1 = theta_module1
                self.theta_module2 = theta_module2

            def forward(self):
                """动态拼接两个theta模块的系数"""
                theta1 = self.theta_module1()
                theta2 = self.theta_module2()

                # torch.cat会保持输入张量的梯度信息和计算图连接
                return torch.cat([theta1, theta2], dim=0)

        # 替换sum_tnn的theta_module
        sum_tnn.theta_module = SumThetaModule(
            self.theta_module, other.theta_module
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
        # 类型检查
        if not isinstance(scalar, int | float | torch.Tensor):
            raise TypeError(
                f"不支持TensorNeuralNetwork与{type(scalar)}的乘法操作"
            )

        # 张量类型检查
        if isinstance(scalar, torch.Tensor) and scalar.numel() != 1:
            raise TypeError("标量乘法只支持单元素张量")

        # 直接引用原始TNN的所有子网络结构
        subnetworks: list[list[SubTensorNeuralNetwork]] = [
            [self.subnet(r, d) for d in range(self.dim)]
            for r in range(self.rank)
        ]

        # 使用构造好的子网络直接创建乘法TNN
        scaled_tnn = TensorNeuralNetwork(
            dim=self.dim,
            rank=self.rank,
            subnetworks=subnetworks,
            domain_bounds=self.domain_bounds,
        )

        # 创建一个新的ScaledThetaModule类来处理标量乘法
        class ScaledThetaModule(nn.Module):
            """标量乘法的ThetaModule"""

            def __init__(
                self,
                original_theta_module: "TensorNeuralNetwork.ThetaModule",
                scalar,
            ):
                super().__init__()
                self.original_theta_module = original_theta_module
                self.scalar = scalar

            def forward(self):
                """返回原系数乘以标量"""
                return self.original_theta_module() * self.scalar

        # 使用ScaledThetaModule替换原来的theta_module
        scaled_tnn.theta_module = ScaledThetaModule(self.theta_module, scalar)

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

        # 构建结果TNN的子网络结构
        result_subnetworks: list[list[SubTensorNeuralNetwork]] = [
            [
                self.subnet(r, d).multiply_by_function(func)
                if d == target_dim
                else self.subnet(r, d)
                for d in range(self.dim)
            ]
            for r in range(self.rank)
        ]

        # 使用构造好的子网络直接创建结果TNN
        result_tnn = TensorNeuralNetwork(
            dim=self.dim,
            rank=self.rank,
            subnetworks=result_subnetworks,
            domain_bounds=self.domain_bounds,
        )

        # 直接共享系数内存
        result_tnn.theta_module = self.theta_module

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

    def __matmul__(
        self, other: "TensorNeuralNetwork"
    ) -> "TensorNeuralNetwork":
        """
        重载矩阵乘法操作符, 实现两个TNN的乘积: tnn1 @ tnn2

        这个方法允许使用 tnn1 @ tnn2 的语法来创建两个TNN的乘积.
        利用张量分解的乘积性质, 两个TNN的乘积仍然保持张量积的结构.

        数学原理:
        对于两个TNN:
        u1(x) = Σ_{r=1}^{rank1} α_r Π_{d=1}^{dim} f_d^{(r)}(x_d)
        u2(x) = Σ_{s=1}^{rank2} β_s Π_{d=1}^{dim} g_d^{(s)}(x_d)

        它们的乘积为:
        u1(x) × u2(x) = (Σᵣ α_r Πᵈ f_d^{(r)}(x_d)) × (Σₛ β_s Πᵈ g_d^{(s)}(x_d))
                      = Σᵣₛ α_r β_s Πᵈ (f_d^{(r)}(x_d) × g_d^{(s)}(x_d))

        新的TNN有 rank1 × rank2 个rank分量

        Args:
            other: 另一个TensorNeuralNetwork实例

        Returns:
            TensorNeuralNetwork: 新的TNN实例, 表示两个TNN的乘积

        Raises:
            AssertionError: 当两个TNN的维度不匹配时
            TypeError: 当other不是TensorNeuralNetwork实例时

        Example:
            >>> tnn1 = TensorNeuralNetwork(dim=2, rank=3)
            >>> tnn2 = TensorNeuralNetwork(dim=2, rank=2)
            >>> product_tnn = tnn1 @ tnn2  # 使用@操作符
            >>> print(product_tnn.rank)  # 输出: 6 (3*2)

        Note:
            - 新TNN的维度与输入TNN相同
            - 新TNN的rank等于两个输入TNN的rank的乘积
            - 每个维度的子网络是两个原始子网络的乘积
            - 支持自动微分和梯度优化
            - 计算复杂度为O(rank1 * rank2 * dim)
        """
        # 类型检查
        if not isinstance(other, TensorNeuralNetwork):
            raise TypeError(
                f"不支持TensorNeuralNetwork与{type(other)}的矩阵乘法操作"
            )

        # 维度检查
        assert self.dim == other.dim, "两个TNN的维度必须相同"

        new_rank = self.rank * other.rank

        # 构建所有rank分量的乘积子网络
        # 对于每个(r,s)组合, 创建一个rank分量, 包含所有维度的乘积子网络
        product_subnetworks: list[list[SubTensorNeuralNetwork]] = [
            [
                self.subnet(r, d).multiply_by_subnet(other.subnet(s, d))
                for d in range(self.dim)
            ]
            for r in range(self.rank)
            for s in range(other.rank)
        ]

        # 使用构造好的子网络直接创建乘积TNN
        product_tnn = TensorNeuralNetwork(
            dim=self.dim,
            rank=new_rank,
            subnetworks=product_subnetworks,
            domain_bounds=self.domain_bounds,
        )

        # 创建新的theta模块来处理系数乘积
        class ProductThetaModule(nn.Module):
            """乘积TNN的系数模块, 动态计算两个原始TNN系数的乘积"""

            def __init__(
                self,
                theta_module1: "TensorNeuralNetwork.ThetaModule",
                theta_module2: "TensorNeuralNetwork.ThetaModule",
            ):
                super().__init__()
                self.theta_module1 = theta_module1
                self.theta_module2 = theta_module2

            def forward(self):
                """动态计算乘积系数"""
                theta1 = self.theta_module1()
                theta2 = self.theta_module2()

                rank1 = theta1.shape[0]
                rank2 = theta2.shape[0]

                # 计算所有系数乘积组合
                product_coeffs = [
                    theta1[r] * theta2[s]
                    for r in range(rank1)
                    for s in range(rank2)
                ]

                # 使用torch.stack()将标量张量列表转换为一维张量
                # 必须用stack而不是torch.tensor(), 因为product_coeffs中的每个元素
                # 都是标量张量(有梯度信息), torch.stack()能保持梯度连接和自动微分功能
                return torch.stack(product_coeffs)

        # 替换product_tnn的theta_module
        product_tnn.theta_module = ProductThetaModule(
            self.theta_module, other.theta_module
        )

        return product_tnn


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
        gaussian_quadrature: 内部高斯积分器实例

    Note:
        - 继承自nn.Module以支持GPU计算和自动微分
        - 积分精度主要取决于子网络的光滑性和积分点数
        - 对于特征值问题和PDE求解, 通常提供足够的精度
    """

    class GaussianQuadrature(nn.Module):
        """
        高斯-勒让德积分器 - TNNIntegrator的内部类

        专门用于处理高斯-勒让德积分的计算, 包括积分点的生成、区间变换和权重计算.
        这个类封装了所有与高斯积分相关的底层操作.

        数学原理:
        高斯-勒让德积分公式:
        ∫_{-1}^{1} f(x) dx ≈ Σ_{i=1}^{n} w_i f(x_i)

        其中x_i是勒让德多项式的根, w_i是对应的权重.
        对于任意区间[a,b], 通过变量替换:
        ∫_{a}^{b} f(x) dx = (b-a)/2 * ∫_{-1}^{1} f((b-a)/2 * t + (b+a)/2) dt

        Args:
            n_quad_points: 高斯-勒让德积分的节点数

        Attributes:
            n_quad_points: 积分节点数
            points: 标准区间[-1,1]上的高斯-勒让德积分点
            weights: 对应的积分权重
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
            return torch.tensor(points, dtype=torch.float32).to(
                DEVICE
            ), torch.tensor(weights, dtype=torch.float32).to(DEVICE)

        def transform_to_interval(self, a: float, b: float):
            """
            将积分点从[-1,1]变换到[a,b]

            通过线性变换将标准区间[-1,1]上的积分点和权重变换到任意区间[a,b].
            变换公式:
            x = (b-a)/2 * t + (b+a)/2
            dx = (b-a)/2 * dt

            Args:
                a: 积分区间下界
                b: 积分区间上界

            Returns:
                tuple: (transformed_points, transformed_weights)
                    - transformed_points: 变换后的积分点
                    - transformed_weights: 变换后的积分权重
            """
            # x = (b-a)/2 * t + (b+a)/2, dx = (b-a)/2 * dt
            transformed_points = (b - a) / 2 * self.points + (b + a) / 2
            transformed_weights = (b - a) / 2 * self.weights
            return transformed_points, transformed_weights

        def transform_to_interval_with_subdivision(
            self, a: float, b: float, sub_intervals: int = 1
        ):
            """
            将积分点从[-1,1]变换到[a,b], 支持区间细分

            通过将积分区间[a,b]分成sub_intervals个等距的子区间,
            在每个子区间内分别应用高斯积分, 然后合并结果。
            这种方法可以提高积分精度, 特别是对于非光滑函数或振荡函数。

            数学原理:
            ∫_{a}^{b} f(x) dx = Σ_{i=1}^{n} ∫_{a_i}^{b_i} f(x) dx
            其中 a_i = a + (i-1)h, b_i = a + ih, h = (b-a)/n

            Args:
                a: 积分区间下界
                b: 积分区间上界
                sub_intervals: 子区间数量, 默认为1(不细分)

            Returns:
                tuple: (transformed_points, transformed_weights)
                    - transformed_points: 合并后的积分点, 总数为sub_intervals * n_quad_points
                    - transformed_weights: 合并后的积分权重

            Note:
                - 当sub_intervals=1时, 等价于调用transform_to_interval
                - 子区间数量越多, 对非光滑函数的积分精度越高
                - 总的积分点数为sub_intervals * n_quad_points
            """
            if sub_intervals == 1:
                # 如果不细分, 直接使用原有方法
                return self.transform_to_interval(a, b)

            # 计算子区间的长度
            sub_interval_length = (b - a) / sub_intervals

            # 存储所有子区间的积分点和权重
            all_points = []
            all_weights = []

            # 对每个子区间应用高斯积分
            for i in range(sub_intervals):
                # 计算第i个子区间的边界
                sub_a = a + i * sub_interval_length
                sub_b = a + (i + 1) * sub_interval_length

                # 获取该子区间的积分点和权重
                sub_points, sub_weights = self.transform_to_interval(
                    sub_a, sub_b
                )

                # 添加到总列表中
                all_points.append(sub_points)
                all_weights.append(sub_weights)

            # 合并所有子区间的积分点和权重
            merged_points = torch.cat(all_points, dim=0)
            merged_weights = torch.cat(all_weights, dim=0)

            return merged_points, merged_weights

    def __init__(self, n_quad_points: int = 16):
        super().__init__()
        self.n_quad_points = n_quad_points
        # 创建内部高斯积分器
        self.gaussian_quadrature = self.GaussianQuadrature(n_quad_points)

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

        优化策略:
        1. 向量化计算减少循环开销
        2. 预计算积分点和权重避免重复计算
        3. 使用矩阵运算提高计算效率
        4. 优化内存使用模式
        """
        # 预计算所有维度的积分点和权重, 避免重复计算
        transformed_quad_data = [
            self.gaussian_quadrature.transform_to_interval(a_d, b_d)
            for a_d, b_d in domain_bounds
        ]

        # 向量化计算所有rank和维度的一维积分
        # 使用列表推导式和torch.stack进行批量计算
        integral_matrix_1d = torch.stack(
            [
                torch.stack(
                    [
                        torch.sum(
                            wts * tnn.subnet(r, d)(pts.unsqueeze(1)).squeeze()
                        )
                        for d, (pts, wts) in enumerate(transformed_quad_data)
                    ]
                )
                for r in range(tnn.rank)
            ]
        ).to(DEVICE)

        # 计算每个rank的维度积分乘积: shape (rank,)
        dimensional_products = torch.prod(integral_matrix_1d, dim=1)

        # 获取系数向量并计算最终积分
        theta_coeffs = tnn.theta_module()
        total_integral = torch.sum(theta_coeffs * dimensional_products)

        return total_integral

    def tnn_int1_with_subdivision(
        self,
        tnn: TensorNeuralNetwork,
        domain_bounds: list[tuple[float, float]],
        sub_intervals: int = 10,
    ):
        """
        单个TNN的积分计算, 支持区间细分

        利用TNN结构和区间细分进行高效高精度积分计算:
        对于TNN: u(x) = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} subtnn_d^{(r)}(x_d)

        积分: ∫ u(x) dx = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} (∫ subtnn_d^{(r)}(x_d) dx_d)

        区间细分优势:
        1. 提高非光滑函数的积分精度
        2. 更好地处理振荡函数
        3. 对于复杂的子网络输出, 提供更稳定的积分结果

        Args:
            tnn: 待积分的TensorNeuralNetwork
            domain_bounds: 积分域边界, 格式为[(a₁, b₁), (a₂, b₂), ...]
            sub_intervals: 每个维度的子区间数量, 默认为1(不细分)

        Returns:
            torch.Tensor: TNN在指定域上的积分值

        Example:
            >>> integrator = TNNIntegrator(n_quad_points=16)
            >>> tnn = TensorNeuralNetwork(dim=2, rank=3)
            >>> domain = [(0.0, 1.0), (0.0, 1.0)]
            >>> # 不细分
            >>> result1 = integrator.tnn_int1_with_subdivision(tnn, domain, 1)
            >>> # 每个维度分成5个子区间
            >>> result2 = integrator.tnn_int1_with_subdivision(tnn, domain, 5)

        Note:
            - 当sub_intervals=1时, 等价于调用tnn_int1
            - 子区间数量越多, 积分精度越高, 但计算成本也越高
            - 总积分点数为sub_intervals * n_quad_points (每个维度)
        """
        # 预计算所有维度的积分点和权重, 使用区间细分
        transformed_quad_data = [
            self.gaussian_quadrature.transform_to_interval_with_subdivision(
                a_d, b_d, sub_intervals
            )
            for a_d, b_d in domain_bounds
        ]

        # 向量化计算所有rank和维度的一维积分
        # 使用列表推导式和torch.stack进行批量计算
        integral_matrix_1d = torch.stack(
            [
                torch.stack(
                    [
                        torch.sum(
                            wts * tnn.subnet(r, d)(pts.unsqueeze(1)).squeeze()
                        )
                        for d, (pts, wts) in enumerate(transformed_quad_data)
                    ]
                )
                for r in range(tnn.rank)
            ]
        ).to(DEVICE)

        # 计算每个rank的维度积分乘积: shape (rank,)
        dimensional_products = torch.prod(integral_matrix_1d, dim=1)

        # 获取系数向量并计算最终积分
        theta_coeffs = tnn.theta_module()
        total_integral = torch.sum(theta_coeffs * dimensional_products)

        return total_integral

    def tnn_int2_with_subdivision(
        self,
        tnn1: TensorNeuralNetwork,
        tnn2: TensorNeuralNetwork,
        domain_bounds: list[tuple[float, float]],
        sub_intervals: int = 10,
    ):
        """
        两个TNN乘积的积分计算, 支持区间细分

        利用TNN的@操作符直接计算两个TNN乘积的积分:
        ∫ (tnn1 @ tnn2) dx

        Args:
            tnn1: 第一个TNN
            tnn2: 第二个TNN
            domain_bounds: 积分域边界
            sub_intervals: 每个维度的子区间数量, 默认为1(不细分)

        Returns:
            torch.Tensor: 两个TNN乘积的积分值

        Example:
            >>> integrator = TNNIntegrator(n_quad_points=16)
            >>> tnn1 = TensorNeuralNetwork(dim=2, rank=3)
            >>> tnn2 = TensorNeuralNetwork(dim=2, rank=2)
            >>> domain = [(0.0, 1.0), (0.0, 1.0)]
            >>> # 使用区间细分计算乘积积分
            >>> result = integrator.tnn_int2_with_subdivision(tnn1, tnn2, domain, 3)
        """
        # 使用TNN的@操作符创建乘积TNN
        product_tnn = tnn1 @ tnn2

        # 使用tnn_int1_with_subdivision计算乘积TNN的积分
        return self.tnn_int1_with_subdivision(
            product_tnn, domain_bounds, sub_intervals
        )

    def tnn_int2(
        self,
        tnn1: TensorNeuralNetwork,
        tnn2: TensorNeuralNetwork,
        domain_bounds: list[tuple[float, float]],
    ):
        """
        两个TNN乘积的积分计算

        利用TNN的@操作符直接计算两个TNN乘积的积分:
        ∫ (tnn1 @ tnn2) dx

        Args:
            tnn1: 第一个TNN
            tnn2: 第二个TNN
            domain_bounds: 积分域边界

        Returns:
            两个TNN乘积的积分值
        """
        # 使用TNN的@操作符创建乘积TNN
        product_tnn = tnn1 @ tnn2

        # 使用tnn_int1计算乘积TNN的积分
        return self.tnn_int1(product_tnn, domain_bounds)
