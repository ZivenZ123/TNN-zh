"""
张量神经网络 (TNN) - 重构版本

核心数学表达式:
tnn(x₁, x₂, ..., x_dim) = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} subtnn_d^{(r)}(x_d)

关键: 有 rank*dim 个子函数输出值 subtnn_d^{(r)}(x_d)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class ThetaModule(nn.Module):
    def __init__(
        self,
        rank: int,
        learnable: bool = True,
        initial_values: torch.Tensor | None = None,
    ):
        super().__init__()
        self.rank = rank
        self.learnable = learnable

        if initial_values is None:
            initial_values = torch.ones(rank)
        else:
            if initial_values.dim() != 1:
                raise ValueError(
                    f"initial_values必须是一阶张量, 但得到了{initial_values.dim()}阶张量"
                )
            if len(initial_values) != rank:
                raise ValueError(
                    f"initial_values的长度{len(initial_values)}与rank{rank}不匹配"
                )

        if self.learnable:
            self.theta = nn.Parameter(initial_values)
        else:
            self.register_buffer("theta", initial_values)

    def forward(self) -> torch.Tensor:
        return self.theta


class TNN(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int,
        func: nn.Module,
        theta_module: ThetaModule | None = None,
    ):
        """
        TNN - 张量神经网络

        数学表达式:
        tnn(x) = Σ_{r=1}^{rank} θᵣ Π_{d=1}^{dim} subtnn_d^{(r)}(x_d)

        参数:
            dim: 输入维度
            rank: 张量分解的秩
            func: nn.Module, 输入 (n_1d, 1, dim) → 输出 (n_1d, rank, dim)
                  计算所有 rank*dim 个子函数的输出值
            theta_module: 系数模块
        """
        super().__init__()

        self.dim = dim
        self.rank = rank
        self.func = func

        if theta_module is None:
            self.theta_module = ThetaModule(self.rank)
        else:
            self.theta_module = theta_module

    @property
    def theta(self):
        return self.theta_module()

    def forward(self, x):
        """
        前向传播计算

        参数:
            x: 输入张量
               - shape为(dim,) - 单个点 → 返回标量
               - shape为(batch_size, dim) - 批量点 → 返回(batch_size,)

        返回:
            标量张量或形状为(batch_size,)的张量
        """
        # 判断是单点还是批量
        if x.dim() == 1:
            # 单点: (dim,) → (1, dim)
            x_batch = x.unsqueeze(0)
            squeeze_output = True
        elif x.dim() == 2:
            # 批量: (batch_size, dim)
            x_batch = x
            squeeze_output = False
        else:
            raise ValueError(f"输入x的维度必须是1或2,但得到了{x.dim()}")

        # (batch_size, dim) → (batch_size, rank, dim)
        subfunc_outputs = self.func(x_batch)

        # 在dim维度做乘积: (batch_size, rank, dim) → (batch_size, rank)
        dim_products = torch.prod(subfunc_outputs, dim=-1)

        # 与theta加权求和: (batch_size, rank) → (batch_size,)
        result = torch.einsum("r,br->b", self.theta, dim_products)

        # 如果输入是单点,返回标量
        if squeeze_output:
            return result.squeeze()
        else:
            return result

    def grad(self, grad_dim: int) -> "TNN":
        """
        计算TNN在指定维度上的偏导数

        参数:
            grad_dim: 计算梯度的维度索引, 范围为[0, self.dim-1]

        返回:
            TNN: 新的TNN实例, 表示∂u/∂x_{grad_dim}
        """
        if not (0 <= grad_dim < self.dim):
            raise ValueError(
                f"grad_dim {grad_dim} 超出有效范围 [0, {self.dim - 1}]"
            )

        class GradFunc(nn.Module):
            def __init__(self, original_func: nn.Module, grad_dim: int):
                """
                计算 ∂(subtnn_d^r)/∂x_{grad_dim}

                original_func: (n_1d, dim) → (n_1d, rank, dim)
                """
                super().__init__()
                self.original_func = original_func
                self.grad_dim = grad_dim

            def forward(self, x):
                """
                参数:
                    x: shape为(n_1d, dim)

                返回:
                    result: shape为(n_1d, rank, dim)
                    其中只有第grad_dim维是导数值,其他维度保持原函数值
                """
                if not hasattr(self.original_func, "forward_with_grad"):
                    raise NotImplementedError(
                        f"{type(self.original_func).__name__}不支持forward_with_grad方法, "
                        "请确保所有TNN组件都实现了解析求导"
                    )

                _, grad_output = self.original_func.forward_with_grad(
                    x, self.grad_dim
                )
                return grad_output

        grad_func = GradFunc(self.func, grad_dim)

        return TNN(
            dim=self.dim,
            rank=self.rank,
            func=grad_func,
            theta_module=self.theta_module,
        )

    def grad2(self, dim1: int, dim2: int) -> "TNN":
        """
        计算TNN的二阶偏导数 ∂²u/∂x_{dim1}∂x_{dim2}

        参数:
            dim1: 第一个维度索引, 范围为[0, self.dim-1]
            dim2: 第二个维度索引, 范围为[0, self.dim-1]

        返回:
            TNN: 新的TNN实例, 表示∂²u/∂x_{dim1}∂x_{dim2}
        """
        if not (0 <= dim1 < self.dim):
            raise ValueError(f"dim1 {dim1} 超出有效范围 [0, {self.dim - 1}]")
        if not (0 <= dim2 < self.dim):
            raise ValueError(f"dim2 {dim2} 超出有效范围 [0, {self.dim - 1}]")

        class Grad2Func(nn.Module):
            def __init__(self, original_func: nn.Module, dim1: int, dim2: int):
                """
                计算 ∂²(subtnn_d^r)/∂x_{dim1}∂x_{dim2}

                original_func: (n_1d, dim) → (n_1d, rank, dim)
                """
                super().__init__()
                self.original_func = original_func
                self.dim1 = dim1
                self.dim2 = dim2

            def forward(self, x):
                """
                参数:
                    x: shape为(n_1d, dim)

                返回:
                    result: shape为(n_1d, rank, dim)
                    相应维度是二阶导数值,其他维度保持原函数值
                """
                # 检查original_func是否有forward_with_grad2方法
                if hasattr(self.original_func, "forward_with_grad2"):
                    _, grad2_output = self.original_func.forward_with_grad2(
                        x, self.dim1, self.dim2
                    )
                    return grad2_output
                else:
                    raise NotImplementedError(
                        f"{type(self.original_func).__name__}不支持forward_with_grad2方法"
                    )

        grad2_func = Grad2Func(self.func, dim1, dim2)

        return TNN(
            dim=self.dim,
            rank=self.rank,
            func=grad2_func,
            theta_module=self.theta_module,
        )

    def cat(self, other: "TNN") -> "TNN":
        """
        将当前TNN与另一个TNN拼接

        参数:
            other: 另一个TNN实例

        返回:
            TNN: 新的拼接TNN实例, rank为self.rank + other.rank
        """
        if self.dim != other.dim:
            raise ValueError(
                f"两个TNN的维度必须相同, 但得到 {self.dim} 和 {other.dim}"
            )

        class ConcatFunc(nn.Module):
            def __init__(self, func1: nn.Module, func2: nn.Module):
                super().__init__()
                self.func1 = func1
                self.func2 = func2

            def forward(self, x):
                """
                x: (n_1d, 1, dim)
                返回: (n_1d, rank1+rank2, dim)
                """
                output1 = self.func1(x)  # (n_1d, rank1, dim)
                output2 = self.func2(x)  # (n_1d, rank2, dim)
                return torch.cat([output1, output2], dim=1)  # 在rank维度拼接

            def forward_with_grad(self, x, grad_dim):
                """
                一阶导数: ∂(u+v)/∂x_i = ∂u/∂x_i + ∂v/∂x_i
                """
                if hasattr(self.func1, "forward_with_grad") and hasattr(
                    self.func2, "forward_with_grad"
                ):
                    out1, grad1 = self.func1.forward_with_grad(x, grad_dim)
                    out2, grad2 = self.func2.forward_with_grad(x, grad_dim)
                    return torch.cat([out1, out2], dim=1), torch.cat(
                        [grad1, grad2], dim=1
                    )
                else:
                    raise NotImplementedError("子函数不支持forward_with_grad")

            def forward_with_grad2(self, x, grad_dim1, grad_dim2):
                """
                二阶导数: ∂²(u+v)/∂x_i∂x_j = ∂²u/∂x_i∂x_j + ∂²v/∂x_i∂x_j
                """
                if hasattr(self.func1, "forward_with_grad2") and hasattr(
                    self.func2, "forward_with_grad2"
                ):
                    out1, grad2_1 = self.func1.forward_with_grad2(
                        x, grad_dim1, grad_dim2
                    )
                    out2, grad2_2 = self.func2.forward_with_grad2(
                        x, grad_dim1, grad_dim2
                    )
                    return torch.cat([out1, out2], dim=1), torch.cat(
                        [grad2_1, grad2_2], dim=1
                    )
                else:
                    raise NotImplementedError("子函数不支持forward_with_grad2")

        concat_func = ConcatFunc(self.func, other.func)

        class CatThetaModule(ThetaModule):
            def __init__(
                self, theta_module1: ThetaModule, theta_module2: ThetaModule
            ):
                super().__init__(
                    theta_module1.rank + theta_module2.rank, learnable=False
                )
                del self.theta
                self.theta_module1 = theta_module1
                self.theta_module2 = theta_module2
                self.register_buffer(
                    "theta",
                    torch.cat([theta_module1(), theta_module2()]),
                )

            def forward(self):
                return self.theta

        new_theta_module = CatThetaModule(
            self.theta_module, other.theta_module
        )

        return TNN(
            dim=self.dim,
            rank=self.rank + other.rank,
            func=concat_func,
            theta_module=new_theta_module,
        )

    def _multiply_by_scalar(self, scalar) -> "TNN":
        """TNN与标量的乘法"""
        if not isinstance(scalar, int | float | torch.Tensor):
            raise TypeError(f"不支持TNN与{type(scalar)}的乘法操作")

        if isinstance(scalar, torch.Tensor) and scalar.numel() != 1:
            raise TypeError("标量乘法只支持单元素张量")

        class ScaledThetaModule(ThetaModule):
            def __init__(self, original_theta_module: ThetaModule, scalar):
                super().__init__(original_theta_module.rank, learnable=False)
                del self.theta
                self.original_theta_module = original_theta_module
                self.register_buffer(
                    "scaled_theta", self.original_theta_module() * scalar
                )

            def forward(self) -> torch.Tensor:
                return self.scaled_theta

        scaled_theta_module = ScaledThetaModule(self.theta_module, scalar)

        return TNN(
            dim=self.dim,
            rank=self.rank,
            func=self.func,
            theta_module=scaled_theta_module,
        )

    def _multiply_by_tnn(self, other: "TNN") -> "TNN":
        """两个TNN的乘法"""
        if not isinstance(other, TNN):
            raise TypeError(f"不支持TNN与{type(other)}的乘法操作")

        if self.dim != other.dim:
            raise ValueError(
                f"两个TNN的维度必须相同, 但得到 {self.dim} 和 {other.dim}"
            )

        class ProductFunc(nn.Module):
            def __init__(self, func1: nn.Module, func2: nn.Module):
                super().__init__()
                self.func1 = func1
                self.func2 = func2

            def forward(self, x):
                """
                x: (n_1d, 1, dim)
                返回: (n_1d, rank1*rank2, dim)
                """
                output1 = self.func1(x)  # (n_1d, rank1, dim)
                output2 = self.func2(x)  # (n_1d, rank2, dim)

                # 对每个维度计算外积
                # output1: (n_1d, rank1, dim)
                # output2: (n_1d, rank2, dim)
                # 目标: (n_1d, rank1*rank2, dim)

                n_1d = output1.shape[0]
                rank1 = output1.shape[1]
                rank2 = output2.shape[1]

                # 重塑为 (n_1d, rank1, 1, dim) 和 (n_1d, 1, rank2, dim)
                output1_expanded = output1.unsqueeze(
                    2
                )  # (n_1d, rank1, 1, dim)
                output2_expanded = output2.unsqueeze(
                    1
                )  # (n_1d, 1, rank2, dim)

                # 逐元素相乘: (n_1d, rank1, rank2, dim)
                product = output1_expanded * output2_expanded

                # 展平rank维度: (n_1d, rank1*rank2, dim)
                return product.reshape(n_1d, rank1 * rank2, -1)

        product_func = ProductFunc(self.func, other.func)

        class ProductThetaModule(ThetaModule):
            def __init__(
                self, theta_module1: ThetaModule, theta_module2: ThetaModule
            ):
                super().__init__(
                    theta_module1.rank * theta_module2.rank, learnable=False
                )
                del self.theta
                self.theta_module1 = theta_module1
                self.theta_module2 = theta_module2

                product_coeffs = torch.einsum(
                    "i,j->ij", theta_module1(), theta_module2()
                ).flatten()
                self.register_buffer("product_theta", product_coeffs)

            def forward(self):
                return self.product_theta

        product_theta_module = ProductThetaModule(
            self.theta_module, other.theta_module
        )

        return TNN(
            dim=self.dim,
            rank=self.rank * other.rank,
            func=product_func,
            theta_module=product_theta_module,
        )

    def __mul__(self, other) -> "TNN":
        """重载乘法操作符"""
        if isinstance(other, int | float | torch.Tensor):
            return self._multiply_by_scalar(other)
        elif isinstance(other, TNN):
            return self._multiply_by_tnn(other)
        else:
            raise TypeError(f"不支持TNN与{type(other)}的乘法操作")

    def __rmul__(self, other) -> "TNN":
        """重载右乘法操作符"""
        return self.__mul__(other)

    def slice(self, fixed_dims: dict[int, float]) -> "TNN":
        """
        在指定维度上固定取值,将TNN降维

        参数:
            fixed_dims: 字典,键为维度索引,值为固定的取值
                例如: {0: 1.5, 2: 3.0} 表示在第0维固定为1.5,第2维固定为3.0

        返回:
            TNN: 降维后的新TNN实例
        """
        if not fixed_dims:
            raise ValueError("fixed_dims不能为空")

        for dim_idx in fixed_dims:
            if not (0 <= dim_idx < self.dim):
                raise ValueError(
                    f"维度索引{dim_idx}超出有效范围[0, {self.dim - 1}]"
                )

        if len(fixed_dims) >= self.dim:
            raise ValueError(
                f"固定维度数{len(fixed_dims)}必须小于TNN总维度{self.dim}"
            )

        if len(set(fixed_dims.keys())) != len(fixed_dims):
            raise ValueError("fixed_dims中存在重复的维度索引")

        # 计算固定维度的函数值
        fixed_point = torch.zeros(
            1, self.dim, device=self.theta.device, dtype=self.theta.dtype
        )
        for dim_idx, fixed_value in fixed_dims.items():
            fixed_point[0, dim_idx] = fixed_value

        # func_values_at_fixed: (1, rank, dim)
        func_values_at_fixed = self.func(fixed_point).squeeze(0)  # (rank, dim)

        # 提取固定维度的值并计算乘积因子
        fixed_dims_factor = torch.ones(
            self.rank, device=self.theta.device, dtype=self.theta.dtype
        )
        for dim_idx in fixed_dims:
            fixed_dims_factor = (
                fixed_dims_factor * func_values_at_fixed[:, dim_idx]
            )

        # 创建切片后的theta模块
        class SlicedThetaModule(ThetaModule):
            def __init__(
                self, original_theta_module: ThetaModule, factor: torch.Tensor
            ):
                super().__init__(original_theta_module.rank, learnable=False)
                del self.theta
                self.original_theta_module = original_theta_module
                self.register_buffer("factor", factor)
                self.register_buffer(
                    "sliced_theta", self.original_theta_module() * self.factor
                )

            def forward(self) -> torch.Tensor:
                return self.sliced_theta

        new_theta_module = SlicedThetaModule(
            self.theta_module, fixed_dims_factor
        )

        # 创建降维后的函数
        remaining_dims = [d for d in range(self.dim) if d not in fixed_dims]
        remaining_dim_indices = torch.tensor(remaining_dims)

        class SlicedFunc(nn.Module):
            def __init__(
                self,
                original_func: nn.Module,
                fixed_dims: dict[int, float],
                remaining_dim_indices: torch.Tensor,
                remaining_dims_list: list[int],
                original_dim: int,
                device,
                dtype,
            ):
                super().__init__()
                self.original_func = original_func
                self.fixed_dims = fixed_dims
                self.register_buffer(
                    "remaining_dim_indices", remaining_dim_indices
                )
                self.remaining_dims_list = remaining_dims_list
                self.original_dim = original_dim
                self.device = device
                self.dtype = dtype

                # 预计算索引映射 (向量化优化)
                # 创建固定维度的索引和值
                fixed_dim_indices = []
                fixed_dim_values = []
                for dim_idx, value in sorted(fixed_dims.items()):
                    fixed_dim_indices.append(dim_idx)
                    fixed_dim_values.append(value)

                if len(fixed_dim_indices) > 0:
                    self.register_buffer(
                        "fixed_dim_indices",
                        torch.tensor(fixed_dim_indices, dtype=torch.long),
                    )
                    self.register_buffer(
                        "fixed_dim_values",
                        torch.tensor(fixed_dim_values, dtype=dtype),
                    )
                else:
                    self.register_buffer(
                        "fixed_dim_indices", torch.tensor([], dtype=torch.long)
                    )
                    self.register_buffer(
                        "fixed_dim_values", torch.tensor([], dtype=dtype)
                    )

                # 创建从 x 的列索引到 full_x 的列索引的映射
                # remaining_to_full[i] = full_x 中对应 x[:, i] 的维度索引
                remaining_to_full = torch.zeros(
                    len(remaining_dims_list), dtype=torch.long
                )
                for i, dim_idx in enumerate(remaining_dims_list):
                    remaining_to_full[i] = dim_idx
                self.register_buffer("remaining_to_full", remaining_to_full)

            def forward(self, x):
                """
                x: shape为(n_1d, new_dim)
                返回: shape为(n_1d, rank, new_dim)
                """
                n_1d = x.shape[0]

                # 创建完整维度的张量,初始化为0 (向量化)
                # 使用x的device和dtype以确保兼容性
                full_x = torch.zeros(
                    n_1d,
                    self.original_dim,
                    device=x.device,
                    dtype=x.dtype,
                )

                # 填充固定维度的值 (向量化)
                if len(self.fixed_dim_indices) > 0:
                    # fixed_dim_values: (num_fixed,) - 确保在正确的设备上
                    fixed_values_on_device = self.fixed_dim_values.to(
                        device=x.device, dtype=x.dtype
                    )
                    fixed_indices_on_device = self.fixed_dim_indices.to(
                        x.device
                    )

                    # 广播为 (n_1d, num_fixed)
                    fixed_values = fixed_values_on_device.unsqueeze(0).expand(
                        n_1d, -1
                    )
                    # 使用高级索引填充
                    full_x[:, fixed_indices_on_device] = fixed_values

                # 填充剩余维度的值 (向量化)
                # x: (n_1d, new_dim)
                # remaining_to_full: (new_dim,) 映射到 full_x 的列索引
                remaining_to_full_on_device = self.remaining_to_full.to(
                    x.device
                )
                full_x[:, remaining_to_full_on_device] = x

                # 调用原始函数: (n_1d, original_dim) → (n_1d, rank, original_dim)
                full_output = self.original_func(full_x)

                # 提取剩余维度: (n_1d, rank, original_dim) → (n_1d, rank, new_dim)
                remaining_dim_indices_on_device = (
                    self.remaining_dim_indices.to(x.device)
                )
                sliced_output = full_output[
                    :, :, remaining_dim_indices_on_device
                ]

                return sliced_output

        sliced_func = SlicedFunc(
            self.func,
            fixed_dims,
            remaining_dim_indices,
            remaining_dims,
            self.dim,
            self.theta.device,
            self.theta.dtype,
        )

        return TNN(
            dim=len(remaining_dims),
            rank=self.rank,
            func=sliced_func,
            theta_module=new_theta_module,
        )

    def __add__(self, other: "TNN | int | float | torch.Tensor") -> "TNN":
        """重载加法操作符"""
        if isinstance(other, (int, float, torch.Tensor)):
            if isinstance(other, torch.Tensor) and other.numel() != 1:
                raise TypeError("标量加法只支持单元素张量")

            # 将标量转换为TNN
            class ConstantFunc(nn.Module):
                def forward(self, x):
                    """
                    x: (n_1d, dim)
                    返回: (n_1d, 1, dim) 全为1
                    """
                    n_1d = x.shape[0]
                    dim = x.shape[1]
                    return torch.ones(
                        n_1d, 1, dim, device=x.device, dtype=x.dtype
                    )

            scalar_value = (
                other if isinstance(other, (int, float)) else other.item()
            )
            theta_module = ThetaModule(
                rank=1,
                learnable=False,
                initial_values=torch.tensor(
                    [scalar_value],
                    device=self.theta.device,
                    dtype=self.theta.dtype,
                ),
            )

            other = TNN(
                dim=self.dim,
                rank=1,
                func=ConstantFunc(),
                theta_module=theta_module,
            )

        if self.dim != other.dim:
            raise ValueError(
                f"两个TNN的维度必须相同, 但得到 {self.dim} 和 {other.dim}"
            )

        return self.cat(other)

    def __radd__(self, other: int | float | torch.Tensor) -> "TNN":
        """重载右加法操作符"""
        if isinstance(other, torch.Tensor) and other.numel() != 1:
            raise TypeError("标量加法只支持单元素张量")

        # 将标量转换为TNN
        class ConstantFunc(nn.Module):
            def forward(self, x):
                """
                x: (n_1d, dim)
                返回: (n_1d, 1, dim) 全为1
                """
                n_1d = x.shape[0]
                dim = x.shape[1]
                return torch.ones(n_1d, 1, dim, device=x.device, dtype=x.dtype)

        scalar_value = (
            other if isinstance(other, (int, float)) else other.item()
        )
        theta_module = ThetaModule(
            rank=1,
            learnable=False,
            initial_values=torch.tensor(
                [scalar_value],
                device=self.theta.device,
                dtype=self.theta.dtype,
            ),
        )

        scalar_tnn = TNN(
            dim=self.dim,
            rank=1,
            func=ConstantFunc(),
            theta_module=theta_module,
        )

        return scalar_tnn.__add__(self)

    def __neg__(self) -> "TNN":
        """重载一元负号操作符"""
        return (-1.0) * self

    def __sub__(self, other) -> "TNN":
        """重载减法操作符"""
        if isinstance(other, (TNN, int, float, torch.Tensor)):
            return self + (-other)
        else:
            raise TypeError(f"不支持TNN与{type(other)}的减法操作")

    def __rsub__(self, other) -> "TNN":
        """重载右减法操作符"""
        return other + (-self)

    def __truediv__(self, other: int | float | torch.Tensor) -> "TNN":
        """重载除法操作符"""
        if not isinstance(other, (int, float, torch.Tensor)):
            raise TypeError(
                f"不支持TNN与{type(other)}的除法操作, 只支持TNN除以标量"
            )

        if isinstance(other, torch.Tensor) and other.numel() != 1:
            raise TypeError("标量除法只支持单元素张量")

        scalar_value = (
            other if isinstance(other, (int, float)) else other.item()
        )

        if scalar_value == 0:
            raise ValueError("除数不能为零")

        return self * (1.0 / scalar_value)


class SeparableDimNetwork(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int,
        hidden_layers: tuple[int, ...] = (50, 50, 50),
    ):
        """
        维度可分离神经网络, 用于构造TNN

        参数:
            dim: 输入维度
            rank: 输出秩(对应TNN中的张量秩)
            hidden_layers: 隐藏层大小tuple, 默认为(50, 50, 50)

        输入: (n_1d, dim)
        输出: (n_1d, rank, dim)

        网络结构: [1 -> hidden_layers[0] -> ... -> hidden_layers[-1] -> rank]

        关键特性:
        - 使用批量矩阵乘法, 避免Python循环
        - 保证维度间独立性: y[b,r,j] 只依赖于 x[b,j]
        - 对于每个 (b,r), 雅可比矩阵 ∂y[b,r,:]/∂x[b,:] 是 dim×dim 对角矩阵
        - 即: ∂y[b,r,j]/∂x[b,i] = 0 当 i≠j
        """
        super().__init__()

        self.dim = dim
        self.rank = rank
        self.hidden_layers = hidden_layers

        # 构建网络结构: [1] + hidden_layers + [rank]
        self.layer_sizes = [1] + list(hidden_layers) + [rank]
        self.num_layers = len(self.layer_sizes) - 1

        # 动态创建权重和偏置参数
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(self.num_layers):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            # W: (dim, out_size, in_size)
            self.weights.append(
                nn.Parameter(torch.empty(dim, out_size, in_size))
            )
            # b: (dim, out_size, 1)
            self.biases.append(nn.Parameter(torch.empty(dim, out_size, 1)))

        # 初始化参数
        self._initialize_parameters()

    def _initialize_parameters(self):
        """使用Xavier初始化权重"""
        for W in self.weights:
            nn.init.xavier_uniform_(W, gain=nn.init.calculate_gain("tanh"))
        for b in self.biases:
            nn.init.zeros_(b)

    def forward(self, x):
        """
        参数:
            x: shape为(n_1d, dim)

        返回:
            shape为(n_1d, rank, dim)

        对角雅可比矩阵的实现原理:
        - permute将维度索引dim放到batch维度, 利用torch.bmm的批量独立处理特性
        - W[j]只作用于x[j], 保证y[:,:,j]只依赖于输入的第j个维度
        - 整个前向传播过程中维度间无交互, 自然形成对角雅可比结构
        """
        # (n_1d, dim) → (dim, n_1d) → (dim, 1, n_1d)
        x = x.t().unsqueeze(1)

        # 逐层前向传播
        for i in range(self.num_layers):
            x = torch.bmm(self.weights[i], x) + self.biases[i]
            x = torch.tanh(x)

        # (dim, rank, n_1d) → (n_1d, rank, dim)
        return x.permute(2, 1, 0)

    def forward_with_grad(self, x, grad_dim):
        """
        同时计算前向传播和对指定维度的梯度

        参数:
            x: shape为(n_1d, dim)
            grad_dim: 计算梯度的维度索引

        返回:
            output: shape为(n_1d, rank, dim) - 原函数值
            grad_output: shape为(n_1d, rank, dim) - 导数值,只有第grad_dim列是真导数
        """
        # (n_1d, dim) → (dim, n_1d) → (dim, 1, n_1d)
        x = x.t().unsqueeze(1)

        # 初始化第grad_dim维度的梯度
        # 第一层输出大小: layer_sizes[1]
        grad_x = self.weights[0][grad_dim]  # (out_size_0, 1)

        # 逐层前向传播和梯度传播
        for i in range(self.num_layers):
            # 前向传播
            x = torch.bmm(self.weights[i], x) + self.biases[i]
            x = torch.tanh(x)

            # 梯度传播 (利用 tanh'(x) = 1 - tanh²(x))
            tanh_x_grad = x[grad_dim]  # 从激活后的值提取
            grad_x = (1.0 - tanh_x_grad * tanh_x_grad) * grad_x

            # 传播到下一层 (如果不是最后一层)
            if i < self.num_layers - 1:
                grad_x = self.weights[i + 1][grad_dim] @ grad_x

        # (dim, rank, n_1d) → (n_1d, rank, dim)
        output = x.permute(2, 1, 0)

        # 构造梯度输出: 只有grad_dim列是真导数,其他列是原函数值
        grad_output = output.clone()
        # grad_x: (rank, n_1d) → (n_1d, rank)
        grad_output[:, :, grad_dim] = grad_x.t()

        return output, grad_output

    def forward_with_grad2(self, x, grad_dim1, grad_dim2):
        """
        同时计算前向传播和对两个维度的二阶混合偏导数

        参数:
            x: shape为(n_1d, dim)
            grad_dim1: 第一个梯度维度索引
            grad_dim2: 第二个梯度维度索引

        返回:
            output: shape为(n_1d, rank, dim) - 原函数值
            grad2_output: shape为(n_1d, rank, dim) - 二阶导数值
        """
        # (n_1d, dim) → (dim, n_1d) → (dim, 1, n_1d)
        x = x.t().unsqueeze(1)

        if grad_dim1 == grad_dim2:
            # 同一维度的二阶导数: ∂²/∂x_i²
            # 初始化一阶和二阶梯度
            grad_x = self.weights[0][grad_dim1]  # (out_size_0, 1)
            grad2_x = torch.zeros_like(grad_x)  # (out_size_0, 1)

            # 逐层前向传播和梯度传播
            for i in range(self.num_layers):
                # 前向传播
                x = torch.bmm(self.weights[i], x) + self.biases[i]
                x = torch.tanh(x)

                # 从激活后的值提取
                tanh_x_grad = x[grad_dim1]  # (out_size, n_1d)

                # 计算导数
                tanh_grad = 1.0 - tanh_x_grad * tanh_x_grad  # tanh'
                tanh_grad2 = -2.0 * tanh_x_grad * tanh_grad  # tanh''

                # 链式法则: grad2 = f''(x) * (grad_x)² + f'(x) * grad2_x
                grad2_x = tanh_grad2 * (grad_x * grad_x) + tanh_grad * grad2_x
                grad_x = tanh_grad * grad_x

                # 传播到下一层
                if i < self.num_layers - 1:
                    grad2_x = self.weights[i + 1][grad_dim1] @ grad2_x
                    grad_x = self.weights[i + 1][grad_dim1] @ grad_x

            # (dim, rank, n_1d) → (n_1d, rank, dim)
            output = x.permute(2, 1, 0)

            # 只clone一次
            grad2_output = output.clone()
            grad2_output[:, :, grad_dim1] = grad2_x.t()

            return output, grad2_output

        else:
            # 混合二阶导数: ∂²/∂x_i∂x_j (i≠j)
            # 结果是两个一阶导数的乘积
            # 需要计算两个维度的一阶导数

            # 初始化两个维度的一阶梯度
            grad_x1 = self.weights[0][grad_dim1]  # (out_size_0, 1)
            grad_x2 = self.weights[0][grad_dim2]  # (out_size_0, 1)

            # 逐层前向传播和梯度传播
            for i in range(self.num_layers):
                # 前向传播
                x = torch.bmm(self.weights[i], x) + self.biases[i]
                x = torch.tanh(x)

                # 计算两个维度的一阶导数
                tanh_x_grad1 = x[grad_dim1]
                tanh_grad1 = 1.0 - tanh_x_grad1 * tanh_x_grad1
                grad_x1 = tanh_grad1 * grad_x1

                tanh_x_grad2 = x[grad_dim2]
                tanh_grad2 = 1.0 - tanh_x_grad2 * tanh_x_grad2
                grad_x2 = tanh_grad2 * grad_x2

                # 传播到下一层
                if i < self.num_layers - 1:
                    grad_x1 = self.weights[i + 1][grad_dim1] @ grad_x1
                    grad_x2 = self.weights[i + 1][grad_dim2] @ grad_x2

            # (dim, rank, n_1d) → (n_1d, rank, dim)
            output = x.permute(2, 1, 0)

            # clone一次,然后设置两列为对应的一阶导数
            grad2_output = output.clone()
            grad2_output[:, :, grad_dim1] = grad_x1.t()
            grad2_output[:, :, grad_dim2] = grad_x2.t()

            return output, grad2_output


class SeparableDimNetworkGELU(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int,
        hidden_layers: tuple[int, ...] = (50, 50, 50),
    ):
        """
        维度可分离神经网络 (GELU激活函数), 用于构造TNN

        参数:
            dim: 输入维度
            rank: 输出秩(对应TNN中的张量秩)
            hidden_layers: 隐藏层大小tuple, 默认为(50, 50, 50)

        输入: (n_1d, dim)
        输出: (n_1d, rank, dim)

        网络结构: [1 -> hidden_layers[0] -> ... -> hidden_layers[-1] -> rank]
        激活函数: GELU

        关键特性:
        - 使用批量矩阵乘法, 避免Python循环
        - 保证维度间独立性: y[b,r,j] 只依赖于 x[b,j]
        - 对于每个 (b,r), 雅可比矩阵 ∂y[b,r,:]/∂x[b,:] 是 dim×dim 对角矩阵
        """
        super().__init__()

        self.dim = dim
        self.rank = rank
        self.hidden_layers = hidden_layers

        # 构建网络结构: [1] + hidden_layers + [rank]
        self.layer_sizes = [1] + list(hidden_layers) + [rank]
        self.num_layers = len(self.layer_sizes) - 1

        # 动态创建权重和偏置参数
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(self.num_layers):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            # W: (dim, out_size, in_size)
            self.weights.append(
                nn.Parameter(torch.empty(dim, out_size, in_size))
            )
            # b: (dim, out_size, 1)
            self.biases.append(nn.Parameter(torch.empty(dim, out_size, 1)))

        # 初始化参数
        self._initialize_parameters()

    def _initialize_parameters(self):
        """使用Xavier初始化权重"""
        for W in self.weights:
            nn.init.xavier_uniform_(W, gain=3.0)
        for b in self.biases:
            nn.init.zeros_(b)

    def _gelu_grad(self, x):
        """
        GELU的一阶导数: GELU'(x) = Φ(x) + x*φ(x)

        其中:
        - Φ(x) = (1/2)[1 + erf(x/√2)]  # 标准正态CDF
        - φ(x) = (1/√(2π)) * exp(-x²/2)  # 标准正态PDF
        """
        sqrt_2 = math.sqrt(2.0)
        sqrt_2pi = math.sqrt(2.0 * math.pi)

        cdf = 0.5 * (1.0 + torch.erf(x / sqrt_2))  # Φ(x)
        pdf = (1.0 / sqrt_2pi) * torch.exp(-0.5 * x * x)  # φ(x)

        return cdf + x * pdf

    def _gelu_grad2(self, x):
        """
        GELU的二阶导数: GELU''(x) = φ(x)*(2-x²)

        其中:
        - φ(x) = (1/√(2π)) * exp(-x²/2)  # 标准正态PDF
        """
        sqrt_2pi = math.sqrt(2.0 * math.pi)
        pdf = (1.0 / sqrt_2pi) * torch.exp(-0.5 * x * x)  # φ(x)

        return pdf * (2.0 - x * x)

    def forward(self, x):
        """
        参数:
            x: shape为(n_1d, dim)

        返回:
            shape为(n_1d, rank, dim)
        """
        # (n_1d, dim) → (dim, n_1d) → (dim, 1, n_1d)
        x = x.t().unsqueeze(1)

        # 逐层前向传播
        for i in range(self.num_layers):
            x = torch.bmm(self.weights[i], x) + self.biases[i]
            x = F.gelu(x)

        # (dim, rank, n_1d) → (n_1d, rank, dim)
        return x.permute(2, 1, 0)

    def forward_with_grad(self, x, grad_dim):
        """
        同时计算前向传播和对指定维度的梯度

        参数:
            x: shape为(n_1d, dim)
            grad_dim: 计算梯度的维度索引

        返回:
            output: shape为(n_1d, rank, dim) - 原函数值
            grad_output: shape为(n_1d, rank, dim) - 导数值,只有第grad_dim列是真导数
        """
        # (n_1d, dim) → (dim, n_1d) → (dim, 1, n_1d)
        x = x.t().unsqueeze(1)

        # 初始化第grad_dim维度的梯度
        grad_x = self.weights[0][grad_dim]  # (out_size_0, 1)

        # 逐层前向传播和梯度传播
        for i in range(self.num_layers):
            # 前向传播(保存激活前的值)
            x_pre_act = torch.bmm(self.weights[i], x) + self.biases[i]
            x = F.gelu(x_pre_act)

            # 梯度传播(使用激活前的值计算GELU导数)
            gelu_grad_val = self._gelu_grad(x_pre_act[grad_dim])
            grad_x = gelu_grad_val * grad_x

            # 传播到下一层(如果不是最后一层)
            if i < self.num_layers - 1:
                grad_x = self.weights[i + 1][grad_dim] @ grad_x

        # (dim, rank, n_1d) → (n_1d, rank, dim)
        output = x.permute(2, 1, 0)

        # 构造梯度输出: 只有grad_dim列是真导数,其他列是原函数值
        grad_output = output.clone()
        # grad_x: (rank, n_1d) → (n_1d, rank)
        grad_output[:, :, grad_dim] = grad_x.t()

        return output, grad_output

    def forward_with_grad2(self, x, grad_dim1, grad_dim2):
        """
        同时计算前向传播和对两个维度的二阶混合偏导数

        参数:
            x: shape为(n_1d, dim)
            grad_dim1: 第一个梯度维度索引
            grad_dim2: 第二个梯度维度索引

        返回:
            output: shape为(n_1d, rank, dim) - 原函数值
            grad2_output: shape为(n_1d, rank, dim) - 二阶导数值
        """
        # (n_1d, dim) → (dim, n_1d) → (dim, 1, n_1d)
        x = x.t().unsqueeze(1)

        if grad_dim1 == grad_dim2:
            # 同一维度的二阶导数: ∂²/∂x_i²
            grad_x = self.weights[0][grad_dim1]  # (out_size_0, 1)
            grad2_x = torch.zeros_like(grad_x)  # (out_size_0, 1)

            # 逐层前向传播和梯度传播
            for i in range(self.num_layers):
                # 前向传播(保存激活前的值)
                x_pre_act = torch.bmm(self.weights[i], x) + self.biases[i]
                x = F.gelu(x_pre_act)

                # 从激活前的值计算导数
                x_pre_grad_dim = x_pre_act[grad_dim1]

                # 计算GELU的一阶和二阶导数
                gelu_grad = self._gelu_grad(x_pre_grad_dim)  # GELU'
                gelu_grad2 = self._gelu_grad2(x_pre_grad_dim)  # GELU''

                # 链式法则: grad2 = f''(x) * (grad_x)² + f'(x) * grad2_x
                grad2_x = gelu_grad2 * (grad_x * grad_x) + gelu_grad * grad2_x
                grad_x = gelu_grad * grad_x

                # 传播到下一层
                if i < self.num_layers - 1:
                    grad2_x = self.weights[i + 1][grad_dim1] @ grad2_x
                    grad_x = self.weights[i + 1][grad_dim1] @ grad_x

            # (dim, rank, n_1d) → (n_1d, rank, dim)
            output = x.permute(2, 1, 0)

            # 只clone一次
            grad2_output = output.clone()
            grad2_output[:, :, grad_dim1] = grad2_x.t()

            return output, grad2_output

        else:
            # 混合二阶导数: ∂²/∂x_i∂x_j (i≠j)
            # 需要计算两个维度的一阶导数

            # 初始化两个维度的一阶梯度
            grad_x1 = self.weights[0][grad_dim1]  # (out_size_0, 1)
            grad_x2 = self.weights[0][grad_dim2]  # (out_size_0, 1)

            # 逐层前向传播和梯度传播
            for i in range(self.num_layers):
                # 前向传播(保存激活前的值)
                x_pre_act = torch.bmm(self.weights[i], x) + self.biases[i]
                x = F.gelu(x_pre_act)

                # 计算两个维度的一阶导数
                gelu_grad1 = self._gelu_grad(x_pre_act[grad_dim1])
                grad_x1 = gelu_grad1 * grad_x1

                gelu_grad2 = self._gelu_grad(x_pre_act[grad_dim2])
                grad_x2 = gelu_grad2 * grad_x2

                # 传播到下一层
                if i < self.num_layers - 1:
                    grad_x1 = self.weights[i + 1][grad_dim1] @ grad_x1
                    grad_x2 = self.weights[i + 1][grad_dim2] @ grad_x2

            # (dim, rank, n_1d) → (n_1d, rank, dim)
            output = x.permute(2, 1, 0)

            # clone一次,然后设置两列为对应的一阶导数
            grad2_output = output.clone()
            grad2_output[:, :, grad_dim1] = grad_x1.t()
            grad2_output[:, :, grad_dim2] = grad_x2.t()

            return output, grad2_output


def wrap_1d_func_as_tnn(dim: int, target_dim: int):
    """
    将支持批量计算的一元函数封装为TNN的装饰器

    可以作为函数调用或装饰器使用:
    - 装饰器用法: @wrap_1d_func_as_tnn(dim=3, target_dim=1)
    - 函数用法: wrap_1d_func_as_tnn(dim=3, target_dim=1)(func)

    参数:
        dim: 封装后TNN的总维度数
        target_dim: 应用该函数的目标维度索引

    返回:
        装饰器函数,接受一元函数并返回TNN实例
    """
    if not (0 <= target_dim < dim):
        raise ValueError(
            f"target_dim {target_dim} 超出有效范围 [0, {dim - 1}]"
        )

    def decorator(func):
        """实际的装饰器函数"""
        if not callable(func):
            raise TypeError(f"func必须是可调用对象, 但得到了{type(func)}")

        class WrappedFunc(nn.Module):
            def __init__(self, original_func, target_dim: int, dim: int):
                super().__init__()
                self.original_func = original_func
                self.target_dim = target_dim
                self.dim = dim

            def forward(self, x):
                """
                x: shape为(n_1d, dim)
                返回: shape为(n_1d, 1, dim)
                """
                # 创建输出张量,所有维度初始化为1
                output = torch.ones(
                    x.shape[0], 1, self.dim, device=x.device, dtype=x.dtype
                )

                # 在目标维度应用函数
                x_target = x[
                    :, self.target_dim : self.target_dim + 1
                ]  # (n_1d, 1)
                func_output = self.original_func(x_target)  # (n_1d, 1)
                output[:, 0, self.target_dim] = func_output.squeeze(-1)

                return output

        wrapped_func = WrappedFunc(func, target_dim, dim)

        tnn = TNN(
            dim=dim,
            rank=1,
            func=wrapped_func,
            theta_module=ThetaModule(rank=1, learnable=False),
        )

        return tnn

    return decorator


def apply_dirichlet_bd(boundary: list[tuple[float | None, float | None]]):
    """
    应用齐次Dirichlet边界条件到函数的装饰器

    可以作为函数调用或装饰器使用:
    - 装饰器用法: @apply_dirichlet_bd([(0, 1), (0, 1)])
    - 函数用法: apply_dirichlet_bd([(0, 1), (0, 1)])(func)

    参数:
        boundary: 边界信息的列表, 格式为[(a₁, b₁), (a₂, b₂), ..., (aₙ, bₙ)]

    返回:
        装饰器函数,接受nn.Module并返回应用边界条件后的nn.Module
    """
    # 验证边界区间合法性
    for i, (a, b) in enumerate(boundary):
        if a is not None and b is not None and a >= b:
            raise ValueError(f"第{i}维的边界区间不合法: a={a} >= b={b}")

    def decorator(func: nn.Module) -> nn.Module:
        """实际的装饰器函数"""

        class BoundaryFunc(nn.Module):
            def __init__(self, original_func: nn.Module, boundary: list):
                super().__init__()
                self.original_func = original_func
                self.boundary = boundary
                self.dim = len(boundary)

                # 预处理边界信息为列表
                self.a_vals = []
                self.b_vals = []
                self.has_left = []
                self.has_right = []
                self.has_both = []
                self.max_bd_vals = []

                for a, b in boundary:
                    self.a_vals.append(a if a is not None else 0.0)
                    self.b_vals.append(b if b is not None else 0.0)
                    self.has_left.append(a is not None)
                    self.has_right.append(b is not None)
                    self.has_both.append(a is not None and b is not None)
                    if a is not None and b is not None:
                        self.max_bd_vals.append(((b - a) / 2) ** 2)
                    else:
                        self.max_bd_vals.append(1.0)

            def _compute_bd_value_and_derivs(self, x):
                """
                计算边界条件函数 b_d(x_d) 及其各维度的一阶、二阶导数

                对于TNN: u(x) = Σ_r θ_r Π_d f_d^r(x_d)
                应用边界条件后: u(x) = Σ_r θ_r Π_d [f_d^r(x_d) * b_d(x_d)]

                每个维度独立应用边界因子,不是所有维度的乘积

                返回:
                    bd_value: b_d(x_d), shape (n_1d, dim)
                    bd_grad: ∂b_d/∂x_d, shape (n_1d, dim)
                    bd_grad2: ∂²b_d/∂x_d², shape (n_1d, dim)
                """
                # 转换边界信息为张量
                a = torch.tensor(self.a_vals, device=x.device, dtype=x.dtype)
                b = torch.tensor(self.b_vals, device=x.device, dtype=x.dtype)
                has_left = torch.tensor(self.has_left, device=x.device)
                has_right = torch.tensor(self.has_right, device=x.device)
                has_both = torch.tensor(self.has_both, device=x.device)
                max_bd = torch.tensor(
                    self.max_bd_vals, device=x.device, dtype=x.dtype
                )

                # 计算每个维度的 b_i(x_i), ∂b_i/∂x_i, ∂²b_i/∂x_i²
                left = x - a  # (n_1d, dim)
                right = b - x  # (n_1d, dim)

                # b_i(x_i)
                b_i = torch.ones_like(x)
                b_i = torch.where(has_left & ~has_right, left, b_i)
                b_i = torch.where(has_right & ~has_left, right, b_i)
                b_i = torch.where(has_both, 2.0 * left * right / max_bd, b_i)

                # ∂b_i/∂x_i
                db_i = torch.zeros_like(x)
                db_i = torch.where(
                    has_left & ~has_right, torch.ones_like(x), db_i
                )
                db_i = torch.where(
                    has_right & ~has_left, -torch.ones_like(x), db_i
                )
                db_i = torch.where(
                    has_both, 2.0 * (a + b - 2.0 * x) / max_bd, db_i
                )

                # ∂²b_i/∂x_i²
                ddb_i = torch.zeros_like(x)
                ddb_i = torch.where(has_both, -4.0 / max_bd, ddb_i)

                # 直接返回每个维度独立的边界因子及其导数
                # 不需要计算所有维度的乘积!
                return b_i, db_i, ddb_i

            def forward(self, x):
                """
                x: shape为(n_1d, dim)
                返回: shape为(n_1d, rank, dim)
                """
                # 调用原始函数
                func_output = self.original_func(x)  # (n_1d, rank, dim)

                # 转换边界信息为张量
                a_tensor = torch.tensor(
                    self.a_vals, device=x.device, dtype=x.dtype
                )  # (dim,)
                b_tensor = torch.tensor(
                    self.b_vals, device=x.device, dtype=x.dtype
                )  # (dim,)
                has_left = torch.tensor(
                    self.has_left, device=x.device
                )  # (dim,)
                has_right = torch.tensor(
                    self.has_right, device=x.device
                )  # (dim,)
                has_both = torch.tensor(
                    self.has_both, device=x.device
                )  # (dim,)
                max_bd = torch.tensor(
                    self.max_bd_vals, device=x.device, dtype=x.dtype
                )  # (dim,)

                # 计算所有维度的边界因子 - 向量化
                factor = torch.ones_like(x)  # (n_1d, dim)

                # 左边界: (x - a)
                left_factor = x - a_tensor  # 广播: (n_1d, dim)
                factor = torch.where(has_left, factor * left_factor, factor)

                # 右边界: (b - x)
                right_factor = b_tensor - x  # 广播: (n_1d, dim)
                factor = torch.where(has_right, factor * right_factor, factor)

                # 归一化 (双边界情况)
                factor = torch.where(has_both, 2.0 * factor / max_bd, factor)

                # 应用到输出: (n_1d, dim) → (n_1d, 1, dim)
                factor = factor.unsqueeze(1)  # (n_1d, 1, dim)
                result = func_output * factor  # (n_1d, rank, dim)

                return result

            def forward_with_grad(self, x, grad_dim):
                """
                乘积法则: u = f * B
                ∂u/∂x_i = ∂f/∂x_i * B + f * ∂B/∂x_i
                """
                if not hasattr(self.original_func, "forward_with_grad"):
                    raise NotImplementedError(
                        f"{type(self.original_func).__name__}不支持forward_with_grad"
                    )

                # 调用原函数
                f, df = self.original_func.forward_with_grad(x, grad_dim)

                # 计算边界条件
                bd, dbd_diag, _ = self._compute_bd_value_and_derivs(x)

                # 扩展维度: (n_1d, dim) -> (n_1d, 1, dim)
                bd = bd.unsqueeze(1)
                dbd = (
                    dbd_diag[:, grad_dim].unsqueeze(1).unsqueeze(2)
                )  # (n_1d, 1, 1)

                # 乘积法则
                output = f * bd
                grad_output = df * bd
                grad_output[:, :, grad_dim] += f[:, :, grad_dim] * dbd.squeeze(
                    2
                )

                return output, grad_output

            def forward_with_grad2(self, x, grad_dim1, grad_dim2):
                """
                同维度: ∂²u/∂x_i² = ∂²f/∂x_i² * B + 2 * ∂f/∂x_i * ∂B/∂x_i + f * ∂²B/∂x_i²
                混合: 输出格式包含两个一阶导数 ∂f/∂x_i * B 和 ∂f/∂x_j * B
                """
                if not hasattr(self.original_func, "forward_with_grad2"):
                    raise NotImplementedError(
                        f"{type(self.original_func).__name__}不支持forward_with_grad2"
                    )

                # 调用原函数
                f, d2f = self.original_func.forward_with_grad2(
                    x, grad_dim1, grad_dim2
                )

                # 计算边界条件
                bd, dbd_diag, ddbd = self._compute_bd_value_and_derivs(x)
                bd = bd.unsqueeze(1)  # (n_1d, 1, dim)

                # 输出: u = f * B
                output = f * bd

                if grad_dim1 == grad_dim2:
                    # 同维度二阶导数,需要一阶导数
                    _, df = self.original_func.forward_with_grad(x, grad_dim1)

                    dbd = (
                        dbd_diag[:, grad_dim1].unsqueeze(1).unsqueeze(2)
                    )  # (n_1d, 1, 1)
                    ddbd_val = (
                        ddbd[:, grad_dim1].unsqueeze(1).unsqueeze(2)
                    )  # (n_1d, 1, 1)

                    grad2_output = f * bd
                    grad2_output[:, :, grad_dim1] = (
                        d2f[:, :, grad_dim1] * bd[:, :, grad_dim1]
                        + 2.0 * df[:, :, grad_dim1] * dbd.squeeze(2)
                        + f[:, :, grad_dim1] * ddbd_val.squeeze(2)
                    )
                else:
                    # 混合导数: 直接乘以B
                    grad2_output = d2f * bd

                return output, grad2_output

        return BoundaryFunc(func, boundary)

    return decorator
