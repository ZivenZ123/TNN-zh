import torch
import torch.nn as nn


# ********** 带导数的激活函数 **********
# 重新定义激活函数和对应的局部梯度
# sin(x)
class TnnSin(nn.Module):
    """TnnSin"""

    def forward(self, x):
        return torch.sin(x)

    def grad(self, x):
        return torch.cos(x)

    def grad_grad(self, x):
        return -torch.sin(x)


# ********** 网络层 **********
# TNN的线性层
class TnnLinear(nn.Module):
    """
    对输入数据应用批量线性变换:
        输入数据: x:[dim, n1, N]
        可学习参数: W:[dim,n2,n1], b:[dim,n2,1]
        输出数据: y=Wx+b:[dim,n2,N]

    参数:
        dim: TNN的维度
        out_features: n2
        in_features: n1
        bias: 是否需要偏置(布尔值)
    """

    def __init__(self, dim, out_features, in_features, bias):
        super().__init__()
        self.dim = dim
        self.out_features = out_features
        self.in_features = in_features

        self.weight = nn.Parameter(
            torch.empty((self.dim, self.out_features, self.in_features))
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((self.dim, self.out_features, 1))
            )
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is None:
            if self.in_features == 1:
                return self.weight * x
            else:
                return self.weight @ x
        else:
            if self.in_features == 1:
                return self.weight * x + self.bias
            else:
                return self.weight @ x + self.bias

    def extra_repr(self):
        weight_info = (
            f"weight: {self.dim}×{self.out_features}×{self.in_features}"
        )

        if self.bias is not None:
            bias_info = f"bias: {self.dim}×{self.out_features}×1"
            return f"{weight_info}, {bias_info}"
        else:
            return f"{weight_info}, bias=False"


# TNN的缩放层
class TnnScaling(nn.Module):
    """
    定义缩放参数.

    尺寸:
        [k,p] 用于多重TNN
        [p] 用于TNN
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.alpha = nn.Parameter(torch.empty(self.size))

    def extra_repr(self):
        return f"size={self.size}"


# 定义额外参数
class TnnExtra(nn.Module):
    """
    定义额外参数.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.beta = nn.Parameter(torch.empty(self.size))

    def extra_repr(self):
        return f"size={self.size}"


# ********** TNN架构 **********
# 一个简单的TNN
class Tnn(nn.Module):
    """
    简单张量神经网络的架构.
    每个维度上的FNN具有相同的尺寸,
    不同维度的输入积分点相同.
    提供数据点处的TNN值和梯度值.

    参数:
        dim: TNN的维度, FNN的数量
        size: [1, n0, n1, ..., nl, p], 每个FNN的尺寸
        activation: 隐藏层中使用的激活函数
        bd: 边界条件的额外函数
        grad_bd: bd的梯度
        initializer: 可学习参数的初始化方法
    """

    def __init__(
        self,
        dim,
        size,
        activation,
        bd=None,
        grad_bd=None,
        grad_grad_bd=None,
        scaling=True,
        extra_size=False,
        initializer=["default", None],
    ):
        super().__init__()
        self.dim = dim
        self.size = size
        self.activation = activation()
        self.bd = bd
        self.grad_bd = grad_bd
        self.grad_grad_bd = grad_grad_bd
        self.scaling = scaling
        self.extra_size = extra_size

        self.p = abs(self.size[-1])

        self.init_type = initializer[0]
        self.init_data = initializer[1]

        self.ms: nn.ModuleDict = self.__init_modules()
        self.__initialize()

    # 注册TNN模块的可学习参数.
    def __init_modules(self):
        modules = nn.ModuleDict()
        for i in range(1, len(self.size)):
            bias = self.size[i] > 0
            modules[f"TnnLinear{i - 1}"] = TnnLinear(
                self.dim, abs(self.size[i]), abs(self.size[i - 1]), bias
            )
        if self.scaling:
            modules["TnnScaling"] = TnnScaling([self.p])
        if self.extra_size:
            modules["TnnExtra"] = TnnExtra(self.extra_size)
        return modules

    # 初始化TNN模块的可学习参数.
    def __initialize(self):
        # 默认初始化.
        if self.init_type == "default":
            for i in range(1, len(self.size)):
                for j in range(self.dim):
                    nn.init.orthogonal_(
                        self.ms[f"TnnLinear{i - 1}"].weight[j, :, :]
                    )
                    # nn.init.normal_(self.ms['TnnLinear{}'.format(i-1)].weight[j,:,:])
                if self.size[i] > 0:
                    # nn.init.orthogonal_(self.ms['TnnLinear{}'.format(i-1)].bias)
                    nn.init.constant_(self.ms[f"TnnLinear{i - 1}"].bias, 0.5)
            if self.scaling:
                nn.init.constant_(self.ms["TnnScaling"].alpha, 1)
            if self.extra_size:
                nn.init.constant_(self.ms["TnnExtra"].beta, 1)

    # 返回缩放参数的函数
    def scaling_par(self):
        if self.scaling:
            return self.ms["TnnScaling"].alpha
        else:
            raise NameError("TNN模块没有缩放参数")

    # 返回额外参数的函数
    def extra_par(self):
        if self.extra_size:
            return self.ms["TnnExtra"].beta
        else:
            raise NameError("TNN模块没有额外参数")

    def forward(self, w, x, need_grad=0, normed=True):
        """
        参数:
            w: 积分权重 [N]
            x: 积分点 [N]
            need_grad: 是否返回梯度

        返回:
            phi: 每个维度FNN的值 [dim, p, N]
            grad_phi: 每个维度FNN的梯度值 [dim, p, N]
        """
        # 计算每个一维输入FNN在每个积分点处的值.
        if need_grad == 0:
            # 获取强制边界条件函数的值.
            bd_value = None if self.bd is None else self.bd(x)
            # 前向过程.
            for i in range(1, len(self.size) - 1):
                x = self.ms[f"TnnLinear{i - 1}"](x)
                x = self.activation(x)
            if bd_value is None:
                phi = self.ms[f"TnnLinear{len(self.size) - 2}"](x)
            else:
                phi = self.ms[f"TnnLinear{len(self.size) - 2}"](x) * bd_value
            # 归一化
            if normed:
                return phi / torch.sqrt(
                    torch.sum(w * phi**2, dim=2)
                ).unsqueeze(dim=-1)
            else:
                return phi

        # 同时计算每个一维输入FNN在每个积分点处的值和梯度值.
        if need_grad == 1:
            # 获取强制边界条件函数的值.
            bd_value = None if self.bd is None else self.bd(x)
            # 获取强制边界条件函数的梯度值.
            grad_bd_value = None if self.grad_bd is None else self.grad_bd(x)
            # 同时计算前向和反向过程.
            grad_x = self.ms[f"TnnLinear{0}"].weight
            for i in range(1, len(self.size) - 1):
                x = self.ms[f"TnnLinear{i - 1}"](x)
                grad_x = self.activation.grad(x) * grad_x
                grad_x = self.ms[f"TnnLinear{i}"].weight @ grad_x
                x = self.activation(x)
            x = self.ms[f"TnnLinear{len(self.size) - 2}"](x)
            if self.bd is None:
                phi = x
                grad_phi = grad_x
            else:
                phi = x * bd_value
                grad_phi = x * grad_bd_value + grad_x * bd_value
            # 归一化
            if normed:
                return phi / (
                    torch.sqrt(torch.sum(w * phi**2, dim=2)).unsqueeze(dim=-1)
                ), grad_phi / (
                    torch.sqrt(torch.sum(w * phi**2, dim=2)).unsqueeze(dim=-1)
                )
            else:
                return phi, grad_phi

        # 同时计算每个一维输入FNN在每个积分点处的值和梯度值.
        if need_grad == 2:
            # 获取强制边界条件函数的值.
            bd_value = None if self.bd is None else self.bd(x)
            # 获取强制边界条件函数的梯度值.
            grad_bd_value = None if self.grad_bd is None else self.grad_bd(x)
            # 获取grad_grad_bd值
            if self.grad_grad_bd is None:
                grad_grad_bd_value = None
            else:
                grad_grad_bd_value = self.grad_grad_bd(x)

            # 同时计算前向和反向过程.
            grad_x = self.ms[f"TnnLinear{0}"].weight
            grad_grad_x = torch.zeros_like(grad_x)
            for i in range(1, len(self.size) - 1):
                x = self.ms[f"TnnLinear{i - 1}"](x)
                grad_grad_x = (
                    self.activation.grad_grad(x) * (grad_x**2)
                    + self.activation.grad(x) * grad_grad_x
                )
                grad_grad_x = self.ms[f"TnnLinear{i}"].weight @ grad_grad_x
                grad_x = self.activation.grad(x) * grad_x
                grad_x = self.ms[f"TnnLinear{i}"].weight @ grad_x
                x = self.activation(x)
            x = self.ms[f"TnnLinear{len(self.size) - 2}"](x)
            if self.bd is None:
                phi = x
                grad_phi = grad_x
                grad_grad_phi = grad_grad_x
            else:
                phi = x * bd_value
                grad_phi = x * grad_bd_value + grad_x * bd_value
                grad_grad_phi = (
                    x * grad_grad_bd_value
                    + 2 * grad_x * grad_bd_value
                    + grad_grad_x * bd_value
                )
            # 归一化
            if normed:
                return (
                    phi
                    / (
                        torch.sqrt(torch.sum(w * phi**2, dim=2)).unsqueeze(
                            dim=-1
                        )
                    ),
                    grad_phi
                    / (
                        torch.sqrt(torch.sum(w * phi**2, dim=2)).unsqueeze(
                            dim=-1
                        )
                    ),
                    grad_grad_phi
                    / (
                        torch.sqrt(torch.sum(w * phi**2, dim=2)).unsqueeze(
                            dim=-1
                        )
                    ),
                )
            else:
                return phi, grad_phi, grad_grad_phi

    def extra_repr(self):
        return (
            f"一个TNN的架构(dim={self.dim},rank={self.p}), 它有{self.dim}个FNN:\n"
            f"每个FNN的尺寸: {self.size}"
        )


def main():
    pass


if __name__ == "__main__":
    main()
