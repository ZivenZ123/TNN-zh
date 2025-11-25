# TNN 库已知函数 (Known Functions) 构建指南

**用户提示**: 请将此文档作为 Prompt 发送给 AI 助手，并提供你需要构建的**已知函数的数学表达式**。

此文档是指导 AI 助手将数学表达式封装为 TNN (Tensor Neural Network) 对象的**完整规范**。AI 助手在阅读此文档后，**无需**访问 TNN 库的源代码即可生成正确的代码。

---

## 1. TNN 类结构与设计哲学

TNN (`tnn_zh.TNN`) 是一个通用的容器类。**用户不需要也不应该重写 TNN 类的 `forward` 方法**。

TNN 的核心逻辑完全由其 `func` 属性决定。工作流如下：
1.  **外部定义**：编写一个满足特定接口规范的 `func` 类（继承自 `nn.Module`）。
2.  **注入**：实例化 `TNN` 类时，将这个 `func` 实例作为参数传入。
3.  **使用**：直接调用 `TNN` 实例的方法（如 `tnn(x)`）。

```python
# 伪代码示例
my_func = MyCustomFunc(dim=3)          # 1. 外部定义 func
tnn = TNN(dim=3, rank=1, func=my_func) # 2. 注入 TNN
val = tnn(x)                           # 3. 使用
```

## 2. 核心数据流规范 (AI 必须严格遵守)

编写 `func` 时，必须严格遵守以下张量形状规范，**切勿混淆**：

*   **输入 (Input)**: `x` 形状为 `(n_1d, dim)`。
    *   `n_1d`: 样本数量（行数）。
    *   `dim`: 空间维度（列数）。
*   **Func 输出 (Internal)**: `func(x)` **必须**返回形状为 **`(n_1d, rank, dim)`** 的张量。
    *   这是 TNN 变量分离的核心体现。
    *   `output[i, r, d]` 代表第 `i` 个样本在第 `r` 个秩分量、第 `d` 个维度上的值。
*   **TNN 输出 (Final)**: `tnn(x)` 返回 `(n_1d,)`。
    *   TNN 类会自动执行 $\sum_{r} \theta_r \prod_{d} \text{func}(x)_{r,d}$ 的计算，用户无需在 `func` 中处理。

## 3. 构建路径 A：高性能模式 (推荐)

**适用场景**: 构建复杂的源项、初值，或追求最高计算效率。
**方法**: 手写一个继承自 `nn.Module` 的类。

**注意**: 通过此方法构建的 TNN **默认不支持求导**（即不能调用 `tnn.grad()` 或 `tnn.laplace()`），因为是已知函数，提示用户需要手动求导。

### 3.1 基础示例 1：异构秩-1 函数

**目标函数**: $f(x) = \sin(\pi x_0) \cdot \cos(2\pi x_1)$
**维度**: `dim=2`, `rank=1`
**特点**: 不同维度应用不同的函数。

```python
import torch
import torch.nn as nn
import math
from tnn_zh import TNN

class SimpleSourceFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        # 输入 x: (n_1d, dim)
        if x.dim() == 1: x = x.unsqueeze(0)
        
        # 1. 计算各维度的分量 (n_1d, 1)
        term0 = torch.sin(math.pi * x[:, 0:1])     # sin(π x0)
        term1 = torch.cos(2 * math.pi * x[:, 1:2]) # cos(2π x1)
        
        # 2. 组合回 (n_1d, dim) 形状
        # 注意：TNN 是各维度值的乘积，所以我们只需把各维度的值填入对应位置
        val = torch.cat([term0, term1], dim=1) 
        
        # 3. 调整为 (n_1d, rank=1, dim)
        return val.unsqueeze(1) 

# 使用
func = SimpleSourceFunc(dim=2)
tnn = TNN(dim=2, rank=1, func=func, theta=False) # theta=False 表示权重固定为1
```

### 3.2 基础示例 2：同构乘积函数 (参考 Poisson 方程)

**目标函数**: $f(x) = \prod_{i=0}^{d-1} \sin(\pi x_i)$
**维度**: `dim=d`, `rank=1`
**特点**: 所有维度应用相同的函数逻辑，可使用向量化操作。

```python
class PoissonSourceFunc(nn.Module):
    """将 f(x) = prod(sin(πx)) 表示为秩1的TNN分量"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x形状: (n_1d, dim)
        if x.dim() == 1: x = x.unsqueeze(0)
            
        # 1. 计算所有维度的 sin(πx) - 直接向量化操作
        val = torch.sin(math.pi * x) # (n_1d, dim)
        
        # 2. 在 dim=1 插入 rank 维度
        return val.unsqueeze(1) # (n_1d, rank=1, dim)

# 构建 TNN (常数因子 dim*π^2 通过外部乘法处理)
dim = 5
f_tnn = (dim * math.pi**2) * TNN(dim=dim, rank=1, func=PoissonSourceFunc(dim), theta=False)
```

### 3.3 进阶示例 1：多项式组合 (Rank > 1)

**目标函数**: $f(x) = \underbrace{x_0 \cdot x_1}_{\text{秩1分量}} + \underbrace{\sin(x_0) \cdot e^{x_1}}_{\text{秩2分量}}$
**维度**: `dim=2`, `rank=2`
**特点**: 函数是多项之和，需要使用多秩 TNN。

```python
class PolyFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        # x: (n_1d, dim)
        if x.dim() == 1: x = x.unsqueeze(0)
        
        # --- 第 1 个秩分量: x0 * x1 ---
        # 每一维的值: [x0, x1]
        rank1_val = x  # (n_1d, dim)
        
        # --- 第 2 个秩分量: sin(x0) * exp(x1) ---
        # 每一维的值: [sin(x0), exp(x1)]
        r2_d0 = torch.sin(x[:, 0:1]) # (n_1d, 1)
        r2_d1 = torch.exp(x[:, 1:2]) # (n_1d, 1)
        rank2_val = torch.cat([r2_d0, r2_d1], dim=1) # (n_1d, dim)
        
        # --- 堆叠为 (n_1d, rank=2, dim) ---
        return torch.stack([rank1_val, rank2_val], dim=1)

# 使用
func = PolyFunc(dim=2)
tnn = TNN(dim=2, rank=2, func=func, theta=False)
```

### 3.4 进阶示例 2：混合时空函数 (参考热传导方程)

**目标函数**: $f(x, t) = \prod_{i=0}^{d-1} \sin(\pi x_i) \quad (\text{t=0 时刻的初值})$
**维度**: `dim=spatial_dim + 1` (最后一维是时间)
**特点**: 某些维度的值依赖输入，某些维度是常数。

```python
class HeatInitialFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dim = spatial_dim + 1 (time)

    def forward(self, x):
        # x形状: (n_1d, dim)
        if x.dim() == 1: x = x.unsqueeze(0)
            
        # 1. 空间维度: sin(π*x) (取前 dim-1 列)
        spatial_val = torch.sin(math.pi * x[:, :-1])
        
        # 2. 时间维度: 1.0 (因为初值只与空间有关，时间维度因子设为1)
        # 即使输入 t 不为 0，此函数也返回与 t 无关的值（仅作为 t=0 的初值形状）
        time_val = torch.ones_like(x[:, -1:])
        
        # 3. 拼接: (n_1d, dim)
        val = torch.cat([spatial_val, time_val], dim=1)
        
        return val.unsqueeze(1)
```

## 4. 构建路径 B：快速原型模式 (wrap_1d_func_as_tnn)

**适用场景**: 简单的单变量系数、常数项，或者不想手写类。
**方法**: 使用 `tnn_zh.wrap_1d_func_as_tnn`。

**注意**: 此方法生成的 TNN 同样 **默认不支持求导**。组合使用性能较差。

### 4.1 示例：Black-Scholes 方程系数 (参考 BS Option)

**目标系数**: 
1.  $\sigma^2(x) = 0.5 \sigma^2$ (其中 $x_2$ 代表 $\sigma$)
2.  $S^2(x) = S^2$ (其中 $x_0$ 代表 $S$)

```python
from tnn_zh import wrap_1d_func_as_tnn

# 定义维度映射: (S, t, sigma) -> (0, 1, 2)
DIM = 3

# 1. 构造 0.5 * sigma^2 (只与第2维有关)
sigma2_tnn = wrap_1d_func_as_tnn(dim=DIM, target_dim=2)(
    lambda s: 0.5 * s**2
)

# 2. 构造 S^2 (只与第0维有关)
S2_tnn = wrap_1d_func_as_tnn(dim=DIM, target_dim=0)(
    lambda S: S**2
)

# 在 Loss 中使用:
# residual = ... + sigma2_tnn * S2_tnn * C.grad2(0, 0) ...
```

## 5. 总结 Checklist (AI 自查用)

1.  **输入形状**: `x` 是 `(n_1d, dim)` 吗？
2.  **输出形状**: `forward` 返回的是 `(n_1d, rank, dim)` 吗？
3.  **维度扩展**: 即使 `rank=1`，也使用了 `.unsqueeze(1)` 吗？
4.  **逻辑正确**: TNN 是**乘积**结构 ($\prod \text{val}_d$)。
    *   如果是加法组合 (如 $x+y$)，需要 `rank=2` ($x\cdot 1 + 1 \cdot y$) 或两个 TNN 相加。
    *   如果是乘法组合 (如 $x \cdot y$)，通常 `rank=1` 即可。
