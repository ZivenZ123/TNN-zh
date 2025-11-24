<div align="center">

# ⚡ TNN (Tensor Neural Network)

**基于张量分解的高维偏微分方程 (PDE) 求解器，有效解决维数灾难**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## 环境搭建

本项目推荐使用 **[uv](https://github.com/astral-sh/uv)** 进行依赖管理和环境配置。

<details>
<summary><strong>🔽 点击展开：如何安装 uv</strong></summary>

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装完成后，请**重启终端**并运行 `uv --version` 验证安装。
</details>

### 项目安装与运行

```bash
# 1. 克隆项目
git clone https://github.com/ZivenZ123/TNN-zh.git
cd TNN-zh

# 2. 初始化环境 (自动安装依赖)
uv sync

# 3. 运行示例
uv run examples/poisson_nd.py
```

---

## 使用示例

### 1. 快速开始

TNN 的核心是将高维函数分解为一维子网络的张量积。

```python
import torch
from tnn_zh import TNN, SeparableDimNetwork

# 1. 定义维度和秩
dim = 3    # 输入维度 (x, y, z)
rank = 10  # 张量分解的秩 (Rank)

# 2. 创建模型
# SeparableDimNetwork 用于构建各维度的子网络
subnet = SeparableDimNetwork(dim=dim, rank=rank)
tnn = TNN(dim=dim, rank=rank, func=subnet)

# 3. 前向传播
x = torch.randn(5, dim)  # Batch size = 5
y = tnn(x)               # Output: (5,)
print(f"Output shape: {y.shape}")

# 4. 自动微分 (计算梯度和 Laplace 算子)
# TNN 内置了高效的微分算子实现
grad = tnn.grad(grad_dim=0)      # 对第0维求导
laplace = tnn.laplace()          # 计算 Laplacian (Δu)
```

### 2. 实战：求解 5 维 Poisson 方程

求解方程 $-\Delta u = f$ 在 $[0,1]^5$ 上, 真解为 $u(x) = \prod_i \sin(\pi x_i)$。

```python
import torch
import torch.nn as nn
import math
from tnn_zh import TNN, SeparableDimNetworkGELU, generate_quad_points, int_tnn_L2

# 配置
DIM = 5
RANK = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
PI = math.pi

# 1. 定义源项 f(x) = d * π^2 * prod(sin(πx_i))
class SourceFunc(nn.Module):
    """将源项 f(x) 表示为秩1的TNN分量"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        val = torch.sin(PI * x)  # 每个维度计算 sin(πx)
        return val.unsqueeze(1)  # 添加 rank 维度

# 2. 定义 PDE 损失函数
class PoissonPDELoss(nn.Module):
    def __init__(self, tnn_model, domain_bounds):
        super().__init__()
        self.tnn = tnn_model
        
        # 生成积分点
        self.quad_points, self.quad_weights = generate_quad_points(
            domain_bounds, device=DEVICE, dtype=DTYPE
        )
        
        # 构造源项 TNN
        source_func = SourceFunc(DIM)
        self.f_tnn = (DIM * PI**2) * TNN(
            dim=DIM, rank=1, func=source_func
        ).to(DEVICE, dtype=DTYPE)
    
    def forward(self):
        residual = -self.tnn.laplace() - self.f_tnn  # 计算残差: -Δu - f
        return int_tnn_L2(residual, self.quad_points, self.quad_weights)

# 3. 构建模型 (应用 Dirichlet 零边界条件)
boundary = [(0.0, 1.0) for _ in range(DIM)]
u_tnn_func = (
    SeparableDimNetworkGELU(dim=DIM, rank=RANK)
    .apply_dirichlet_bd(boundary)
    .to(DEVICE, dtype=DTYPE)
)
u_tnn = TNN(dim=DIM, rank=RANK, func=u_tnn_func).to(DEVICE, dtype=DTYPE)

# 4. 训练
loss_fn = PoissonPDELoss(u_tnn, boundary)
u_tnn.fit(
    loss_fn=loss_fn,
    phases=[{"type": "adam", "lr": 0.01, "epochs": 2000, "grad_clip": 1.0}]
)
```

> 完整代码请参考 `examples/poisson_nd.py`。
