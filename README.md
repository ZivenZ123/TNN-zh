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
uv run examples/laplace_nd.py
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

### 2. 实战：求解 5 维 Laplace 方程

求解方程 $-\Delta u = f$ 在 $[0,1]^5$ 上。以下演示如何利用 TNN 和 PyTorch 优化器求解高维 PDE。

```python
import torch
import math
from tnn_zh import TNN, SeparableDimNetworkGELU, generate_quad_points, int_tnn_L2

# 配置
DIM = 5
RANK = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 构建模型 (应用 Dirichlet 零边界条件)
boundary = [(0.0, 1.0) for _ in range(DIM)]
func = SeparableDimNetworkGELU(dim=DIM, rank=RANK).apply_dirichlet_bd(boundary)
u_tnn = TNN(dim=DIM, rank=RANK, func=func).to(DEVICE)

# 2. 准备积分点 (用于计算 PDE Loss)
quad_points, quad_weights = generate_quad_points(
    domain_bounds=boundary, n_quad_points=16, device=DEVICE
)

# 3. 定义 PDE 源项 f(x) (此处略去 f 的具体构造，假设为已知 TNN f_tnn)
# f_tnn = ... 

# 4. 训练 (使用 TNN.fit)
def loss_fn():
    # 计算残差: R = -Δu - f
    # u_tnn.laplace() 返回一个新的 TNN 对象表示 Δu
    residual = -u_tnn.laplace() - f_tnn
    
    # 计算 Loss: ||R||^2
    return int_tnn_L2(residual, quad_points, quad_weights)

# 训练配置: 支持 Adam, LBFGS 等多种优化器
phases = [
    {"type": "adam", "lr": 0.01, "epochs": 1000},
    {"type": "adam", "lr": 0.001, "epochs": 1000},
]

# 开始训练
u_tnn.fit(loss_fn, phases)
```

> 完整代码请参考 `examples/laplace_nd.py`。
