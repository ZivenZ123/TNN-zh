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

### 1. 快速开始: TNN 求解 PDE 的标准流程

```
┌─────────────────────────────────────────────────────────────────
│  步骤 1: 定义 PDE 损失函数类 (继承 nn.Module)
├─────────────────────────────────────────────────────────────────
│  __init__(self, tnn_model):
│    ├─ 构建高维积分点和权重 (generate_quad_points)
│    └─ 构造源项 TNN (如果有)
│
│  forward(self):
│    ├─ 计算 PDE 残差 TNN (例: -Δu - f)
│    └─ 返回 L2 范数 (l2_norm)
└─────────────────────────────────────────────────────────────────
                            ↓
┌─────────────────────────────────────────────────────────────────
│  步骤 2: 定义 solve() 求解主函数
├─────────────────────────────────────────────────────────────────
│  ① 创建 func 网络 (SeparableDimNetwork)
│    └─ apply_dirichlet_bd() 应用强制边界条件
│
│  ② 构建解的 TNN 模型
│
│  ③ 实例化 PDE 损失函数
│
│  ④ 调用 tnn.fit() 方法进行训练
│    └─ 支持多阶段优化 (Adam → LBFGS)
└─────────────────────────────────────────────────────────────────
                            ↓
┌─────────────────────────────────────────────────────────────────
│  步骤 3: 样本外评估与可视化
├─────────────────────────────────────────────────────────────────
│  ① 生成测试点
│  ② 计算预测值 u_tnn(test_points)
│  ③ 与解析解对比 (如有)
│  ④ 可视化结果
└─────────────────────────────────────────────────────────────────
```

### 2. 实战：求解 5 维 Poisson 方程

求解方程 $-\Delta u = f$ 在 $\Omega = [0,1]^5$ 上, 边界条件 $u|_{\partial\Omega} = 0$, 真解为 $u(x) = \prod_i \sin(\pi x_i)$。

```python
import math
import torch
import torch.nn as nn
from tnn_zh import TNN, SeparableDimNetwork, generate_quad_points, l2_norm

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
        val = torch.sin(PI * x)
        return val.unsqueeze(1)

# 2. 定义 PDE 损失函数
class PoissonPDELoss(nn.Module):
    def __init__(self, tnn_model: TNN):
        super().__init__()
        self.tnn: TNN = tnn_model
        
        # 生成积分点
        domain_bounds = [(0.0, 1.0) for _ in range(DIM)]
        self.pts, self.w = generate_quad_points(
            domain_bounds, device=DEVICE, dtype=DTYPE
        )
        
        # 构造源项 TNN
        source_func = SourceFunc(DIM)
        self.f_tnn: TNN = (DIM * PI**2) * TNN(
            dim=DIM, rank=1, func=source_func, theta=False
        ).to(DEVICE, DTYPE)
    
    def forward(self):
        residual: TNN = -self.tnn.laplace() - self.f_tnn
        return l2_norm(residual, self.pts, self.w)

# 3. 构建模型 (应用 Dirichlet 零边界条件)
boundary_conditions = [(0.0, 1.0) for _ in range(DIM)]
u_tnn_func = (
    SeparableDimNetwork(dim=DIM, rank=RANK)
    .apply_dirichlet_bd(boundary_conditions)
    .to(DEVICE, DTYPE)
)
u_tnn = TNN(dim=DIM, rank=RANK, func=u_tnn_func).to(DEVICE, DTYPE)

# 4. 训练
loss_fn = PoissonPDELoss(u_tnn)
u_tnn.fit(
    loss_fn,
    phases=[
        {"type": "adam", "lr": 0.01, "epochs": 2000},
        {"type": "lbfgs", "lr": 1.0, "epochs": 100},
    ],
)
```

> 完整代码请参考 `examples/poisson_nd.py`
