<div align="center">

# ⚡ TNN (Tensor Neural Network)

**张量神经网络 - 基于张量分解的神经网络架构用于高精度 PDE 求解**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

_高效解决偏微分方程求解中的维数灾难问题_

</div>

---

## 目录

- [项目简介](#项目简介)
- [环境搭建](#环境搭建)
- [项目结构](#项目结构)
- [使用示例](#使用示例)
- [许可证](#许可证)

---

## 项目简介

**TNN (Tensor Neural Network)** 是一种基于张量分解的创新神经网络架构,专门用于求解高维偏微分方程.

### 核心特性

- **张量分解**: 将高维函数表示为多个低维函数的张量积形式
- **高效求解**: 有效解决偏微分方程求解中的维数灾难问题
- **高精度积分**: 支持区间细分的高斯积分,提升非光滑函数的积分精度
- **通用训练器**: 提供统一的训练接口,支持多种优化器和多阶段训练
- **GPU加速**: 自动检测并使用GPU加速计算

### 理论基础

TNN 基于张量分解理论,通过将高维函数 \(u(x_1, x_2, \ldots, x_d)\) 表示为:

$$\mathrm{tnn}(x_1, x_2, \ldots, x_d) = \sum_{r=1}^{\mathrm{rank}} \theta_r \prod_{d=1}^{\mathrm{dim}} \mathrm{subtnn}_d^{(r)}(x_d)$$

其中 \(\mathrm{subtnn}_d^{(r)}\) 是 TNN 子网络,为 \(\mathbb{R}\to\mathbb{R}\) 的映射,\(\theta_r\) 是张量系数.

---

## 环境搭建

### Python 环境要求

- **Python**: 3.11+
- **PyTorch**: 自动安装(支持 CPU 和 GPU)
- **依赖管理**: 推荐使用 [uv](https://github.com/astral-sh/uv)

### 安装 uv

<details>
<summary><strong>🔽 点击这里查看 uv 安装步骤</strong></summary>

#### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 验证安装

安装完成后,**重启终端**并运行以下命令验证安装:

```bash
uv --version
```

</details>

### 项目安装

```bash
# 1. 克隆项目
git clone https://github.com/ZivenZ123/TNN-zh.git
cd TNN-zh

# 2. 安装依赖
uv sync
```

> 💡 **重要提示**: `uv sync` 会自动将 `tnn` 包以可编辑模式安装到虚拟环境中

### 可编辑模式的优势

| 特性         | 说明                                       |
| ------------ | ------------------------------------------ |
| **自动安装** | 无需手动运行 `pip install -e .`            |
| **实时更新** | 代码修改立即生效,无需重新安装             |
| **直接导入** | 支持 `from tnn_zh import TNN` |
| **简化运行** | 支持 `uv run examples/xxx.py`              |

---

## 项目结构

```
TNN-zh/
├── tnn_zh/                      # TNN 核心包
│   ├── __init__.py              # 包初始化
│   ├── core.py                  # 核心实现 (TNN, SeparableDimNetwork等)
│   ├── integration.py           # 积分模块 (int_tnn, int_tnn_product等)
│   └── trainer.py               # 通用训练器
├── examples/                    # 示例代码
│   ├── __init__.py             # 示例包初始化
│   └── black_scholes_option.py # Black-Scholes期权定价
├── pyproject.toml              # 项目配置
├── uv.lock                     # 依赖锁定
└── README.md                   # 项目文档
```

---

## 使用示例

### 快速开始

```python
import torch
from tnn_zh import (
    TNN,
    SeparableDimNetworkGELU,
    TNNTrainer,
    int_tnn,
    int_tnn_product,
    generate_quad_points,
)

# 设备配置 (自动检测GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 创建TNN模型
dim = 3  # 输入维度
rank = 10  # 张量秩
domain_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

# 创建子网络
subnet = SeparableDimNetworkGELU(dim=dim, rank=rank).to(DEVICE)

# 创建TNN
tnn = TNN(dim=dim, rank=rank, func=subnet).to(DEVICE)

# 生成积分点和权重
quad_points, quad_weights = generate_quad_points(
    domain_bounds=domain_bounds,
    n_quad_points=16,
    sub_intervals=10,
    device=DEVICE
)

# 定义损失函数
def loss_fn():
    # 示例: 计算TNN的L2范数
    result = int_tnn_product(tnn, tnn, quad_points, quad_weights)
    return result

# 创建训练器
trainer = TNNTrainer(tnn, loss_fn, print_interval=100)

# 多阶段训练
training_phases = [
    {'type': 'adam', 'lr': 0.001, 'epochs': 1000},
    {'type': 'adam', 'lr': 0.0001, 'epochs': 1000},
]

losses, training_time = trainer.multi_phase(training_phases)
print(f"训练完成! 用时: {training_time:.2f}s")
```

### Black-Scholes 期权定价示例

项目包含一个完整的 Black-Scholes 期权定价求解器示例,展示如何使用 TNN 求解实际的偏微分方程问题.

```bash
# 运行示例
uv run examples/black_scholes_option.py
```

这个示例展示了:
- 如何使用两步法求解带边界条件的PDE
- 如何使用 `apply_dirichlet_bd` 应用边界条件
- 如何使用 `wrap_1d_func_as_tnn` 包装一维函数
- 如何使用 `TNNTrainer` 进行多阶段训练
- 如何可视化求解结果

### 主要组件说明

| 组件 | 功能描述 |
| --- | --- |
| **TNN** | 主要的张量神经网络类,支持高维张量分解 |
| **SeparableDimNetwork / SeparableDimNetworkGELU** | 可分离维度子网络,支持不同激活函数 |
| **TNNTrainer** | 通用训练器,支持 Adam, LBFGS, SGD 等优化器 |
| **int_tnn** | 计算单个TNN的积分 |
| **int_tnn_product** | 计算两个TNN乘积的积分(内存优化版本) |
| **generate_quad_points** | 生成高斯积分点和权重,支持区间细分 |
| **apply_dirichlet_bd** | 应用 Dirichlet 边界条件 |
| **wrap_1d_func_as_tnn** | 将一维函数包装为TNN对象 |

### 优化器配置

训练器支持多种优化器,每个训练阶段的配置格式:

```python
{
    'type': 'adam',        # 优化器类型: 'adam', 'lbfgs', 'sgd'
    'lr': 0.001,           # 学习率
    'epochs': 100,         # 训练轮数
    'grad_clip': 1.0,      # 梯度裁剪 (可选)
    # 其他优化器特定参数...
}
```

---

## 许可证

本项目采用 [MIT 许可证](LICENSE.txt).

---

<div align="center">

**⭐ 如果这个项目对你有帮助,请给我们一个 Star!**

[![GitHub stars](https://img.shields.io/github/stars/ZivenZ123/TNN-zh.svg?style=social&label=Star)](https://github.com/ZivenZ123/TNN-zh)

---

_Made with ❤️ by TNN Team_

</div>
