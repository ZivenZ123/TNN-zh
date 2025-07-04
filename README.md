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
- [核心实现](#核心实现)
- [使用示例](#使用示例)
- [GPU 使用说明](#gpu-使用说明)
- [API 参考](#api-参考)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 项目简介

**TNN (Tensor Neural Network)** 是一种基于张量分解的创新神经网络架构，专门用于求解高维偏微分方程。

### 核心特性

- **张量分解**: 将高维函数表示为多个低维函数的张量积形式
- **高效求解**: 有效解决偏微分方程求解中的维数灾难问题
- **高精度积分**: 支持区间细分的高斯积分，提升非光滑函数的积分精度
- **通用训练器**: 提供统一的训练接口，支持多种优化器和多阶段训练
- **科学计算**: 专为科学计算和数值分析设计

### 理论基础

TNN 基于张量分解理论，通过将高维函数 $u(x_1, x_2, \ldots, x_d)$ 表示为：

$$\mathrm{tnn}(x_1, x_2, \ldots, x_d) = \sum_{r=1}^{\mathrm{rank}} \theta_r \prod_{d=1}^{\mathrm{dim}} \mathrm{subtnn}_d^{(r)}(x_d)$$

其中 $\mathrm{subtnn}_d^{(r)}$ 是 TNN 子网络，为 $\mathbb{R}\to\mathbb{R}$ 的映射，$\theta_r$ 是张量系数。

---

## 环境搭建

### Python 环境要求

- **Python**: 3.11+
- **PyTorch**: 自动安装（支持 CPU 和 GPU）
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

安装完成后，**重启终端**并运行以下命令验证安装:

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
| **实时更新** | 代码修改立即生效，无需重新安装             |
| **直接导入** | 支持 `from tnn import TensorNeuralNetwork` |
| **简化运行** | 支持 `uv run examples/xxx.py`              |

---

## 项目结构

```
TNN-zh/
├── tnn/                           # TNN 核心包
│   ├── __init__.py               # 包初始化
│   ├── core.py                   # 核心实现
│   └── trainer.py                # 通用训练器
├── examples/                     # 示例代码
│   ├── __init__.py              # 示例包初始化
│   ├── laplacian_eigenvalue.py  # 拉普拉斯特征值问题
│   └── mixed_derivative_pde.py  # 混合导数PDE问题
├── pyproject.toml               # 项目配置
├── uv.lock                      # 依赖锁定
└── README.md                    # 项目文档
```

---

## 核心实现

### 主要组件

| 组件                       | 功能描述                                                       |
| -------------------------- | -------------------------------------------------------------- |
| **SubTensorNeuralNetwork** | 子张量神经网络，处理一维输入输出                               |
| **TensorNeuralNetwork**    | 主要的 TNN 实现，支持高维张量分解                              |
| **TNNIntegrator**          | 张量积分器，利用高斯积分实现高效数值积分，支持区间细分提升精度 |
| **TNNTrainer**             | 通用训练器，支持多种优化器和多阶段训练                         |

### 核心算法

1. **张量分解**: 将高维函数分解为低维函数的张量积
2. **参数共享**: 相同维度的子网络共享参数，减少参数量
3. **边界条件**: 自动应用齐次 Dirichlet 边界条件
4. **数值积分**: 使用高斯积分器进行高效数值积分
5. **多阶段训练**: 支持 Adam + LBFGS 组合优化策略

---

## 使用示例

### 快速开始

```python
from tnn import TensorNeuralNetwork, TNNIntegrator, TNNTrainer, DEVICE

# 检查当前设备
print(f"当前设备: {DEVICE}")  # 自动检测 GPU 或 CPU

# 创建5维TNN (自动使用GPU加速)
tnn = TensorNeuralNetwork(dim=5, rank=15, domain_bounds=[(0, 1)] * 5).to(DEVICE)

# 创建积分器 (支持区间细分以提升积分精度)
integrator = TNNIntegrator(n_quad_points=16)

# 定义损失函数
def loss_fn():
    # 这里定义你的损失函数，例如 Rayleigh 商
    return compute_your_loss(tnn, integrator)

# 创建训练器
trainer = TNNTrainer(tnn, loss_fn)

# 配置多阶段训练
training_phases = [
    {'type': 'adam', 'lr': 0.001, 'epochs': 10, 'name': 'Adam 快速下降'},
    {'type': 'adam', 'lr': 0.0001, 'epochs': 10, 'name': 'Adam 精细调优'},
    {'type': 'lbfgs', 'lr': 1.0, 'epochs': 1, 'name': 'LBFGS 精确求解'},
]

# 执行训练
losses, training_time = trainer.multi_phase(training_phases)
```

### GPU 使用说明

TNN 具有内置的 GPU 支持，无需额外配置：

- **自动检测**: 代码会自动检测并使用 GPU（如果可用）
- **设备管理**: 全局 `DEVICE` 变量自动设置为 `"cuda"` 或 `"cpu"`
- **模型迁移**: 所有张量和模型会自动移动到正确的设备
- **性能提升**: 在高维问题中可获得显著的性能提升

如需强制使用 CPU：

```python
import torch
import tnn.core as core

# 强制使用CPU
core.DEVICE = torch.device("cpu")
```

### 拉普拉斯特征值问题

求解 $-\Delta u = \lambda u$ 在 $[0,1]^d$ 上的特征值问题：

```bash
# 推荐方式
uv run examples/laplacian_eigenvalue.py

# 传统方式
source .venv/bin/activate
python examples/laplacian_eigenvalue.py
```

**示例输出**:

```
>>> 5维拉普拉斯特征值问题 <<<
张量秩: 15
计算设备: cuda
理论最小特征值: 49.348022

>>> 开始多阶段训练 <<<
TNN参数总数: 18090
训练阶段数: 3

>>> Adam 快速下降 阶段 <<<
Epoch 0, Loss: 245.234567
...

最终特征值: 49.348123
相对误差: 0.0002%
```

### 混合导数 PDE 问题

求解带混合导数的二元函数特征值问题：

```bash
# 推荐方式
uv run examples/mixed_derivative_pde.py

# 传统方式
source .venv/bin/activate
python examples/mixed_derivative_pde.py
```

---

## API 参考

<details>
<summary><strong> TNNTrainer 类 - 通用训练器</strong></summary>

`TNNTrainer` 是通用训练器，支持多种优化器和多阶段训练策略。

#### 构造函数

```python
TNNTrainer(tnn, loss_fn, verbose=True)
```

**参数**:

- `tnn`: TensorNeuralNetwork 实例
- `loss_fn`: 损失函数，返回标量 tensor
- `verbose`: 是否输出详细训练信息

#### 主要方法

##### multi_phase()

```python
multi_phase(phases) -> Tuple[List[float], float]
```

执行多阶段训练。

**参数**:

- `phases`: 训练阶段配置列表

**返回**:

- 损失历史和训练时间

##### train_simple()

```python
train_simple(optimizer_type="adam", lr=0.001, epochs=100, **kwargs)
```

简单的单阶段训练。

#### 训练阶段配置

每个训练阶段的配置格式:

```python
{
    'type': 'adam',        # 优化器类型: 'adam', 'lbfgs', 'sgd'
    'lr': 0.001,           # 学习率
    'epochs': 10,          # 训练轮数
    'name': 'Adam 阶段',   # 阶段名称 (可选)
    # 其他优化器特定参数...
}
```

#### 支持的优化器

| 优化器 | 类型标识 | 主要参数                                     |
| ------ | -------- | -------------------------------------------- |
| Adam   | 'adam'   | `lr`, `weight_decay`, `betas`, `eps`         |
| LBFGS  | 'lbfgs'  | `lr`, `max_iter`, `tolerance_grad`           |
| SGD    | 'sgd'    | `lr`, `momentum`, `weight_decay`, `nesterov` |

</details>

<details>
<summary><strong> TensorNeuralNetwork 类 - 张量神经网络</strong></summary>

主要的 TNN 实现，支持高维张量分解和 GPU 加速。

#### 构造函数

```python
TensorNeuralNetwork(dim, rank, domain_bounds, subnet_factory=None)
```

**参数**:

- `dim`: 输入维度
- `rank`: 张量秩
- `domain_bounds`: 域边界，格式为 `[(a₁,b₁), (a₂,b₂), ..., (aₙ,bₙ)]`
- `subnet_factory`: 子网络工厂函数 (可选)

#### 主要方法

- `forward(x)`: 前向传播
- `grad(dim)`: 计算关于指定维度的梯度
- `multiply_1d_function(func, target_dim)`: 与一维函数相乘
- `to(device)`: 将模型移动到指定设备 (GPU/CPU)

#### 设备管理

TNN 支持自动 GPU 加速：

```python
from tnn import TensorNeuralNetwork, DEVICE

# 创建TNN并移动到GPU
tnn = TensorNeuralNetwork(dim=5, rank=15).to(DEVICE)

# 检查模型设备
print(f"模型设备: {next(tnn.parameters()).device}")
```

</details>

<details>
<summary><strong> TNNIntegrator 类 - 张量积分器</strong></summary>

张量积分器，利用高斯积分实现高效数值积分，支持区间细分功能以提升积分精度。

#### 构造函数

```python
TNNIntegrator(n_quad_points=16)
```

**参数**:

- `n_quad_points`: 高斯积分点数，默认为 16

#### 主要方法

##### 基础积分方法

- `tnn_int1(tnn, domain_bounds)`: 计算单个 TNN 函数的积分
- `tnn_int2(tnn1, tnn2, domain_bounds)`: 计算两个 TNN 函数的内积

##### 区间细分积分方法

- `tnn_int1_with_subdivision(tnn, domain_bounds, sub_intervals=10)`: 带区间细分的单个 TNN 积分
- `tnn_int2_with_subdivision(tnn1, tnn2, domain_bounds, sub_intervals=10)`: 带区间细分的两个 TNN 内积

#### 区间细分功能

区间细分通过将积分区间分成多个等距的子区间来提高积分精度，特别适用于：

- 非光滑函数的积分
- 振荡函数的积分
- 复杂子网络输出的积分

**使用示例**:

```python
integrator = TNNIntegrator(n_quad_points=16)
tnn = TensorNeuralNetwork(dim=2, rank=5)
domain = [(0.0, 1.0), (0.0, 1.0)]

# 基础积分
result1 = integrator.tnn_int1(tnn, domain)

# 使用区间细分 (每个维度分成10个子区间)
result2 = integrator.tnn_int1_with_subdivision(tnn, domain, sub_intervals=10)

# 两个TNN的内积，使用区间细分
result3 = integrator.tnn_int2_with_subdivision(tnn1, tnn2, domain, sub_intervals=10)
```

**性能对比**:

| 方法     | 精度 | 计算成本 | 适用场景        |
| -------- | ---- | -------- | --------------- |
| 基础积分 | 标准 | 低       | 光滑函数        |
| 区间细分 | 高   | 中等     | 非光滑/振荡函数 |

</details>

---

## 贡献指南

我们欢迎所有形式的贡献！

### 提交规范

在提交代码前，请确保：

- [ ] 代码通过所有测试
- [ ] 遵循项目的代码风格
- [ ] 添加必要的文档和注释
- [ ] 更新相关的 README 文档

### 报告问题

如果发现问题，请通过 [GitHub Issues](https://github.com/ZivenZ123/TNN-zh/issues) 报告。

---

## 许可证

本项目采用 [MIT 许可证](LICENSE.txt)。

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给我们一个 Star！**

[![GitHub stars](https://img.shields.io/github/stars/ZivenZ123/TNN-zh.svg?style=social&label=Star)](https://github.com/ZivenZ123/TNN-zh)

---

_Made with ❤️ by TNN Team_

</div>
