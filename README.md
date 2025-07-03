<div align="center">

# ⚡ TNN (Tensor Neural Network)

**张量神经网络 - 基于张量分解的神经网络架构用于高精度 PDE 求解**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

_高效解决偏微分方程求解中的维数灾难问题_

</div>

---

## 📋 目录

- [🚀 项目简介](#-项目简介)
- [⚡ 快速开始](#-快速开始)
- [🛠️ 环境搭建](#️-环境搭建)
- [📁 项目结构](#-项目结构)
- [🔧 核心实现](#-核心实现)
- [📚 使用示例](#-使用示例)
- [🤝 贡献指南](#-贡献指南)
- [📄 许可证](#-许可证)

---

## 🚀 项目简介

**TNN (Tensor Neural Network)** 是一种基于张量分解的创新神经网络架构，专门用于求解高维偏微分方程。

### ✨ 核心特性

- 🎯 **张量分解**: 将高维函数表示为多个低维函数的张量积形式
- 🚀 **高效求解**: 有效解决高维偏微分方程求解中的维数灾难问题
- 🔬 **科学计算**: 专为科学计算和数值分析设计
- 🧮 **特征值问题**: 支持拉普拉斯特征值问题和混合导数 PDE 问题

### 🎓 理论基础

TNN 基于张量分解理论，通过将高维函数 $u(x_1, x_2, \ldots, x_d)$ 表示为：

$$\mathrm{tnn}(x_1, x_2, \ldots, x_d) = \sum_{r=1}^{\mathrm{rank}} \theta_r \prod_{d=1}^{\mathrm{dim}} \mathrm{subtnn}_d^{(r)}(x_d)$$

其中 $\mathrm{subtnn}_d^{(r)}$ 是 TNN 子网络，为 $\mathbb{R}\to\mathbb{R}$ 的映射，$\theta_r$ 是张量系数。

---

## ⚡ 快速开始

### 📦 一键安装

```bash
# 克隆项目
git clone https://github.com/ZivenZ123/TNN-zh.git
cd TNN-zh

# 安装依赖 (自动安装 tnn 包)
uv sync

# 运行示例
uv run examples/laplacian_eigenvalue.py
```

---

## 🛠️ 环境搭建

### 🐍 Python 环境要求

- **Python**: 3.11+
- **依赖管理**: 推荐使用 [uv](https://github.com/astral-sh/uv)

### 📥 安装 uv

<details>
<summary>点击展开安装说明</summary>

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

### 🔧 项目安装

```bash
# 1. 克隆项目
git clone https://github.com/ZivenZ123/TNN-zh.git
cd TNN-zh

# 2. 安装依赖
uv sync
```

> 💡 **重要提示**: `uv sync` 会自动将 `tnn` 包以可编辑模式安装到虚拟环境中

### 🎯 可编辑模式的优势

| 特性            | 说明                                       |
| --------------- | ------------------------------------------ |
| 📦 **自动安装** | 无需手动运行 `pip install -e .`            |
| 🔄 **实时更新** | 代码修改立即生效，无需重新安装             |
| 🎯 **直接导入** | 支持 `from tnn import TensorNeuralNetwork` |
| 🚀 **简化运行** | 支持 `uv run examples/xxx.py`              |

---

## 📁 项目结构

```
TNN-zh/
├── 📦 tnn/                           # 🧠 TNN 核心包
│   ├── __init__.py                   # 📋 包初始化
│   └── core.py                       # 🔧 核心实现
├── 📚 examples/                      # 💡 示例代码
│   ├── __init__.py                   # 📋 示例包初始化
│   ├── laplacian_eigenvalue.py       # 🌊 拉普拉斯特征值问题
│   └── mixed_derivative_pde.py       # 🔀 混合导数PDE问题
├── ⚙️ pyproject.toml                 # 🔧 项目配置
├── 🔒 uv.lock                        # 📌 依赖锁定
└── 📖 README.md                      # 📚 项目文档
```

---

## 🔧 核心实现

### 🏗️ 主要组件

| 组件                       | 功能描述                                    |
| -------------------------- | ------------------------------------------- |
| **SubTensorNeuralNetwork** | 🧮 子张量神经网络，处理一维输入输出         |
| **TensorNeuralNetwork**    | 🧠 主要的 TNN 实现，支持高维张量分解        |
| **TNNIntegrator**          | ⚡ 张量积分器，利用高斯积分实现高效数值积分 |
| **DefaultSubNet**          | 🏗️ 默认子网络实现，提供全连接网络           |

### 🎯 核心算法

1. **张量分解**: 将高维函数分解为低维函数的张量积
2. **参数共享**: 相同维度的子网络共享参数，减少参数量
3. **边界条件**: 自动应用齐次 Dirichlet 边界条件
4. **数值积分**: 使用高斯积分器进行高效数值积分

---

## 📚 使用示例

### 🌊 拉普拉斯特征值问题

求解 $-\Delta u = \lambda u$ 在 $[0,1]^d$ 上的特征值问题：

```bash
# 🚀 推荐方式
uv run examples/laplacian_eigenvalue.py

# 🐍 传统方式
source .venv/bin/activate
python examples/laplacian_eigenvalue.py
```

### 🔀 混合导数 PDE 问题

求解带混合导数的二元函数特征值问题：

```bash
# 🚀 推荐方式
uv run examples/mixed_derivative_pde.py

# 🐍 传统方式
source .venv/bin/activate
python examples/mixed_derivative_pde.py
```

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 📝 提交规范

在提交代码前，请确保：

- [ ] 代码通过所有测试
- [ ] 遵循项目的代码风格
- [ ] 添加必要的文档和注释
- [ ] 更新相关的 README 文档

### 🐛 报告问题

如果发现问题，请通过 [GitHub Issues](https://github.com/ZivenZ123/TNN-zh/issues) 报告。

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE.txt)。

---

<div align="center">

**🌟 如果这个项目对你有帮助，请给我们一个 Star！**

[![GitHub stars](https://img.shields.io/github/stars/ZivenZ123/TNN-zh.svg?style=social&label=Star)](https://github.com/ZivenZ123/TNN-zh)

---

_Made with ❤️ by TNN Team_

</div>
