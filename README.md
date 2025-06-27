# TNN (Tensor Neural Network)

张量神经网络(TNN)实现 - 基于张量分解的神经网络架构用于高维 PDE 求解

## 项目简介

本项目实现了张量神经网络(Tensor Neural Network, TNN), 这是一种基于张量分解的创新神经网络架构. TNN 通过将高维函数表示为多个低维函数的张量积形式, 有效解决了高维偏微分方程求解中的维数灾难问题.

## 推荐安装 uv 来管理项目 Python 环境

uv 是一个快速的 Python 包管理器和项目管理工具. 以下是在不同操作系统上安装 uv 的方法:

### macOS 和 Linux

使用官方安装脚本:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或者使用 Homebrew (仅 macOS):

```bash
brew install uv
```

### Windows

使用 PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

或者使用 Scoop:

```bash
scoop install uv
```

### 验证安装

安装完成后，重启终端并运行以下命令验证安装:

```bash
uv --version
```

## 项目环境搭建

### 1. 克隆项目

```bash
git clone <repository-url>
cd TNN
```

### 2. 使用 uv 安装依赖

uv 会自动检测 `pyproject.toml` 文件并安装所需的依赖:

```bash
# 安装项目依赖
uv sync
```

### 3. 激活虚拟环境

```bash
# 激活 uv 创建的虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或者
.venv\Scripts\activate     # Windows
```

### 4. 运行项目

有两种方式运行项目代码:

#### 方式一: 使用 python (需要先激活虚拟环境)

```bash
# 确保已激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或者 .venv\Scripts\activate  # Windows

# 运行主要的TNN实现示例
python tnn_reconstructed.py
```

#### 方式二: 使用 uv run (推荐，无需手动激活虚拟环境)

```bash
# 直接使用 uv 运行，自动使用项目虚拟环境
uv run python tnn_reconstructed.py
```

#### 两种方式的区别:

- **python 方式**:

  - 需要手动激活虚拟环境 (`source .venv/bin/activate`)
  - 使用当前激活环境中的 Python 解释器和依赖
  - 传统的 Python 项目运行方式

- **uv run 方式** (推荐):
  - 无需手动激活虚拟环境，uv 自动管理
  - 自动使用项目的虚拟环境和依赖
  - 更简洁，避免环境切换的麻烦
  - 确保使用正确的项目依赖版本

## 项目结构

```
TNN/
├── tnn_reconstructed.py      # 主要的TNN实现文件
├── pyproject.toml            # 项目配置和依赖
├── uv.lock                   # 依赖锁定文件
└── README.md                 # 项目说明文档
```

## 核心实现

### 主要类和功能

1. **SubTensorNeuralNetwork**: 子张量神经网络，处理一维输入输出
2. **TensorNeuralNetwork**: 主要的 TNN 实现，支持高维张量分解
3. **TNNIntegrator**: 张量积分器，利用高斯积分实现高效数值积分

## 使用示例

### 1. 拉普拉斯特征值问题

```python
# 运行主文件中的拉普拉斯特征值问题示例
uv run python -c "
from tnn_reconstructed import laplacian_eigenvalue_problem
laplacian_eigenvalue_problem()
"
```

### 2. 混合导数特征值问题

```python
# 运行混合导数特征值问题示例
uv run python -c "
from tnn_reconstructed import mixed_derivative_eigenvalue_problem
mixed_derivative_eigenvalue_problem()
"
```

## 许可证

本项目采用开源许可证，具体许可证信息请查看项目根目录下的 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。在提交代码前，请确保:

1. 代码通过所有测试
2. 遵循项目的代码风格
3. 添加必要的文档和注释

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系我们。
