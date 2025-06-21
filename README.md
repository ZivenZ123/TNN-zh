# TNN (Tensor Neural Network)

Tensor Neural Network implementation and examples

## 安装 uv

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

# 或者如果你想安装开发依赖
uv sync --group dev
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

# 运行示例代码
python ex_5_1/dim5/ex_5_1_dim5.py
```

#### 方式二: 使用 uv run (推荐，无需手动激活虚拟环境)

```bash
# 直接使用 uv 运行，自动使用项目虚拟环境
uv run ex_5_1/dim5/ex_5_1_dim5.py
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
├── ex_5_1/
│   └── dim5/
│       ├── ex_5_1_dim5.py    # 主要示例代码
│       ├── integration.py     # 积分相关实现
│       ├── quadrature.py      # 数值积分方法
│       └── tnn.py            # 张量神经网络实现
├── pyproject.toml            # 项目配置和依赖
└── README.md                 # 项目说明文档
```

## 依赖说明

项目主要依赖:

- **numpy**: 数值计算库
- **torch**: PyTorch 深度学习框架
- **matplotlib**: 数据可视化
- **jupyter**: Jupyter Notebook 支持

开发依赖:

- **ruff**: Python 代码格式化和检查工具

## 常用 uv 命令

```bash
# 添加新的依赖
uv add package-name

# 添加开发依赖
uv add --group dev package-name

# 移除依赖
uv remove package-name

# 更新所有依赖
uv sync --upgrade

# 运行 Python 脚本
uv run python script.py

# 安装特定版本的 Python
uv python install 3.11
```

## 故障排除

如果遇到问题，可以尝试:

1. 清理缓存和重新安装:

```bash
uv cache clean
uv sync --reinstall
```

2. 检查 Python 版本兼容性:

```bash
uv python list
```

3. 查看详细错误信息:

```bash
uv sync --verbose
```
