"""
通用训练器 - 封装各种优化器和训练流程

本模块提供了用于训练TensorNeuralNetwork的通用训练器类.
"""

import time
from collections.abc import Callable

import torch
import torch.optim as optim

from .core import TensorNeuralNetwork


class TNNTrainer:
    """
    张量神经网络通用训练器

    这个类封装了完整的TNN训练流程, 支持多种优化器和多阶段训练策略.
    适用于各种基于TNN的PDE求解问题.

    主要特性:
    - 支持多种优化器: Adam, LBFGS, SGD等
    - 多阶段训练策略
    - 自动训练历史记录和分析
    - 灵活的损失函数接口
    - 详细的训练过程监控
    """

    def __init__(
        self,
        tnn: TensorNeuralNetwork,
        loss_fn: Callable[[], torch.Tensor],
        verbose: bool = True,
    ):
        """
        初始化训练器

        Args:
            tnn: 待训练的TensorNeuralNetwork实例
            loss_fn: 损失函数, 应该返回一个标量tensor
            verbose: 是否输出详细训练信息
        """
        self.tnn = tnn
        self.loss_fn = loss_fn
        self.verbose = verbose

        # 训练历史
        self.losses = []
        self.training_time = 0.0
        self.current_optimizer = None

        # 阶段记录
        self.phase_history = []

    def adam_phase(
        self,
        lr: float,
        epochs: int,
        phase_name: str = "Adam",
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        """
        Adam优化器训练阶段

        Args:
            lr: 学习率
            epochs: 训练轮数
            phase_name: 阶段名称
            weight_decay: 权重衰减
            betas: Adam的beta参数
            eps: Adam的epsilon参数
        """
        if self.verbose:
            print(
                f"\n>>> {phase_name} 阶段 (学习率: {lr}, 轮数: {epochs}) <<<"
            )

        optimizer = optim.Adam(
            self.tnn.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        self.current_optimizer = optimizer

        phase_start_idx = len(self.losses)
        phase_start_time = time.time()

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            self.losses.append(loss_value)

            if self.verbose:
                print(f"Epoch {len(self.losses) - 1}, Loss: {loss_value:.6f}")

        phase_end_time = time.time()
        self.phase_history.append(
            {
                "name": phase_name,
                "optimizer": "Adam",
                "epochs": epochs,
                "lr": lr,
                "start_idx": phase_start_idx,
                "end_idx": len(self.losses) - 1,
                "start_loss": self.losses[phase_start_idx]
                if phase_start_idx < len(self.losses)
                else None,
                "end_loss": self.losses[-1],
                "time": phase_end_time - phase_start_time,
            }
        )

    def lbfgs_phase(
        self,
        lr: float,
        epochs: int,
        phase_name: str = "LBFGS",
        max_iter: int = 20,
        max_eval: int | None = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
    ):
        """
        LBFGS优化器训练阶段

        Args:
            lr: 学习率
            epochs: 训练轮数
            phase_name: 阶段名称
            max_iter: 每步最大迭代次数
            max_eval: 最大函数评估次数
            tolerance_grad: 梯度容忍度
            tolerance_change: 变化容忍度
            history_size: 历史大小
        """
        if self.verbose:
            print(
                f"\n>>> {phase_name} 阶段 (学习率: {lr}, 轮数: {epochs}) <<<"
            )

        optimizer = optim.LBFGS(
            self.tnn.parameters(),
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
        )
        self.current_optimizer = optimizer

        phase_start_idx = len(self.losses)
        phase_start_time = time.time()

        for _ in range(epochs):

            def closure():
                optimizer.zero_grad()
                loss = self.loss_fn()
                loss.backward()
                return loss.item()

            loss = optimizer.step(closure)
            loss_value = (
                loss.item() if isinstance(loss, torch.Tensor) else loss
            )
            self.losses.append(loss_value)

            if self.verbose:
                print(f"Epoch {len(self.losses) - 1}, Loss: {loss_value:.8f}")

        phase_end_time = time.time()
        self.phase_history.append(
            {
                "name": phase_name,
                "optimizer": "LBFGS",
                "epochs": epochs,
                "lr": lr,
                "start_idx": phase_start_idx,
                "end_idx": len(self.losses) - 1,
                "start_loss": self.losses[phase_start_idx]
                if phase_start_idx < len(self.losses)
                else None,
                "end_loss": self.losses[-1],
                "time": phase_end_time - phase_start_time,
            }
        )

    def sgd_phase(
        self,
        lr: float,
        epochs: int,
        phase_name: str = "SGD",
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        """
        SGD优化器训练阶段

        Args:
            lr: 学习率
            epochs: 训练轮数
            phase_name: 阶段名称
            momentum: 动量
            weight_decay: 权重衰减
            dampening: 动量阻尼
            nesterov: 是否使用Nesterov动量
        """
        if self.verbose:
            print(
                f"\n>>> {phase_name} 阶段 (学习率: {lr}, 轮数: {epochs}) <<<"
            )

        optimizer = optim.SGD(
            self.tnn.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
        self.current_optimizer = optimizer

        phase_start_idx = len(self.losses)
        phase_start_time = time.time()

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            self.losses.append(loss_value)

            if self.verbose:
                print(f"Epoch {len(self.losses) - 1}, Loss: {loss_value:.6f}")

        phase_end_time = time.time()
        self.phase_history.append(
            {
                "name": phase_name,
                "optimizer": "SGD",
                "epochs": epochs,
                "lr": lr,
                "start_idx": phase_start_idx,
                "end_idx": len(self.losses) - 1,
                "start_loss": self.losses[phase_start_idx]
                if phase_start_idx < len(self.losses)
                else None,
                "end_loss": self.losses[-1],
                "time": phase_end_time - phase_start_time,
            }
        )

    def multi_phase(
        self,
        phases: list[dict],
    ) -> tuple[list[float], float]:
        """
        执行多阶段训练

        Args:
            phases: 训练阶段配置列表, 每个元素是一个字典, 包含:
                - 'type': 优化器类型 ('adam', 'lbfgs', 'sgd')
                - 'lr': 学习率
                - 'epochs': 训练轮数
                - 'name': 阶段名称 (可选)
                - 其他优化器相关参数

        Returns:
            Tuple[List[float], float]: (损失历史, 训练时间)
        """
        if self.verbose:
            print("\n>>> 开始多阶段训练 <<<")
            print(
                f"TNN参数总数: {sum(p.numel() for p in self.tnn.parameters())}"
            )
            print(f"训练阶段数: {len(phases)}")

        start_time = time.time()

        # 清空历史记录
        self.losses = []
        self.phase_history = []

        for i, phase_config in enumerate(phases):
            phase_type = phase_config["type"].lower()
            phase_name = phase_config.get("name", f"阶段{i + 1}")

            if phase_type == "adam":
                self.adam_phase(
                    lr=phase_config["lr"],
                    epochs=phase_config["epochs"],
                    phase_name=phase_name,
                    weight_decay=phase_config.get("weight_decay", 0.0),
                    betas=phase_config.get("betas", (0.9, 0.999)),
                    eps=phase_config.get("eps", 1e-8),
                )
            elif phase_type == "lbfgs":
                self.lbfgs_phase(
                    lr=phase_config["lr"],
                    epochs=phase_config["epochs"],
                    phase_name=phase_name,
                    max_iter=phase_config.get("max_iter", 20),
                    max_eval=phase_config.get("max_eval", None),
                    tolerance_grad=phase_config.get("tolerance_grad", 1e-7),
                    tolerance_change=phase_config.get(
                        "tolerance_change", 1e-9
                    ),
                    history_size=phase_config.get("history_size", 100),
                )
            elif phase_type == "sgd":
                self.sgd_phase(
                    lr=phase_config["lr"],
                    epochs=phase_config["epochs"],
                    phase_name=phase_name,
                    momentum=phase_config.get("momentum", 0.0),
                    weight_decay=phase_config.get("weight_decay", 0.0),
                    dampening=phase_config.get("dampening", 0.0),
                    nesterov=phase_config.get("nesterov", False),
                )
            else:
                raise ValueError(f"不支持的优化器类型: {phase_type}")

        end_time = time.time()
        self.training_time = end_time - start_time

        if self.verbose:
            self.print_summary()

        return self.losses, self.training_time

    def train_simple(
        self,
        optimizer_type: str = "adam",
        lr: float = 0.001,
        epochs: int = 100,
        **kwargs,
    ) -> tuple[list[float], float]:
        """
        简单的单阶段训练

        Args:
            optimizer_type: 优化器类型
            lr: 学习率
            epochs: 训练轮数
            **kwargs: 其他优化器参数

        Returns:
            Tuple[List[float], float]: (损失历史, 训练时间)
        """
        phase_config = {
            "type": optimizer_type,
            "lr": lr,
            "epochs": epochs,
            "name": f"{optimizer_type.upper()} 训练",
            **kwargs,
        }

        return self.multi_phase([phase_config])

    def print_summary(self):
        """打印训练结果摘要"""
        if not self.losses:
            print("没有训练历史记录")
            return

        print(f"\n{'=' * 50}")
        print("训练完成!")
        print(f"初始损失: {self.losses[0]:.8f}")
        print(f"最终损失: {self.losses[-1]:.8f}")
        print(f"总训练轮数: {len(self.losses)}")
        print(f"总训练时间: {self.training_time:.2f} 秒")
        print(f"平均每轮时间: {self.training_time / len(self.losses):.3f} 秒")

    def get_final_loss(self) -> float | None:
        """获取最终的损失值"""
        return self.losses[-1] if self.losses else None

    def get_loss_history(self) -> list[float]:
        """获取损失历史"""
        return self.losses.copy()

    def get_phase_history(self) -> list[dict]:
        """获取阶段历史"""
        return self.phase_history.copy()

    def reset_history(self):
        """重置训练历史"""
        self.losses = []
        self.phase_history = []
        self.training_time = 0.0
