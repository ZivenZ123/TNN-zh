"""
通用训练器 - 封装各种优化器和训练流程

本模块提供了用于训练TNN的通用训练器类.
"""

import time
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim


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
        tnn: nn.Module,
        loss_fn: Callable[[], torch.Tensor | tuple],
        verbose: bool = True,
        print_interval: int = 100,
    ):
        """
        初始化训练器

        Args:
            tnn: 待训练的TNN实例
            loss_fn: 损失函数, 应该返回一个标量tensor或包含多个损失分量的元组
            verbose: 是否输出详细训练信息
            print_interval: 打印间隔(每隔多少个epoch打印一次), 默认100
        """
        self.tnn = tnn
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.print_interval = print_interval

        # 训练历史
        self.losses = []
        self.training_time = 0.0
        self.current_optimizer = None

        # 阶段记录
        self.phase_history = []

    def _train_loop(self, optimizer, epochs, grad_clip=None):
        """通用训练循环"""
        interval_start_time = time.time()

        for _ in range(epochs):
            optimizer.zero_grad()
            loss_result = self.loss_fn()

            # 处理单个或多个loss
            loss = (
                loss_result[0]
                if isinstance(loss_result, tuple)
                else loss_result
            )
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.tnn.parameters(), grad_clip
                )

            optimizer.step()
            loss_value = loss.item()
            self.losses.append(loss_value)

            # 每隔print_interval打印一次
            if self.verbose and len(self.losses) % self.print_interval == 0:
                interval_time = time.time() - interval_start_time
                loss_str = f"Epoch {len(self.losses)}, Loss: {loss_value:.8f}, Time: {interval_time:.2f}s"

                # 如果有多个loss分量,打印它们
                if isinstance(loss_result, tuple) and len(loss_result) > 1:
                    for i, comp in enumerate(loss_result[1:]):
                        name = (
                            ["PDE", "BC1", "BC2", "BC3", "BC4"][i]
                            if i < 5
                            else f"C{i + 1}"
                        )
                        loss_str += f", {name}: {comp.item():.8f}"

                print(loss_str)
                interval_start_time = time.time()

    def adam_phase(
        self,
        lr: float,
        epochs: int,
        phase_name: str = "Adam",
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        grad_clip: float | None = None,
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
            grad_clip: 梯度裁剪阈值, None表示不裁剪
        """
        if self.verbose:
            print(
                f"\n>>> {phase_name} 阶段 (学习率: {lr}, 轮数: {epochs}) <<<"
            )
            if grad_clip is not None:
                print(f"梯度裁剪: {grad_clip}")

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

        self._train_loop(optimizer, epochs, grad_clip)

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
        interval_start_time = time.time()

        for _ in range(epochs):
            loss_components = {"components": []}

            def closure(components_dict=loss_components):
                optimizer.zero_grad()
                loss_result = self.loss_fn()
                loss = (
                    loss_result[0]
                    if isinstance(loss_result, tuple)
                    else loss_result
                )
                components_dict["components"] = (
                    list(loss_result)
                    if isinstance(loss_result, tuple)
                    else [loss]
                )
                loss.backward()
                return loss.item()

            loss = optimizer.step(closure)
            loss_value = (
                loss.item() if isinstance(loss, torch.Tensor) else loss
            )
            self.losses.append(loss_value)

            if self.verbose and len(self.losses) % self.print_interval == 0:
                interval_time = time.time() - interval_start_time
                loss_str = f"Epoch {len(self.losses)}, Loss: {loss_value:.8f}, Time: {interval_time:.2f}s"

                components = loss_components["components"]
                if len(components) > 1:
                    for i, comp in enumerate(components[1:]):
                        name = (
                            ["PDE", "BC1", "BC2", "BC3", "BC4"][i]
                            if i < 5
                            else f"C{i + 1}"
                        )
                        val = (
                            comp.item()
                            if isinstance(comp, torch.Tensor)
                            else comp
                        )
                        loss_str += f", {name}: {val:.8f}"

                print(loss_str)
                interval_start_time = time.time()

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
        grad_clip: float | None = None,
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
            grad_clip: 梯度裁剪阈值, None表示不裁剪
        """
        if self.verbose:
            print(
                f"\n>>> {phase_name} 阶段 (学习率: {lr}, 轮数: {epochs}) <<<"
            )
            if grad_clip is not None:
                print(f"梯度裁剪: {grad_clip}")

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

        self._train_loop(optimizer, epochs, grad_clip)

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
                    grad_clip=phase_config.get("grad_clip", None),
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
                    grad_clip=phase_config.get("grad_clip", None),
                )
            else:
                raise ValueError(f"不支持的优化器类型: {phase_type}")

        end_time = time.time()
        self.training_time = end_time - start_time

        if self.verbose:
            self.print_summary()

        return self.losses, self.training_time

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
