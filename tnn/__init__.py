"""
TNN (Tensor Neural Network) Package

张量神经网络包 - 基于张量分解的神经网络架构用于高维PDE求解
"""

from .core import (
    DEVICE,
    DefaultSubNet,
    SubTensorNeuralNetwork,
    TensorNeuralNetwork,
    TNNIntegrator,
)
from .trainer import TNNTrainer

__version__ = "0.1.0"
__author__ = "TNN Team"

__all__ = [
    "DefaultSubNet",
    "SubTensorNeuralNetwork",
    "TensorNeuralNetwork",
    "TNNIntegrator",
    "TNNTrainer",
    "DEVICE",
]
