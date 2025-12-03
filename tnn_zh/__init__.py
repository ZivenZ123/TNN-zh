"""
TNN (Tensor Neural Network) Package
"""

from .core import (
    TNN,
    SeparableDimNetwork,
    SeparableDimNetworkGELU,
    SeparableDimNetworkSin,
    ThetaModule,
    apply_dirichlet_bd,
    wrap_1d_func_as_tnn,
)
from .integration import (
    GaussLegendre,
    generate_quad_points,
    int_tnn,
    int_tnn_product,
    l2_norm,
)

__version__ = "0.1.0"
__author__ = "TNN Team"

__all__ = [
    "ThetaModule",
    "TNN",
    "SeparableDimNetwork",
    "SeparableDimNetworkGELU",
    "SeparableDimNetworkSin",
    "apply_dirichlet_bd",
    "wrap_1d_func_as_tnn",
    "int_tnn",
    "int_tnn_product",
    "l2_norm",
    "generate_quad_points",
    "GaussLegendre",
]
