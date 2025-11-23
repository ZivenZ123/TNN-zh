"""
TNN (Tensor Neural Network) Package
"""

from .core import (
    TNN,
    SeparableDimNetwork,
    SeparableDimNetworkGELU,
    ThetaModule,
    apply_dirichlet_bd,
    wrap_1d_func_as_tnn,
)
from .integration import (
    int_tnn,
    int_tnn_product,
    int_tnn_L2,
    generate_quad_points,
    GaussLegendre,
)

__version__ = "0.1.0"
__author__ = "TNN Team"

__all__ = [
    "ThetaModule",
    "TNN",
    "SeparableDimNetwork",
    "SeparableDimNetworkGELU",
    "apply_dirichlet_bd",
    "wrap_1d_func_as_tnn",
    "GaussLegendre",
]
