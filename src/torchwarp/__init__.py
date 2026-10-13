"""torchwarp: accelerated PyTorch implementation of uDTW and JEANIE with optimized CUDA kernels.

uncertainty-DTW (uDTW, ECCV 2022) and JEANIE (IJCV 2024) were proposed by
Lei Wang, Piotr Koniusz et al. The official code is at github.com/LeiWangR.
This package also provides soft-DTW and Free Viewpoint Matching, with
CUDA/HIP kernels and a portable PyTorch backend. Functions and classes that
exist in the official code keep its names and positional arguments;
options added by torchwarp are keyword-only.
"""

from . import paths, plot
from .jeanie import (
    JEANIE,
    euclidean_cost,
    fvm_1d_from_cost,
    fvm_2d_from_cost,
    fvm_from_cost,
    fvm_query_only_1d,
    fvm_query_only_2d,
    jeanie_1d_from_cost,
    jeanie_1d_from_features,
    jeanie_2d_from_cost,
    jeanie_2d_from_features,
    rbf_cost,
    squared_euclidean_cost,
)
from .sdtw import SoftDTW, soft_dtw, softmin, softmin_weights
from .udtw import (
    UDTW,
    pairwise_matrices,
    uDTW,
    udtw_from_cost,
    udtw_from_features,
)

__version__ = "0.1.0"

__all__ = [
    # uDTW
    "UDTW",
    "uDTW",
    "udtw_from_cost",
    "udtw_from_features",
    "pairwise_matrices",
    # JEANIE
    "JEANIE",
    "jeanie_1d_from_cost",
    "jeanie_1d_from_features",
    "jeanie_2d_from_cost",
    "jeanie_2d_from_features",
    # soft-DTW / FVM / distances
    "SoftDTW",
    "soft_dtw",
    "softmin",
    "softmin_weights",
    "fvm_from_cost",
    "fvm_1d_from_cost",
    "fvm_2d_from_cost",
    "fvm_query_only_1d",
    "fvm_query_only_2d",
    "euclidean_cost",
    "squared_euclidean_cost",
    "rbf_cost",
    # modules
    "paths",
    "plot",
]
