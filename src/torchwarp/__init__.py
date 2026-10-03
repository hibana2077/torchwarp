"""torchwarp: unofficial PyTorch/CUDA reimplementation of uDTW and JEANIE.

uncertainty-DTW (uDTW, ECCV 2022) and JEANIE (IJCV 2024) were proposed by
Lei Wang, Piotr Koniusz et al. The official code is at github.com/LeiWangR.
This package also provides soft-DTW and Free Viewpoint Matching, with
CUDA/HIP kernels and a portable PyTorch backend.
"""

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
    jeanie_dp,
    jeanie_dp_features,
    rbf_cost,
    soft_dtw,
    softmin,
    squared_euclidean_cost,
)
from .udtw import (
    pairwise_matrices,
    squared_euclidean,
    uDTW,
    udtw_from_features,
    udtw_from_matrices,
)

from . import paths

__version__ = "0.1.0"

__all__ = [
    # uDTW
    "uDTW",
    "udtw_from_features",
    "udtw_from_matrices",
    "pairwise_matrices",
    "squared_euclidean",
    # JEANIE
    "JEANIE",
    "jeanie_1d_from_cost",
    "jeanie_1d_from_features",
    "jeanie_2d_from_cost",
    "jeanie_2d_from_features",
    "jeanie_dp",
    "jeanie_dp_features",
    # soft-DTW / FVM / distances
    "soft_dtw",
    "softmin",
    "fvm_from_cost",
    "fvm_1d_from_cost",
    "fvm_2d_from_cost",
    "fvm_query_only_1d",
    "fvm_query_only_2d",
    "euclidean_cost",
    "squared_euclidean_cost",
    "rbf_cost",
]
