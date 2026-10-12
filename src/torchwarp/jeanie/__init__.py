"""JEANIE (IJCV 2024) and Free Viewpoint Matching."""

from .distances import euclidean_cost, rbf_cost, squared_euclidean_cost
from .functional import (
    jeanie_1d_from_cost,
    jeanie_1d_from_features,
    jeanie_2d_from_cost,
    jeanie_2d_from_features,
)
from .fvm import (
    fvm_1d_from_cost,
    fvm_2d_from_cost,
    fvm_from_cost,
    fvm_query_only_1d,
    fvm_query_only_2d,
)
from .module import JEANIE

__all__ = [
    "JEANIE",
    "jeanie_1d_from_cost",
    "jeanie_1d_from_features",
    "jeanie_2d_from_cost",
    "jeanie_2d_from_features",
    "fvm_from_cost",
    "fvm_1d_from_cost",
    "fvm_2d_from_cost",
    "fvm_query_only_1d",
    "fvm_query_only_2d",
    "euclidean_cost",
    "squared_euclidean_cost",
    "rbf_cost",
]
