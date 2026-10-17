"""JEANIE (IJCV 2024) and Free Viewpoint Matching."""

from .distances import euclidean_cost, rbf_cost, squared_euclidean_cost
from .functional import jeanie, jeanie_from_cost
from .fvm import fvm, fvm_from_cost
from .module import FVM, JEANIE

__all__ = [
    "JEANIE",
    "FVM",
    "jeanie",
    "jeanie_from_cost",
    "fvm",
    "fvm_from_cost",
    "euclidean_cost",
    "squared_euclidean_cost",
    "rbf_cost",
]
