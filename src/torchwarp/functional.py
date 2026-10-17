"""Functional interface: one function per module, plus ``*_from_cost``
variants that take a precomputed cost tensor.

    import torchwarp.functional as TF
"""

from .jeanie.distances import euclidean_cost, rbf_cost, squared_euclidean_cost
from .jeanie.functional import jeanie, jeanie_from_cost
from .jeanie.fvm import fvm, fvm_from_cost
from .sdtw import soft_dtw, soft_dtw_from_cost, softmin, softmin_weights
from .udtw.functional import UDTWOutput, pairwise_matrices, udtw, udtw_from_cost

__all__ = [
    "udtw",
    "udtw_from_cost",
    "pairwise_matrices",
    "UDTWOutput",
    "soft_dtw",
    "soft_dtw_from_cost",
    "jeanie",
    "jeanie_from_cost",
    "fvm",
    "fvm_from_cost",
    "softmin",
    "softmin_weights",
    "euclidean_cost",
    "squared_euclidean_cost",
    "rbf_cost",
]
