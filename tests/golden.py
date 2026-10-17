"""Reference values of the official uDTW and JEANIE code, for the tests.

``golden/reference.pt`` holds, for every test case, the inputs and the
outputs of the official implementation (float64). The tests compare
torchwarp with these values and do not need the official code. To
regenerate the file (downloads the official code once):

    python tests/golden/generate.py

The case lists below are shared by the tests and the generator.
"""

from pathlib import Path

import torch

PATH = Path(__file__).resolve().parent / "golden" / "reference.pt"

# uDTW
UDTW_SHAPES = [(1, 1), (1, 6), (6, 1), (5, 8), (8, 5), (9, 9)]
GAMMAS = [0.01, 0.1, 1.0]
UDTW_BANDS = [(9, 9, 1), (9, 9, 2), (6, 10, 4), (10, 6, 4.5), (7, 7, 0)]
UDTW_BETAS = [0.0, 1.0, 3.0]
UDTW_MAML = [(False, None), (True, None), (False, 2)]

# JEANIE / soft-DTW / FVM
SDTW_SHAPES = [(1, 1), (1, 6), (6, 1), (5, 7), (8, 4), (9, 9)]
JEANIE_1D_SHAPES = [(1, 4, 5), (5, 6, 7), (5, 8, 5), (3, 1, 4)]
JEANIE_1D_SHIFTS = [0, 1, 2, 7]
JEANIE_2D_SHAPES = [(3, 3, 5, 6), (2, 4, 6, 4), (1, 3, 4, 4)]
JEANIE_2D_SHIFTS = [(0, 0), (1, 1), (1, 0), (2, 1)]
# every official FVM variant is TF.fvm_from_cost on the batched cost
FVM_CASES = [
    ("fvm_query_only_1d", (4, 5, 6)),
    ("fvm_query_only_2d", (3, 3, 5, 6)),
    ("fvm_1d_from_cost", (3, 4, 6, 5)),
    ("fvm_2d_from_cost", (2, 2, 3, 2, 5, 6)),
]
DISTANCES = ["euclidean_cost", "squared_euclidean_cost", "rbf_cost"]
METRICS = ["euclidean", "sqeuclidean", "rbf"]
FEATURES_1D = [(5, 6, 8, 1), (3, 9, 4, 2), (1, 5, 5, 0)]
ACCUMULATOR_COST = [(4, 5, 7, 1), (3, 7, 4, 2), (1, 4, 4, 0)]
# second-order gradients of the distance alone (no accumulator)
SECOND_ORDER = ["jeanie_cost", "jeanie_features", "fvm", "soft_dtw"]


def key(name, /, **params):
    return name + "".join("|{}={!r}".format(k, params[k]) for k in sorted(params))


_DATA = None


def golden(name, /, **params):
    """The stored inputs and official outputs of one case, as a dict of tensors."""
    global _DATA
    if _DATA is None:
        _DATA = torch.load(PATH, weights_only=True)
    k = key(name, **params)
    if k not in _DATA:
        raise KeyError("no reference values for {!r}; run python tests/golden/generate.py".format(k))
    return _DATA[k]
