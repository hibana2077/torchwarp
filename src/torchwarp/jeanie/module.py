"""``nn.Module`` front end for JEANIE on features."""

import torch.nn as nn

from .alignment import jeanie_dp_features


class JEANIE(nn.Module):
    """Joint tEmporal and cAmera viewpoiNt alIgnmEnt (JEANIE).

    Args:
        gamma: SoftMin temperature (> 0).
        max_shift: Viewpoint smoothness (iota-max shift). An int for one
            viewpoint axis, or a pair ``(azimuth, altitude)`` for two.
        metric: Base distance: "euclidean" (default), "sqeuclidean" or "rbf".
        backend: "auto" (CUDA kernel when possible), "cuda" or "torch".

    Forward:
        query: [B, K, T, D] (one viewpoint axis) or [B, K1, K2, T, D] (two
            axes) viewpoint-augmented temporal blocks.
        support: [B, U, D] temporal blocks.
        return_accumulator: also return the differentiable DP tensor
            [B, K, T, U] / [B, K1, K2, T, U].

    Returns:
        distance [B] (and the accumulator if requested).
    """

    def __init__(self, gamma=0.1, max_shift=1, metric="euclidean", backend="auto"):
        super().__init__()
        if isinstance(max_shift, int):
            max_shift = (max_shift, 0)
        self.gamma = float(gamma)
        self.max_shift = tuple(int(s) for s in max_shift)
        self.metric = metric
        self.backend = backend

    def forward(self, query, support, return_accumulator=False):
        one_axis = query.ndim == 4
        if one_axis:
            query = query.unsqueeze(2)
        elif query.ndim != 5:
            raise ValueError("query must be [B,K,T,D] or [B,K1,K2,T,D]")
        out, R = jeanie_dp_features(query, support, self.gamma, self.max_shift[0],
                                    self.max_shift[1], self.metric, self.backend,
                                    return_accumulator)
        if not return_accumulator:
            return out
        return out, (R[:, :, 0] if one_axis else R)

    def extra_repr(self):
        return "gamma={}, max_shift={}, metric={!r}, backend={!r}".format(
            self.gamma, self.max_shift, self.metric, self.backend)
