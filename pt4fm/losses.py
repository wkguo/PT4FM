# Copyright 2026 The PT4FM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure, dependency-light loss/weighting primitives for PT4FM.

This module deliberately avoids importing ``rlinf`` or ``openpi`` so that the
math can be unit-tested in isolation (``pytest pt4fm/tests/test_losses.py``).
Everything here is a strict superset of RECAP: with the default/disabled
settings each function degenerates to the plain RECAP behaviour.

Three method primitives live here:

* :func:`awr_weight`     — AWR/AWAC style continuous advantage weighting.
* :func:`calql_calibrate`— Cal-QL style lower-bounding of the bootstrap value.
* :func:`expectile_weight` — IQL expectile re-weighting for the value model.

The torch-based helpers accept ``float`` tensors and never mutate inputs.
The numpy helpers are used in the offline (pre-)processing pipeline.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

__all__ = [
    "awr_weight",
    "normalize_weights",
    "masked_mean",
    "calql_calibrate",
    "advantage_to_weight_np",
    "expectile_weight",
]


# --------------------------------------------------------------------------- #
# AWR / AWAC continuous advantage weighting (policy side)
# --------------------------------------------------------------------------- #
def awr_weight(
    advantage: torch.Tensor,
    beta: float,
    w_max: float = 20.0,
    mask: Optional[torch.Tensor] = None,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Exponential advantage weight ``w = clip(exp((A - max A) / beta), 0, w_max)``.

    The ``- max A`` shift is purely for numerical stability and cancels out
    after the optional mean-1 normalization, so it does not change the relative
    weighting. ``beta`` is the AWR temperature: as ``beta -> inf`` (or
    non-positive / non-finite) the weights collapse to all-ones, recovering the
    un-weighted RECAP objective.

    Args:
        advantage: ``[B]`` continuous (calibrated) advantage values.
        beta: AWR temperature. ``<=0`` or non-finite disables weighting.
        w_max: Upper clip on the (pre-normalization) weight to bound variance.
        mask: Optional ``[B]`` boolean/float mask selecting the samples over
            which the weights are defined and normalized. Masked-out entries
            get weight ``0``.
        normalize: If True, rescale so the masked weights average to 1, which
            keeps the effective learning-rate of the weighted term comparable
            to the unweighted RECAP term.
        eps: Numerical floor for the normalization denominator.

    Returns:
        ``[B]`` float tensor of per-sample weights on ``advantage.device``.
    """
    a = advantage.detach().float()
    if mask is None:
        mask_f = torch.ones_like(a)
    else:
        mask_f = mask.detach().float()

    # Disabled / degenerate temperature -> uniform weights (RECAP behaviour).
    if beta is None or not math.isfinite(beta) or beta <= 0.0:
        w = torch.ones_like(a)
        return normalize_weights(w, mask_f, eps=eps) if normalize else w * mask_f

    # Stable exponential over the *active* set.
    if mask_f.sum() > 0:
        a_ref = a[mask_f > 0].max()
    else:
        a_ref = a.max()
    w = torch.exp((a - a_ref) / beta)
    w = torch.clamp(w, max=w_max)
    w = w * mask_f
    if normalize:
        w = normalize_weights(w, mask_f, eps=eps)
    return w


def normalize_weights(
    weights: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Rescale ``weights`` so they average to 1 over the active ``mask``."""
    mask_f = mask.float()
    n_active = mask_f.sum()
    total = (weights * mask_f).sum()
    if total <= eps or n_active <= 0:
        # Fall back to uniform over the active set.
        return mask_f
    return weights * mask_f * (n_active / (total + eps))


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean of ``values`` over the boolean/float ``mask`` (grad-safe)."""
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(eps)
    return (values * mask_f).sum() / denom


# --------------------------------------------------------------------------- #
# Cal-QL calibration (offline advantage pipeline, numpy)
# --------------------------------------------------------------------------- #
def calql_calibrate(
    value_next: np.ndarray,
    mc_return_next: Optional[np.ndarray],
    enabled: bool = True,
) -> np.ndarray:
    """Lower-bound the bootstrapped next-state value by the realized MC return.

    Mirrors Cal-QL's ``maximum(Q, mc_returns)`` calibration
    (``Cal-QL/JaxCQL/conservative_sac.py``). The Monte-Carlo return-to-go
    ``G_{t+N}`` is a *valid, on-policy* reference for the value at the look-ahead
    state, so taking ``max(V, G)`` prevents the learned value from
    under-estimating states that demonstrably led to high return, while leaving
    pessimistic (e.g. failure) bootstraps untouched. Sharpens the advantage
    signal on success-leading transitions and stabilizes mixed success/failure
    (rollout) data.

    Args:
        value_next: ``[B]`` learned ``V(o_{t+N})``.
        mc_return_next: ``[B]`` realized discounted return-to-go from
            ``o_{t+N}``; ``None`` disables calibration.
        enabled: Master switch.

    Returns:
        ``[B]`` calibrated next-state value.
    """
    if not enabled or mc_return_next is None:
        return value_next
    return np.maximum(value_next, mc_return_next)


def advantage_to_weight_np(
    advantage: np.ndarray,
    beta: float,
    w_max: float = 20.0,
    normalize: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Numpy twin of :func:`awr_weight`, used to bake an ``advantage_weight``
    column into the advantages parquet during stage-3.

    Storing the weight offline keeps the (cheap) computation out of the hot
    training loop and makes the AWR distribution inspectable/visualizable
    alongside the boolean advantage label.
    """
    a = advantage.astype(np.float64)
    if beta is None or not np.isfinite(beta) or beta <= 0.0:
        return np.ones_like(a)
    w = np.exp((a - a.max()) / beta)
    w = np.clip(w, a_min=0.0, a_max=w_max)
    if normalize:
        denom = w.mean()
        if denom > eps:
            w = w / denom
    return w


# --------------------------------------------------------------------------- #
# IQL expectile re-weighting (value model side)
# --------------------------------------------------------------------------- #
def expectile_weight(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """IQL expectile weight ``|tau - 1{diff < 0}|`` for ``diff = target - pred``.

    For ``tau > 0.5`` under-estimations (``target > pred``) are up-weighted,
    biasing the value toward the *best in-support* return rather than the
    behaviour-policy average — i.e. an optimistic-within-support value that
    yields a sharper advantage. ``tau == 0.5`` is symmetric and recovers the
    plain (un-weighted) regression.

    Returns a detached weight tensor to be multiplied onto a per-sample loss.
    """
    d = diff.detach()
    return torch.where(d < 0, torch.full_like(d, 1.0 - tau), torch.full_like(d, tau))
