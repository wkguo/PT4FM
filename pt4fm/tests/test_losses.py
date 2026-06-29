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

"""Unit tests for the pure PT4FM loss primitives.

Run with: ``pytest PT4FM/pt4fm/tests/test_losses.py`` (only needs numpy+torch).
"""

import math

import numpy as np
import torch

from pt4fm.losses import (
    advantage_to_weight_np,
    awr_weight,
    calql_calibrate,
    expectile_weight,
    masked_mean,
    normalize_weights,
)


def test_awr_disabled_is_uniform():
    a = torch.tensor([-2.0, 0.0, 1.0, 5.0])
    for beta in (0.0, -1.0, math.inf, float("nan")):
        w = awr_weight(a, beta=beta)
        assert torch.allclose(w, torch.ones_like(a)), f"beta={beta} should be uniform"


def test_awr_mean_one_and_monotonic():
    a = torch.tensor([-2.0, 0.0, 1.0, 3.0])
    w = awr_weight(a, beta=1.0, w_max=1e9, normalize=True)
    # Normalized to mean 1.
    assert abs(w.mean().item() - 1.0) < 1e-5
    # Higher advantage -> higher weight (monotonic).
    assert torch.all(w[1:] >= w[:-1])


def test_awr_w_max_clip():
    a = torch.tensor([0.0, 100.0])
    w = awr_weight(a, beta=1.0, w_max=3.0, normalize=False)
    assert w.max().item() <= 3.0 + 1e-6


def test_awr_mask_restricts_support():
    a = torch.tensor([10.0, 0.0, 0.0])
    mask = torch.tensor([0.0, 1.0, 1.0])
    w = awr_weight(a, beta=1.0, mask=mask, normalize=True)
    assert w[0].item() == 0.0
    # mean over the active set is 1.
    assert abs((w[1] + w[2]).item() / 2.0 - 1.0) < 1e-5


def test_normalize_weights_all_zero_falls_back_uniform():
    w = torch.zeros(4)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    out = normalize_weights(w, mask)
    assert torch.allclose(out, mask)


def test_masked_mean():
    v = torch.tensor([1.0, 2.0, 100.0])
    m = torch.tensor([1.0, 1.0, 0.0])
    assert abs(masked_mean(v, m).item() - 1.5) < 1e-6


def test_calql_lower_bounds_and_monotonic():
    v = np.array([-0.8, -0.5, -0.2])
    g = np.array([-0.3, -0.9, -0.1])  # MC return reference
    out = calql_calibrate(v, g, enabled=True)
    assert np.all(out >= v - 1e-9)
    assert np.all(out >= g - 1e-9)
    assert np.allclose(out, np.maximum(v, g))
    # Disabled is a no-op.
    assert np.allclose(calql_calibrate(v, g, enabled=False), v)
    assert np.allclose(calql_calibrate(v, None, enabled=True), v)


def test_advantage_to_weight_np_matches_torch():
    a = np.array([-1.0, 0.0, 0.5, 2.0])
    w_np = advantage_to_weight_np(a, beta=0.7, w_max=1e9, normalize=True)
    w_t = awr_weight(torch.tensor(a), beta=0.7, w_max=1e9, normalize=True).numpy()
    assert np.allclose(w_np, w_t, atol=1e-5)


def test_expectile_symmetric_at_half():
    diff = torch.tensor([-1.0, 1.0, -2.0, 3.0])
    w = expectile_weight(diff, tau=0.5)
    assert torch.allclose(w, torch.full_like(diff, 0.5))


def test_expectile_upweights_underestimation():
    # diff = target - pred. tau>0.5 -> underestimation (diff>0) weighted more.
    diff = torch.tensor([1.0, -1.0])
    w = expectile_weight(diff, tau=0.9)
    assert abs(w[0].item() - 0.9) < 1e-6  # target>pred
    assert abs(w[1].item() - 0.1) < 1e-6  # target<pred


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
