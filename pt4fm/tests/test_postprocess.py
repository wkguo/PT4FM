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

"""Unit tests for the PT4FM stage-3 post-process DataFrame core (numpy+pandas)."""

import numpy as np
import pandas as pd

from pt4fm.process.postprocess_advantages import (
    assign_pt4fm_columns,
    calql_calibrate_df,
    compute_lookahead_return,
    normalize_return,
)


def _toy_df():
    # One episode, frames 0..4. return-to-go decreasing; value_next pessimistic.
    return pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 0, 0],
            "frame_index": [0, 1, 2, 3, 4],
            "advantage_continuous": [0.1, 0.0, -0.1, -0.2, -0.3],
            "value_next": [-0.9, -0.8, -0.7, -0.6, -0.5],
            "return": [-4.0, -3.0, -2.0, -1.0, 0.0],
        }
    )


def test_normalize_return_maps_to_minus_one_zero():
    g = np.array([-4.0, 0.0])
    out = normalize_return(g, ret_min=-4.0, ret_max=0.0)
    assert np.isclose(out[0], -1.0)
    assert np.isclose(out[1], 0.0)


def test_lookahead_return_shift_and_episode_boundary():
    df = _toy_df()
    g = compute_lookahead_return(df, lookahead=2)
    # frame 0 -> return at frame 2 = -2.0; frame 3,4 -> past end -> NaN
    assert g[0] == -2.0
    assert g[1] == -1.0
    assert np.isnan(g[3]) and np.isnan(g[4])


def test_calql_only_raises_or_keeps_advantage():
    df = _toy_df()
    a0 = df["advantage_continuous"].to_numpy()
    a_cal = calql_calibrate_df(df, lookahead=2, gamma=1.0, ret_min=-4.0, ret_max=0.0)
    # Calibration uses max(V_next, G_next) >= V_next, so A' >= A elementwise
    # wherever a valid lookahead exists; episode-end rows are unchanged.
    assert np.all(a_cal >= a0 - 1e-9)
    assert np.isclose(a_cal[3], a0[3]) and np.isclose(a_cal[4], a0[4])


def test_assign_columns_sft_all_true_and_weight_present():
    df = _toy_df()
    out = assign_pt4fm_columns(
        df, dataset_type="sft",
        advantage_used=df["advantage_continuous"].to_numpy(),
        bool_threshold=0.0, awr_beta=0.7, awr_wmax=20.0,
    )
    assert out["is_demo"].all()
    assert out["advantage"].all()  # sft forced True regardless of threshold
    assert "advantage_weight" in out.columns
    assert (out["advantage_weight"] >= 0).all()


def test_assign_columns_rollout_thresholds():
    df = _toy_df()
    a = df["advantage_continuous"].to_numpy()
    out = assign_pt4fm_columns(
        df, dataset_type="rollout", advantage_used=a,
        bool_threshold=0.0, awr_beta=0.7, awr_wmax=20.0,
    )
    assert not out["is_demo"].any()
    np.testing.assert_array_equal(out["advantage"].to_numpy(), a >= 0.0)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
