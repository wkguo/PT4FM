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

"""PT4FM CFG dataset/dataloader: carry advantage_weight + is_demo end-to-end.

These subclass RLinf's RECAP equivalents and add two per-sample fields needed
by :class:`pt4fm.models.cfg_action_model.PT4FMCfgActionModel`:

* ``advantage_weight`` — continuous AWR weight (from stage-3 parquet),
* ``is_demo``          — whether the sample comes from an ``sft`` dataset
                          (used to scope the SFT auxiliary anchor).

They are packed into a single ``[B, 3]`` ``advantage`` tensor by the dataloader
so RLinf's CFG worker (`run_training`) — which forwards an opaque ``advantage``
tensor — needs no modification.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import torch

from pt4fm.compat import install_lerobot_common_shim

install_lerobot_common_shim()  # must precede any rlinf.recap import

from rlinf.data.datasets.recap.cfg_model import (
    AdvantagePreservingDataset,
    CFGDataLoaderImpl,
)
from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
    Observation as CFGObservation,
)

logger = logging.getLogger(__name__)


def _coerce_record(value: Any) -> tuple[bool, float, bool]:
    """Normalize a lookup value into ``(advantage, weight, is_demo)``.

    Accepts either RLinf's plain ``bool`` value or a PT4FM ``dict``/tuple.
    """
    if isinstance(value, dict):
        return (
            bool(value.get("advantage", False)),
            float(value.get("advantage_weight", 1.0)),
            bool(value.get("is_demo", True)),
        )
    if isinstance(value, (tuple, list)):
        adv = bool(value[0])
        weight = float(value[1]) if len(value) > 1 else 1.0
        is_demo = bool(value[2]) if len(value) > 2 else True
        return adv, weight, is_demo
    # Plain bool (RLinf format): default weight 1, treat as demo.
    return bool(value), 1.0, True


class PT4FMAdvantagePreservingDataset(AdvantagePreservingDataset):
    """Like RECAP's wrapper, but also restores advantage_weight + is_demo."""

    def __init__(
        self,
        base_dataset: Any,
        transformed_dataset: Any,
        advantages_lookup: dict | None = None,
        is_demo: bool = True,
    ):
        self._default_is_demo = is_demo
        self._record_by_index: dict[int, tuple[bool, float, bool]] | None = None
        super().__init__(base_dataset, transformed_dataset, advantages_lookup)

    def _build_advantage_index(self, base_dataset, advantages_lookup):
        # Build the parent's bool index for compatibility, then a richer index.
        hf_dataset = self._get_hf_dataset(base_dataset)
        if hf_dataset is None or advantages_lookup is None:
            # Fall back to the parent path (per-sample / column based).
            return super()._build_advantage_index(base_dataset, advantages_lookup)

        ep_indices = hf_dataset["episode_index"]
        frame_indices = hf_dataset["frame_index"]
        bool_index: dict[int, bool] = {}
        record_index: dict[int, tuple[bool, float, bool]] = {}
        missing = []
        for i in range(len(hf_dataset)):
            key = (int(ep_indices[i]), int(frame_indices[i]))
            if key not in advantages_lookup:
                missing.append(key)
                continue
            adv, weight, is_demo = _coerce_record(advantages_lookup[key])
            bool_index[i] = adv
            record_index[i] = (adv, weight, is_demo)
        if missing:
            raise ValueError(
                f"[PT4FMAdvantagePreservingDataset] {len(missing)} samples missing "
                f"from advantages lookup (first 5: {missing[:5]}). Re-run stage-3 "
                f"compute_advantages."
            )
        self._record_by_index = record_index
        return bool_index

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = super().__getitem__(idx)  # sets sample["advantage"]
        if self._record_by_index is not None and idx in self._record_by_index:
            adv, weight, is_demo = self._record_by_index[idx]
            sample["advantage"] = adv
            sample["advantage_weight"] = weight
            sample["is_demo"] = is_demo
        else:
            # Parent-path fallback (column-based / per-sample): keep advantage,
            # default the new fields.
            sample.setdefault("advantage_weight", 1.0)
            sample.setdefault("is_demo", self._default_is_demo)
        return sample


class PT4FMCFGDataLoaderImpl(CFGDataLoaderImpl):
    """Yield ``(observation, actions, advantage[B,3])`` for PT4FM training.

    The packed tensor is ``[advantage_bool, advantage_weight, is_demo]`` (all as
    float), decoded back in :meth:`PT4FMCfgActionModel._unpack_advantage`.
    """

    def __iter__(self) -> Iterator[tuple[Any, Any, torch.Tensor]]:
        for batch in self._data_loader:
            observation = CFGObservation.from_dict(batch)
            actions = batch["actions"]

            advantage = _as_float_vec(batch["advantage"], dtype=torch.float32)
            weight = _as_float_vec(
                batch.get("advantage_weight", torch.ones_like(advantage)),
                dtype=torch.float32,
            )
            is_demo = _as_float_vec(
                batch.get("is_demo", torch.ones_like(advantage)),
                dtype=torch.float32,
            )
            packed = torch.stack([advantage, weight, is_demo], dim=1)  # [B, 3]
            yield observation, actions, packed


def _as_float_vec(x: Any, dtype=torch.float32) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    return x.reshape(-1).to(dtype)
