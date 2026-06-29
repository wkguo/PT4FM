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

"""PT4FM value-model SFT worker (RECAP value worker + optional expectile).

Only ``model_provider_func`` is overridden — the expectile re-weighting lives
inside :class:`pt4fm.models.value_critic.PT4FMValueCriticModel`, so the data
pipeline, optimizer, and Spearman eval are inherited unchanged.
"""

from __future__ import annotations

import torch

from pt4fm.compat import install_lerobot_common_shim

install_lerobot_common_shim()  # must precede any rlinf.recap import

from rlinf.workers.sft.fsdp_value_sft_worker import FSDPValueSftWorker


class PT4FMValueWorker(FSDPValueSftWorker):
    """RECAP value worker + PT4FM (expectile-capable) critic."""

    def model_provider_func(self) -> torch.nn.Module:
        from pt4fm.models import get_value_model

        return get_value_model(self.cfg.actor.model)
