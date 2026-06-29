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

"""PT4FM model builders (lazy imports to avoid pulling torch/rlinf eagerly)."""


def get_cfg_model(cfg, torch_dtype=None):
    """Build the PT4FM CFG policy model, reusing RLinf's loader verbatim.

    See :func:`pt4fm.models.cfg_action_model.get_model`.
    """
    from pt4fm.models.cfg_action_model import get_model

    return get_model(cfg, torch_dtype)


def get_value_model(cfg, torch_dtype=None):
    """Build the PT4FM value/critic model (expectile/calibration-capable)."""
    from pt4fm.models.value_critic import get_model

    return get_model(cfg, torch_dtype)
