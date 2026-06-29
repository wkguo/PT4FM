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

"""PT4FM value/critic model: RECAP categorical value + optional IQL expectile.

``PT4FMValueCriticModel`` subclasses RLinf's ``ValueCriticModel`` and overrides
only ``_compute_categorical_loss`` to optionally re-weight the per-sample
cross-entropy by the IQL expectile factor ``|tau - 1{target<pred}|``. With
``expectile.enabled=False`` (or ``tau=0.5``) it is byte-for-byte RECAP.

Why expectile here? The RECAP value is a Monte-Carlo return regressor, i.e. it
estimates the *behaviour-policy* value ``V^mu``. An upper expectile (``tau>0.5``)
biases the estimate toward the better-than-average returns that pass through
similar observations — an "optimistic-within-support" value that yields a
sharper, more discriminative advantage in stage-3. Cal-QL style calibration is
applied later (stage-3 advantage bootstrap), since for a pure-MC target the
``max(target, MC_return)`` lower bound is a no-op at value-training time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlinf.models.embodiment.value_model.modeling_critic import (
    ValueCriticModel,
    prepare_target_values,
)

from pt4fm.losses import expectile_weight


class PT4FMValueCriticModel(ValueCriticModel):
    """RECAP value model + optional IQL expectile re-weighting."""

    # Set by :func:`get_model` after construction (defaults => RECAP).
    _pt4fm_expectile_enabled: bool = False
    _pt4fm_expectile_tau: float = 0.5

    def set_expectile(self, enabled: bool, tau: float) -> None:
        self._pt4fm_expectile_enabled = bool(enabled)
        self._pt4fm_expectile_tau = float(tau)

    def _compute_categorical_loss(self, logits, target_values):
        loss, metrics = super()._compute_categorical_loss(logits, target_values)
        if not self._pt4fm_expectile_enabled or abs(self._pt4fm_expectile_tau - 0.5) < 1e-9:
            return loss, metrics

        # Scalar prediction = sum_k softmax(logits)_k * atom_k.
        atoms = self.value_head.atoms.to(logits.device, dtype=torch.float32)
        pred_value = (F.softmax(logits.float(), dim=-1) * atoms).sum(dim=-1)
        tv = prepare_target_values(
            target_values, v_min=self.v_min, v_max=self.v_max,
            expected_batch_size=logits.shape[0],
        )
        diff = tv - pred_value  # target - pred
        w = expectile_weight(diff, self._pt4fm_expectile_tau).to(loss.dtype)
        metrics = {**metrics, "expectile_weight_mean": w.mean().detach()}
        return loss * w, metrics


def get_model(cfg, torch_dtype=None):
    """Build a :class:`PT4FMValueCriticModel`, reusing RLinf's value loader.

    Rebinds ``ValueCriticModel`` in RLinf's value-model package for the duration
    of the build (the loader looks the class up as a module global), then injects
    the expectile knobs from ``cfg.expectile`` onto the constructed model.
    """
    import rlinf.models.embodiment.value_model as vm

    saved = vm.ValueCriticModel
    vm.ValueCriticModel = PT4FMValueCriticModel
    try:
        model = vm.get_model(cfg, torch_dtype)
    finally:
        vm.ValueCriticModel = saved

    exp = getattr(cfg, "expectile", None)
    if exp is not None and isinstance(model, PT4FMValueCriticModel):
        enabled = bool(getattr(exp, "enabled", False))
        tau = float(getattr(exp, "tau", 0.5))
        model.set_expectile(enabled=enabled, tau=tau)
    return model
