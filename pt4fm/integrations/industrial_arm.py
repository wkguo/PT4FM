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

"""openpi data transform + config for the *Industrial_Arm* 3-camera embodiment.

Dataset schema (LeRobot v2.1, see ``meta/info.json``):
  * ``action``                   float32 [7]  = (x, y, z, rx, ry, rz, gripper)  -- absolute EEF pose + gripper
  * ``observation.state``        float32 [13] = pose(7) + force/torque(6: fx,fy,fz,tx,ty,tz)
  * ``observation.images.hand``  video 480x640x3  (wrist camera)
  * ``observation.images.view1`` video 480x640x3  (primary external)
  * ``observation.images.view2`` video 480x640x3  (secondary external)
  * prompt: from task (``prompt_from_task=True``)

This mirrors RLinf's ``realworld_policy`` / ``libero_policy`` templates. pi0.5 has
three image slots (``base_0_rgb``, ``left_wrist_0_rgb``, ``right_wrist_0_rgb``);
we map all three real cameras into them (all masks True):
    base_0_rgb        <- view1   (primary external)
    left_wrist_0_rgb  <- hand    (wrist)
    right_wrist_0_rgb <- view2   (secondary external)
State (13) and the action chunk (7) are padded to the model action dim (32).

Actions are **absolute** Cartesian pose (not deltas), so no DeltaActions transform
by default. Set ``extra_delta_transform=True`` to convert the first 6 dims to
deltas relative to the chunk's first state (gripper stays absolute) if your
checkpoint expects that.

Register the ``pi05_industrial_arm`` config (so ``config_name`` resolves through
RLinf's ``get_openpi_config``) by importing this module and calling
:func:`register_industrial_arm_configs` — PT4FM's CFG worker does this
automatically.
"""

from __future__ import annotations

import dataclasses
import pathlib

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    image = np.squeeze(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:  # CHW -> HWC
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# --------------------------------------------------------------------------- #
# data <-> model transforms (applied in training AND inference)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class IndustrialArmOutputs(transforms.DataTransformFn):
    """Model action space -> dataset action space (first 7 dims)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class IndustrialArmInputs(transforms.DataTransformFn):
    """Industrial_Arm dataset sample -> openpi model inputs."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    state_dim: int = 13

    def __call__(self, data: dict) -> dict:
        state = data["observation/state"]
        assert np.asarray(state).shape[-1] == self.state_dim, (
            f"Expected state dim {self.state_dim}, got {np.asarray(state).shape}"
        )
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = transforms.pad_to_dim(state, self.action_dim)

        base_image = _parse_image(data["observation/image"])          # view1
        wrist_image = _parse_image(data["observation/wrist_image"])    # hand
        extra_image = _parse_image(data["observation/extra_view_image"])  # view2

        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        images = (base_image, wrist_image, extra_image)
        image_masks = (np.True_, np.True_, np.True_)  # all three cameras are real

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            actions = data["actions"]
            assert np.asarray(actions).shape[-1] == 7, (
                f"Expected action dim 7, got {np.asarray(actions).shape}"
            )
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


# --------------------------------------------------------------------------- #
# DataConfigFactory (repack + transforms + model transforms)
# --------------------------------------------------------------------------- #
def make_industrial_arm_dataconfig_cls():
    """Build the DataConfigFactory subclass lazily (needs openpi training imports)."""
    from openpi.training.config import (
        DataConfig,
        DataConfigFactory,
        ModelTransformFactory,
    )
    import openpi.models.model as _m
    import openpi.transforms as _transforms
    from typing_extensions import override

    @dataclasses.dataclass(frozen=True)
    class LeRobotIndustrialArmDataConfig(DataConfigFactory):
        """LeRobot data config for the Industrial_Arm 3-cam embodiment."""

        # Dataset camera keys -> intermediate ("observation/*") keys.
        base_image_key: str = "observation.images.view1"
        wrist_image_key: str = "observation.images.hand"
        extra_image_key: str = "observation.images.view2"
        state_key: str = "observation.state"
        action_key: str = "action"
        default_prompt: str | None = None
        extra_delta_transform: bool = False

        @override
        def create(self, assets_dirs: pathlib.Path, model_config: _m.BaseModelConfig) -> DataConfig:
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/image": self.base_image_key,
                            "observation/wrist_image": self.wrist_image_key,
                            "observation/extra_view_image": self.extra_image_key,
                            "observation/state": self.state_key,
                            "actions": self.action_key,
                            "prompt": "prompt",
                        }
                    )
                ]
            )

            data_transforms = _transforms.Group(
                inputs=[
                    IndustrialArmInputs(
                        action_dim=model_config.action_dim,
                        model_type=model_config.model_type,
                    )
                ],
                outputs=[IndustrialArmOutputs()],
            )

            if self.extra_delta_transform:
                # First 6 dims (xyz + rxryrz) -> delta; gripper (dim 7) stays absolute.
                delta_mask = _transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_mask)],
                )

            model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
                model_config
            )

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
            )

    return LeRobotIndustrialArmDataConfig


# --------------------------------------------------------------------------- #
# config registration into RLinf's get_openpi_config registry
# --------------------------------------------------------------------------- #
CONFIG_NAME = "pi05_industrial_arm"


def register_industrial_arm_configs(repo_id: str | None = None, action_horizon: int = 10) -> str:
    """Inject ``pi05_industrial_arm`` into RLinf's ``_CONFIGS_DICT``. Idempotent.

    ``repo_id`` may be a HF id or a local LeRobot path; it is also used as the
    ``asset_id`` under which openpi looks for norm_stats
    (``<model_path>/assets/<asset_id>/norm_stats.json``). At model-build time the
    CFG worker passes the real ``repo_id``/``model_path`` overrides, so this just
    needs sane defaults.
    """
    import rlinf.models.embodiment.openpi.dataconfig as dc
    import openpi.models.pi0_config as pi0_config
    from openpi.training.config import AssetsConfig, DataConfig, TrainConfig

    if CONFIG_NAME in dc._CONFIGS_DICT:
        return CONFIG_NAME

    rid = repo_id or "industrial_arm"
    DataCfgCls = make_industrial_arm_dataconfig_cls()
    cfg = TrainConfig(
        name=CONFIG_NAME,
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=action_horizon, discrete_state_input=False
        ),
        data=DataCfgCls(
            repo_id=rid,
            assets=AssetsConfig(asset_id=rid),
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),  # dataset stores actions under "action"
            ),
        ),
    )
    dc._CONFIGS_DICT[CONFIG_NAME] = cfg
    dc._CONFIGS.append(cfg)
    return CONFIG_NAME
